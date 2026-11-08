"""Manual-evaluation workflow.
Design (paired): sample RAW triples once, label each once, then see which stages keep it.
So precision differences between stages are paired, and 'correct raw facts removed' is direct."""
import csv
import hashlib
import json
import pathlib
import random

from .config import rpath
from .data import doc_to_prompt_text, load_split, sentences
from .evaluate import sample_docs
from .extract import load_raw
from .judge import Judge
from .pipeline import ALL_FLAGS, MAIN, DocRun, config_name
from .schema import load_schema
from .stats import bootstrap_ci, cohen_kappa, wilson
from .textutil import norm_tokens

COLS = ["raw_id", "doc_id", "title", "sent_id", "head", "head_type", "relation", "relation_name",
        "tail", "tail_type", "quote", "cited_sentence", "label", "notes"]
VALID = {"correct", "incorrect", "uncertain"}


def sample_annotation(cfg, raw_path, split="dev", log=print, n_sample=None, overlap_frac=None,
                      use_judge=True, out_dir=None, exclude_key=None):
    schema = load_schema(cfg)
    docs = load_split(rpath(cfg, cfg["paths"][split]))
    raw = load_raw(raw_path)
    e = cfg["eval"]
    ids = sample_docs(sorted(raw), e["primary_n"], e["primary_seed"])
    if len(ids) < e["primary_n"]:
        log(f"WARNING: only {len(ids)} documents available (primary_n={e['primary_n']})")
    judge = Judge(cfg, schema) if use_judge and cfg["judge"].get("enabled", True) else None
    all_triples, kept = {}, {}
    for i in ids:
        run = DocRun(docs[i], raw[i]["parsed"], schema, cfg, judge)
        sn = sentences(docs[i])
        for t in run.triples:
            hn = run.ent_by_id.get(t["head"], {})
            tn = run.ent_by_id.get(t["tail"], {})
            sid = t["sent_id"]
            all_triples[t["raw_id"]] = {
                "raw_id": t["raw_id"], "doc_id": i, "title": docs[i]["title"], "sent_id": sid,
                "head": hn.get("name", f"<missing {t['head']}>"), "head_type": hn.get("type", ""),
                "relation": t["relation"],
                "relation_name": schema["relations"].get(t["relation"], {}).get("name", "(not in schema)"),
                "tail": tn.get("name", f"<missing {t['tail']}>"), "tail_type": tn.get("type", ""),
                "quote": t["quote"],
                "cited_sentence": sn[sid] if isinstance(sid, int) and 0 <= sid < len(sn) else "",
                "label": "", "notes": ""}
            kept[t["raw_id"]] = {}
        for f in ALL_FLAGS:
            res = run.run(*f)
            for rid in kept:
                if rid.split(":")[0] == str(i):
                    kept[rid][config_name(*f)] = res["kept"].get(rid)
    a = cfg["annotation"]
    sample_size = a["n_sample"] if n_sample is None else n_sample
    overlap_share = a["overlap_frac"] if overlap_frac is None else overlap_frac
    if sample_size < 1:
        raise ValueError("annotation sample size must be at least 1")
    if not 0 <= overlap_share <= 1:
        raise ValueError("annotation overlap fraction must be between 0 and 1")
    rng = random.Random(a["seed"])
    pool = sorted(all_triples)
    if exclude_key:
        with open(exclude_key, encoding="utf-8") as f:
            excluded = set(json.load(f)["sampled"])
        pool = [rid for rid in pool if rid not in excluded]
    n = min(sample_size, len(pool))
    sample = rng.sample(pool, n)
    overlap = rng.sample(sample, round(overlap_share * n)) if n else []
    out = pathlib.Path(out_dir) if out_dir else pathlib.Path(rpath(cfg, cfg["paths"]["annotation_dir"]))
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "annotator1.csv", [all_triples[r] for r in sample])
    _write(out / "annotator2.csv", [all_triples[r] for r in overlap])
    _write(out / "annotator3.csv", [all_triples[r] for r in overlap])
    with open(out / "docs.txt", "w", encoding="utf-8") as f:
        for i in sorted({all_triples[r]["doc_id"] for r in sample}):
            f.write(f"=== doc {i}: {docs[i]['title']} ===\n{doc_to_prompt_text(docs[i])}\n\n")
    key = {"primary_n": e["primary_n"], "primary_seed": e["primary_seed"], "sampled": sample,
           "overlap": overlap, "kept": {r: kept[r] for r in sample},
           "configs": [config_name(*f) for f in ALL_FLAGS], "n_raw_pool": len(pool)}
    with open(out / "key.json", "w", encoding="utf-8") as f:
        json.dump(key, f)
    log(f"sampled {n} of {len(pool)} raw triples from {len(ids)} docs; "
        f"{len(overlap)} shared with annotators 2 and 3.\n"
        f"Files in {out}: annotator1.csv, annotator2.csv, annotator3.csv, docs.txt, "
        "key.json (do NOT share key.json with annotators).")
    if judge is None:
        log("P-related stages use quote validation only; LLM judge disabled.")


def _proposal_identity(doc_id, edge, nodes):
    head = nodes.get(edge["head"], {}).get("name", edge["head"])
    tail = nodes.get(edge["tail"], {}).get("name", edge["tail"])
    return (doc_id, edge["rel"], norm_tokens(head), norm_tokens(tail))


def _balanced_stage_sample(candidates, n, seeds, rng):
    if len(candidates) < n:
        raise ValueError(f"proposal sample needs {n} unique edges per stage; found {len(candidates)}")
    ordered = list(candidates.values())
    rng.shuffle(ordered)
    assigned = {seed: [] for seed in seeds}
    for candidate in ordered:
        eligible = sorted(candidate["seeds"])
        if not eligible:
            continue
        owner = min(eligible, key=lambda seed: (len(assigned[seed]), seeds.index(seed)))
        assigned[owner].append(candidate)
    quota, remainder = divmod(n, len(seeds))
    chosen = []
    chosen_ids = set()
    for idx, seed in enumerate(seeds):
        target = quota + (1 if idx < remainder else 0)
        group = assigned[seed]
        take = min(target, len(group))
        chosen.extend(group[:take])
        chosen_ids.update(c["identity"] for c in group[:take])
    if len(chosen) < n:
        for candidate in ordered:
            if candidate["identity"] not in chosen_ids:
                chosen.append(candidate)
                chosen_ids.add(candidate["identity"])
                if len(chosen) == n:
                    break
    return chosen


def sample_proposal_annotation(cfg, raw_path, split="test", log=print, n_per_stage=None,
                               overlap_frac=None, use_judge=True, out_dir=None):
    """Create the proposal's stage-balanced manual sample from five seeded 200-doc test runs."""
    if split != "test":
        raise ValueError("the proposal's final manual annotation sample must use the held-out test split")
    from .freeze import assert_test_frozen
    assert_test_frozen(cfg)
    schema = load_schema(cfg)
    docs = load_split(rpath(cfg, cfg["paths"][split]))
    raw = load_raw(raw_path)
    eval_cfg = cfg["eval"]
    seeds = list(eval_cfg["seeds"])
    if len(seeds) != 5:
        raise ValueError(f"proposal annotation requires five sampling seeds; configured {len(seeds)}")
    n_docs = eval_cfg["primary_n"]
    pool = sorted(raw)
    if n_docs > len(pool):
        raise ValueError(f"proposal annotation requires {n_docs} test documents per seed; found {len(pool)}")
    docs_by_seed = {seed: sample_docs(pool, n_docs, seed) for seed in seeds}
    seeds_by_doc = {}
    for seed, doc_ids in docs_by_seed.items():
        for doc_id in doc_ids:
            seeds_by_doc.setdefault(doc_id, set()).add(seed)

    sample_size = cfg["annotation"]["n_sample"] if n_per_stage is None else n_per_stage
    overlap_share = cfg["annotation"]["overlap_frac"] if overlap_frac is None else overlap_frac
    if sample_size < 1:
        raise ValueError("proposal stage sample size must be at least 1")
    if not 0 <= overlap_share <= 1:
        raise ValueError("annotation overlap fraction must be between 0 and 1")

    judge = Judge(cfg, schema) if use_judge and cfg["judge"].get("enabled", True) else None
    triples_by_doc, stage_pools, kept_by_doc = {}, {name: {} for name in MAIN}, {}
    edge_counts_by_seed = {seed: {name: 0 for name in MAIN} for seed in seeds}
    for doc_id in sorted(seeds_by_doc):
        doc, rec = docs[doc_id], raw[doc_id]
        run = DocRun(doc, rec["parsed"], schema, cfg, judge)
        triples_by_doc[doc_id] = {t["raw_id"]: t for t in run.triples}
        kept_by_doc[doc_id] = {}
        for name in MAIN:
            c, s, p = {"raw": (0, 0, 0), "C": (1, 0, 0), "C+S": (1, 1, 0),
                       "C+S+P": (1, 1, 1)}[name]
            result = run.run(c, s, p)
            for seed in seeds_by_doc[doc_id]:
                edge_counts_by_seed[seed][name] += len(result["edges"])
            kept_by_doc[doc_id][name] = result["kept"]
            candidates = stage_pools[name]
            for edge in result["edges"]:
                identity = _proposal_identity(doc_id, edge, result["nodes"])
                candidate = candidates.setdefault(identity, {
                    "identity": identity, "doc_id": doc_id, "title": doc["title"],
                    "head": result["nodes"].get(edge["head"], {}).get("name", edge["head"]),
                    "head_type": ", ".join(sorted(result["nodes"].get(edge["head"], {}).get("types", set()))),
                    "relation": edge["rel"],
                    "relation_name": schema["relations"].get(edge["rel"], {}).get("name", edge["rel"]),
                    "tail": result["nodes"].get(edge["tail"], {}).get("name", edge["tail"]),
                    "tail_type": ", ".join(sorted(result["nodes"].get(edge["tail"], {}).get("types", set()))),
                    "source_raw_ids": set(), "seeds": set(), "quotes": [],
                })
                candidate["source_raw_ids"].update(edge["src"])
                candidate["seeds"].update(seeds_by_doc[doc_id])
                for rid in edge["src"]:
                    triple = triples_by_doc[doc_id].get(rid)
                    if triple and triple.get("quote"):
                        candidate["quotes"].append((triple.get("quote", ""), triple.get("sent_id")))

    rng = random.Random(cfg["annotation"]["seed"])
    stage_samples = {
        stage: _balanced_stage_sample(stage_pools[stage], sample_size, seeds,
                                      random.Random(rng.randrange(2**32)))
        for stage in MAIN
    }

    selected = [(stage, candidate) for stage, candidates in stage_samples.items()
                for candidate in candidates]
    parent = list(range(len(selected)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    first_by_identity, first_by_source = {}, {}
    for i, (_, candidate) in enumerate(selected):
        identity = candidate["identity"]
        if identity in first_by_identity:
            union(i, first_by_identity[identity])
        else:
            first_by_identity[identity] = i
        for rid in candidate["source_raw_ids"]:
            if rid in first_by_source:
                union(i, first_by_source[rid])
            else:
                first_by_source[rid] = i
    groups = {}
    for i, (stage, candidate) in enumerate(selected):
        group = groups.setdefault(find(i), {"candidates": [], "sampled_for_stages": set(),
                                             "source_raw_ids": set(), "seeds": set()})
        group["candidates"].append(candidate)
        group["sampled_for_stages"].add(stage)
        group["source_raw_ids"].update(candidate["source_raw_ids"])
        group["seeds"].update(candidate["seeds"])

    items, key_items = [], []
    for group in groups.values():
        candidates = group["candidates"]
        preferred = max(candidates, key=lambda c: len(c["head"]) + len(c["tail"]))
        source_ids = sorted(group["source_raw_ids"])
        item_material = json.dumps([preferred["identity"], source_ids], ensure_ascii=False)
        item_id = hashlib.sha256(item_material.encode("utf-8")).hexdigest()[:16]
        quote_rows = [q for c in candidates for q in c["quotes"] if q[0]]
        quote, sent_id = max(quote_rows, key=lambda q: len(q[0])) if quote_rows else ("", None)
        cited_sentence = ""
        if isinstance(sent_id, int):
            sn = sentences(docs[preferred["doc_id"]])
            if 0 <= sent_id < len(sn):
                cited_sentence = sn[sent_id]
        kept_by_stage = {
            stage: any(rid in kept_by_doc[preferred["doc_id"]][stage] for rid in source_ids)
            for stage in MAIN
        }
        row = {
            "item_id": item_id, "doc_id": preferred["doc_id"], "title": preferred["title"],
            "head": preferred["head"], "head_type": preferred["head_type"],
            "relation": preferred["relation"], "relation_name": preferred["relation_name"],
            "tail": preferred["tail"], "tail_type": preferred["tail_type"],
            "quote": quote, "cited_sentence": cited_sentence, "label": "", "notes": "",
        }
        items.append(row)
        key_items.append({
            "item_id": item_id, "sampled_for_stages": sorted(group["sampled_for_stages"],
                                                               key=MAIN.index),
            "source_raw_ids": source_ids, "seeds": sorted(group["seeds"]),
            "kept_by_stage": kept_by_stage,
        })

    n_overlap = round(overlap_share * len(items))
    overlap_ids = set(rng.sample([row["item_id"] for row in items], n_overlap)) if n_overlap else set()
    out = pathlib.Path(out_dir) if out_dir else pathlib.Path(rpath(cfg, cfg["paths"]["annotation_dir"])) / "proposal_test"
    out.mkdir(parents=True, exist_ok=True)
    _write_proposal(out / "annotator1.csv", items)
    _write_proposal(out / "annotator2.csv", [row for row in items if row["item_id"] in overlap_ids])
    with open(out / "docs.txt", "w", encoding="utf-8") as f:
        for doc_id in sorted({row["doc_id"] for row in items}):
            f.write(f"=== doc {doc_id}: {docs[doc_id]['title']} ===\n{doc_to_prompt_text(docs[doc_id])}\n\n")

    total_edges = {stage: sum(edge_counts_by_seed[seed][stage] for seed in seeds) for stage in MAIN}
    raw_total = total_edges["raw"]
    key = {
        "protocol": "DKR_proposal-1.pdf: 100 unique accepted triples per main stage, five 200-doc seeds, 20% second annotator",
        "split": split, "primary_n": n_docs, "seeds": seeds,
        "sample_size_per_stage": sample_size, "stage_sample_sizes": {stage: len(stage_samples[stage]) for stage in MAIN},
        "stage_pool_sizes_unique": {stage: len(stage_pools[stage]) for stage in MAIN},
        "sampled_candidate_identities": {stage: [list(c["identity"]) for c in sample]
                                         for stage, sample in stage_samples.items()},
        "items": key_items, "overlap_item_ids": sorted(overlap_ids),
        "automatic_edge_counts_across_seed_runs": total_edges,
        "automatic_retention_across_seed_runs": {
            stage: (total_edges[stage] / raw_total if raw_total else None) for stage in MAIN
        },
        "judge_enabled": judge is not None,
    }
    with open(out / "key.json", "w", encoding="utf-8") as f:
        json.dump(key, f, ensure_ascii=False, indent=2)
    log(f"Proposal sample: {len(items)} unique annotation items after cross-stage deduplication.")
    log(f"Stage samples: {key['stage_sample_sizes']}; unique stage pools: {key['stage_pool_sizes_unique']}")
    log(f"Second annotator overlap: {len(overlap_ids)} items ({overlap_share:.0%} requested).")
    log(f"Automatic retention across five seeded test runs: {key['automatic_retention_across_seed_runs']}")
    log(f"Files in {out}: annotator1.csv, annotator2.csv, docs.txt, key.json (do NOT share key.json).")
    if judge is None:
        log("P uses quote validation only; LLM judge disabled.")


def _write_proposal(path, rows):
    cols = ["item_id", "doc_id", "title", "head", "head_type", "relation", "relation_name",
            "tail", "tail_type", "quote", "cited_sentence", "label", "notes"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def score_proposal_annotation(cfg, sample_dir, labels_path, labels2_path, log=print):
    sample_dir = pathlib.Path(sample_dir)
    with open(sample_dir / "key.json", encoding="utf-8") as f:
        key = json.load(f)

    def read(path):
        result = {}
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                label = (row.get("label") or "").strip().lower()
                if label and label not in VALID:
                    raise ValueError(f"{path}: item_id {row.get('item_id')} has invalid label {label!r}")
                result[row["item_id"]] = label
        return result

    labels = read(labels_path)
    expected = {item["item_id"] for item in key["items"]}
    if expected - labels.keys():
        raise ValueError(f"annotator 1 is missing {len(expected - labels.keys())} proposal sample labels")
    labels2 = read(labels2_path) if labels2_path and pathlib.Path(labels2_path).exists() else {}
    expected_overlap = set(key["overlap_item_ids"])
    if expected_overlap - labels2.keys():
        raise ValueError(f"annotator 2 is missing {len(expected_overlap - labels2.keys())} overlap labels")
    if labels2.keys() - expected_overlap:
        raise ValueError("annotator 2 file contains items outside the assigned 20% overlap")

    policy = cfg["annotation"]["uncertain"]
    normalized = {item_id: ("incorrect" if label == "uncertain" and policy == "incorrect" else label)
                  for item_id, label in labels.items()}
    usable = {item_id for item_id, label in normalized.items() if label in ("correct", "incorrect")}
    by_id = {item["item_id"]: item for item in key["items"]}
    stage_rows = []
    for stage in MAIN:
        stage_items = [item for item in key["items"] if stage in item["sampled_for_stages"]]
        known = [item["item_id"] for item in stage_items if item["item_id"] in usable]
        correct = sum(normalized[item_id] == "correct" for item_id in known)
        p, lo, hi = wilson(correct, len(known))
        boot_lo, boot_hi = bootstrap_ci(
            known, lambda sample: sum(normalized[item_id] == "correct" for item_id in sample) / len(sample)
            if sample else None, cfg["annotation"]["bootstrap"], seed=MAIN.index(stage))
        stage_rows.append({
            "config": stage, "sampled_unique_edges": len(stage_items), "labeled_edges": len(known),
            "correct": correct, "precision": p, "wilson_lo": lo, "wilson_hi": hi,
            "boot_lo": boot_lo, "boot_hi": boot_hi,
            "automatic_retention_across_seed_runs": key["automatic_retention_across_seed_runs"][stage],
            "sample_retention_vs_raw": None,
            "correct_raw_facts_removed_share": None,
        })
    raw_items = [item for item in key["items"]
                 if "raw" in item["sampled_for_stages"] and item["item_id"] in usable]
    for row in stage_rows:
        stage = row["config"]
        row["sample_retention_vs_raw"] = (
            sum(item["kept_by_stage"][stage] for item in raw_items) / len(raw_items)
            if raw_items else float("nan")
        )
        correct_raw = [item for item in raw_items if normalized[item["item_id"]] == "correct"]
        removed = sum(not item["kept_by_stage"][stage] for item in correct_raw)
        row["correct_raw_facts_removed_share"] = removed / len(correct_raw) if correct_raw else float("nan")
        log(f"{stage:8} n={row['labeled_edges']:3d}/{row['sampled_unique_edges']:3d} "
            f"precision={row['precision']:.3f} Wilson95=[{row['wilson_lo']:.3f},{row['wilson_hi']:.3f}] "
            f"bootstrap95=[{row['boot_lo']:.3f},{row['boot_hi']:.3f}] "
            f"auto_retention={row['automatic_retention_across_seed_runs']:.3f} "
            f"raw-sample-retention={row['sample_retention_vs_raw']:.3f} "
            f"correct-raw-removed={row['correct_raw_facts_removed_share']:.3f}")

    overlap_pairs = [(labels[item_id], labels2[item_id]) for item_id in expected_overlap
                     if labels.get(item_id) and labels2.get(item_id)]
    if overlap_pairs:
        a, b = zip(*overlap_pairs)
        agreement = sum(x == y for x, y in overlap_pairs) / len(overlap_pairs)
        log(f"Annotator agreement: n={len(overlap_pairs)}, raw={agreement:.1%}, "
            f"Cohen's kappa={cohen_kappa(list(a), list(b)):.3f}")
    else:
        log("Annotator agreement: no completed labels on overlap")
    log(f"Uncertain labels: {sum(label == 'uncertain' for label in labels.values())} "
        f"(policy: {policy})")
    from .evaluate import _write_csv
    results_dir = pathlib.Path(rpath(cfg, cfg["paths"]["results_dir"]))
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(results_dir / "proposal_manual_summary.csv", stage_rows)
    log(f"written: {results_dir / 'proposal_manual_summary.csv'}")
    return stage_rows


def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


def _read_labels(path):
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            lab = (r.get("label") or "").strip().lower()
            if lab and lab not in VALID:
                raise ValueError(f"{path}: raw_id {r['raw_id']} has invalid label {lab!r} (use {sorted(VALID)})")
            out[r["raw_id"]] = lab
    return out


def _prec(sample, kept, label):
    seen, k, n = set(), 0, 0
    for rid in sample:
        ek = kept.get(rid)
        if ek and ek not in seen:
            seen.add(ek)
            n += 1
            k += label[rid] == "correct"
    return k / n if n else None


def score_annotation(cfg, labels_path, labels2_path=None, labels3_path=None, log=print):
    out = pathlib.Path(rpath(cfg, cfg["paths"]["annotation_dir"]))
    key = json.load(open(out / "key.json", encoding="utf-8"))
    lab = _read_labels(labels_path)
    missing = [r for r in key["sampled"] if not lab.get(r)]
    if missing:
        raise ValueError(f"{len(missing)} sampled triples are unlabelled (e.g. {missing[:3]}). Label all rows first.")
    pol = cfg["annotation"]["uncertain"]
    n_unc = sum(lab[r] == "uncertain" for r in key["sampled"])
    label = {r: ("incorrect" if (lab[r] == "uncertain" and pol == "incorrect") else lab[r]) for r in key["sampled"]}
    usable = [r for r in key["sampled"] if label[r] in ("correct", "incorrect")]
    B = cfg["annotation"]["bootstrap"]
    rows = []
    log(f"{len(key['sampled'])} sampled raw triples; {n_unc} 'uncertain' (policy: {pol}); {len(usable)} used.\n")
    log(f"{'config':8} {'edges':>5} {'precision':>9} {'95% CI (Wilson)':>17} {'95% CI (boot)':>15} {'retention':>9} {'corr.removed':>12}")
    for name in key["configs"]:
        kept = {r: key["kept"][r].get(name) for r in key["sampled"]}
        edge_label, conflicts = {}, 0
        for r in usable:
            ek = kept[r]
            if ek:
                if ek in edge_label and edge_label[ek] != label[r]:
                    conflicts += 1
                edge_label.setdefault(ek, label[r])
        n_e, k_c = len(edge_label), sum(v == "correct" for v in edge_label.values())
        p, lo, hi = wilson(k_c, n_e)
        blo, bhi = bootstrap_ci(usable, lambda s, kept=kept: _prec(s, kept, label), B, seed=1)
        ret = sum(1 for r in key["sampled"] if kept[r]) / len(key["sampled"])
        corr = [r for r in usable if label[r] == "correct"]
        crem = sum(1 for r in corr if not kept[r]) / len(corr) if corr else float("nan")
        rows.append({"config": name, "unique_edges": n_e, "correct": k_c, "precision": p, "wilson_lo": lo,
                     "wilson_hi": hi, "boot_lo": blo, "boot_hi": bhi, "retention": ret,
                     "correct_removed_share": crem, "label_conflicts": conflicts})
        if name in MAIN:
            log(f"{name:8} {n_e:5d} {p:9.3f} [{lo:6.3f},{hi:6.3f}] [{blo:6.3f},{bhi:6.3f}] {ret:9.3f} {crem:12.3f}"
                + (f"  ({conflicts} label conflicts)" if conflicts else ""))
    log("\nPaired precision differences (bootstrap over sampled raw triples):")
    chain = [("raw", "C"), ("C", "C+S"), ("C+S", "C+S+P"), ("raw", "C+S+P")]
    for a, b in chain:
        ka = {r: key["kept"][r].get(a) for r in key["sampled"]}
        kb = {r: key["kept"][r].get(b) for r in key["sampled"]}
        fn = lambda s: (None if _prec(s, ka, label) is None or _prec(s, kb, label) is None
                        else _prec(s, kb, label) - _prec(s, ka, label))
        d = fn(usable)
        lo, hi = bootstrap_ci(usable, fn, B, seed=2)
        log(f"  {a:6} -> {b:7}: {d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    annotators = {"annotator1": lab}
    for name, path in (("annotator2", labels2_path), ("annotator3", labels3_path)):
        if path:
            annotators[name] = _read_labels(path)
    if len(annotators) > 1:
        log("\nInter-annotator agreement on double-labelled triples:")
        names = list(annotators)
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                pairs = [(annotators[first][r], annotators[second][r])
                         for r in key["overlap"]
                         if annotators[first].get(r) and annotators[second].get(r)]
                if not pairs:
                    log(f"  {first} vs {second}: no completed labels on the overlap")
                    continue
                a_, b_ = zip(*pairs)
                agree = sum(x == y for x, y in pairs) / len(pairs)
                log(f"  {first} vs {second}: n={len(pairs)}, {agree:.1%} raw agreement, "
                    f"Cohen's kappa = {cohen_kappa(list(a_), list(b_)):.3f}")
    from .evaluate import _write_csv
    res = pathlib.Path(rpath(cfg, cfg["paths"]["results_dir"]))
    res.mkdir(parents=True, exist_ok=True)
    _write_csv(res / "manual_summary.csv", rows)
    log(f"\nwritten: {res / 'manual_summary.csv'}")
    return rows
