"""Automatic evaluation against Re-DocRED gold (restricted to the schema's relations).
Automatic precision is a LOWER bound (gold is incomplete, entity matching is heuristic);
the manual study is the primary metric."""
import csv
import pathlib
import random

from .config import rpath
from .data import gold_triples, load_split
from .extract import load_raw
from .judge import Judge
from .matching import map_entities
from .pipeline import ALL_FLAGS, MAIN, DocRun, config_name
from .schema import load_schema
from .stats import mean_std

COUNT_KEYS = ["edges", "correct_edges", "tp_unique", "gold", "raw_total", "raw_correct", "kept_raw",
              "correct_removed", "parse_fail", "parse_recovered"]


def node_gold(node, ent_map):
    golds = {ent_map.get(m) for m in node["members"]} - {None}
    return next(iter(golds)) if len(golds) == 1 else None


def eval_doc(run, result, ent_map, gold, raw_correct_ids):
    correct_edges, tp = 0, set()
    for e in result["edges"]:
        hn, tn = result["nodes"].get(e["head"]), result["nodes"].get(e["tail"])
        gh = node_gold(hn, ent_map) if hn else None
        gt = node_gold(tn, ent_map) if tn else None
        if gh is not None and gt is not None and (gh, e["rel"], gt) in gold:
            correct_edges += 1
            tp.add((gh, e["rel"], gt))
    kept = result["kept"]
    return {"edges": len(result["edges"]), "correct_edges": correct_edges, "tp_unique": len(tp),
            "gold": len(gold), "raw_total": len(run.triples), "raw_correct": len(raw_correct_ids),
            "kept_raw": len(kept), "correct_removed": len([r for r in raw_correct_ids if r not in kept]),
            "parse_fail": 0}


def raw_correct(run, ent_map, gold):
    out = set()
    for t in run.triples:
        gh, gt = ent_map.get(t["head"]), ent_map.get(t["tail"])
        if gh is not None and gt is not None and (gh, t["relation"], gt) in gold:
            out.add(t["raw_id"])
    return out


def sample_docs(pool, n, seed):
    """Nested by construction: sample_docs(pool, 25, s) is a prefix of sample_docs(pool, 50, s)."""
    perm = sorted(pool)
    random.Random(seed).shuffle(perm)
    return perm[:n]


def aggregate(counts_list):
    s = {k: sum(c[k] for c in counts_list) for k in COUNT_KEYS}
    E = s["edges"]
    prec = s["correct_edges"] / E if E else float("nan")
    uprec = s["tp_unique"] / E if E else float("nan")
    rec = s["tp_unique"] / s["gold"] if s["gold"] else float("nan")
    f1 = 2 * uprec * rec / (uprec + rec) if E and (uprec + rec) > 0 else float("nan")
    ret = s["kept_raw"] / s["raw_total"] if s["raw_total"] else float("nan")
    crem = s["correct_removed"] / s["raw_correct"] if s["raw_correct"] else float("nan")
    return {**s, "precision": prec, "unique_precision": uprec, "recall": rec, "f1": f1,
            "retention_raw": ret, "correct_removed_share": crem}


def evaluate_all(cfg, raw_path, split="dev", configs="all", log=print, use_judge=True, results_dir=None):
    if split == "test":
        from .freeze import assert_test_frozen
        assert_test_frozen(cfg)
    schema = load_schema(cfg)
    docs = load_split(rpath(cfg, cfg["paths"][split]))
    raw = load_raw(raw_path)
    pool = sorted(raw)
    judge = Judge(cfg, schema) if use_judge and cfg["judge"].get("enabled", True) else None
    flags = ALL_FLAGS if configs == "all" else [f for f in ALL_FLAGS if config_name(*f) in MAIN]
    rels = set(schema["relations"])
    per_doc = {}
    for n, doc_id in enumerate(pool, 1):
        doc, rec = docs[doc_id], raw[doc_id]
        run = DocRun(doc, rec["parsed"], schema, cfg, judge)
        ent_map = map_entities(doc, run.entities, cfg.get("matching", {}))
        gold = gold_triples(doc, rels)
        rc = raw_correct(run, ent_map, gold)
        for f in flags:
            res = run.run(*f)
            c = eval_doc(run, res, ent_map, gold, rc)
            c["parse_fail"] = int(rec["parsed"] is None)
            c["parse_recovered"] = int(rec.get("parse_error") == "partial_json_recovered")
            per_doc[(doc_id, config_name(*f))] = c
        if n % 25 == 0:
            log(f"  evaluated {n}/{len(pool)} docs" + (f" (judge calls so far: {judge.calls})" if judge else ""))
    res_dir = pathlib.Path(results_dir) if results_dir else pathlib.Path(rpath(cfg, cfg["paths"]["results_dir"]))
    res_dir.mkdir(parents=True, exist_ok=True)
    names = [config_name(*f) for f in flags]
    rows = []
    for seed in cfg["eval"]["seeds"]:
        for n in cfg["eval"]["sizes"]:
            if n > len(pool):
                log(f"  skip n={n}: only {len(pool)} documents have raw responses")
                continue
            ids = sample_docs(pool, n, seed)
            for name in names:
                rows.append({"config": name, "n_docs": n, "seed": seed,
                             **aggregate([per_doc[(i, name)] for i in ids])})
    _write_csv(res_dir / "auto_metrics.csv", rows)
    summ = []
    for name in names:
        for n in cfg["eval"]["sizes"]:
            rs = [r for r in rows if r["config"] == name and r["n_docs"] == n]
            if not rs:
                continue
            d = {"config": name, "n_docs": n, "n_seeds": len(rs)}
            for k in ["precision", "unique_precision", "recall", "f1", "retention_raw", "correct_removed_share"]:
                m, sd = mean_std([r[k] for r in rs])
                d[k + "_mean"], d[k + "_sd"] = m, sd
            summ.append(d)
    _write_csv(res_dir / "auto_summary.csv", summ)
    _write_csv(res_dir / "per_doc.csv", [{"doc_id": i, "config": c, **v} for (i, c), v in sorted(per_doc.items())])
    if judge:
        log(f"judge: {judge.calls} new calls, tokens {judge.tokens}")
    return rows, summ


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def print_primary(cfg, rows, log=print):
    p = cfg["eval"]
    rs = [r for r in rows if r["seed"] == p["primary_seed"] and r["n_docs"] == p["primary_n"]]
    if not rs:
        return
    log(f"\nPrimary run: n={p['primary_n']} docs, seed={p['primary_seed']} (automatic metrics, lower bounds)")
    log(f"{'config':8} {'edges':>6} {'prec':>6} {'uprec':>6} {'recall':>6} {'F1':>6} {'retain':>6} {'corr.removed':>12}")
    for r in rs:
        log(f"{r['config']:8} {r['edges']:6d} {r['precision']:6.3f} {r['unique_precision']:6.3f} "
            f"{r['recall']:6.3f} {r['f1']:6.3f} {r['retention_raw']:6.3f} {r['correct_removed_share']:12.3f}")
