"""Week-1 pilot report: how well does the model follow the format, and what do raw numbers look like?"""
from .config import rpath
from .data import gold_triples, load_split
from .evaluate import aggregate, eval_doc, raw_correct
from .extract import load_raw
from .matching import map_entities
from .pipeline import DocRun, FLAGS_BY_NAME
from .schema import load_schema


def pilot_report(cfg, raw_path, split, log=print):
    schema = load_schema(cfg)
    docs = load_split(rpath(cfg, cfg["paths"][split]))
    raw = load_raw(raw_path)
    n = len(raw)
    req_fail = sum(1 for r in raw.values() if r.get("error"))
    parse_fail = sum(1 for r in raw.values() if r["parsed"] is None and not r.get("error"))
    partial_recovered = sum(1 for r in raw.values() if r.get("parse_error") == "partial_json_recovered")
    ents = trips = in_schema = ids_ok = q_claimed = q_any = malformed = 0
    lat, comp, prompt = [], 0, 0
    counts = {name: [] for name in ("raw", "C", "C+S", "P", "C+S+P")}
    for i, r in sorted(raw.items()):
        malformed += r["malformed"]
        if r.get("latency"):
            lat.append(r["latency"])
        u = r.get("usage") or {}
        comp += u.get("completion_tokens") or 0
        prompt += u.get("prompt_tokens") or 0
        run = DocRun(docs[i], r["parsed"], schema, cfg, judge=None)
        ents += len(run.entities)
        for t in run.triples:
            trips += 1
            in_schema += t["relation"] in schema["relations"]
            ids_ok += t["head"] in run.ent_by_id and t["tail"] in run.ent_by_id
            from .repair import find_quote
            w, claimed = find_quote(t["quote"], run.sent_norms, run.doc_norm, t["sent_id"])
            q_any += w is not None
            q_claimed += claimed
        em = map_entities(docs[i], run.entities, cfg.get("matching", {}))
        gold = gold_triples(docs[i], set(schema["relations"]))
        raw_correct_ids = raw_correct(run, em, gold)
        for name in counts:
            c, s, p = FLAGS_BY_NAME[name]
            row = eval_doc(run, run.run(c, s, p), em, gold, raw_correct_ids)
            row["parse_fail"] = int(r["parsed"] is None)
            row["parse_recovered"] = int(r.get("parse_error") == "partial_json_recovered")
            counts[name].append(row)
    pct = lambda a, b: f"{100 * a / b:.1f}%" if b else "n/a"
    log(f"Pilot report: {n} documents ({split})")
    log(f"  request failures: {req_fail}   unrecoverable JSON/shape: {parse_fail}   "
        f"partial JSON recovered: {partial_recovered}   malformed items dropped: {malformed}")
    log(f"  mean entities/doc: {ents / max(n, 1):.1f}   mean triples/doc: {trips / max(n, 1):.1f}")
    log(f"  triples with relation in schema: {pct(in_schema, trips)}   with both entity ids defined: {pct(ids_ok, trips)}")
    log(f"  quote found verbatim in document: {pct(q_any, trips)}   in the cited sentence: {pct(q_claimed, trips)}")
    if lat:
        log(f"  mean latency/doc: {sum(lat) / len(lat):.1f}s   tokens/doc: prompt {prompt / n:.0f}, completion {comp / n:.0f}")
    log("  Automatic metrics (lower bounds; P is quote-rule only, LLM judge disabled):")
    rows = []
    for name, doc_counts in counts.items():
        a = aggregate(doc_counts)
        rows.append({"config": name, "n_docs": n, **a})
        log(f"    {name:7} precision {a['precision']:.3f}, unique precision {a['unique_precision']:.3f}, "
            f"recall {a['recall']:.3f}, F1 {a['f1']:.3f}, retention {a['retention_raw']:.3f}")
    result_dir = rpath(cfg, cfg["paths"]["results_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    from .evaluate import _write_csv
    _write_csv(result_dir / "pilot_metrics.csv", rows)
    log(f"  pilot metrics written to {result_dir / 'pilot_metrics.csv'}")
    log("  Read a small set of triples by hand and record typical errors for the blocker list.")
