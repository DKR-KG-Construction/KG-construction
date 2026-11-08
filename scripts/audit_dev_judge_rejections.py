
import argparse
import csv
import hashlib
import json
import pathlib
import sys
import time

import requests


LABEL_SCHEMA = {
    "type": "object",
    "required": ["fact_correct", "quote_supports_fact", "reason"],
    "properties": {
        "fact_correct": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "quote_supports_fact": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "reason": {"type": "string"},
    },
}
FIELDS = [
    "doc_id", "title", "raw_id", "head", "relation", "relation_name", "tail",
    "relation_description",
    "source_raw_ids", "evidence_quotes", "fact_correct",
    "quote_supports_fact", "reason", "judge_reason", "review_model", "review_error",
]


def judge_key(judge, doc, triple, head_names, tail_names, quote_sentence):
    rel = triple["relation"]
    spec = judge.schema["relations"].get(rel, {"name": rel, "description": ""})
    context = judge._context(doc, triple.get("sent_id"), quote_sentence)
    user = (
        f"Context:\n{context}\n\nClaim:\n"
        f"- head: {head_names[0] if head_names else ''}\n"
        f"- relation: {rel} ({spec['name']}): {spec['description']}\n"
        f"- tail: {tail_names[0] if tail_names else ''}\n\n"
        f"Quote cited by the extractor:\n\"{triple['quote']}\""
    )
    key = hashlib.sha1(
        json.dumps([judge.jcfg["model"], judge.system, user], sort_keys=True).encode()
    ).hexdigest()
    return key, context


class CachedJudge:
    """Expose only existing cached verdicts; never make more original-judge calls."""

    def __init__(self, judge):
        self.judge = judge
        self.missing = 0

    def check(self, doc, triple, head_names, tail_names, quote_sentence=None):
        key, _ = judge_key(self.judge, doc, triple, head_names, tail_names, quote_sentence)
        verdict = self.judge.cache.get(key)
        if verdict is None:
            self.missing += 1
            return True, "not_in_original_judge_cache"
        return bool(verdict["supported"]), str(verdict.get("reason", ""))


def load_candidates(repo, cfg):
    from kgrepair.config import rpath
    from kgrepair.data import load_split, sentences
    from kgrepair.extract import load_raw
    from kgrepair.judge import Judge
    from kgrepair.pipeline import DocRun
    from kgrepair.repair import find_quote
    from kgrepair.schema import load_schema

    schema = load_schema(cfg)
    docs = load_split(rpath(cfg, cfg["paths"]["dev"]))
    raw = load_raw(rpath(cfg, "outputs/raw/dev.jsonl"))
    judge = Judge(cfg, schema)
    cached_judge = CachedJudge(judge)
    candidates = []

    for doc_id, rec in sorted(raw.items()):
        doc = docs[doc_id]
        run = DocRun(doc, rec["parsed"], schema, cfg, judge=cached_judge)
        before_p = run.run(1, 1, 0)
        after_p = run.run(1, 1, 1)
        kept_edge_keys = {edge["key"] for edge in after_p["edges"]}
        by_id = {triple["raw_id"]: triple for triple in run.triples}
        sents = sentences(doc)
        for edge in before_p["edges"]:
            if edge["key"] in kept_edge_keys:
                continue
            rejected_sources = []
            evidence = []
            reasons = []
            for raw_id in edge["src"]:
                triple = by_id[raw_id]
                ok, reason = run.p_ok(triple)
                if ok or reason != "judge_unsupported":
                    continue
                head = run.ent_by_id.get(triple["head"], {})
                tail = run.ent_by_id.get(triple["tail"], {})
                head_names = [head.get("name", "")] + list(head.get("mentions", []))
                tail_names = [tail.get("name", "")] + list(tail.get("mentions", []))
                found_sentence, _ = find_quote(
                    triple["quote"], run.sent_norms, run.doc_norm, triple.get("sent_id")
                )
                quote_sentence = (
                    found_sentence
                    if found_sentence is not None and found_sentence >= 0 else None
                )
                original_key, context = judge_key(
                    judge, doc, triple, head_names, tail_names, quote_sentence
                )
                original_verdict = judge.cache.get(original_key, {})
                evidence.append({
                    "raw_id": raw_id,
                    "sent_id": triple.get("sent_id"),
                    "quote_sentence_id": quote_sentence,
                    "quote": triple["quote"],
                    "judge_context": context,
                })
                if original_verdict.get("reason"):
                    reasons.append(original_verdict["reason"])
                rejected_sources.append(raw_id)
            if not rejected_sources:
                continue
            head_node = before_p["nodes"].get(edge["head"], {})
            tail_node = before_p["nodes"].get(edge["tail"], {})
            rel = edge["rel"]
            rel_spec = schema["relations"].get(rel, {"name": rel})
            candidates.append({
                "doc_id": doc_id,
                "title": doc["title"],
                "raw_id": rejected_sources[0],
                "source_raw_ids": json.dumps(rejected_sources, ensure_ascii=False),
                "head": head_node.get("name", edge["head"]),
                "relation": rel,
                "relation_name": rel_spec["name"],
                "relation_description": rel_spec["description"],
                "tail": tail_node.get("name", edge["tail"]),
                "evidence_quotes": json.dumps(evidence, ensure_ascii=False),
                "full_document": "\n".join(
                    f"[{idx}] {sentence}" for idx, sentence in enumerate(sents)
                ),
                "judge_reason": " | ".join(sorted(set(reasons))),
            })
    return candidates, cached_judge.missing


def review_one(row, model, host, timeout):
    system = (
        "You are an exploratory AI reviewer for a knowledge-graph extraction study. "
        "You are not a human annotator. Judge the two questions independently. "
        "Use only the supplied document text; do not use outside knowledge. "
        "A fact is correct only when the document states or directly implies the directed "
        "head-relation-tail claim with the right entities. Quote support is yes only when "
        "the supplied quote and its local context provide evidence for that fact. A fact can "
        "be correct in the full document while its cited quote fails to support it. Use "
        "uncertain when the supplied text is insufficient or genuinely ambiguous. Preserve "
        "the relation direction and meaning."
    )
    user = (
        f"Full DEV document ({row['title']}):\n{row['full_document']}\n\n"
        f"Claim:\nHead: {row['head']}\nRelation: {row['relation']} "
        f"({row['relation_name']}; {row['relation_description']})\n"
        f"Tail: {row['tail']}\n\n"
        f"All rejected source quotations and their judge-visible local contexts:\n"
        f"{row['evidence_quotes']}\n\n"
        "Return fact_correct and quote_supports_fact separately as yes, no, or uncertain. "
        "For quote_supports_fact, answer yes if any of the listed quote-context pairs "
        "supports the claim. Give one short reason distinguishing factuality from evidence support."
    )
    payload = {
        "model": model,
        "stream": False,
        "format": LABEL_SCHEMA,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": 0, "seed": 0, "num_predict": 180, "num_ctx": 12288},
    }
    response = requests.post(
        f"{host.rstrip('/')}/api/chat", json=payload, timeout=timeout
    )
    response.raise_for_status()
    result = json.loads(response.json()["message"]["content"])
    if result.get("fact_correct") not in {"yes", "no", "uncertain"}:
        raise ValueError("invalid fact_correct label")
    if result.get("quote_supports_fact") not in {"yes", "no", "uncertain"}:
        raise ValueError("invalid quote_supports_fact label")
    if not isinstance(result.get("reason"), str):
        raise ValueError("missing reason")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument(
        "--out", type=pathlib.Path,
        default=pathlib.Path("outputs/dev_judge_rejections_ai_review.csv"),
    )
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None,
                        help="optional small smoke-test limit; omit to review all")
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))

    from kgrepair.config import load_config
    from kgrepair.data import load_split, sentences

    cfg = load_config(repo / "config.yaml")
    candidates, missing_cache = load_candidates(repo, cfg)
    docs = load_split(repo / cfg["paths"]["dev"])
    if args.limit is not None:
        candidates = candidates[:args.limit]

    out = args.out if args.out.is_absolute() else repo / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if out.exists():
        with out.open(newline="", encoding="utf-8-sig") as file:
            completed = {
                row["raw_id"] for row in csv.DictReader(file)
                if row.get("fact_correct") and row.get("quote_supports_fact")
            }

    todo = [row for row in candidates if row["raw_id"] not in completed]
    print(
        f"{len(candidates)} cached DEV judge rejections selected; "
        f"{len(completed)} already reviewed; {len(todo)} remaining; "
        f"{missing_cache} cache misses were excluded."
    )
    write_header = not out.exists() or out.stat().st_size == 0
    with out.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for idx, row in enumerate(todo, 1):
            base = {key: value for key, value in row.items() if key != "full_document"}
            try:
                labels = review_one(row, args.model, args.host, args.timeout)
                output = {
                    **base, **labels, "review_model": args.model, "review_error": "",
                }
            except Exception as exc:
                output = {
                    **base, "fact_correct": "", "quote_supports_fact": "",
                    "reason": "", "review_model": args.model,
                    "review_error": f"{type(exc).__name__}: {exc}",
                }
            writer.writerow(output)
            file.flush()
            if idx % 25 == 0 or idx == len(todo):
                print(f"{idx}/{len(todo)} reviewed")
            if output["review_error"]:
                time.sleep(0.1)
    print(f"Review sheet: {out}")


if __name__ == "__main__":
    main()
