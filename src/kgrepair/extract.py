"""Step 1: one LLM call per document; raw response saved to JSONL and reused by every repair config."""
import hashlib
import json
import pathlib
import re
import time

from .config import rpath
from .data import doc_to_prompt_text, load_split
from .llm import make_backend
from .schema import entity_types_prompt_block, load_schema, relations_prompt_block


def build_system_prompt(cfg, schema):
    text = rpath(cfg, cfg["extractor"]["prompt_file"]).read_text(encoding="utf-8")
    return (text.replace("{{RELATIONS}}", relations_prompt_block(schema))
                .replace("{{ENTITY_TYPES}}", entity_types_prompt_block(schema)))


def output_schema(cfg, schema):
    ecfg = cfg["extractor"]
    etype = {"type": "string"}
    if ecfg.get("constrain_entity_types", True):
        etype["enum"] = schema["entity_types"]
    rel = {"type": "string"}
    if ecfg.get("constrain_relation_enum", False):
        rel["enum"] = sorted(schema["relations"])
    return {"type": "object", "required": ["entities", "triples"], "properties": {
        "entities": {"type": "array", "items": {"type": "object",
                     "required": ["id", "name", "type", "mentions"],
                     "properties": {"id": {"type": "string"}, "name": {"type": "string"}, "type": etype,
                                    "mentions": {"type": "array", "items": {"type": "string"}}}}},
        "triples": {"type": "array", "items": {"type": "object",
                    "required": ["head", "relation", "tail", "sent_id", "quote"],
                    "properties": {"head": {"type": "string"}, "relation": rel, "tail": {"type": "string"},
                                   "sent_id": {"type": "integer"}, "quote": {"type": "string"}}}}}}


def fingerprint(cfg, schema, split=None):
    ecfg = cfg["extractor"]
    settings = {k: ecfg.get(k) for k in (
        "backend", "host", "model", "temperature", "seed", "max_tokens", "num_ctx",
        "constrain_relation_enum", "constrain_entity_types",
    )}
    blob = json.dumps({"split": split, "prompt": build_system_prompt(cfg, schema), "schema": output_schema(cfg, schema),
                       "settings": settings}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def _decode_array(text, start):
    decoder = json.JSONDecoder()
    items = []
    i = start + 1
    while i < len(text):
        while i < len(text) and (text[i].isspace() or text[i] == ","):
            i += 1
        if i >= len(text):
            return items, False, i
        if text[i] == "]":
            return items, True, i + 1
        try:
            item, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            return items, False, i
        items.append(item)
        i = end
    return items, False, i


def _recover_truncated_extraction(text):
    """Recover complete triples only when the entity array was fully emitted."""
    entity_match = re.match(r'\s*\{\s*"entities"\s*:\s*\[', text)
    if not entity_match:
        return None
    entities, entities_complete, entity_end = _decode_array(text, entity_match.end() - 1)
    if not entities_complete:
        return None
    triple_match = re.match(r'\s*,\s*"triples"\s*:\s*\[', text[entity_end:])
    if not triple_match:
        return None
    triple_start = entity_end + triple_match.end() - 1
    triples, triples_complete, _ = _decode_array(text, triple_start)
    triples = [t for t in triples if isinstance(t, dict) and
               all(k in t for k in ("head", "relation", "tail"))]
    if not triples and not triples_complete:
        return None
    return {"entities": entities, "triples": triples}


def parse_extraction(text):
    """Return (parsed|None, error|None, n_malformed). Malformed triples/entities are dropped and
    counted (reported by `pilot-report`); they are not part of the raw graph."""
    t = _FENCE.sub("", (text or "").strip())
    obj = None
    try:
        obj = json.loads(t)
    except Exception:
        a, b = t.find("{"), t.rfind("}")
        if a >= 0 and b > a:
            try:
                obj = json.loads(t[a:b + 1])
            except Exception:
                obj = None
    partial = False
    if not isinstance(obj, dict) or not isinstance(obj.get("entities"), list) \
            or not isinstance(obj.get("triples"), list):
        obj = _recover_truncated_extraction(t)
        if obj is None:
            return None, "invalid_json_or_shape", 0
        partial = True
    bad = 0
    ents, seen = [], set()
    for e in obj["entities"]:
        if not isinstance(e, dict) or "id" not in e or str(e["id"]) in seen:
            bad += 1
            continue
        seen.add(str(e["id"]))
        ments = e.get("mentions", [])
        ments = list(dict.fromkeys(str(m) for m in ments)) if isinstance(ments, list) else []
        ents.append({"id": str(e["id"]), "name": str(e.get("name", "")),
                     "type": str(e.get("type", "")).strip().upper(), "mentions": ments})
    trips = []
    for t_ in obj["triples"]:
        if not isinstance(t_, dict) or not all(k in t_ for k in ("head", "relation", "tail")):
            bad += 1
            continue
        sid = t_.get("sent_id")
        sid = sid if isinstance(sid, int) and not isinstance(sid, bool) else None
        trips.append({"head": str(t_["head"]), "relation": str(t_["relation"]).strip(),
                      "tail": str(t_["tail"]), "sent_id": sid, "quote": str(t_.get("quote", ""))})
    return {"entities": ents, "triples": trips}, "partial_json_recovered" if partial else None, bad


def attach_ids(doc_id, parsed):
    for k, t in enumerate(parsed["triples"]):
        t["raw_id"] = f"{doc_id}:{k}"
    return parsed


def load_raw(path):
    """doc_id -> record with record['parsed'] (or None) and record['malformed']."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            parsed, err, bad = parse_extraction(r.get("raw"))
            if parsed is not None:
                attach_ids(r["doc_id"], parsed)
            r["parsed"], r["parse_error"], r["malformed"] = parsed, err or r.get("error"), bad
            out[r["doc_id"]] = r
    return out


def validate_test_raw(cfg, raw, schema, docs):
    """Prevent evaluation of an incomplete or foreign extraction as the frozen TEST run."""
    expected = fingerprint(cfg, schema, 'test')
    if set(raw) != set(range(len(docs))):
        raise ValueError('TEST evaluation requires one cached record for every TEST document')
    for doc_id, record in raw.items():
        if record.get('split') != 'test' or record.get('fingerprint') != expected:
            raise ValueError(f'TEST raw record {doc_id} has a different split or extractor fingerprint')


def run_extract(cfg, split, n, start, out_path, log=print):
    if split == "test":
        from .freeze import assert_test_frozen
        assert_test_frozen(cfg)
    schema = load_schema(cfg)
    docs = load_split(rpath(cfg, cfg["paths"][split]))
    ids = list(range(start, len(docs) if n is None else min(len(docs), start + n)))
    backend = make_backend(cfg["extractor"])
    system = build_system_prompt(cfg, schema)
    oschema = output_schema(cfg, schema)
    fp = fingerprint(cfg, schema, split)
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("fingerprint") != fp:
                    raise ValueError(
                        f"{out_path} contains records from a different extractor configuration "
                        f"(doc_id={record.get('doc_id')}). Use a new --out path for this run."
                    )
                if record.get("error") is None and record.get("raw") is not None:
                    done.add(record["doc_id"])
    todo = [i for i in ids if i not in done]
    log(f"{len(ids)} docs requested, {len(done & set(ids))} already done, {len(todo)} to run "
        f"(model={cfg['extractor']['model']}, fingerprint={fp})")
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for k, i in enumerate(todo, 1):
            doc = docs[i]
            rec = {"doc_id": i, "title": doc["title"], "split": split, "fingerprint": fp,
                   "model": cfg["extractor"]["model"]}
            last = None
            for attempt in range(cfg["extractor"].get("retries", 2) + 1):
                try:
                    # Gold annotations are NOT in the prompt. `meta` is read only by the test-only mock.
                    res = backend.generate(system, doc_to_prompt_text(doc), oschema,
                                           meta={"kind": "extract", "doc": doc, "schema": schema})
                    rec.update(raw=res["text"], usage=res["usage"], latency=res["latency"], error=None)
                    break
                except Exception as e:  # network / server error
                    last = f"request_failed: {type(e).__name__}: {e}"
                    time.sleep(2 * (attempt + 1))
            else:
                rec.update(raw=None, usage=None, latency=None, error=last)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if k % 10 == 0 or k == len(todo):
                el = time.time() - t0
                log(f"  {k}/{len(todo)} done, {el:.0f}s elapsed, ~{el / k * (len(todo) - k):.0f}s left")
