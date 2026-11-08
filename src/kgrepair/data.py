import json


def load_split(path):
    with open(path, encoding="utf-8") as f:
        docs = json.load(f)
    for i, d in enumerate(docs):
        d["doc_id"] = i
    return docs


def sentences(doc):
    return [" ".join(s) for s in doc["sents"]]


def doc_to_prompt_text(doc):
    return "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences(doc)))


def gold_triples(doc, relations=None):
    """Set of (head_entity_idx, relation, tail_entity_idx). Optionally filtered to `relations`."""
    out = set()
    for l in doc["labels"]:
        if relations is None or l["r"] in relations:
            out.add((l["h"], l["r"], l["t"]))
    return out


def gold_entity_names(doc):
    """List (per gold entity) of unique mention strings."""
    return [sorted({m["name"] for m in v}) for v in doc["vertexSet"]]


def gold_entity_type(doc, idx):
    return doc["vertexSet"][idx][0]["type"]
