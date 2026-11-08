"""Map predicted entities to gold entities. Used ONLY for automatic metrics (and the mock
backend), never for repair decisions and never inside prompts."""
from .textutil import norm_tokens


def _score(a, b, mode, thr):
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if mode == "exact":
        return 0.0
    sa, sb = set(a), set(b)
    j = len(sa & sb) / len(sa | sb)
    return j if j >= thr else 0.0


def build_gold_index(doc):
    """gold entity idx -> list of token tuples (one per distinct mention name)."""
    idx = []
    for v in doc["vertexSet"]:
        names = {m["name"] for m in v}
        idx.append([t for t in (norm_tokens(n) for n in names) if t])
    return idx


def match_names(names, gold_index, mode="partial", thr=0.5):
    """Return the gold entity idx that best matches any of `names`, or None."""
    toks = [t for t in (norm_tokens(n) for n in names) if t]
    best_score, best_hits, best_idx = 0.0, 0, None
    for gi, gnames in enumerate(gold_index):
        top, hits = 0.0, 0
        for a in toks:
            for b in gnames:
                s = _score(a, b, mode, thr)
                if s > 0:
                    hits += 1
                top = max(top, s)
        # strict '>' keeps the lowest gold index on exact ties (deterministic)
        if top > 0 and (top, hits) > (best_score, best_hits):
            best_score, best_hits, best_idx = top, hits, gi
    return best_idx


def map_entities(doc, entities, mcfg):
    """entities: list of dicts with id/name/mentions. Returns {entity_id: gold_idx|None}."""
    gi = build_gold_index(doc)
    out = {}
    for e in entities:
        out[e["id"]] = match_names([e["name"]] + list(e.get("mentions", [])), gi,
                                   mcfg.get("mode", "partial"), mcfg.get("jaccard", 0.5))
    return out
