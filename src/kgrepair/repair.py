"""The three repair steps. C and S are pure functions; P's rule stage lives here, its LLM stage in judge.py."""
from .schema import allowed_pairs
from .textutil import norm_alnum, norm_tokens


# ----------------------------------------------------------------------------------------------
# C: canonicalization (within one document, rule-based)
# ----------------------------------------------------------------------------------------------
def identity_nodes(entities):
    node_of = {e["id"]: e["id"] for e in entities}
    nodes = {e["id"]: {"members": [e["id"]], "name": e["name"], "types": {e["type"]}} for e in entities}
    return node_of, nodes


def _variants(e):
    vs = [norm_tokens(n) for n in [e["name"]] + list(e.get("mentions", []))]
    return [v for v in vs if v]


def _exact_same(a, b):
    va, vb = _variants(a), _variants(b)
    return bool(va and vb and set(va) & set(vb))


def _contained_in(short_e, long_e, ccfg):
    """True if some normalized form of `short_e` is a token-subset of a longer form of `long_e`."""
    mt = ccfg.get("min_tokens", 1)
    for x in _variants(short_e):
        for y in _variants(long_e):
            if len(x) >= mt and len(x) < len(y) and set(x) <= set(y):
                return True
    return False


def canonicalize(entities, ccfg):
    """Within-document alias merging. Returns (node_of, nodes). Node id = id of its first member.
    Pass 1: merge entities sharing an exactly equal normalized surface form (union-find).
    Pass 2 (containment, optional): merge a shorter name into a longer one ONLY if exactly one
    cluster could absorb it ('Alger' is left alone when both 'Ross Alger' and 'Harry Alger' exist),
    and only for the entity types in `containment_types` (default PER). Naive containment on all
    types merged distinct entities (Albany County / Albany) and lowered precision in testing."""
    parent = {e["id"]: e["id"] for e in entities}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            if _exact_same(entities[i], entities[j]):
                union(entities[i]["id"], entities[j]["id"])
    if ccfg.get("containment", True):
        types = set(ccfg.get("containment_types", ["PER"]))
        for a in entities:
            if a["type"] not in types:
                continue
            cands = {find(b["id"]) for b in entities
                     if b is not a and b["type"] == a["type"] and find(b["id"]) != find(a["id"])
                     and _contained_in(a, b, ccfg)}
            if len(cands) == 1:
                union(next(iter(cands)), a["id"])
    groups = {}
    for e in entities:
        groups.setdefault(find(e["id"]), []).append(e)
    node_of, nodes = {}, {}
    for members in groups.values():
        nid = members[0]["id"]
        name = max((m["name"] for m in members), key=len)
        nodes[nid] = {"members": [m["id"] for m in members], "name": name, "types": {m["type"] for m in members}}
        for m in members:
            node_of[m["id"]] = nid
    return node_of, nodes


# ----------------------------------------------------------------------------------------------
# S: schema checking
# ----------------------------------------------------------------------------------------------
def schema_check(edge, nodes, schema, scfg):
    """Return a rejection reason or None."""
    rel = edge["rel"]
    if rel not in schema["relations"]:
        return "unknown_relation"
    if edge["head"] not in nodes or edge["tail"] not in nodes:
        return "missing_entity"
    ht, tt = nodes[edge["head"]]["types"], nodes[edge["tail"]]["types"]
    valid = set(schema["entity_types"])
    if not (ht <= valid and tt <= valid):
        return "unknown_entity_type"
    if scfg.get("drop_self_loops", True) and edge["head"] == edge["tail"]:
        return "self_loop"
    ok = allowed_pairs(schema, rel)
    if any((h, t) in ok for h in ht for t in tt):
        return None
    if any((t, h) in ok for h in ht for t in tt):
        return "wrong_direction"
    return "type_incompatible"


# ----------------------------------------------------------------------------------------------
# P (stage 1, rule): does the quotation exist in the document?
# ----------------------------------------------------------------------------------------------
def find_quote(quote, sent_norms, doc_norm, sent_id):
    """Return (found_sentence_idx | -1 for 'only across sentences' | None, in_claimed_sentence)."""
    nq = norm_alnum(quote)
    if not nq:
        return None, False
    if isinstance(sent_id, int) and 0 <= sent_id < len(sent_norms) and nq in sent_norms[sent_id]:
        return sent_id, True
    for i, s in enumerate(sent_norms):
        if nq in s:
            return i, False
    if nq in doc_norm:
        return -1, False
    return None, False


def provenance_rule(triple, sent_norms, doc_norm, pcfg, head_names=None, tail_names=None):
    """Return (ok, reason, quote_sentence_idx)."""
    q = triple["quote"]
    if not q.strip():
        return False, "empty_quote", None
    if len(norm_alnum(q)) < pcfg.get("min_quote_chars", 8):
        return False, "quote_too_short", None
    where, claimed = find_quote(q, sent_norms, doc_norm, triple.get("sent_id"))
    if where is None:
        return False, "quote_not_found", None
    if pcfg.get("require_claimed_sentence", False) and not claimed:
        return False, "quote_not_in_claimed_sentence", where if where >= 0 else None
    if pcfg.get("require_endpoints_in_quote", False):
        nq = norm_alnum(q)
        if not (any(norm_alnum(n) in nq for n in (head_names or []) if norm_alnum(n)) and
                any(norm_alnum(n) in nq for n in (tail_names or []) if norm_alnum(n))):
            return False, "endpoints_not_in_quote", where if where >= 0 else None
    return True, None, where if where >= 0 else None
