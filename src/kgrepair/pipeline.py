"""Run one document through any of the 8 combinations of C, S, P."""
from .data import sentences
from .repair import canonicalize, identity_nodes, provenance_rule, schema_check
from .textutil import norm_alnum

ALL_FLAGS = [(c, s, p) for c in (0, 1) for s in (0, 1) for p in (0, 1)]
MAIN = ["raw", "C", "C+S", "C+S+P"]


def config_name(c, s, p):
    return "+".join(n for n, f in zip("CSP", (c, s, p)) if f) or "raw"


FLAGS_BY_NAME = {config_name(*f): f for f in ALL_FLAGS}


class DocRun:
    def __init__(self, doc, parsed, schema, cfg, judge=None):
        self.doc, self.schema, self.cfg, self.judge = doc, schema, cfg, judge
        self.doc_id = doc["doc_id"]
        self.entities = parsed["entities"] if parsed else []
        self.triples = parsed["triples"] if parsed else []
        self.ent_by_id = {e["id"]: e for e in self.entities}
        sents = sentences(doc)
        self.sent_norms = [norm_alnum(s) for s in sents]
        self.doc_norm = norm_alnum(" ".join(sents))
        self._canon = None
        self._p = {}

    # -- nodes ------------------------------------------------------------------------------
    def nodes_for(self, c):
        if not c:
            return identity_nodes(self.entities)
        if self._canon is None:
            self._canon = canonicalize(self.entities, self.cfg.get("c", {}))
        return self._canon

    # -- provenance (independent of C and S, so cached per raw triple) ------------------------
    def _names(self, eid):
        e = self.ent_by_id.get(eid)
        return ([e["name"]] + e["mentions"]) if e else []

    def p_ok(self, t):
        rid = t["raw_id"]
        if rid in self._p:
            return self._p[rid]
        hn, tn = self._names(t["head"]), self._names(t["tail"])
        ok, reason, qs = provenance_rule(t, self.sent_norms, self.doc_norm, self.cfg.get("p", {}), hn, tn)
        if ok and self.judge is not None:
            ok, reason = self.judge.check(self.doc, t, hn, tn, qs)
            reason = None if ok else ("judge_unsupported" if not str(reason).startswith("judge_error")
                                      else "judge_error")
        self._p[rid] = (ok, reason)
        return self._p[rid]

    # -- main entry ----------------------------------------------------------------------------
    def run(self, c, s, p):
        node_of, nodes = self.nodes_for(c)
        edges = {}
        for t in self.triples:
            h, tl = node_of.get(t["head"]), node_of.get(t["tail"])
            hn, tn = (h if h is not None else "?" + t["head"]), (tl if tl is not None else "?" + t["tail"])
            key = f"{self.doc_id}|{hn}|{t['relation']}|{tn}"
            e = edges.setdefault(key, {"key": key, "head": hn, "rel": t["relation"], "tail": tn, "src": []})
            e["src"].append(t["raw_id"])
        rejected = []
        if s:
            for k in list(edges):
                reason = schema_check(edges[k], nodes, self.schema, self.cfg.get("s", {}))
                if reason:
                    rejected.append({"stage": "S", "reason": reason, "raw_ids": edges.pop(k)["src"]})
        if p:
            by_id = {t["raw_id"]: t for t in self.triples}
            for k in list(edges):
                ok_src, bad = [], []
                for rid in edges[k]["src"]:
                    ok, reason = self.p_ok(by_id[rid])
                    (ok_src if ok else bad).append((rid, reason))
                for rid, reason in bad:
                    rejected.append({"stage": "P", "reason": reason, "raw_ids": [rid]})
                if ok_src:
                    edges[k]["src"] = [r for r, _ in ok_src]
                else:
                    del edges[k]
        kept = {rid: e["key"] for e in edges.values() for rid in e["src"]}
        return {"name": config_name(c, s, p), "edges": list(edges.values()), "rejected": rejected,
                "nodes": nodes, "kept": kept}
