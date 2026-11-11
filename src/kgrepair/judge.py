"""LLM judge for provenance (P, second stage). Results are cached on disk by content hash,
so the 8-way ablation never pays twice for the same claim."""
import hashlib
import json
import pathlib

from .config import rpath
from .data import sentences
from .llm import make_backend

JUDGE_SCHEMA = {"type": "object", "required": ["supported", "reason"],
                "properties": {"supported": {"type": "boolean"}, "reason": {"type": "string"}}}


class Judge:
    def __init__(self, cfg, schema):
        self.cfg, self.jcfg, self.schema = cfg, cfg["judge"], schema
        self.backend = make_backend(self.jcfg)
        self.system = rpath(cfg, self.jcfg["prompt_file"]).read_text(encoding="utf-8")
        self.path = pathlib.Path(rpath(cfg, cfg["paths"]["judge_cache"]))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        self.cache[r["key"]] = r
        self.calls = 0
        self.tokens = {"prompt": 0, "completion": 0}

    def _context(self, doc, sent_id, quote_sent):
        sents = sentences(doc)
        k = self.jcfg.get("context_sentences", 1)
        center = quote_sent if isinstance(quote_sent, int) and 0 <= quote_sent < len(sents) else sent_id
        if not isinstance(center, int) or not 0 <= center < len(sents):
            return ""
        lo, hi = max(0, center - k), min(len(sents), center + k + 1)
        return "\n".join(f"[{i}] {sents[i]}" for i in range(lo, hi))

    def check(self, doc, triple, head_names, tail_names, quote_sent=None):
        """Return (supported: bool, reason: str)."""
        rel = triple["relation"]
        spec = self.schema["relations"].get(rel, {"name": rel, "description": ""})
        ctx = self._context(doc, triple.get("sent_id"), quote_sent)
        user = (f"Context:\n{ctx}\n\nClaim:\n- head: {head_names[0] if head_names else ''}\n"
                f"- relation: {rel} ({spec['name']}): {spec['description']}\n"
                f"- tail: {tail_names[0] if tail_names else ''}\n\nQuote cited by the extractor:\n\"{triple['quote']}\"")
        cache_settings = {k: self.jcfg.get(k) for k in ('backend','model','host','temperature','seed','max_tokens','num_ctx','context_sentences','on_error')}
        key = hashlib.sha256(json.dumps([cache_settings, self.system, user], sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            r = self.cache[key]
            if type(r.get('supported')) is bool:
                return r['supported'], r.get('reason', '')
            del self.cache[key]
        meta = {"kind": "judge", "doc": doc, "head_names": head_names, "tail_names": tail_names,
                "relation": rel, "raw_id": triple["raw_id"], "matching": self.cfg.get("matching", {})}
        last = "judge_error"
        for _ in range(self.jcfg.get("retries", 2) + 1):
            try:
                res = self.backend.generate(self.system, user, JUDGE_SCHEMA, meta=meta)
                self.calls += 1
                u = res.get("usage") or {}
                self.tokens["prompt"] += u.get("prompt_tokens") or 0
                self.tokens["completion"] += u.get("completion_tokens") or 0
                obj = json.loads(res["text"])
                if type(obj.get('supported')) is not bool:
                    raise ValueError('judge supported field must be a JSON boolean')
                if not isinstance(obj.get('reason'), str):
                    raise ValueError('judge reason field must be a string')
                supported, reason = obj['supported'], obj['reason']
                rec = {"key": key, "supported": supported, "reason": reason, "model": self.jcfg["model"]}
                self.cache[key] = rec
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                return supported, reason
            except Exception as e:
                last = f"judge_error: {type(e).__name__}"
        return self.jcfg.get("on_error", "reject") == "accept", last
