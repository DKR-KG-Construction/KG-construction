"""LLM backends. All expose generate(system, user, schema=None, meta=None) ->
{"text": str, "usage": {"prompt_tokens": int|None, "completion_tokens": int|None}, "latency": float}.
`meta` is ignored by real backends; only the TEST-ONLY MockBackend reads it."""
import json
import os
import random
import time

import requests


class BackendError(RuntimeError):
    pass


def _check_model(cfg):
    if str(cfg.get("model", "")).startswith("CHANGE_ME"):
        raise BackendError("Set 'model' in config.yaml (extractor.model / judge.model) before running.")


class OllamaBackend:
    def __init__(self, cfg):
        _check_model(cfg)
        self.cfg = cfg
        self.host = cfg["host"].rstrip("/")

    def generate(self, system, user, schema=None, meta=None):
        c = self.cfg
        payload = {
            "model": c["model"], "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": c.get("temperature", 0), "seed": c.get("seed", 0),
                        "num_predict": c.get("max_tokens", 4096), "num_ctx": c.get("num_ctx", 8192)},
        }
        if schema is not None:
            payload["format"] = schema
        t0 = time.time()
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=c.get("timeout_s", 600))
        r.raise_for_status()
        j = r.json()
        return {"text": j["message"]["content"],
                "usage": {"prompt_tokens": j.get("prompt_eval_count"),
                          "completion_tokens": j.get("eval_count")},
                "latency": time.time() - t0}


class OpenAICompatBackend:
    """Works with vLLM, llama.cpp server, LM Studio, OpenRouter and similar (/v1/chat/completions)."""

    def __init__(self, cfg):
        _check_model(cfg)
        self.cfg = cfg
        self.base = cfg["host"].rstrip("/")
        env = cfg.get("api_key_env") or ""
        self.key = os.environ.get(env, "") if env else ""

    def generate(self, system, user, schema=None, meta=None):
        c = self.cfg
        payload = {"model": c["model"], "temperature": c.get("temperature", 0),
                   "max_tokens": c.get("max_tokens", 4096), "seed": c.get("seed", 0),
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if schema is not None:
            payload["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": "output", "schema": schema}}
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        t0 = time.time()
        r = requests.post(f"{self.base}/chat/completions", json=payload, headers=headers,
                          timeout=c.get("timeout_s", 600))
        r.raise_for_status()
        j = r.json()
        u = j.get("usage") or {}
        return {"text": j["choices"][0]["message"]["content"],
                "usage": {"prompt_tokens": u.get("prompt_tokens"),
                          "completion_tokens": u.get("completion_tokens")},
                "latency": time.time() - t0}


class MockBackend:
    """TEST-ONLY. Fabricates noisy extractions/judgements FROM THE GOLD ANNOTATIONS so the whole
    pipeline can be exercised offline. It reads gold through `meta`, which real backends ignore.
    Never use it for results: it exists solely to test the code."""

    def __init__(self, cfg):
        self.cfg = cfg

    def generate(self, system, user, schema=None, meta=None):
        kind = meta["kind"]
        text = self._extract(meta) if kind == "extract" else self._judge(meta)
        return {"text": text, "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "latency": 0.0}

    def _extract(self, meta):
        from .data import gold_entity_names, gold_entity_type, gold_triples, sentences
        doc, schema = meta["doc"], meta["schema"]
        rng = random.Random(doc["doc_id"] * 7919 + 1)
        sents = sentences(doc)
        names = gold_entity_names(doc)
        types = sorted(schema["entity_types"])
        ents, primary = [], {}

        def ensure(g, alias=False):
            if g in primary and not alias:
                return primary[g]
            nm = names[g][-1] if (alias and len(names[g]) > 1) else names[g][0]
            typ = gold_entity_type(doc, g)
            if rng.random() < 0.10:
                typ = rng.choice(types)
            eid = f"e{len(ents) + 1}"
            ents.append({"id": eid, "name": nm, "type": typ, "mentions": [nm]})
            if not alias:
                primary[g] = eid
            return eid

        triples = []
        for (h, r, t) in sorted(gold_triples(doc, set(schema["relations"]))):
            if rng.random() > 0.6:
                continue
            hid = ensure(h, alias=rng.random() < 0.2)
            tid = ensure(t)
            if rng.random() < 0.05:
                hid, tid = tid, hid
            ev = next((l["evidence"] for l in doc["labels"] if (l["h"], l["r"], l["t"]) == (h, r, t)), [])
            sid = ev[0] if ev else 0
            quote = " ".join(sents[sid].split()[:30])
            if rng.random() < 0.10:
                quote = "it is said that " + " ".join(reversed(quote.split()))
            triples.append({"head": hid, "relation": r, "tail": tid, "sent_id": sid, "quote": quote})
        rels = sorted(schema["relations"])
        for _ in range(int(0.3 * len(triples)) + 1):
            if len(doc["vertexSet"]) < 2:
                break
            a, b = rng.sample(range(len(doc["vertexSet"])), 2)
            sid = rng.randrange(len(sents))
            triples.append({"head": ensure(a), "relation": rng.choice(rels + ["P9999"]), "tail": ensure(b),
                            "sent_id": sid, "quote": " ".join(sents[sid].split()[:30])})
        return json.dumps({"entities": ents, "triples": triples})

    def _judge(self, meta):
        from .matching import build_gold_index, match_names
        doc = meta["doc"]
        gi = build_gold_index(doc)
        m = meta.get("matching", {})
        h = match_names(meta["head_names"], gi, m.get("mode", "partial"), m.get("jaccard", 0.5))
        t = match_names(meta["tail_names"], gi, m.get("mode", "partial"), m.get("jaccard", 0.5))
        truth = h is not None and t is not None and (h, meta["relation"], t) in {
            (l["h"], l["r"], l["t"]) for l in doc["labels"]}
        if random.Random(meta["raw_id"]).random() < 0.10:
            truth = not truth
        return json.dumps({"supported": bool(truth), "reason": "mock"})


def make_backend(cfg):
    kind = cfg.get("backend", "ollama")
    if kind == "ollama":
        return OllamaBackend(cfg)
    if kind == "openai_compat":
        return OpenAICompatBackend(cfg)
    if kind == "mock":
        return MockBackend(cfg)
    raise ValueError(f"unknown backend {kind!r}")
