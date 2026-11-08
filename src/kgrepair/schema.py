"""Schema = entity types + selected relations + allowed (head_type, tail_type) pairs.
The allowed pairs are derived from the TRAIN split only, then frozen in schema/schema.json."""
import collections
import json
import pathlib

from .config import rpath


def build_schema(cfg, train_docs):
    scfg = cfg["schema"]
    rels = {r["id"]: r for r in scfg["relations"]}
    pair_counts = {rid: collections.Counter() for rid in rels}
    entity_types = collections.Counter()
    for doc in train_docs:
        vs = doc["vertexSet"]
        for v in vs:
            for m in v:
                entity_types[m["type"]] += 1
        for l in doc["labels"]:
            if l["r"] in rels:
                pair_counts[l["r"]][(vs[l["h"]][0]["type"], vs[l["t"]][0]["type"])] += 1
    out = {"entity_types": sorted(entity_types), "relations": {},
           "built_from": {"min_pair_count": scfg["min_pair_count"],
                          "min_pair_share": scfg["min_pair_share"],
                          "n_train_docs": len(train_docs)}}
    for rid, spec in rels.items():
        total = sum(pair_counts[rid].values())
        allowed = [list(p) for p, c in pair_counts[rid].most_common()
                   if c >= scfg["min_pair_count"] and total and c / total >= scfg["min_pair_share"]]
        out["relations"][rid] = {"name": spec["name"], "description": spec["description"],
                                 "train_count": total, "allowed_pairs": allowed}
    return out


def save_schema(schema, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)


def load_schema(cfg):
    p = rpath(cfg, cfg["paths"]["schema"])
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run: python -m kgrepair build-schema")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def allowed_pairs(schema, rel):
    return {tuple(p) for p in schema["relations"][rel]["allowed_pairs"]}


def relations_prompt_block(schema):
    lines = []
    for rid, r in schema["relations"].items():
        lines.append(f"- {rid} ({r['name']}): {r['description']}")
    return "\n".join(lines)


ENTITY_TYPE_HELP = {
    "PER": "person", "ORG": "organization", "LOC": "location, country, region, city",
    "TIME": "date or year", "NUM": "number or quantity",
    "MISC": "any other named thing (works, events, titles, products, nationalities)",
}


def entity_types_prompt_block(schema):
    return "\n".join(f"- {t}: {ENTITY_TYPE_HELP.get(t, t)}" for t in schema["entity_types"])
