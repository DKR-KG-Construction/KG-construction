"""Offline tests (no LLM, no dataset download): synthetic Re-DocRED-format data + the test-only mock backend."""
import copy
import csv
import json
import pathlib

import pytest
import yaml

from kgrepair.annotate import (sample_annotation, sample_proposal_annotation, score_annotation,
                               score_proposal_annotation)
from kgrepair.config import load_config
from kgrepair.evaluate import evaluate_all, sample_docs
from kgrepair.extract import attach_ids, load_raw, parse_extraction, run_extract
from kgrepair.freeze import assert_test_frozen, freeze_test_configuration
from kgrepair.matching import match_names
from kgrepair.report import pilot_report
from kgrepair.repair import canonicalize, find_quote, provenance_rule, schema_check
from kgrepair.judge import Judge
from kgrepair.schema import build_schema, save_schema
from kgrepair.stats import cohen_kappa, wilson
from kgrepair.textutil import norm_alnum, norm_tokens

ROOT = pathlib.Path(__file__).resolve().parents[1]


def make_doc(i):
    sents = [["Ada", "Lovelace", "was", "born", "in", "London", "."],
             ["She", "was", "a", "citizen", "of", "England", "."],
             ["Ada", "was", "born", "on", "10", "December", "1815", "."]]
    vs = [[{"name": "Ada Lovelace", "type": "PER", "sent_id": 0, "pos": [0, 2]},
           {"name": "Ada", "type": "PER", "sent_id": 2, "pos": [0, 1]}],
          [{"name": "London", "type": "LOC", "sent_id": 0, "pos": [5, 6]}],
          [{"name": "England", "type": "LOC", "sent_id": 1, "pos": [5, 6]}],
          [{"name": "10 December 1815", "type": "TIME", "sent_id": 2, "pos": [4, 7]}]]
    labels = [{"r": "P19", "h": 0, "t": 1, "evidence": [0]}, {"r": "P27", "h": 0, "t": 2, "evidence": [1]},
              {"r": "P569", "h": 0, "t": 3, "evidence": [2]}]
    return {"title": f"Doc {i}", "sents": sents, "vertexSet": vs, "labels": labels}


@pytest.fixture
def cfg(tmp_path):
    base = yaml.safe_load((ROOT / "config.yaml").read_text())
    for split in ("train", "dev", "test"):
        (tmp_path / f"{split}.json").write_text(json.dumps([make_doc(i) for i in range(30)]))
    base["paths"].update(train=str(tmp_path / "train.json"), dev=str(tmp_path / "dev.json"),
                         test=str(tmp_path / "test.json"),
                         schema=str(tmp_path / "schema.json"), raw_dir=str(tmp_path / "raw"),
                         results_dir=str(tmp_path / "results"), annotation_dir=str(tmp_path / "ann"),
                         judge_cache=str(tmp_path / "judge.jsonl"),
                         test_freeze=str(tmp_path / "test_freeze.json"))
    base["extractor"].update(backend="mock", model="mock")
    base["judge"].update(backend="mock", model="mock")
    base["eval"].update(sizes=[5, 10], seeds=[0, 1, 2, 3, 4], primary_n=10, primary_seed=0)
    base["annotation"].update(n_sample=20, bootstrap=100)
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(base))
    c = load_config(p)
    c["_root"] = str(ROOT)  # prompts/ live in the repo
    from kgrepair.data import load_split
    save_schema(build_schema(c, load_split(c["paths"]["train"])), c["paths"]["schema"])
    freeze_test_configuration(c)
    return c


# ---- text utilities -------------------------------------------------------------------------
def test_quote_normalisation_ignores_spacing_and_punctuation():
    assert norm_alnum("Schneider ( born") == norm_alnum("Schneider (born")
    assert norm_tokens("The  Hague") == norm_tokens("hague")


def test_find_quote_claimed_vs_elsewhere():
    sents = [norm_alnum(s) for s in ["Ada was born in London .", "She was a citizen of England ."]]
    doc = "".join(sents)
    assert find_quote("was born in London", sents, doc, 0) == (0, True)
    assert find_quote("was born in London", sents, doc, 1) == (0, False)
    assert find_quote("London She was", sents, doc, 0) == (-1, False)  # spans a sentence boundary
    assert find_quote("not in the text", sents, doc, 0) == (None, False)


def test_provenance_rule_reasons():
    sents = [norm_alnum("Ada was born in London .")]
    doc = sents[0]
    pc = {"min_quote_chars": 8}
    t = {"quote": "", "sent_id": 0}
    assert provenance_rule(t, sents, doc, pc)[1] == "empty_quote"
    assert provenance_rule({"quote": "Ada", "sent_id": 0}, sents, doc, pc)[1] == "quote_too_short"
    assert provenance_rule({"quote": "born in Paris", "sent_id": 0}, sents, doc, pc)[1] == "quote_not_found"
    assert provenance_rule({"quote": "born in London", "sent_id": 0}, sents, doc, pc)[0] is True


def test_judge_context_prefers_sentence_containing_quote():
    judge = Judge.__new__(Judge)
    judge.jcfg = {"context_sentences": 0}
    doc = {"sents": [["The", "claim", "sentence", "."],
                     ["Evidence", "for", "the", "claim", "."]]}
    assert judge._context(doc, sent_id=0, quote_sent=1) == "[1] Evidence for the claim ."
    assert judge._context(doc, sent_id=0, quote_sent=None) == "[0] The claim sentence ."


# ---- C -----------------------------------------------------------------------------------------
def E(i, name, typ, mentions=None):
    return {"id": i, "name": name, "type": typ, "mentions": mentions or [name]}


def test_canonicalize_merges_exact_and_unambiguous_person_alias():
    ents = [E("e1", "Ada Lovelace", "PER"), E("e2", "Lovelace", "PER"), E("e3", "London", "LOC"),
            E("e4", "the London", "LOC")]
    node_of, nodes = canonicalize(ents, {"containment": True, "containment_types": ["PER"]})
    assert node_of["e1"] == node_of["e2"]          # unambiguous containment (PER)
    assert node_of["e3"] == node_of["e4"]          # exact after dropping the article
    assert len(nodes) == 2


def test_canonicalize_leaves_ambiguous_and_non_person_alone():
    ents = [E("e1", "Ross Alger", "PER"), E("e2", "Harry Alger", "PER"), E("e3", "Alger", "PER"),
            E("e4", "Albany County", "LOC"), E("e5", "Albany", "LOC")]
    node_of, _ = canonicalize(ents, {"containment": True, "containment_types": ["PER"]})
    assert len({node_of["e1"], node_of["e2"], node_of["e3"]}) == 3   # 'Alger' ambiguous -> untouched
    assert node_of["e4"] != node_of["e5"]                             # LOC containment is off


# ---- S -----------------------------------------------------------------------------------------
def test_schema_check_reasons(cfg):
    from kgrepair.schema import load_schema
    schema = load_schema(cfg)
    nodes = {"a": {"types": {"PER"}}, "b": {"types": {"LOC"}}, "c": {"types": {"TIME"}}, "d": {"types": {"PERSON"}}}
    sc = {"drop_self_loops": True}
    ck = lambda h, r, t: schema_check({"head": h, "rel": r, "tail": t}, nodes, schema, sc)
    assert ck("a", "P19", "b") is None
    assert ck("b", "P19", "a") == "wrong_direction"
    assert ck("a", "P19", "c") == "type_incompatible"
    assert ck("a", "P9999", "b") == "unknown_relation"
    assert ck("a", "P19", "zzz") == "missing_entity"
    assert ck("a", "P19", "a") == "self_loop"
    assert ck("d", "P19", "b") == "unknown_entity_type"


# ---- parsing / sampling / stats -------------------------------------------------------------
def test_parse_extraction_handles_fences_and_malformed():
    txt = '```json\n{"entities":[{"id":"e1","name":"A","type":"per","mentions":["A","A","a"]},{"name":"no id"}],' \
          '"triples":[{"head":"e1","relation":"P19","tail":"e1","sent_id":0,"quote":"q"},{"head":"e1"}]}\n```'
    parsed, err, bad = parse_extraction(txt)
    assert err is None and bad == 2
    assert parsed["entities"][0]["type"] == "PER"
    assert parsed["entities"][0]["mentions"] == ["A", "a"]
    assert attach_ids(7, parsed)["triples"][0]["raw_id"] == "7:0"
    assert parse_extraction("not json")[0] is None


def test_parse_extraction_recovers_only_complete_triples_from_truncation():
    txt = (
        '{"entities":[{"id":"e1","name":"Ada","type":"PER","mentions":["Ada"]},'
        '{"id":"e2","name":"London","type":"LOC","mentions":["London"]}],'
        '"triples":[{"head":"e1","relation":"P19","tail":"e2","sent_id":0,'
        '"quote":"born in London"},{"head":"e1","relation":"P27"'
    )
    parsed, error, malformed = parse_extraction(txt)
    assert error == "partial_json_recovered"
    assert malformed == 0
    assert len(parsed["entities"]) == 2
    assert len(parsed["triples"]) == 1
    assert parsed["triples"][0]["relation"] == "P19"

    entities_truncated = '{"entities":[{"id":"e1","name":"Ada","type":"PER","mentions":["Ada"]}'
    assert parse_extraction(entities_truncated)[0] is None


def test_sample_docs_are_nested_and_deterministic():
    pool = list(range(100))
    a, b = sample_docs(pool, 10, 3), sample_docs(pool, 25, 3)
    assert b[:10] == a and a == sample_docs(pool, 10, 3) and a != sample_docs(pool, 10, 4)


def test_held_out_test_requires_unchanged_freeze(cfg, tmp_path):
    assert_test_frozen(cfg)
    original = pathlib.Path(cfg["paths"]["test"]).read_text()
    pathlib.Path(cfg["paths"]["test"]).write_text(original + " ")
    with pytest.raises(ValueError, match="test_data"):
        assert_test_frozen(cfg)
    pathlib.Path(cfg["paths"]["test"]).write_text(original)


def test_exact_matching_requires_a_shared_normalized_mention():
    gold_index = [[norm_tokens("Ada Lovelace")], [norm_tokens("London")]]
    assert match_names(["Ada Lovelace"], gold_index, "exact") == 0
    assert match_names(["Ada"], gold_index, "exact") is None


def test_wilson_and_kappa():
    p, lo, hi = wilson(80, 100)
    assert 0.70 < lo < p < hi < 0.88
    assert cohen_kappa(list("aabb"), list("aabb")) == 1.0
    assert abs(cohen_kappa(list("aabb"), list("abab"))) < 1e-9


# ---- end to end -----------------------------------------------------------------------------------
def test_end_to_end_with_mock(cfg, tmp_path):
    raw = tmp_path / "raw" / "dev.jsonl"
    run_extract(cfg, "dev", 20, 0, raw, log=lambda *_: None)
    n_lines = len(raw.read_text().splitlines())
    run_extract(cfg, "dev", 20, 0, raw, log=lambda *_: None)             # resume: nothing re-run
    assert len(raw.read_text().splitlines()) == n_lines == 20
    cfg["extractor"]["max_tokens"] += 1
    with pytest.raises(ValueError, match="different extractor configuration"):
        run_extract(cfg, "dev", 20, 0, raw, log=lambda *_: None)
    cfg["extractor"]["max_tokens"] -= 1
    recs = load_raw(raw)
    assert all(r["parsed"] is not None for r in recs.values())

    rows, _ = evaluate_all(cfg, raw, "dev", "all", log=lambda *_: None)
    by = {(r["config"], r["n_docs"], r["seed"]): r for r in rows}
    names = {r["config"] for r in rows}
    assert names == {"raw", "C", "S", "P", "C+S", "C+P", "S+P", "C+S+P"}
    for n in (5, 10):
        assert by[("raw", n, 0)]["retention_raw"] == 1.0
        # Adding S or P can only remove raw triples, never add them.
        kr = lambda c: by[(c, n, 0)]["kept_raw"]
        assert kr("C+S+P") <= kr("C+S") <= kr("C") and kr("C+S+P") <= kr("C+P") <= kr("C")
        assert kr("S+P") <= kr("S") <= kr("raw") and kr("S+P") <= kr("P") <= kr("raw")
        # NOT an invariant: C+S+P <= S+P. C unions node types, so it can rescue a triple whose
        # endpoint got a wrong type from the LLM (the C -> S interaction the 8-way ablation measures).
    assert (tmp_path / "results" / "auto_metrics.csv").exists()

    train_rows, _ = evaluate_all(cfg, raw, "train", "main", log=lambda *_: None, use_judge=False,
                                 results_dir=tmp_path / "results" / "train")
    assert {r["config"] for r in train_rows} == {"raw", "C", "C+S", "C+S+P"}
    assert (tmp_path / "results" / "train" / "auto_metrics.csv").exists()

    pilot_report(cfg, raw, "train", log=lambda *_: None)
    pilot_rows = list(csv.DictReader(open(tmp_path / "results" / "pilot_metrics.csv", encoding="utf-8")))
    assert {r["config"] for r in pilot_rows} == {"raw", "C", "C+S", "P", "C+S+P"}
    assert all(r["n_docs"] == "20" for r in pilot_rows)

    sample_annotation(cfg, raw, "dev", log=lambda *_: None)
    d = pathlib.Path(tmp_path / "ann")
    key = json.loads((d / "key.json").read_text())
    assert len(key["sampled"]) == 20 and set(key["overlap"]) <= set(key["sampled"])
    header = (d / "annotator1.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    assert "kept" not in header and "gold" not in header          # annotators must not see stage info
    for fn in ("annotator1.csv", "annotator2.csv", "annotator3.csv"):
        rows_ = list(csv.DictReader(open(d / fn, encoding="utf-8-sig")))
        for k, r in enumerate(rows_):
            r["label"] = "correct" if k % 3 else "incorrect"
        with open(d / fn, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_[0]))
            w.writeheader()
            w.writerows(rows_)
    out = score_annotation(cfg, d / "annotator1.csv", d / "annotator2.csv", d / "annotator3.csv",
                           log=lambda *_: None)
    assert {r["config"] for r in out} >= {"raw", "C", "C+S", "C+S+P"}
    with pytest.raises(ValueError):                              # unlabelled rows must be rejected
        bad = list(csv.DictReader(open(d / "annotator1.csv", encoding="utf-8-sig")))
        bad[0]["label"] = ""
        with open(d / "bad.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(bad[0]))
            w.writeheader()
            w.writerows(bad)
        score_annotation(cfg, d / "bad.csv", None, log=lambda *_: None)

    sample_annotation(cfg, raw, "train", log=lambda *_: None, n_sample=7, overlap_frac=1.0,
                      use_judge=False)
    key = json.loads((d / "key.json").read_text())
    assert len(key["sampled"]) == len(key["overlap"]) == 7
    calibration_dir = d / "calibration"
    sample_annotation(cfg, raw, "train", log=lambda *_: None, n_sample=3, overlap_frac=1.0,
                      use_judge=False, out_dir=calibration_dir, exclude_key=d / "key.json")
    calibration_key = json.loads((calibration_dir / "key.json").read_text())
    assert len(calibration_key["sampled"]) == 3
    assert set(calibration_key["sampled"]).isdisjoint(key["sampled"])

    proposal_raw = tmp_path / "raw" / "test.jsonl"
    run_extract(cfg, "test", 20, 0, proposal_raw, log=lambda *_: None)
    proposal_dir = tmp_path / "proposal"
    sample_proposal_annotation(cfg, proposal_raw, log=lambda *_: None, n_per_stage=2,
                               overlap_frac=0.5, use_judge=False, out_dir=proposal_dir)
    proposal_key = json.loads((proposal_dir / "key.json").read_text())
    assert proposal_key["split"] == "test"
    assert proposal_key["stage_sample_sizes"] == {"raw": 2, "C": 2, "C+S": 2, "C+S+P": 2}
    proposal_rows = list(csv.DictReader(open(proposal_dir / "annotator1.csv", encoding="utf-8-sig")))
    proposal_rows2 = list(csv.DictReader(open(proposal_dir / "annotator2.csv", encoding="utf-8-sig")))
    assert len(proposal_rows) == len(proposal_key["items"])
    assert len(proposal_rows2) == len(proposal_key["overlap_item_ids"])
    assert "sampled_for_stages" not in proposal_rows[0]  # blind item sheet has no stage status
    for i, row in enumerate(proposal_rows):
        row["label"] = "correct" if i % 2 else "incorrect"
    with open(proposal_dir / "annotator1.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(proposal_rows[0]))
        w.writeheader()
        w.writerows(proposal_rows)
    overlap_ids = set(proposal_key["overlap_item_ids"])
    proposal_rows2 = [dict(row, label=row["label"] or "correct") for row in proposal_rows
                      if row["item_id"] in overlap_ids]
    with open(proposal_dir / "annotator2.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(proposal_rows2[0]))
        w.writeheader()
        w.writerows(proposal_rows2)
    scored = score_proposal_annotation(cfg, proposal_dir, proposal_dir / "annotator1.csv",
                                       proposal_dir / "annotator2.csv", log=lambda *_: None)
    assert {row["config"] for row in scored} == {"raw", "C", "C+S", "C+S+P"}
