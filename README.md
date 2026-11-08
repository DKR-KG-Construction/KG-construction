# LLM-assisted KG construction with a precision budget

How much do three repair steps improve a knowledge graph extracted by an LLM, and how many correct
facts do they throw away? Dataset: [Re-DocRED](https://github.com/tonytan48/Re-DocRED) (Tan et al., EMNLP 2022).

* **raw** – the LLM's output, one call per document, saved once and reused by every configuration
* **C** – canonicalization: merge aliases of one entity inside a document (rules, no LLM)
* **S** – schema checking: unknown relation, missing entity, unknown type, self-loop, wrong direction, incompatible endpoint types
* **P** – provenance: the quote must exist in the document (rule), then an LLM judge (different model family) must find it supports the claim

Main comparison `raw, C, C+S, C+S+P`; secondary ablation: all 8 combinations. Every rejected item is logged with a reason.

## Quick start (local, Ollama)

Requires Python 3.11 or newer.

The commands below show environment setup. This repository already contains the
TRAIN-derived schema and the DEV-selected configuration. **Do not rerun `pilot`,
`train-baseline`, `extract-dev`, or `evaluate-dev` to continue the frozen TEST run.**
Those commands are for reproducing/developing the study from scratch and may
regenerate artifacts. The current final selection is C+S with the entire P stage
disabled (neither quote validation nor the LLM judge is applied).

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.lock
bash run.sh test        # offline tests, no model needed (uses a test-only mock LLM)
bash run.sh data        # download Re-DocRED at a pinned commit and verify sha256
# In a separate terminal, install the model configured in config.yaml:
ollama pull qwen2.5:3b
```

For the already-selected configuration, run `bash run.sh freeze-test` only if no
valid freeze manifest exists, then run `bash run.sh extract-test`. Do not change
`config.yaml`, the extractor prompt, or the schema after freezing. See the TEST
commands below.

### Reproduce the experiment from scratch

These are development steps, not required to continue the already-frozen TEST run.
They use the pretrained Ollama model; there is no model training or fine-tuning.
Run the schema-building/pilot step before selecting the configuration because it
regenerates `schema/schema.json`:

```bash
bash run.sh pilot
bash run.sh train-baseline 200 outputs/raw/train_baseline.jsonl
bash run.sh extract-dev
bash run.sh evaluate-dev
```

`train-baseline 200` supplies the configured nested TRAIN sample sizes up to 200.
`extract-dev` extracts the DEV split and `evaluate-dev` scores all C/S/P ablations.
The checked-in configuration disables P, so these commands reproduce the selected
C+S setup and quote-rule-only P ablations; they do not reproduce the historical
LLM-judge comparison described in the report.

To independently reproduce that judge comparison, before running `evaluate-dev`,
temporarily set `judge.enabled: true` in `config.yaml` and ensure Ollama has both
configured models. Run `bash run.sh evaluate-dev`; then run the quote-rule-only
comparison separately:

```bash
PYTHONPATH=src python -m kgrepair evaluate --raw outputs/raw/dev.jsonl --split dev --no-judge \
  --out-dir results/dev_quote_only
```

After reviewing the DEV results, restore the selected setting `judge.enabled: false`.
Do not access/extract/evaluate TEST until the final configuration, prompt, and schema
are selected and frozen. `bash run.sh freeze-test` records checksums; `extract-test`
and `evaluate-test` refuse to run if those frozen inputs/settings have changed.
TRAIN pilot annotation/calibration commands are optional human-labeling exercises,
not model training and not required for the automatic experiment.

`requirements.txt` pins the direct dependencies; `requirements.lock` pins the complete tested
environment. To regenerate the lock after changing a dependency, install `requirements.txt` in a
clean virtual environment and run `python -m pip freeze > requirements.lock`.

#### Windows

Install Python 3.11 and Git for Windows (Git Bash), then run from Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.lock
bash run.sh test
bash run.sh data
```

Ollama must be installed and running separately. The extractor is set to `qwen2.5:3b` as a
compact starting point for a 6 GiB GPU; the judge model is `llama3.2:3b`, but `judge.enabled` is
false in the DEV-selected configuration. The selected next-run configuration is C+S without P:
the corrected-context DEV comparison found the judge excessively restrictive. The TRAIN pilot
also disabled the judge and used quote-rule P only. The pilot has shown that the default
4,096-token output cap can truncate long JSON responses; `max_tokens`/`num_ctx` are currently
6,144/12,288, but some long-document outputs still reach the cap. Treat the existing results as
preliminary, not final.

Use DEV for development and freeze the selected configuration before accessing TEST. The current
DEV-based selection is C+S with P disabled; see the report for the automatic comparison and its
limitations. Then,
from Git Bash with the environment activated:

```bash
# Run only if docs/test_freeze.json is absent or you deliberately re-freeze after review:
bash run.sh freeze-test
# Continue the already-started final run:
bash run.sh extract-test
bash run.sh evaluate-test
bash run.sh proposal-sample
# After annotation, score the proposal sample.
bash run.sh proposal-score
```

## Layout

```
config.yaml                 all settings: models, schema relations, thresholds, seeds
prompts/extract_v1.txt      extractor prompt ({{RELATIONS}}, {{ENTITY_TYPES}} filled from the schema)
prompts/judge_v1.txt        P judge prompt
schema/schema.json          generated by `build-schema` from TRAIN (frozen artifact)
src/kgrepair/
  data.py  textutil.py      loading Re-DocRED, text normalisation
  schema.py                 relations + allowed (head_type, tail_type) pairs from TRAIN
  llm.py                    Ollama / OpenAI-compatible backends (+ test-only mock)
  extract.py                one call per doc, JSONL, resume, parsing
  repair.py                 C, S and P-rule
  judge.py                  P LLM judge, on-disk cache
  pipeline.py               run any of the 8 C/S/P combinations on one document
  matching.py  evaluate.py  gold matching, automatic P/R/F1, nested samples x seeds
  annotate.py               paired manual sample, scoring, CIs, kappa
  report.py                 pilot report
tests/                      offline tests
docs/annotation_guidelines.md
docs/midway_check.md        current preliminary metrics, blockers, and TA questions
```

## What the numbers mean

* **Automatic precision/recall/F1** use Re-DocRED labels restricted to the schema's relations. Gold is incomplete and entity matching is heuristic, so automatic precision is a **lower bound**. `precision` counts edges; `unique_precision` counts distinct gold triples recovered / edges (penalises duplicates); `f1` uses `unique_precision`.
* Automatic matching uses exact overlap between normalized predicted and gold mention strings. Keep this rule aligned with the proposal; any alternative matching rule must be selected and frozen using TRAIN only, never tuned on DEV.
* **Manual precision (primary, proposal protocol)**: after calibration and freezing the guidelines, sample 100 unique accepted edges for each of `raw`, `C`, `C+S`, and `C+S+P` from the combined five-seed, 200-document TEST runs, balancing across seeds when possible. A second annotator checks 20%; cross-stage duplicate triples are labelled once. Report precision and its interval, retention, and correct raw facts removed by repairs. The prepared 50-item TRAIN sample is only a pilot and does not satisfy this final protocol.
* **Retention** = share of raw triples a stage still represents (a triple merged into another edge by C counts as kept). **Correct-removed share** = share of raw triples labelled correct that a stage no longer represents. Always report these next to precision.
* Nested samples (25 ⊂ 50 ⊂ 100 ⊂ 200) of a 500-document dev pool overlap heavily across seeds, so the between-seed spread understates the true uncertainty. Say so in the report.

## Decisions encoded in the code (change deliberately, then re-freeze)

1. **C is within-document and rule-based**: exact normalized surface form, plus *unambiguous* containment for PER only. Naive containment for all types merged distinct entities (Albany County / Albany) in testing.
2. **C changes node types**: a merged node carries the union of its members' types, so C can rescue a triple that S would otherwise reject. Hence C+S+P is *not* a subset of S+P.
3. **S** uses allowed type pairs derived from TRAIN (`min_pair_count`, `min_pair_share`). For P17/P131 the pairs are broad, so S may reject little; Re-DocRED has only 6 coarse types.
4. **P is two-stage**: rule (quote exists) then LLM judge. The judge sees the cited sentence ±1 and the claim, not the whole document. `on_error: reject` makes unusable judge output count as unsupported.
5. **Prompt asks for one id per entity**, which reduces duplicates the extractor emits and therefore C's measurable effect. Report this.
6. Gold annotations are never in a prompt. Only `MockBackend` (tests) reads gold, through `meta`.

## Reproducibility checklist

Pinned data commit + sha256 (`scripts/`), `temperature 0` + seed, model name/tag in `config.yaml`, extractor fingerprint stored in every raw record, raw responses and judge cache on disk, audit log of rejections, all 8 configs from one cached extraction. Record exact model tags, quantisation, Ollama version and weight hashes in your report. Add a `requirements.lock` from `pip freeze`.

## Midway TRAIN-only pilot

`bash run.sh pilot` builds the schema from TRAIN if needed, extracts 10 TRAIN documents, and writes
`results/pilot_metrics.csv` plus a console report for `raw`, `C`, `C+S`, `P`, and `C+S+P`.
The pilot report runs `P` with the quote-matching rule only (no LLM judge), and labels it as such.
It is a smoke-test-sized preliminary result, not a held-out evaluation. Continue prompt/model
iterations on TRAIN, use DEV to select the final configuration, then freeze all choices before
running TEST.
After a real pilot extraction, `bash run.sh annotate-train` prepares a TRAIN pilot sample. Then run
`bash run.sh annotate-calibration <same-raw-jsonl>` to generate a separate 30-triple calibration
sample that excludes the pilot sample's raw IDs. All annotators complete and discuss calibration
first; freeze the guidelines before any final annotation. This TRAIN pilot and its labels are not
the proposal's final TEST annotation. The DEV-selected configuration disables P; if P ablations
are run, the current setting makes them quote-rule-only, not judge-assisted.
For larger preliminary samples, `bash run.sh train-baseline 200` extracts 200 TRAIN documents,
evaluates the configured nested sizes and seeds, and writes results under `results/train/` without
calling the judge. Choose an extraction count large enough for the configured `eval.sizes`. Supply
a new raw-output path as the second `pilot` argument or third `train-baseline` argument when the
extractor fingerprint changes; the runner refuses to mix responses from different configurations.

See [`docs/midway_check.md`](docs/midway_check.md) for the TRAIN pilot, DEV configuration
comparison, exploratory rejected-edge audit, remaining blockers, and TA questions.

Generated data, raw LLM outputs, judge cache, annotations, and result files are ignored by Git.
They remain local so Re-DocRED data and model outputs are not accidentally redistributed; retain
the raw JSONL and its model/config fingerprint for reproducibility.

## AI-use disclosure, citation, licence

AI tools assisted with some code and documentation during development. The project authors remain
responsible for checking and understanding the implementation and results; disclose tool use in
the submitted report according to course policy. Cite Re-DocRED: Tan et al., *Revisiting DocRED –
Addressing the False Negative Problem in Relation Extraction*, EMNLP 2022. Re-DocRED repo: MIT;
text derived from Wikipedia (CC BY-SA). Data is not redistributed here.
