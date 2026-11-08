# LLM-assisted KG construction with a precision budget

How much do three repair steps improve a knowledge graph extracted by an LLM, and how many correct
facts do they throw away? Dataset: [Re-DocRED](https://github.com/tonytan48/Re-DocRED) (Tan et al., EMNLP 2022).

* **raw** – the LLM's output, one call per document, saved once and reused by every configuration
* **C** – canonicalization: merge aliases of one entity inside a document (rules, no LLM)
* **S** – schema checking: unknown relation, missing entity, unknown type, self-loop, wrong direction, incompatible endpoint types
* **P** – provenance: the quote must exist in the document (rule), then an LLM judge (different model family) must find it supports the claim

Main comparison `raw, C, C+S, C+S+P`; secondary ablation: all 8 combinations. Every rejected item is logged with a reason.

## Run the selected project from a clean checkout

This is the single recommended path for a new team member to reproduce the **DEV-selected final
experiment**. It uses the pretrained `qwen2.5:3b` model; **there is no model training or
fine-tuning**. The repository includes the TRAIN-derived schema, selected configuration, and TEST
freeze manifest. The selected primary configuration is C+S. The P ablation in the proposal
sample is quote-existence-only because the LLM judge is disabled.

### 1. Install prerequisites

Install Python 3.11+, Git for Windows (including Git Bash), and Ollama. Open Git Bash and clone
the repository:

```bash
git clone https://github.com/DKR-KG-Construction/KG-construction.git
cd KG-construction
```

Create the Python environment, install dependencies, and run offline tests:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.lock
bash run.sh test
```

Download and verify the pinned Re-DocRED data, then download the configured local model. Ollama
must be running; if needed, start it in a separate terminal before pulling the model:

```bash
bash run.sh data
ollama pull qwen2.5:3b
```

### 2. Run the frozen TEST evaluation

Do not run `pilot`, `train-baseline`, `extract-dev`, `evaluate-dev`, or `build-schema` in this
reproduction path. They are development commands and can regenerate artifacts used by the frozen
experiment. Do not edit `config.yaml`, `prompts/extract_v1.txt`, or `schema/schema.json`.
The checked-in `docs/test_freeze.json` records their approved hashes/settings; TEST commands
validate it and stop if those inputs have changed.

Extract TEST outputs (resumable; this may take a long time):

```bash
bash run.sh extract-test
```

When extraction completes successfully, evaluate the configured five seeds and sample the
proposal annotation items:

```bash
bash run.sh evaluate-test
bash run.sh proposal-sample
```

### 3. Complete the two-person annotation

The sample is written under `annotation/proposal_test/`. The primary annotator completes every
row in `annotator1.csv`; the second annotator completes only the rows in `annotator2.csv` (the
20% overlap). Both annotators use `docs.txt` and the agreed [`docs/annotation_guidelines.md`](docs/annotation_guidelines.md).
Label each row `correct`, `incorrect`, or `uncertain`; add a short note for incorrect/uncertain
cases. Do not change the CSV headers or item IDs. Do not show either annotator `key.json`, which
contains the stage mapping and is only for scoring.

Share the annotation CSVs and `docs.txt` with the annotators using your team's approved method.
After both files are completed and returned to `annotation/proposal_test/`, score them:

```bash
bash run.sh proposal-score
```

The command reports per-stage manual precision and intervals, retention, correct raw facts
removed, uncertain-label count, and agreement on the double-labelled overlap. It writes
`results/proposal_manual_summary.csv`. Annotation files and generated results are local and
Git-ignored; share/publish them only if your team and data policy permit.

`requirements.txt` lists direct dependencies; `requirements.lock` pins the tested environment.
For the earlier TRAIN/DEV decision evidence and limitations, see
[`docs/midway_check.md`](docs/midway_check.md). This clean-checkout workflow reproduces the
selected experiment; it does not rerun the historical DEV selection.

## Developer path: reproduce and extend the experiment

Follow this full path when you are a developer starting from a fresh clone and want to rebuild
the TRAIN/DEV evidence, choose a configuration, and then run TEST. This is separate from the
selected-run instructions above. It uses pretrained Ollama models—**there is no model training
or fine-tuning**. Do not start this path while another TEST evaluation is running in the same
checkout.

### A. Clone and install

Install Python 3.11+, Git, and Ollama. Open Git Bash and run:

```bash
git clone https://github.com/DKR-KG-Construction/KG-construction.git
cd KG-construction
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.lock
bash run.sh test
bash run.sh data
ollama pull qwen2.5:3b
ollama pull llama3.2:3b
```

Keep Ollama running. The extractor is `qwen2.5:3b`; the optional provenance judge is
`llama3.2:3b`.

### B. Build the TRAIN schema and run TRAIN development

Run these commands in order:

```bash
bash run.sh pilot
bash run.sh train-baseline 200 outputs/raw/train_baseline.jsonl
```

`pilot` builds `schema/schema.json` from TRAIN and runs a 10-document smoke-test pilot.
`train-baseline` extracts 200 TRAIN documents and evaluates the configured nested sizes and
seeds. These steps generate local files under `outputs/` and `results/`; they do not alter model
weights. Complete any optional TRAIN annotation calibration separately, before final TEST
annotation.

### C. Extract DEV once and compare configurations

Extract DEV once. Reuse this same raw extraction for both judge settings:

```bash
bash run.sh extract-dev
```

First evaluate with the checked-in setting (`judge.enabled: false`), which means P ablations
use quote-existence validation only:

```bash
PYTHONPATH=src python -m kgrepair evaluate --raw outputs/raw/dev.jsonl --split dev \
  --out-dir results/dev_quote_only
```

Next, make a temporary config copy with the LLM judge enabled and evaluate the same DEV output.
The commands below do not change `config.yaml`:

```bash
cp config.yaml config_judge.yaml
python -c "import yaml; from pathlib import Path; p=Path('config_judge.yaml'); c=yaml.safe_load(p.read_text(encoding='utf-8')); c['judge']['enabled']=True; p.write_text(yaml.safe_dump(c, sort_keys=False), encoding='utf-8')"
PYTHONPATH=src python -m kgrepair --config config_judge.yaml evaluate \
  --raw outputs/raw/dev.jsonl --split dev --out-dir results/dev_with_judge
rm config_judge.yaml
```

Compare `results/dev_quote_only/auto_summary.csv` with
`results/dev_with_judge/auto_summary.csv`. Select the final setup using the proposal's
pre-specified criterion—not TEST. The current project selection is C+S with P disabled and
`judge.enabled: false`; the DEV comparison and exploratory AI audit are documented in
[`docs/midway_check.md`](docs/midway_check.md).

### D. Freeze the chosen setup before TEST

Set `config.yaml` to the selected final values. Make sure the final TEST data is downloaded and
verified, then write the freeze manifest:

```bash
bash run.sh data
bash run.sh freeze-test
```

Review and commit the final `config.yaml`, prompt/schema if changed, and `docs/test_freeze.json`
before running TEST. Once frozen, do not edit these files. If you deliberately change the
configuration, prompt, or schema, review the DEV evidence and run `bash run.sh freeze-test`
again before TEST.

### E. Run TEST, create the manual sample, and score annotation

With the same checkout and unchanged frozen inputs:

```bash
bash run.sh extract-test
bash run.sh evaluate-test
bash run.sh proposal-sample
```

After the proposal sample is generated, follow section 3 above: the primary annotator completes
`annotation/proposal_test/annotator1.csv`; the second annotator completes the 20% overlap in
`annotator2.csv`; neither sees `key.json`. Then score:

```bash
bash run.sh proposal-score
```

Generated data, model responses, annotation sheets, and result files are local and Git-ignored.
Share annotation materials with teammates through your approved channel; do not commit them
unless the team explicitly approves. This workflow reproduces the experiment structure, but
LLM outputs and resulting metrics may vary with model/runtime versions. The committed historical
DEV results are reference results, not guaranteed bit-for-bit outputs from a new run.

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
