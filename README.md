# LLM-Assisted Knowledge Graph Construction with a Precision Budget

This project measures how repair steps change the precision of an LLM-extracted knowledge
 graph and how much correct content they remove. It uses the official Re-DocRED splits,
 a pretrained local extractor, cached responses, and independent manual annotation.
 There is no model training or fine-tuning.

## Current status

The midway experiment has completed automatic TEST evaluation and two-person manual annotation.
The extractor is `qwen2.5:3b`. The DEV-selected configuration is **C+S**; the optional
`llama3.2:3b` provenance judge is disabled in the frozen experiment.

The main comparison is `raw`, `C`, `C+S`, and `C+S+P`; automatic evaluation also covers all
eight C/S/P combinations. Each document is extracted once, and its saved response is reused.

| Stage | Operation |
|---|---|
| raw | Parsed extractor output before repairs |
| C | Within-document alias merging using normalized names and unambiguous person-name containment |
| S | Structural and schema checks, including relation validity, endpoints, direction, and allowed type pairs |
| P | Quote-existence validation in the current experiment; optional LLM support validation is implemented |

**Protocol limitation:** the proposal describes P as checking both quotation existence and
support for the relation. The reported P ablation checks quote existence only. It must be
reported as `quote_only`, rather than full support validation. C+S was selected using DEV;
the manual TEST results below do not change that selection.

## Manual results

The primary annotator completed 375 distinct items, sampled as 100 accepted triples per main
stage from the combined five-seed, 200-document TEST runs. Cross-stage duplicate claims were
labelled once. The second annotator independently checked 75 items (20% overlap).

| Stage | Correct / determinate labels | Manual precision | 95% Wilson interval | Raw-sample retention | Correct raw facts removed |
|---|---:|---:|---:|---:|---:|
| raw | 4 / 98 | 4.1% | 1.6-10.0% | 100% | 0 / 4 |
| C | 13 / 97 | 13.4% | 8.0-21.6% | 100% | 0 / 4 |
| C+S | 23 / 100 | 23.0% | 15.8-32.2% | 25% | 1 / 4 |
| C+S+P (quote only) | 35 / 100 | 35.0% | 26.4-44.7% | 23% | 1 / 4 |

Five uncertain labels (two raw, three C) are excluded from precision denominators under the
frozen `annotation.uncertain: exclude` policy. Agreement is **73/75 (97.3%)**, with
**Cohen's kappa = 0.917**. Primary labels determine precision; secondary labels measure
agreement and are not automatically substituted or adjudicated.

Measured precision increases across the sampled stages while schema and provenance filtering
remove substantial content. Only four raw-sample facts were labelled correct, so the 25%
correct-removal estimate is based on one fact and is highly uncertain. Stage samples are not
fully paired; these point estimates alone do not establish a statistically significant
improvement between stages.

These values come from the local `results/proposal_manual_summary.csv`. Raw annotations and
generated results are Git-ignored; this table preserves the aggregate results in the repository.
The [midway report](docs/midway_check.md) contains earlier TRAIN/DEV evidence and experiment
context; its annotation-pending status describes the period before this scoring run.

## Automatic evaluation and limitations

- Official splits: 3,053 TRAIN, 500 DEV, and 500 TEST documents. A schema of 14 relations and
  six entity types is derived from TRAIN. Gold labels are never included in real-model prompts.
- TEST uses nested samples of 25, 50, 100, and 200 documents with five seeds: 0-4. This produces
  160 metric rows across eight configurations. Seed 0 at 200 documents is the configured
  primary automatic run; manual precision is the primary quality measure.
- All 500 TEST responses are cached: 356 parse normally, 89 recover complete prefixes, and
  55 remain unparseable. Failed responses contribute no extracted edges but remain in the
  document population. Partial recovery can omit facts and affects interpretation.
- Automatic matching uses exact normalized mention overlap against gold restricted to schema
  relations. Incomplete gold and heuristic entity matching limit automatic precision estimates.
  Automatic F1 uses unique-triple precision to penalize duplicate predictions.
- Raw-sample retention above measures whether sampled raw facts remain represented, counting
  merged facts as retained. The scoring CSV also reports aggregate graph-edge retention across
  seed runs; deduplication can reduce that measure even when no raw facts disappear.
- Samples overlap across seeds and items can share documents, so seed variation and item-level
  confidence intervals do not capture every source of uncertainty.
- Public Wikipedia-derived text may overlap with model pretraining. Model/runtime versions
  can also affect regenerated responses despite deterministic generation settings.

## Install

Requirements: Python 3.11+, Git, Git Bash for the shell runner, and Ollama for new LLM extraction.
Ollama is not needed to score existing annotation files or evaluate cached outputs with the
current judge-disabled configuration.

From Git Bash on Windows:

```bash
git clone https://github.com/DKR-KG-Construction/KG-construction.git
cd KG-construction
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.lock
bash run.sh test
```

On Linux/macOS, activate with `source .venv/bin/activate`. Direct dependencies are in
`requirements.txt`; `requirements.lock` pins the environment. Tests use synthetic data and mock
models; HTTP backend checks use a local fake server. Test outputs are not experimental results.

## Reproduce the frozen experiment from a clean checkout

The repository includes the selected configuration, prompts, TRAIN schema, and TEST freeze
manifest. Dataset files and generated artifacts must be obtained or recreated separately.
Run the following from Git Bash with the environment activated and Ollama running:

```bash
bash run.sh data
ollama pull qwen2.5:3b
bash run.sh extract-test
bash run.sh evaluate-test
bash run.sh proposal-sample
```

Data downloads are pinned and checked against SHA-256 checksums. Extraction is resumable and
may take a long time. TEST commands validate the existing freeze and extraction fingerprints.
Keep the frozen settings unchanged when reproducing this experiment. A freeze mismatch needs
investigation; regenerating the manifest does not establish that an altered run is equivalent.

Evaluation writes `auto_metrics.csv` (each seed), `auto_summary.csv`, `per_doc.csv`,
`rejections.jsonl`, and `run_metadata.json` under `results/test/`. The existing local verification
run is under `results/test_verified/`. Rejections include source IDs, stage, and reason;
metadata identifies the provenance mode and parsing counts.

A new extraction or sample requires new manual labels; the table above is not guaranteed to be
reproduced by rerunning the model. Sampling refuses to overwrite existing proposal annotation
files. Preserve completed annotations rather than regenerating their sample.

## Annotation and scoring

The completed local sample is under `annotation/proposal_test/`:

- `annotator1.csv`: all 375 primary items.
- `annotator2.csv`: the assigned 75-item independent overlap.
- `docs.txt`: source documents used for judging claims.
- `key.json`: private stage mapping used by scoring; keep it hidden from annotators.

Follow [the annotation guidelines](docs/annotation_guidelines.md). Judge claims against the full
source document, fill only `label` and `notes`, and preserve IDs and evidence columns. Labels
are `correct`, `incorrect`, or `uncertain`. An incorrect quote alone does not invalidate a fact
supported elsewhere in the document. Preserve original independent labels before discussing
any disagreements.

To rescore the completed files in Git Bash:

```bash
bash run.sh proposal-score
```

Or in PowerShell, from the repository root:

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m kgrepair score-proposal-annotation
```

Scoring validates the frozen setup, assigned IDs, and complete labels. It writes
`results/proposal_manual_summary.csv`, including Wilson and bootstrap intervals, uncertainty
counts, retention, correct-fact removal, and overlap agreement. It does not require new LLM calls.

## Development commands

Use a separate experiment checkout for further development. Prompt/rule changes belong on TRAIN,
with DEV used for selection before a new TEST experiment. Existing TEST outcomes must not guide tuning.

| Command | Purpose |
|---|---|
| `bash run.sh pilot` | Build the TRAIN schema and extract a 10-document TRAIN smoke test |
| `bash run.sh train-baseline 200` | Extract and evaluate a larger TRAIN sample |
| `bash run.sh extract-dev` | Cache DEV extraction |
| `bash run.sh evaluate-dev` | Compare repairs on cached DEV outputs |
| `bash run.sh freeze-test` | Record inputs for a deliberately selected new experiment |

The historical DEV judge comparison informed disabling the judge. Changing judge generation
settings now invalidates its cache keys; historical cached judgments are not reused as results
from the revised settings. Optional judge-enabled development requires `llama3.2:3b` in Ollama.

## Repository layout

```text
config.yaml                    selected experiment settings
prompts/                       extractor and optional judge prompts
schema/schema.json             frozen TRAIN-derived schema
src/kgrepair/                  extraction, repairs, evaluation, annotation, statistics
scripts/                       pinned dataset download and checksums
tests/                         synthetic pipeline and HTTP backend checks
docs/test_freeze.json           existing TEST freeze manifest
docs/annotation_guidelines.md   shared annotation rules
docs/midway_check.md            historical preliminary evidence and run context
results/pilot_metrics.csv       tracked TRAIN pilot results
```

`data/Re-DocRED/`, `outputs/`, `annotation/`, and most `results/` files are ignored by Git.
Keep local backups of cached responses, annotation files, stage keys, and result tables if
sharing or archiving the study. A source-code clone alone does not include the completed labels.

## Citation and disclosure

Dataset: Tan et al., *Revisiting DocRED: Addressing the False Negative Problem in Relation
Extraction*, EMNLP 2022. See the [Re-DocRED repository](https://github.com/tonytan48/Re-DocRED).
The dataset repository uses the MIT License; its Wikipedia-derived text has CC BY-SA provenance.
Dataset files are not redistributed in this repository.

AI tools assisted with code and documentation. The project authors are responsible for
checking the implementation, annotation process, and results, and for disclosing tool use
according to course policy.
