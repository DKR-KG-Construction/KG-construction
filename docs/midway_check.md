# Interim report — TRAIN and DEV results, limitations, and next steps

This report summarizes the preliminary TRAIN pilot and completed DEV automatic comparison. It is
not the final evaluation: proposal-compliant manual annotation and the held-out TEST run remain
incomplete. The workflow is to develop on TRAIN, select and freeze the configuration on DEV, and
only then access TEST. The DEV rejection audit below used an AI reviewer and is exploratory, not
independent human annotation.

## Current verified state

- Re-DocRED was downloaded from the pinned commit in `scripts/download_data.sh`; all three SHA-256 checks passed.
- `schema/schema.json` was generated from TRAIN only using the 14 relations configured in `config.yaml`. The data has six coarse entity types: `LOC`, `MISC`, `NUM`, `ORG`, `PER`, and `TIME`.
- The default automatic entity match is exact overlap after normalizing mention strings. Confirm that this matches the submitted proposal before presenting results.
- The real local model `qwen2.5:3b` ran on the RTX 4050 Laptop GPU. Ollama reports model ID `357c53fb659c`, model size 1.9 GB, GPU processor 100%, and Ollama version `0.40.11`.
- The offline suite passes in the pinned environment: 16 tests passed, including a regression fix for freezing configurations whose config file is outside the repository root. These tests use `MockBackend`, which reads gold labels; **their numbers are not project results and must not be reported as model metrics**.
- The extractor (`qwen2.5:3b`) and a different-family judge (`llama3.2:3b`) are installed locally. The selected TRAIN pilot used quote-rule P with the LLM judge disabled. A separate corrected-context judge-assisted comparison has since been completed on DEV and is summarized below.
- The selected 50-document TRAIN run used `qwen2.5:3b`, `max_tokens=6144`, and `num_ctx=12288`. Raw responses are in `outputs/raw/train_baseline_6144.jsonl`; main-stage metrics are in `results/pilot_metrics.csv` and all-eight-stage metrics recomputed with current parsing are in `results/train_6144_reparsed/`. Fifteen outputs reached the 6,144-token cap. Ten were safely recovered as partial JSON because their entity arrays were complete; five remain unrecoverable because the entity list was truncated (doc IDs 4, 9, 22, 33, 38). Metrics use only complete decoded triple objects from recovered responses. It produced 1,048 parsed triples across 45 usable documents; 5 documents have no parsed graph. Quotes were found in the document for 70.8% of triples and in the claimed sentence for 53.0%. Mean latency was 43.0 s/doc; usage totaled 43,890 prompt and 128,243 completion tokens (local model; no API charge).
- For n=50 all five configured seeds use the same complete pool, so across-seed standard deviations are zero and do not represent sampling uncertainty. At n=25, the seeds produce different subsets.
- DEV automatic evaluation is complete for 200 documents and five configured seeds. It compares C+S, C+S+P with the quote-existence rule only, and C+S+P with the corrected-context LLM judge. The judge-assisted setting retained fewer than 0.1% of raw triples on average; see the DEV table below.
- Based on the DEV automatic comparison, the selected configuration is C+S without P; `judge.enabled` is now false. This is a DEV-based selection, not a TEST result. TEST has not been run or accessed.
- The DEV rejection audit reviewed 812 unique C+S edges rejected by the original cached judge, with zero tool errors. It used a local AI model, not human annotators; 528/812 cases (65.0%) were uncertain on at least one of the two labels.
- `bash run.sh annotate-train outputs/raw/train_baseline_6144.jsonl` prepared a 50-triple TRAIN pilot sample for all three annotators; `annotation/annotator1.csv`, `annotator2.csv`, and `annotator3.csv` have no labels entered. A separate 30-triple calibration sample is prepared under `annotation/calibration/`; its annotator CSVs are also blank. These are pilot/calibration materials, not the proposal's final evaluation sample.
- A prompt revision was tested to request only triple-endpoint entities, unique surface forms, and compact JSON. On the five previously unrecoverable TRAIN docs at 6,144 tokens, it parsed three fully, recovered one partially, and failed on doc 4; raising the cap to 8,192 did not fix docs 4 or 22. A 50-doc run of that revision was stopped after 16 docs: 12 parsed, 1 was partially recovered, and 4 were unparseable. That incomplete trial is retained at `outputs/raw/train_baseline_prompt_v2_8192.jsonl` for diagnosis only. The original prompt/6,144-token baseline remains selected. Parser normalization now removes duplicate mention strings; `results/pilot_metrics.csv` and `results/train_6144_reparsed/` were recomputed from the selected baseline using current parsing.

## Preliminary model results

These are preliminary automatic lower bounds from only 50 TRAIN documents (1,048 complete parsed
triples, including partial recoveries, in 45 usable documents), using exact normalized mention
overlap. Do not treat them as stable estimates or held-out results; 15 outputs reached the token
cap, five of which remain unrecoverable, and manual labels are still needed.

| Configuration | Precision | Recall | F1 | Raw triples retained | Retention | Correct raw facts removed* | Correct-removed share* |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.0403 | 0.0228 | 0.0291 | 1,048/1,048 | 100.0% | 0 | 0.0% |
| C | 0.0356 | 0.0195 | 0.0252 | 1,048/1,048 | 100.0% | 0 | 0.0% |
| C+S | 0.1558 | 0.0130 | 0.0240 | 217/1,048 | 20.7% | 15 | 15.0% |
| P (quote rule) | 0.0544 | 0.0206 | 0.0299 | 742/1,048 | 70.8% | 16 | 16.0% |
| C+S+P (quote rule) | 0.1803 | 0.0119 | 0.0224 | 178/1,048 | 17.0% | 30 | 30.0% |

*“Correct raw facts removed” and its share are automatic matches against benchmark gold from
`results/pilot_metrics.csv` (`correct_removed` and `correct_removed_share`), not human-adjudicated
counts. The 100 raw gold-correct triples are an automatic reference for this pilot only.

Automatic precision is a lower bound because Re-DocRED gold can be incomplete and matching is
heuristic. Do not substitute `MockBackend` output.

## DEV configuration comparison

These are automatic results for the 200-document DEV sample across five configured seeds. They
are development evidence, not held-out estimates. The "correct-removed share" is based on
automatic gold matching, not human review.

| Configuration | Precision | Recall | F1 | Raw retention | Correct-removed share |
|---|---:|---:|---:|---:|---:|
| C+S | 0.1200 | 0.01249 | 0.02262 | 22.07% | 30.05% |
| C+S+P (quote rule only) | 0.1133 | 0.00882 | 0.01636 | 16.84% | 51.10% |
| C+S+P (corrected-context LLM judge) | 0.4200 | 0.000329 | undefined | 0.096% | 98.80% |

The judge-assisted precision is based on only 16 accepted edges across the five seeds; precision
standard deviation was 0.266. Its high precision is therefore not a useful standalone result:
recall was near zero and almost all automatically correct raw facts were removed. The current judge should not be used for the final configuration. Among these automatic proxies,
C+S has the highest F1 and recall, while quote-rule-only P has lower retention and removes more
correct raw facts. The selected final candidate is therefore C+S with P disabled, matching the
user's DEV-based choice. Freeze this configuration and confirm it follows the proposal protocol
before any TEST access; do not tune on TEST.

## Exploratory AI review of rejected DEV edges

The local AI audit evaluated 812 unique C+S edges rejected by the judge. It labeled fact
correctness separately from whether any cited quote/context supported the claim:

| Fact correctness | Quote supports fact | Count |
|---|---|---:|
| uncertain | uncertain | 505 |
| no | no | 235 |
| uncertain | no | 14 |
| yes | yes | 36 |
| yes | no | 13 |
| no | uncertain | 9 |

There were zero review-call errors. Only 36 cases (4.4%) were AI-labeled both factually correct
and quote-supported; 13 (1.6%) were labeled correct but unsupported by the cited evidence. The
remaining 65.0% were uncertain on at least one dimension. These results do not establish judge
false-rejection accuracy: labels are AI-generated, many are uncertain, and no independent human
adjudication has been completed. Treat the audit as exploratory context only, not as proposal
annotation or a precision estimate. Its CSV and raw DEV artifacts are local, Git-ignored files.

## Blockers and actions already tried

| Blocker | Evidence / attempted action | Next action |
|---|---|---|
| Git repository boundary | The original working tree is nested under the user profile and the saved Copilot project mapping referred to the wrong GitHub owner. A clean local snapshot was prepared without Git metadata or generated artifacts; no files were pushed. |
| Output truncation and runtime | Fifteen of 50 outputs reached `max_tokens=6144`; ten were salvaged as complete triple prefixes and five remain unrecoverable because their entity lists were incomplete. The tested prompt revision was worse (4/16 unparseable), so it was not adopted. Mean latency for the selected run is 43.0 seconds/doc. | Present the selected 50-document baseline with the truncation limitation disclosed. Any further prompt/model experiment must use TRAIN and a fresh raw-output path. |
| `S` may be weak with coarse types | On this TRAIN sample, C+S retained 217 of 1,048 raw triples (C+S+P retained 178); TRAIN-derived schema uses six broad types. | Ask TA whether these coarse types support a meaningful schema filter. |
| Human quality evidence is missing | The 30-triple calibration sample and 50-triple TRAIN pilot annotation files are prepared, but all annotator CSVs are blank. | Calibrate and freeze the guidelines, then complete the proposal's final manual evaluation on TEST before reporting human metrics. |
| TEST setup is not frozen | C+S without P is selected from DEV and `judge.enabled` is false, but the TEST data is not included in this clean snapshot and no freeze manifest exists. | Confirm the proposal protocol and exact config on the working machine, then run `bash run.sh freeze-test` before any TEST extraction. Do not access TEST until this freeze is approved. |
| Proposal matching and uncertainty choices need confirmation | The available chat summary says mention overlap; the implementation now uses exact normalized mention overlap. `annotation.uncertain` is currently `exclude`. | Compare the matching rule to the submitted proposal and ask TA how `uncertain` should enter the denominator. |
| Final TEST result is not run | No TEST freeze, raw TEST output, TEST metrics, or completed proposal annotation sample has been verified. | After the configuration freeze, run the prescribed five-seed, 200-document TEST evaluation once. Do not use TEST for configuration decisions or present TRAIN/DEV automatic metrics as final/held-out results. |
| Proposal manual-precision sample remains outstanding | The prepared 50-triple TRAIN sample is only a pilot; the 30-triple calibration set is separate, and neither has labels. | On the combined five-seed, 200-document TEST runs, manually assess 100 unique accepted triples per main stage (`raw`, `C`, `C+S`, `C+S+P`), balanced across seeds when possible. A second annotator checks 20%; cross-stage duplicate triples are labelled once. Report precision with an interval, retention, and correct raw facts removed. |
| Final GitHub repository is not initialized | The supplied `DKR-KG-Construction/KG-construction` repository is accessible but has no commits or branches. The prior Copilot project mapping refers to a different owner. | Reconnect the project to the intended repository, review the staged file list, and create/push only after approval. No push has been made. |

## TA questions

1. Are Re-DocRED's six coarse entity types sufficient for a meaningful `S` step?
2. Should `uncertain` annotations be excluded from the denominator or counted as incorrect?
3. Should the primary metric use a separately preselected seed, or is the configured seed sufficient?

## Commands for the next run

The local models are installed; the 50-document TRAIN pilot and 200-document DEV comparison have
run. This report does not claim proposal-compliant manual or final TEST results. Remaining work:
resolve any required proposal/TA decisions, freeze the selected configuration and annotation
guidelines, run the prescribed TEST evaluation once, and complete human annotation/scoring. To
recreate TRAIN pilot annotation samples:

```bash
bash run.sh test
bash run.sh data
bash run.sh annotate-train outputs/raw/train_baseline_6144.jsonl
bash run.sh annotate-calibration outputs/raw/train_baseline_6144.jsonl
```

Calibrate on the separate files first; only after the guidelines are frozen should annotators label
the final proposal sample. After labels are complete, run the appropriate scoring command.
`train-baseline` can be extended to larger
TRAIN-only samples if time allows. Generated data, responses, labels, and results are local and
Git-ignored.
