# Verified local status (2026-11-10)

The sections below are historical development notes. Their statements that TEST data,
a freeze manifest, or TEST outputs are absent have been superseded by this verification.
For current progress and remaining tasks, use the verified status above. Historical claims and
commands in the earlier interim report must not be used as the current run instructions.

- The existing TEST freeze validates against the current configuration, prompt, schema,
  and dataset. All 500 cached records have the frozen extractor fingerprint.
- Of 500 responses, 356 parse normally, 89 recover complete prefixes, and 55 remain
  unparseable. Failed responses remain in the evaluation population with no extracted edges.
- Cached-output evaluation was verified under `results/test_verified/`: all eight
  configurations, four nested sizes, and five seeds. This uses the original cached
  responses and does not rerun the extractor or select a new configuration.
- `rejections.jsonl` records rejected raw IDs with document, configuration, stage, and reason.
  `run_metadata.json` records the provenance mode, fingerprints, parsing counts, and sampling settings.
- P is **quote-existence-only** in this frozen experiment. Full proposal P also requires
  support validation; the judge implementation exists but is disabled after the DEV comparison.
  The current experiment must be reported with that deviation. TEST results cannot justify
  changing the judge or selecting a replacement configuration. Any further method development
  belongs on TRAIN/DEV and requires an explicitly documented follow-up experiment.
- The existing manual sample has 100 accepted triples per main stage, 375 distinct annotation
  rows, and 75 overlap rows. Primary labels remain blank. Manual precision and annotator
  agreement are pending independent human annotation.
- Scoring now rejects blank, duplicate, unexpected, and missing assigned labels. Retention
  includes uncertain items; manual precision follows the configured uncertain-label policy.
  Per-stage uncertainty counts, provenance mode, and overlap agreement are saved with scores.
- The final code verification is recorded in the change report; the historical tests and
  machine/repository descriptions below describe the earlier development environment.

# Historical TRAIN and DEV preliminary evidence

The tables below preserve the earlier development evidence. Local raw TRAIN/DEV files
are absent from this checkout, so their reported values cannot be independently recomputed
here. The TEST artifacts and annotation sample in the verified status above are present.
These development results motivated selecting C+S and disabling the judge before TEST.

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

## Remaining midway work

Complete the independent annotation using `docs/annotation_guidelines.md` and
`annotation/proposal_test/docs.txt`. Annotator 1 labels all 375 assigned items; annotator 2
labels the assigned 75 overlap items. Keep `key.json` private to scoring.
After both files are complete, run `bash run.sh proposal-score`.

The midway presentation can use the TRAIN/DEV preliminary tables above with their historical
status disclosed, and the verified automatic TEST results under `results/test_verified/`.
Manual precision and agreement remain pending; quote-only P and parsing failures must be
reported alongside the results. Further tuning on the existing TEST results is not part of
this annotation workflow.
