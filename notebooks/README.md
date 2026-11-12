# Colab runners

The workflow has a configuration-search phase and a final frozen-experiment phase:

1. `01_colab_experiment_setup.ipynb` creates prompt, relation-schema, size-preset, and YAML variants on Google Drive. It makes no LLM calls.
2. `02_colab_experiment_runner.ipynb` runs selected TRAIN/DEV variants, exports trial KGs, and records standard plus common-scope metrics. TEST is intentionally blocked.
3. After a human selects and explicitly freezes one prompt and relation set, `03_colab_extract.ipynb` runs resumable final extraction. Raw JSONL is written to Google Drive after every document.
4. `04_colab_evaluate_annotate.ipynb` evaluates cached extraction, creates the manual-annotation package, and scores completed labels. It does not require a GPU or Ollama for the frozen quote-only experiment.

All notebooks use the same Google Drive root by default:

```text
MyDrive/DKR_KG_Construction/
```

The two extraction notebooks can be safely restarted after a Colab disconnect. Re-run their setup cells and then the extraction cell; `kgrepair` skips successfully cached documents with the same extractor fingerprint.

The configuration runner supports five selection modes (`prompts`, `relations`, `factorial`, `baseline`, and `custom`) and five data presets (`pilot`, `small`, `medium`, `large`, and `all`). Size presets distinguish the extraction pool from evaluation sample sizes so multiple seeds remain meaningful.

The setup notebook discovers the complete positive relation catalog from Re-DocRED TRAIN and supports relation-count variants up to all 96 relations. English Wikidata labels and descriptions are cached on Drive; the project's curated descriptions take precedence for its original 14 relations.

Do not choose a configuration on TEST. Narrow candidates on TRAIN, confirm the final choice on DEV, then hard-code and freeze that choice before final TEST extraction.

Do not regenerate an annotation sample after annotation has started. Keep `key.json` private from annotators until independent labelling is complete.
