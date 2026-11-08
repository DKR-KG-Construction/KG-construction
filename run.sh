#!/usr/bin/env bash
# One command per stage. Usage: bash run.sh <stage>
#   test     unit + end-to-end tests (offline, mock LLM; no GPU, no model needed)
#   data     download + verify Re-DocRED
#   pilot    schema + 10-doc pilot on TRAIN + pilot report      (week 1)
#   train-baseline [N]  extract N TRAIN docs and evaluate nested samples (default 200)
#   annotate-calibration  draw a separate calibration sample
#   annotate-train  draw a 50-item TRAIN pilot sample for annotators
#   freeze-test  freeze the final TEST configuration
#   extract-dev / extract-test  extract all documents from one split (resumable)
#   evaluate-dev / evaluate-test  run all eight ablations on cached split outputs
#   proposal-sample  draw proposal-aligned five-seed manual sample from TEST
#   proposal-score  score proposal annotation files
#   extract  extract all 500 DEV documents (resumable)          (only after freezing config/prompt/schema!)
#   evaluate automatic metrics for all 8 C/S/P combinations on DEV
#   sample   draw the paired manual-annotation sample
#   score    manual precision / retention / kappa from labelled CSVs
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src
if command -v python >/dev/null 2>&1 && python -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
  PYTHON=python
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
  PYTHON=python3
else
  echo "Python 3.11 or newer is required but was not found on PATH." >&2
  exit 1
fi
PY="$PYTHON -m kgrepair"
case "${1:-}" in
  test)     "$PYTHON" -m pytest -q tests ;;
  data)     bash scripts/download_data.sh ;;
  pilot)    RAW="${2:-outputs/raw/train_pilot.jsonl}"
            $PY build-schema
            $PY extract --split train --n 10 --out "$RAW"
            $PY pilot-report --raw "$RAW" --split train ;;
  train-baseline) N="${2:-200}"
            RAW="${3:-outputs/raw/train_baseline.jsonl}"
            $PY extract --split train --n "$N" --out "$RAW"
            $PY evaluate --raw "$RAW" --split train --configs main --no-judge --out-dir results/train ;;
  annotate-calibration) RAW="${2:-outputs/raw/train_baseline_6144.jsonl}"
            $PY sample-annotation --raw "$RAW" --split train --n 30 --overlap-frac 1 --no-judge \
              --out-dir annotation/calibration --exclude-key annotation/key.json ;;
  annotate-train) RAW="${2:-outputs/raw/train_pilot.jsonl}"
            $PY sample-annotation --raw "$RAW" --split train --n 50 --overlap-frac 1 --no-judge ;;
  freeze-test) $PY freeze-test ;;
  extract-dev) $PY extract --split dev --n all --out outputs/raw/dev.jsonl ;;
  extract-test) $PY extract --split test --n all --out outputs/raw/test.jsonl ;;
  evaluate-dev) $PY evaluate --raw outputs/raw/dev.jsonl --split dev --out-dir results/dev ;;
  evaluate-test) $PY evaluate --raw outputs/raw/test.jsonl --split test --out-dir results/test ;;
  proposal-sample) RAW="${2:-outputs/raw/test.jsonl}"
            $PY sample-proposal-annotation --raw "$RAW" --out-dir annotation/proposal_test ;;
  proposal-score) $PY score-proposal-annotation ;;
  extract)  $PY extract --split dev --n all --out outputs/raw/dev.jsonl ;;
  evaluate) $PY evaluate --raw outputs/raw/dev.jsonl --split dev ;;
  sample)   $PY sample-annotation --raw outputs/raw/dev.jsonl --split dev ;;
  score)    $PY score-annotation --labels annotation/annotator1.csv --labels2 annotation/annotator2.csv --labels3 annotation/annotator3.csv ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
