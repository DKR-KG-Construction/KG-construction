#!/usr/bin/env bash
# Downloads Re-DocRED (MIT-licensed repo; text derived from Wikipedia, CC BY-SA) at a PINNED commit
# and verifies checksums. The data is not redistributed in this repo (see .gitignore).
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=ccfb54f5ddf5836027c87badda10f6dfc56efaac
DEST=data/Re-DocRED
mkdir -p "$DEST"
for f in train_revised dev_revised test_revised; do
  if [ ! -f "$DEST/$f.json" ]; then
    curl -fL "https://raw.githubusercontent.com/tonytan48/Re-DocRED/$COMMIT/data/$f.json" -o "$DEST/$f.json"
  fi
done
cp scripts/CHECKSUMS.sha256 "$DEST/CHECKSUMS.sha256"
(cd "$DEST" && sha256sum -c CHECKSUMS.sha256)
echo "Re-DocRED ready in $DEST (commit $COMMIT)"
