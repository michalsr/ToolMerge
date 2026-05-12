#!/usr/bin/env bash
# Download T-REN weights from the Hugging Face Hub into tren/weights/.
#
# Replace `TOOLMERGE_TREN_REPO` with the actual HF Hub repo once the release
# is up. For now this is a placeholder.

set -euo pipefail

HF_REPO="${TOOLMERGE_TREN_REPO:-toolmerge/tren-weights}"
DEST="$(dirname "$(realpath "$0")")/../tren/weights"
mkdir -p "$DEST"

echo "Downloading T-REN weights from $HF_REPO -> $DEST"

python - <<PY
from huggingface_hub import snapshot_download

dest = "$DEST"
snapshot_download(
    repo_id="$HF_REPO",
    repo_type="model",
    local_dir=dest,
    allow_patterns=["*.pth"],
)
print("Done.")
PY

echo
echo "Files now in tren/weights/:"
ls -lh "$DEST"
