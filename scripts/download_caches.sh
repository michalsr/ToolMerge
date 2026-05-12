#!/usr/bin/env bash
# Download precomputed SigLIP / T-REN / OCR caches from the Hugging Face Hub.
#
# PLACEHOLDER — set TOOLMERGE_CACHES_REPO to the actual HF Hub repo once
# we publish it. Until then, build the caches yourself with cache_build/,
# or contact the authors for an out-of-band bundle.

set -euo pipefail

HF_REPO="${TOOLMERGE_CACHES_REPO:-toolmerge/caches}"
DEST="${TOOLMERGE_CACHE_DIR:-$(dirname "$(realpath "$0")")/../caches}"
mkdir -p "$DEST"

echo "Downloading caches from $HF_REPO -> $DEST"
echo "(skipped — repo not yet published; see cache_build/README.md to build locally)"
