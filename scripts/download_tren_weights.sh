#!/usr/bin/env bash
# Download T-REN model weights into tren/ for the toolmerge runtime.
#
# Pulls the T-REN region-encoder checkpoint from the canonical Hugging Face
# Hub release (https://huggingface.co/savyak2/T-REN) into the location the
# vendored tren/ package looks for it.
#
# The DINOv3 backbone weights are NOT released by the T-REN repo — see
# tren/README.md for how to obtain them from Facebook Research's dinov3 repo.

set -euo pipefail

HF_REPO="${TOOLMERGE_TREN_REPO:-savyak2/T-REN}"
DEST="$(dirname "$(realpath "$0")")/../tren"
mkdir -p "$DEST"

echo "Downloading T-REN region encoder from $HF_REPO -> $DEST/"

python - <<PY
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="$HF_REPO",
    repo_type="model",
    filename="tren_region_encoder.pth",
    local_dir="$DEST",
)
print(f"Downloaded: {path}")
PY

# The toolmerge runtime currently expects the file at tren/best_checkpoint.pth
# (see tren/video_query_search/config.yaml). Symlink so both names resolve.
if [ ! -e "$DEST/best_checkpoint.pth" ]; then
  ln -s tren_region_encoder.pth "$DEST/best_checkpoint.pth"
fi

echo
echo "T-REN weights in tren/:"
ls -lh "$DEST"/tren_region_encoder.pth "$DEST"/best_checkpoint.pth 2>/dev/null || true

echo
echo "NOTE: the DINOv3 ViT-L/16 backbone weights are not in the T-REN release."
echo "      Place them under tren/ before running cache_build/tren.py:"
echo "        tren/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
echo "        tren/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
echo "      Source: https://github.com/facebookresearch/dinov3"
