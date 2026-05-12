#!/bin/bash
#SBATCH --job-name=tm_cache
#SBATCH --output=cache_build/slurm/logs/%x_%A_%a.out
#SBATCH --error=cache_build/slurm/logs/%x_%A_%a.out
#SBATCH --time=4:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
# Adjust the partition line for your cluster.
#SBATCH --partition=gpuA100x4
#SBATCH --array=0
#
# Build SigLIP / T-REN / OCR / frame caches for a dataset.
#
# Required env vars (set before sbatch):
#   TOOLMERGE_CACHE_DIR   -- root cache directory
#   VIDEO_DIR             -- source mp4 directory
#   DATASET_JSON          -- JSON file listing the videos to process (matches
#                            data.input_path from the inference config)
#   DATASET_NAME          -- subdir name under each cache type, e.g. "molmo2_moments"
#   TOOL                  -- one of: siglip, tren, ocr, frame  (default: siglip)
#
# Example:
#   TOOLMERGE_CACHE_DIR=/scratch/caches \
#   VIDEO_DIR=/datasets/m2m/videos \
#   DATASET_JSON=/datasets/m2m/test.json \
#   DATASET_NAME=molmo2_moments \
#   TOOL=siglip \
#   sbatch cache_build/slurm/build_cache.sbatch

set -euo pipefail

: "${TOOLMERGE_CACHE_DIR:?required}"
: "${VIDEO_DIR:?required}"
: "${DATASET_JSON:?required}"
: "${DATASET_NAME:?required}"
TOOL="${TOOL:-siglip}"

OUT_DIR="${TOOLMERGE_CACHE_DIR}/${TOOL}/${DATASET_NAME}"
mkdir -p "$OUT_DIR"

echo "Building ${TOOL} cache: ${DATASET_JSON} -> ${OUT_DIR}"

cd "$(dirname "$(realpath "$0")")"/../..

# Choose the right --tools / --output_dir flags for the unified builder.
case "$TOOL" in
  siglip)
    OUT_FLAG="--siglip_output_dir $OUT_DIR" ;;
  tren)
    OUT_FLAG="--tren_output_dir $OUT_DIR" ;;
  ocr)
    OUT_FLAG="--ocr_output_dir $OUT_DIR" ;;
  frame)
    OUT_FLAG="--frame_output_dir $OUT_DIR" ;;
  *)
    echo "Unknown TOOL=$TOOL (expected siglip|tren|ocr|frame)" >&2
    exit 1 ;;
esac

python -m cache_build.build_caches \
    --video_dir "$VIDEO_DIR" \
    --dataset_json "$DATASET_JSON" \
    --tools "$TOOL" \
    $OUT_FLAG \
    --target_fps 2.0
