#!/bin/bash
#SBATCH --account=bcgp-delta-gpu
#SBATCH --job-name=ra_h250_topk_sig
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/%x_%A_%a.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/%x_%A_%a.err
#SBATCH --time=0:15:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpuA100x4,gpuA100x4-preempt,gpuA100x8,gpuA40x4,gpuA40x4-preempt
#SBATCH --gpus-per-node=1
#SBATCH --array=0-12
#SBATCH --requeue
#
# Reanswer — m2c_v2 hard 250 — Qwen3-VL-8B (LIF prompt)
# Source: top-k SigLIP frame selection with auto greedy gap.
# K controlled via env var K (default 8). 250 items, 20/chunk, 13 chunks.

set -euo pipefail

export TRITON_CACHE_DIR=/tmp/michal5
export HF_HOME=/tmp/michal5

K=${K:-8}
WORK_DIR="/work/hdd/bcgp/michal5/verify_video/multi_turn"
SOURCE_DIR="/work/hdd/bcgp/michal5/WFS-SB/output/molmo2cap_v2_hard250/reanswer_source_topk_siglip_${K}f"
DATASET="/work/hdd/bcgp/michal5/verify_video/multi_turn/evidence_pipeline_v2/dataset_generation/group_v2/hard_gpt52_wrong_250.json"
VIDEO_DIR="/work/hdd/bcgp/michal5/molmo2_cap/videos"
OUTPUT_ROOT="/work/hdd/bcgp/michal5/WFS-SB/output/molmo2cap_v2_hard250/reanswer_topk_siglip_${K}f_q3_lif"

CHUNK_SIZE=20
TOTAL=250
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
START=$((TASK_ID * CHUNK_SIZE))
END=$((START + CHUNK_SIZE))
[ "$END" -gt "$TOTAL" ] && END="$TOTAL"
[ "$START" -ge "$TOTAL" ] && { echo "Nothing to do."; exit 0; }

SAVE_DIR="${OUTPUT_ROOT}/chunk_${TASK_ID}"

echo "Reanswer m2c_v2 hard250 K=${K} Qwen3-VL-8B (LIF) — chunk ${TASK_ID} (${START}:${END})"
echo "Source frames: ${SOURCE_DIR}"
echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME} at $(date)"

source /sw/external/python/anaconda3/bin/activate
conda activate /projects/bcgp/michal5/envs/deepspeed_video

cd "$WORK_DIR"

python -m evidence_pipeline_v2.baselines.reanswer \
    --source-dir "$SOURCE_DIR" \
    --dataset "$DATASET" \
    --video-dir "$VIDEO_DIR" \
    --model-path Qwen/Qwen3-VL-8B-Instruct \
    --prompt-template lif \
    --temperature 0 \
    --video-backend opencv \
    --start-idx $START \
    --end-idx $END \
    --output-dir "$SAVE_DIR"

echo "Done at $(date). Results: $SAVE_DIR"
