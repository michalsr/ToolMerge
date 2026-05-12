#!/bin/bash
#SBATCH --account=bcgp-delta-gpu
#SBATCH --job-name=ra_wfs_w8v_cv
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/%x_%A_%a.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/%x_%A_%a.out
#SBATCH --time=0:30:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x4,gpuA100x8,gpuA40x4,gpuA100x4-preempt,gpuA40x4-preempt
#SBATCH --gpus-per-node=1
#SBATCH --array=0-69
#SBATCH --signal=B:USR1@90
#SBATCH --requeue
#
# Reanswer — Video-MME Long — 8 frames — Qwen3-VL-8B (LIF prompt)
# Source: WFS-SB wavelet frame selection
# 900 items, 13/chunk, 70 chunks — Delta — opencv backend
# Re-run of reanswer_vmme_8f_wfs_wavelet.slurm with opencv instead of decord

set -euo pipefail

_requeue() { echo "Preempted at $(date), requeuing"; scontrol requeue "$SLURM_JOB_ID"; }
trap '_requeue' USR1

export TRITON_CACHE_DIR=/tmp/michal5
export HF_HOME=/tmp/michal5

WORK_DIR="/work/hdd/bcgp/michal5/verify_video/multi_turn"
SOURCE_DIR="/work/hdd/bcgp/michal5/WFS-SB/output/vmme_all/reanswer_source_8f"
DATASET="/work/hdd/bcgp/michal5/verify_video/data/video_mme/video_mme_long.json"
VIDEO_DIR="/projects/bcgp/michal5/Video-MME/videos/data"
OUTPUT_ROOT="/work/hdd/bcgp/michal5/WFS-SB/output/vmme_long/reanswer_wfs_wavelet_8f_q3_lif_cv2"

CHUNK_SIZE=13
TOTAL=900
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
START=$((TASK_ID * CHUNK_SIZE))
END=$((START + CHUNK_SIZE))
[ "$END" -gt "$TOTAL" ] && END="$TOTAL"
[ "$START" -ge "$TOTAL" ] && { echo "Nothing to do."; exit 0; }

SAVE_DIR="${OUTPUT_ROOT}/chunk_${TASK_ID}"

echo "Reanswer VMME 8f WFS-wavelet Qwen3-VL-8B (LIF, opencv) — chunk ${TASK_ID} (${START}:${END})"
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
    --output-dir "$SAVE_DIR" &

wait $!
echo "Done at $(date). Results: $SAVE_DIR"
