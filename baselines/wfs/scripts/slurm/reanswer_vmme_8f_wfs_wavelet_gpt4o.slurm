#!/bin/bash
#SBATCH --account=bcgp-delta-cpu
#SBATCH --job-name=ra_wfs_wav8vgpt
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/%x_%A_%a.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/%x_%A_%a.out
#SBATCH --time=2:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu
#SBATCH --array=0-8%2
#
# Reanswer — Video-MME — 8 frames — GPT-4o (LIF prompt)
# Source: WFS-SB wavelet frame selection
# 900 items, 100/chunk, 9 chunks, 2 concurrent (1 per endpoint)

set -euo pipefail

WORK_DIR="/work/hdd/bcgp/michal5/verify_video/multi_turn"
SOURCE_DIR="/work/hdd/bcgp/michal5/WFS-SB/output/vmme/reanswer_source_wavelet_8f"
DATASET="/work/hdd/bcgp/michal5/verify_video/data/video_mme/video_mme_long.json"
VIDEO_DIR="/projects/bcgp/michal5/Video-MME/videos/data"
OUTPUT_ROOT="/work/hdd/bcgp/michal5/WFS-SB/output/vmme/reanswer_wfs_wavelet_8f_gpt4o_lif"

CHUNK_SIZE=100
TOTAL=900
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
START=$((TASK_ID * CHUNK_SIZE))
END=$((START + CHUNK_SIZE))
[ "$END" -gt "$TOTAL" ] && END="$TOTAL"
[ "$START" -ge "$TOTAL" ] && { echo "Nothing to do."; exit 0; }

if [ $((TASK_ID % 2)) -eq 0 ]; then
    SECRETS=/work/hdd/bcgp/michal5/molmo2_cap/agentai.key
    ENDPOINT="https://michalagentai.cognitiveservices.azure.com/openai/v1/"
else
    SECRETS=/work/hdd/bcgp/michal5/molmo2_cap/micha_mlt2ioil.key
    ENDPOINT="https://micha-mlt2ioil-eastus2.cognitiveservices.azure.com/openai/v1/"
fi

SAVE_DIR="${OUTPUT_ROOT}/chunk_${TASK_ID}"

echo "Reanswer VMME 8f WFS-wavelet GPT-4o (LIF) — chunk ${TASK_ID} (${START}:${END})"
echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME} at $(date)"
echo "Endpoint: ${ENDPOINT}"

source /sw/external/python/anaconda3/bin/activate
conda activate /projects/bcgp/michal5/envs/deepspeed_video

cd "$WORK_DIR"

python -m evidence_pipeline_v2.baselines.reanswer \
    --source-dir "$SOURCE_DIR" \
    --dataset "$DATASET" \
    --video-dir "$VIDEO_DIR" \
    --model azure-gpt4o \
    --secrets-file "$SECRETS" \
    --api-endpoint "$ENDPOINT" \
    --prompt-template lif \
    --temperature 0 \
    --video-backend decord \
    --start-idx $START \
    --end-idx $END \
    --output-dir "$SAVE_DIR"

echo "Done at $(date). Results: $SAVE_DIR"
