#!/bin/bash
#SBATCH --account=bcgp-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA100x4-preempt,gpuA100x8,gpuA40x4,gpuA40x4-preempt
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --job-name=tm_smoke
#SBATCH --output=/work/hdd/bcgp/michal5/toolmerge/scripts/slurm/logs/%x_%A_%a.out
#SBATCH --error=/work/hdd/bcgp/michal5/toolmerge/scripts/slurm/logs/%x_%A_%a.out
#SBATCH --array=0

set -euo pipefail

cd /work/hdd/bcgp/michal5/toolmerge

source /sw/external/python/anaconda3/bin/activate
conda activate /projects/bcgp/michal5/envs/deepspeed_video

# OCR judge talks to GPT-4o-mini via Azure-hosted OpenAI-compatible endpoint
# (matches the research-tree config). Key file is plaintext, one line.
export OPENAI_API_KEY="$(tr -d '[:space:]' < /work/hdd/bcgp/michal5/verify_video/multi_turn/evidence_pipeline_v2/config/secrets.env)"
export OPENAI_BASE_URL="https://michalagentai.cognitiveservices.azure.com/openai/v1/"

python -m toolmerge.run \
    config=configs/tables/smoke_m2m_val.yaml \
    data.start_idx=0 \
    data.end_idx=2 \
    data.save_path=/work/hdd/bcgp/michal5/toolmerge/outputs/smoke_m2m_val
