#!/bin/bash
#SBATCH --account=bcgp-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --job-name=wfs_cap_v2
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/compute_caption_retrieval_scores_v2_%A.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/compute_caption_retrieval_scores_v2_%A.err

set -euo pipefail

export HF_HOME=/tmp/michal5

source /sw/external/python/anaconda3/bin/activate
conda activate /projects/bcgp/michal5/envs/deepspeed_video

cd /work/hdd/bcgp/michal5/WFS-SB

echo "=== SigLIP2 caption retrieval (group_v2, per-clip captions) ==="
echo "Date: $(date) | Node: $(hostname)"

python scripts/compute_caption_retrieval_scores_v2.py \
    --dedup-json /work/hdd/bcgp/michal5/molmo2_cap/test_clip_captions_dedup.json \
    --feature-cache-dir /work/hdd/bcgp/michal5/feature_cache/molmo2_cap/qwen3vl_256 \
    --output-dir /work/hdd/bcgp/michal5/WFS-SB/features/caption_retrieval/siglip2_2fps_group_v2 \
    --skip-existing

echo "Completed at $(date)  Exit code: $?"
