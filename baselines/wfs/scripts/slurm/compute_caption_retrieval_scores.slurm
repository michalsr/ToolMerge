#!/bin/bash
#SBATCH --account=bcgp-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --job-name=wfs_cap
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/compute_caption_retrieval_scores_%A.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/compute_caption_retrieval_scores_%A.out

set -euo pipefail

export HF_HOME=/tmp/michal5

source /sw/external/python/anaconda3/bin/activate
conda activate /projects/bcgp/michal5/envs/deepspeed_video

cd /work/hdd/bcgp/michal5/WFS-SB

echo "=== Compute SigLIP2 caption retrieval scores ==="
echo "Date: $(date) | Node: $(hostname)"

python scripts/compute_caption_retrieval_scores.py \
    --test-split /work/hdd/bcgp/michal5/verify_video/multi_turn/evidence_pipeline_v2/dataset_generation/full_pipeline_final/group_1_0412/test_split.json \
    --annotations /work/hdd/bcgp/michal5/molmo2_cap/annotations_filtered.parquet \
    --feature-cache-dir /work/hdd/bcgp/michal5/feature_cache/molmo2_cap/qwen3vl_256 \
    --output-dir /work/hdd/bcgp/michal5/WFS-SB/features/caption_retrieval/siglip2_2fps \
    --skip-existing

echo "Completed at $(date)  Exit code: $?"
