#!/bin/bash
#SBATCH --account=bcgp-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA40x4-preempt,gpuA100x4,gpuA100x4-preempt,gpuA100x8
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --requeue
#SBATCH --job-name=siglip_q_m2c_v2_test
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/siglip_q_m2c_v2_test_%A.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/siglip_q_m2c_v2_test_%A.err

set -euo pipefail
export HF_HOME=/tmp/michal5
source /sw/external/python/anaconda3/bin/activate
conda activate /projects/bcgp/michal5/envs/deepspeed_video

cd /work/hdd/bcgp/michal5/WFS-SB

python scripts/compute_siglip_scores.py \
    --benchmark molmo2cap \
    --json-file /work/hdd/bcgp/michal5/verify_video/multi_turn/evidence_pipeline_v2/dataset_generation/group_v2/test.json \
    --feature-cache-dir /work/hdd/bcgp/michal5/feature_cache/molmo2_cap/qwen3vl_256 \
    --output-dir /work/hdd/bcgp/michal5/WFS-SB/features/molmo2cap_v2/siglip_2fps \
    --skip-existing
