#!/bin/bash
#SBATCH --job-name=wfs_sig_vmme
#SBATCH --partition=gpuA100x4,gpuA100x8
#SBATCH --account=bcgp-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=gpua039
#SBATCH --time=00:30:00
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/siglip_scores_vmme_%j.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/siglip_scores_vmme_%j.err

export HF_HOME=/tmp/michal5

source /sw/external/python/anaconda3/bin/activate
conda activate /projects/bcgp/michal5/envs/deepspeed_video

cd /work/hdd/bcgp/michal5/WFS-SB

python scripts/compute_siglip_scores.py \
    --benchmark vmme \
    --json-file /work/hdd/bcgp/michal5/WFS-SB/datasets/videomme/videomme_long_local.json \
    --feature-cache-dir /work/hdd/bcgp/michal5/feature_cache/video_mme/qwen3vl_256 \
    --output-dir /work/hdd/bcgp/michal5/WFS-SB/features/vmme_long/siglip2_2fps \
    --skip-existing
