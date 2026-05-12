#!/bin/bash
#SBATCH --job-name=sig_vmme_all
#SBATCH --partition=gpuA100x4,gpuA100x8,gpuA40x4
#SBATCH --account=bcgp-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=gpua039
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/siglip_scores_vmme_all_%j.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/siglip_scores_vmme_all_%j.err

export HF_HOME=/tmp/michal5

source /sw/external/python/anaconda3/bin/activate
conda activate /projects/bcgp/michal5/envs/deepspeed_video

cd /work/hdd/bcgp/michal5/WFS-SB

python scripts/compute_siglip_scores.py \
    --benchmark vmme \
    --json-file /work/hdd/bcgp/michal5/WFS-SB/datasets/videomme/videomme_json_file.json \
    --feature-cache-dir /work/hdd/bcgp/michal5/feature_cache/video_mme/qwen3vl_256 \
    --output-dir /work/hdd/bcgp/michal5/WFS-SB/features/vmme_all/siglip2_2fps \
    --skip-existing
