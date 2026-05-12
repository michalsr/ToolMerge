#!/bin/bash
#SBATCH --job-name=siglip2_lvb
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --account=bcgp-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --exclude=gpua039
#SBATCH --time=00:30:00
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/convert_siglip2_lvb_%j.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/convert_siglip2_lvb_%j.err


export PYTHONUNBUFFERED=1

source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/envs/wfs_sb

cd /work/hdd/bcgp/michal5/WFS-SB

stdbuf -oL -eL python scripts/convert_siglip2_features_lvb.py \
    --feature_cache_dir /work/hdd/bcgp/michal5/feature_cache/longvideobench/qwen3vl_256 \
    --lvb_json /work/hdd/bcgp/michal5/WFS-SB/datasets/longvideobench/lvb_val.json \
    --output_dir /work/hdd/bcgp/michal5/WFS-SB/features/lvb/siglip_2fps \
    --model_name google/siglip2-giant-opt-patch16-384 \
    --device cuda \
    --skip_existing
