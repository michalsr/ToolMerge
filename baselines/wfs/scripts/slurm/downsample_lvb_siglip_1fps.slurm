#!/bin/bash
#SBATCH --job-name=ds_lvb_1fps
#SBATCH --partition=cpu
#SBATCH --account=bcgp-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/downsample_lvb_siglip_1fps_%j.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/downsample_lvb_siglip_1fps_%j.err
#
# Downsample LVB SigLIP features from 2 FPS to 1 FPS (every other frame).
# Input:  features/lvb/siglip_2fps/
# Output: features/lvb/siglip_1fps/
# 1337 samples.

set -euo pipefail

source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/envs/wfs_sb

cd /work/hdd/bcgp/michal5/WFS-SB

python scripts/downsample_features_2fps_to_1fps.py \
    --in-dir  features/lvb/siglip_2fps \
    --out-dir features/lvb/siglip_1fps \
    --feat-filename siglip_vision_features.pkl \
    --skip-existing

echo "Done at $(date)."
