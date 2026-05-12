#!/bin/bash
#SBATCH --job-name=ds_vmmeA_1fps
#SBATCH --partition=cpu
#SBATCH --account=bcgp-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/downsample_vmme_all_siglip2_1fps_%j.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/downsample_vmme_all_siglip2_1fps_%j.err
#
# Downsample Video-MME (all = long+med+short, 2700) SigLIP2 features from 2 FPS to 1 FPS.
# Input:  features/vmme_all/siglip2_2fps/
# Output: features/vmme_all/siglip2_1fps/

set -euo pipefail

source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/envs/wfs_sb

cd /work/hdd/bcgp/michal5/WFS-SB

python scripts/downsample_features_2fps_to_1fps.py \
    --in-dir  features/vmme_all/siglip2_2fps \
    --out-dir features/vmme_all/siglip2_1fps \
    --feat-filename siglip2_vision_features.pkl \
    --skip-existing

echo "Done at $(date)."
