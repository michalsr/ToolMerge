#!/bin/bash
#SBATCH --job-name=wfs_lvb_1f32
#SBATCH --partition=cpu
#SBATCH --account=bcgp-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/wfs_lvb_1fps_32f_%j.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/wfs_lvb_1fps_32f_%j.err

set -euo pipefail

source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/envs/wfs_sb

cd /work/hdd/bcgp/michal5/WFS-SB

WFS_OUTPUT=/work/hdd/bcgp/michal5/WFS-SB/output/lvb/wfs_siglip_32f_1fps.json
FEATURES_DIR=/work/hdd/bcgp/michal5/WFS-SB/features/lvb/siglip_1fps
REANSWER_DIR=/work/hdd/bcgp/michal5/WFS-SB/output/lvb/reanswer_source_wavelet_32f_1fps

# Step 1: WFS frame selection at 1 FPS
python -m wfs.pipeline \
    --benchmark lvb \
    --feature_model siglip \
    --max_frames 32 \
    --dataset_root /work/nvme/bcgp/michal5/longvideobench \
    --questions_file /work/hdd/bcgp/michal5/WFS-SB/datasets/longvideobench/lvb_val.json \
    --features_dir ${FEATURES_DIR} \
    --output_path ${WFS_OUTPUT}

# Step 2: Convert to reanswer.py format (fps=1.0 since features are now at 1 FPS)
python scripts/convert_to_reanswer.py \
    --benchmark lvb \
    --wfs-output ${WFS_OUTPUT} \
    --features-dir ${FEATURES_DIR} \
    --output-dir ${REANSWER_DIR} \
    --fps 1.0

echo "Done at $(date)."
