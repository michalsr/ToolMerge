#!/bin/bash
#SBATCH --job-name=wfs_lvb
#SBATCH --partition=cpu
#SBATCH --account=bcgp-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/wfs_lvb_%j.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/wfs_lvb_%j.err

source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/envs/wfs_sb

cd /work/hdd/bcgp/michal5/WFS-SB

WFS_OUTPUT=/work/hdd/bcgp/michal5/WFS-SB/output/lvb/wfs_blip2_16f.json
FEATURES_DIR=/work/hdd/bcgp/michal5/WFS-SB/features/lvb/blip2_2fps
REANSWER_DIR=/work/hdd/bcgp/michal5/WFS-SB/output/lvb/reanswer_source_16f

# Step 1: WFS frame selection
python -m wfs.pipeline \
    --benchmark lvb \
    --feature_model blip2 \
    --max_frames 16 \
    --dataset_root /work/nvme/bcgp/michal5/longvideobench \
    --questions_file /work/hdd/bcgp/michal5/WFS-SB/datasets/longvideobench/lvb_val_local.json \
    --features_dir ${FEATURES_DIR} \
    --output_path ${WFS_OUTPUT}

# Step 2: Convert to reanswer.py format
python scripts/convert_to_reanswer.py \
    --benchmark lvb \
    --wfs-output ${WFS_OUTPUT} \
    --features-dir ${FEATURES_DIR} \
    --output-dir ${REANSWER_DIR}
