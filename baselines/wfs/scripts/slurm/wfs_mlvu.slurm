#!/bin/bash
#SBATCH --job-name=wfs_mlvu
#SBATCH --partition=cpu
#SBATCH --account=bcgp-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/wfs_mlvu_%j.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/wfs_mlvu_%j.err

source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/envs/wfs_sb

cd /work/hdd/bcgp/michal5/WFS-SB

WFS_OUTPUT=/work/hdd/bcgp/michal5/WFS-SB/output/mlvu/wfs_siglip2_16f.json
FEATURES_DIR=/work/hdd/bcgp/michal5/WFS-SB/features/mlvu/siglip2_2fps
REANSWER_DIR=/work/hdd/bcgp/michal5/WFS-SB/output/mlvu/reanswer_source_16f

# Step 1: WFS frame selection
python -m wfs.pipeline \
    --benchmark mlvu \
    --feature_model siglip2 \
    --max_frames 16 \
    --dataset_root /work/hdd/bcgp/michal5/WFS-SB/datasets/mlvu \
    --questions_file /work/hdd/bcgp/michal5/WFS-SB/datasets/mlvu/mlvu_dev_local.json \
    --features_dir ${FEATURES_DIR} \
    --output_path ${WFS_OUTPUT}

# Step 2: Convert to reanswer.py format
python scripts/convert_to_reanswer.py \
    --benchmark mlvu \
    --wfs-output ${WFS_OUTPUT} \
    --features-dir ${FEATURES_DIR} \
    --output-dir ${REANSWER_DIR}
