#!/bin/bash
#SBATCH --job-name=wfs_m2c_8f
#SBATCH --partition=gpuMI100x8
#SBATCH --account=bcgp-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/wfs_molmo2cap_siglip_8f_%j.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/wfs_molmo2cap_siglip_8f_%j.err

export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/michal5

source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/envs/wfs_sb

cd /work/hdd/bcgp/michal5/WFS-SB

WFS_OUTPUT=/work/hdd/bcgp/michal5/WFS-SB/output/molmo2cap/wfs_siglip_8f.json
FEATURES_DIR=/work/hdd/bcgp/michal5/WFS-SB/features/molmo2cap/siglip_2fps
REANSWER_DIR=/work/hdd/bcgp/michal5/WFS-SB/output/molmo2cap/reanswer_source_siglip_8f

# Step 1: WFS frame selection (8 frames)
python -m wfs.pipeline \
    --benchmark lvb \
    --feature_model siglip2 \
    --max_frames 8 \
    --dataset_root /work/hdd/bcgp/michal5/molmo2_cap \
    --questions_file /work/hdd/bcgp/michal5/WFS-SB/features/molmo2cap/siglip_2fps/questions_wfs.json \
    --features_dir ${FEATURES_DIR} \
    --output_path ${WFS_OUTPUT}

# Step 2: Convert to reanswer.py format
python scripts/convert_to_reanswer_molmo2cap.py \
    --wfs-output ${WFS_OUTPUT} \
    --features-dir ${FEATURES_DIR} \
    --output-dir ${REANSWER_DIR}
