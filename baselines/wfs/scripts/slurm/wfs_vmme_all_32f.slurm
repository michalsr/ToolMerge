#!/bin/bash
#SBATCH --job-name=wfs_vmme_a32
#SBATCH --partition=gpuMI100x8
#SBATCH --account=bcgp-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/wfs_vmme_all_32f_%j.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/wfs_vmme_all_32f_%j.err

source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/envs/wfs_sb

cd /work/hdd/bcgp/michal5/WFS-SB

WFS_OUTPUT=/work/hdd/bcgp/michal5/WFS-SB/output/vmme_all/wfs_siglip2_32f.json
FEATURES_DIR=/work/hdd/bcgp/michal5/WFS-SB/features/vmme_all/siglip2_2fps
REANSWER_DIR=/work/hdd/bcgp/michal5/WFS-SB/output/vmme_all/reanswer_source_32f

# Step 1: WFS frame selection
python -m wfs.pipeline \
    --benchmark videomme \
    --feature_model siglip2 \
    --max_frames 32 \
    --dataset_root /projects/bcgp/michal5/Video-MME/videos \
    --questions_file /work/hdd/bcgp/michal5/WFS-SB/datasets/videomme/videomme_json_file.json \
    --features_dir ${FEATURES_DIR} \
    --output_path ${WFS_OUTPUT}

# Step 2: Convert to reanswer.py format
python scripts/convert_to_reanswer.py \
    --benchmark videomme \
    --wfs-output ${WFS_OUTPUT} \
    --features-dir ${FEATURES_DIR} \
    --output-dir ${REANSWER_DIR}
