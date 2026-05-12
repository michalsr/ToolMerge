#!/bin/bash
#SBATCH --job-name=aks_lvb_32f
#SBATCH --partition=cpu
#SBATCH --account=bcgp-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/aks_lvb_32f_%j.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/aks_lvb_32f_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/envs/wfs_sb
cd /work/hdd/bcgp/michal5/WFS-SB

AKS_OUTPUT=/work/hdd/bcgp/michal5/WFS-SB/output/lvb/aks_siglip_32f.json
FEATURES_DIR=/work/hdd/bcgp/michal5/WFS-SB/features/lvb/siglip_2fps
REANSWER_DIR=/work/hdd/bcgp/michal5/WFS-SB/output/lvb/reanswer_source_aks_32f

python scripts/run_aks.py \
    --benchmark lvb \
    --feature_model siglip \
    --max_frames 32 \
    --dataset_root /work/nvme/bcgp/michal5/longvideobench \
    --questions_file /work/hdd/bcgp/michal5/WFS-SB/datasets/longvideobench/lvb_val.json \
    --features_dir ${FEATURES_DIR} \
    --output_path ${AKS_OUTPUT}

python scripts/convert_to_reanswer.py \
    --benchmark lvb \
    --wfs-output ${AKS_OUTPUT} \
    --features-dir ${FEATURES_DIR} \
    --output-dir ${REANSWER_DIR}
