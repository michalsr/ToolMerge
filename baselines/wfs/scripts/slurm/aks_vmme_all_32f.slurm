#!/bin/bash
#SBATCH --job-name=aks_vmme_32f
#SBATCH --partition=cpu
#SBATCH --account=bcgp-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/aks_vmme_all_32f_%j.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/aks_vmme_all_32f_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/envs/wfs_sb
cd /work/hdd/bcgp/michal5/WFS-SB

AKS_OUTPUT=/work/hdd/bcgp/michal5/WFS-SB/output/vmme_all/aks_siglip2_32f.json
FEATURES_DIR=/work/hdd/bcgp/michal5/WFS-SB/features/vmme_all/siglip2_2fps
REANSWER_DIR=/work/hdd/bcgp/michal5/WFS-SB/output/vmme_all/reanswer_source_aks_32f

python scripts/run_aks.py \
    --benchmark videomme \
    --feature_model siglip2 \
    --max_frames 32 \
    --dataset_root /projects/bcgp/michal5/Video-MME/videos \
    --questions_file /work/hdd/bcgp/michal5/WFS-SB/datasets/videomme/videomme_json_file.json \
    --features_dir ${FEATURES_DIR} \
    --output_path ${AKS_OUTPUT}

python scripts/convert_to_reanswer.py \
    --benchmark videomme \
    --wfs-output ${AKS_OUTPUT} \
    --features-dir ${FEATURES_DIR} \
    --output-dir ${REANSWER_DIR}
