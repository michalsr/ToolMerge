#!/bin/bash
#SBATCH --job-name=wfs_m2cv2_4f
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --account=bcgp-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@90
#SBATCH --requeue
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/%x_%j.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/%x_%j.out

# Wavelet frame selection on precomputed SigLIP2 features — molmo2cap group_v2 test (999 items) — 4 frames.
# CPU-only; features are already computed. Selection-only pass for Hit@k eval.

source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/envs/wfs_sb

cd /work/hdd/bcgp/michal5/WFS-SB

FEATURES_DIR=/work/hdd/bcgp/michal5/WFS-SB/features/molmo2cap_v2/siglip_2fps
QUESTIONS=${FEATURES_DIR}/questions_wfs.json
WFS_OUTPUT=/work/hdd/bcgp/michal5/WFS-SB/output/molmo2cap_v2/wfs_siglip_4f.json

python -m wfs.pipeline \
    --benchmark lvb \
    --feature_model siglip2 \
    --max_frames 4 \
    --dataset_root /work/hdd/bcgp/michal5/molmo2_cap \
    --questions_file ${QUESTIONS} \
    --features_dir ${FEATURES_DIR} \
    --output_path ${WFS_OUTPUT}
