#!/bin/bash
#SBATCH --job-name=wfs_m2cv2_32f
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

# Wavelet frame selection on precomputed SigLIP2 features — molmo2cap group_v2 test (999 items) — 32 frames.
# CPU-only; features are already computed by sig2_m2c_v2.

source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/envs/wfs_sb

cd /work/hdd/bcgp/michal5/WFS-SB

FEATURES_DIR=/work/hdd/bcgp/michal5/WFS-SB/features/molmo2cap_v2/siglip_2fps
QUESTIONS=${FEATURES_DIR}/questions_wfs.json
WFS_OUTPUT=/work/hdd/bcgp/michal5/WFS-SB/output/molmo2cap_v2/wfs_siglip_32f.json
REANSWER_DIR=/work/hdd/bcgp/michal5/WFS-SB/output/molmo2cap_v2/reanswer_source_wavelet_32f

# Step 1: WFS frame selection (32 frames)
python -m wfs.pipeline \
    --benchmark lvb \
    --feature_model siglip2 \
    --max_frames 32 \
    --dataset_root /work/hdd/bcgp/michal5/molmo2_cap \
    --questions_file ${QUESTIONS} \
    --features_dir ${FEATURES_DIR} \
    --output_path ${WFS_OUTPUT}

# Step 2: Convert to reanswer source (chunk_0/results.json with frames_used)
python scripts/convert_to_reanswer_molmo2cap.py \
    --wfs-output ${WFS_OUTPUT} \
    --features-dir ${FEATURES_DIR} \
    --output-dir ${REANSWER_DIR}
