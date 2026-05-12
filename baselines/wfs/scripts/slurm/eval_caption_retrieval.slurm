#!/bin/bash
#SBATCH --account=bcgp-delta-gpu
#SBATCH --partition=gpuMI100x8
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:45:00
#SBATCH --job-name=wfs_eval
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/eval_caption_retrieval_%A.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/eval_caption_retrieval_%A.out

set -euo pipefail

export HF_HOME=/tmp/michal5

source /sw/external/python/anaconda3/bin/activate
conda activate /projects/bcgp/michal5/envs/deepspeed_video

cd /work/hdd/bcgp/michal5/WFS-SB

FEATURES_DIR=/work/hdd/bcgp/michal5/WFS-SB/features/caption_retrieval/siglip2_2fps
OUT_BASE=/work/hdd/bcgp/michal5/WFS-SB/output/caption_retrieval_eval

echo "=== WFS caption retrieval eval (8f, 16f, 32f) ==="
echo "Date: $(date) | Node: $(hostname)"

for MARGIN in 0 5; do
    for K in 8 16 32; do
        echo "--- max_frames=${K} margin=${MARGIN} ---"
        python scripts/eval_caption_retrieval.py \
            --features-dir ${FEATURES_DIR} \
            --max-frames ${K} \
            --margin ${MARGIN} \
            --output-dir ${OUT_BASE}/wfs_${K}f_margin${MARGIN}
    done
done

echo "Completed at $(date)  Exit code: $?"
