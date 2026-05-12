#!/bin/bash
#SBATCH --account=bcgp-delta-cpu
#SBATCH --partition=cpu
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --job-name=wfs_topk_cap_extra
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/eval_caption_retrieval_v2_extra_ks_%A.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/eval_caption_retrieval_v2_extra_ks_%A.err

set -euo pipefail

export HF_HOME=/tmp/michal5

source /sw/external/python/anaconda3/bin/activate
conda activate /projects/bcgp/michal5/envs/deepspeed_video

cd /work/hdd/bcgp/michal5/WFS-SB

FEATURES_DIR=/work/hdd/bcgp/michal5/WFS-SB/features/caption_retrieval/siglip2_2fps_group_v2
OUT_BASE=/work/hdd/bcgp/michal5/WFS-SB/output/caption_retrieval_eval_group_v2

echo "=== WFS + TopK caption retrieval eval (K=2,4,16, margin=5s) ==="
echo "Date: $(date) | Node: $(hostname)"

for K in 2 4 16; do
    echo "--- WFS max_frames=${K} margin=5 ---"
    python scripts/eval_caption_retrieval.py \
        --features-dir ${FEATURES_DIR} \
        --max-frames ${K} \
        --margin 5 \
        --output-dir ${OUT_BASE}/wfs_${K}f_margin5

    echo "--- TopK max_frames=${K} margin=5 ---"
    python scripts/eval_caption_retrieval_topk.py \
        --features-dir ${FEATURES_DIR} \
        --max-frames ${K} \
        --margin 5 \
        --output-dir ${OUT_BASE}/topk_${K}f_margin5
done

echo "Completed at $(date)  Exit code: $?"
