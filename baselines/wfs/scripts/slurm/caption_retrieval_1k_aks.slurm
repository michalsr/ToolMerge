#!/bin/bash
#SBATCH --account=bcgp-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --job-name=cap_ret_1k_aks
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/caption_retrieval_1k_aks_%A.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/caption_retrieval_1k_aks_%A.err

set -euo pipefail

export HF_HOME=/tmp/michal5

source /sw/external/python/anaconda3/bin/activate
conda activate /projects/bcgp/michal5/envs/deepspeed_video

cd /work/hdd/bcgp/michal5/WFS-SB

FEATURES_DIR=/work/hdd/bcgp/michal5/WFS-SB/features/caption_retrieval/siglip2_2fps_group_v2
EVAL_OUT_BASE=/work/hdd/bcgp/michal5/WFS-SB/output/caption_retrieval_eval_group_v2_1k

echo "=== AKS caption retrieval (group_v2 1K), K=2/4/8/16/32, margin=0 ==="
echo "Date: $(date) | Node: $(hostname)"

for K in 2 4 8 16 32; do
    echo "--- AKS-SigLIP K=${K} ---"
    python scripts/eval_caption_retrieval_aks.py \
        --features-dir "$FEATURES_DIR" \
        --max-frames ${K} \
        --margin 0 \
        --output-dir "${EVAL_OUT_BASE}/aks_${K}f"
done

echo "Completed at $(date)  Exit code: $?"
