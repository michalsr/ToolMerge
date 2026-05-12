#!/bin/bash
#SBATCH --account=bcgp-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA40x4-preempt,gpuA100x4,gpuA100x4-preempt,gpuA100x8
#SBATCH --gpus-per-node=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --requeue
#SBATCH --job-name=cap_ret_1k
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/caption_retrieval_1k_topk_gap_%A.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/caption_retrieval_1k_topk_gap_%A.err

set -euo pipefail

export HF_HOME=/tmp/michal5

source /sw/external/python/anaconda3/bin/activate
conda activate /projects/bcgp/michal5/envs/deepspeed_video

cd /work/hdd/bcgp/michal5/WFS-SB

DEDUP_JSON=/work/hdd/bcgp/michal5/verify_video/multi_turn/evidence_pipeline_v2/dataset_generation/group_v2/test_clip_captions_dedup_1k.json
FEATURES_DIR=/work/hdd/bcgp/michal5/WFS-SB/features/caption_retrieval/siglip2_2fps_group_v2
EVAL_OUT_BASE=/work/hdd/bcgp/michal5/WFS-SB/output/caption_retrieval_eval_group_v2_1k

echo "=== Step 1: Compute SigLIP2 caption retrieval scores (group_v2 1K) ==="
echo "Date: $(date) | Node: $(hostname)"

python scripts/compute_caption_retrieval_scores_v2.py \
    --dedup-json "$DEDUP_JSON" \
    --feature-cache-dir /work/hdd/bcgp/michal5/feature_cache/molmo2_cap/qwen3vl_256 \
    --output-dir "$FEATURES_DIR" \
    --skip-existing

echo ""
echo "=== Step 2: TopK-SigLIP with gap (margin=0), K=2/4/8/16/32 ==="
for K in 2 4 8 16 32; do
    echo "--- TopK-SigLIP-Gap K=${K} ---"
    python scripts/eval_caption_retrieval_topk.py \
        --features-dir "$FEATURES_DIR" \
        --max-frames ${K} \
        --margin 0 \
        --apply-gap \
        --output-dir "${EVAL_OUT_BASE}/topk_gap_${K}f"
done

echo "Completed at $(date)  Exit code: $?"
