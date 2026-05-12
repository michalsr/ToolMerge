#!/bin/bash
#SBATCH --account=bcgp-delta-cpu
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --requeue
#SBATCH --job-name=topk_src_m2c_v2_hard250
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/topk_siglip_reanswer_source_m2c_v2_hard250_%A.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/topk_siglip_reanswer_source_m2c_v2_hard250_%A.err

set -euo pipefail
source /sw/external/python/anaconda3/bin/activate
conda activate /projects/bcgp/michal5/envs/deepspeed_video

cd /work/hdd/bcgp/michal5/WFS-SB

FEATURES=/work/hdd/bcgp/michal5/WFS-SB/features/molmo2cap_v2/siglip_2fps_hard250
Q=/work/hdd/bcgp/michal5/verify_video/multi_turn/evidence_pipeline_v2/dataset_generation/group_v2/hard_gpt52_wrong_250.json
OUT_BASE=/work/hdd/bcgp/michal5/WFS-SB/output/molmo2cap_v2_hard250

for K in 8 32; do
    echo "=== TopK-SigLIP reanswer source K=${K} (auto gap: min(dur/(K*2), 10s)) ==="
    python scripts/topk_siglip_to_reanswer.py \
        --features-dir ${FEATURES} \
        --questions-file ${Q} \
        --k ${K} \
        --min-frame-gap-seconds -1 \
        --output-dir ${OUT_BASE}/reanswer_source_topk_siglip_${K}f
done
echo "done at $(date)"
