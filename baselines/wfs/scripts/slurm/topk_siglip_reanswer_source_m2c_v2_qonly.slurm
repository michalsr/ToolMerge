#!/bin/bash
#SBATCH --account=bcgp-delta-cpu
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --requeue
#SBATCH --job-name=topk_src_m2c_v2_qonly
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/topk_siglip_reanswer_source_m2c_v2_qonly_%A.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/topk_siglip_reanswer_source_m2c_v2_qonly_%A.err

set -euo pipefail
source /sw/external/python/anaconda3/bin/activate
conda activate /projects/bcgp/michal5/envs/deepspeed_video

cd /work/hdd/bcgp/michal5/WFS-SB

FEATURES=/work/hdd/bcgp/michal5/WFS-SB/features/molmo2cap_v2/siglip_2fps_qonly
Q=/work/hdd/bcgp/michal5/verify_video/multi_turn/evidence_pipeline_v2/dataset_generation/group_v2/test.json
OUT_BASE=/work/hdd/bcgp/michal5/WFS-SB/output/molmo2cap_v2

for K in 8 32; do
    echo "=== TopK-SigLIP (question-only) reanswer source K=${K} (auto gap) ==="
    python scripts/topk_siglip_to_reanswer.py \
        --features-dir ${FEATURES} \
        --questions-file ${Q} \
        --k ${K} \
        --min-frame-gap-seconds -1 \
        --output-dir ${OUT_BASE}/reanswer_source_topk_siglip_qonly_${K}f
done
echo "done at $(date)"
