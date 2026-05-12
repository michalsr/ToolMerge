#!/bin/bash
#SBATCH --account=bcgp-delta-cpu
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --requeue
#SBATCH --job-name=topk_src_m2c_v2
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/topk_siglip_reanswer_source_m2c_v2_%A.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/topk_siglip_reanswer_source_m2c_v2_%A.err

set -euo pipefail
source /sw/external/python/anaconda3/bin/activate
conda activate /projects/bcgp/michal5/envs/deepspeed_video

cd /work/hdd/bcgp/michal5/WFS-SB

FEATURES=/work/hdd/bcgp/michal5/WFS-SB/features/molmo2cap_v2/siglip_2fps
Q=${FEATURES}/questions_wfs.json
OUT_BASE=/work/hdd/bcgp/michal5/WFS-SB/output/molmo2cap_v2

for K in 8 32; do
    echo "=== TopK-SigLIP reanswer source K=${K} (with gap policy) ==="
    python scripts/topk_siglip_to_reanswer.py \
        --features-dir ${FEATURES} \
        --questions-file ${Q} \
        --k ${K} \
        --output-dir ${OUT_BASE}/reanswer_source_topk_siglip_${K}f
done
echo "done at $(date)"
