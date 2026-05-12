#!/bin/bash
#SBATCH --job-name=wfs_m2cv2_q_k2k4k16
#SBATCH --partition=gpuA40x4-preempt,gpuA40x4
#SBATCH --account=bcgp-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --requeue
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/%x_%j.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/%x_%j.err

# Convert existing wfs_siglip_{2,4,16}f.json (m2c_v2 question retrieval, 999 items)
# into reanswer_source_wavelet_{K}f/chunk_0/results.json so question_retrieval_aggregate.py picks them up.

set -euo pipefail

source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/envs/wfs_sb

cd /work/hdd/bcgp/michal5/WFS-SB

FEATURES_DIR=/work/hdd/bcgp/michal5/WFS-SB/features/molmo2cap_v2/siglip_2fps
BASE=/work/hdd/bcgp/michal5/WFS-SB/output/molmo2cap_v2

for K in 2 4 16; do
    echo "=== Convert wfs_siglip_${K}f.json -> reanswer_source_wavelet_${K}f ==="
    python scripts/convert_to_reanswer_molmo2cap.py \
        --wfs-output "${BASE}/wfs_siglip_${K}f.json" \
        --features-dir "${FEATURES_DIR}" \
        --output-dir "${BASE}/reanswer_source_wavelet_${K}f"
done

echo "Completed at $(date)"
