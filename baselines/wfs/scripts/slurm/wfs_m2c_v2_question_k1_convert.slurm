#!/bin/bash
#SBATCH --job-name=wfs_m2cv2_q_k1
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
#SBATCH --dependency=singleton

# Convert wfs_siglip_1f.json into reanswer_source_wavelet_1f/chunk_0/results.json
# so the question_retrieval Hit@1 aggregator picks it up.
# Run after wfs_molmo2cap_v2_cpu_1f.slurm produces wfs_siglip_1f.json.

set -euo pipefail

source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/envs/wfs_sb

cd /work/hdd/bcgp/michal5/WFS-SB

# Same FEATURES_DIR as wfs_molmo2cap_v2_cpu_1f.slurm (old_prebalance ordering).
FEATURES_DIR=/work/hdd/bcgp/michal5/WFS-SB/features/molmo2cap_v2/siglip_2fps.old_prebalance
BASE=/work/hdd/bcgp/michal5/WFS-SB/output/molmo2cap_v2

echo "=== Convert wfs_siglip_1f.json -> reanswer_source_wavelet_1f ==="
python scripts/convert_to_reanswer_molmo2cap.py \
    --wfs-output "${BASE}/wfs_siglip_1f.json" \
    --features-dir "${FEATURES_DIR}" \
    --output-dir "${BASE}/reanswer_source_wavelet_1f"

echo "Completed at $(date)"
