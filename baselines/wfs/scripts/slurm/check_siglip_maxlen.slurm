#!/bin/bash
#SBATCH --account=bcgp-delta-gpu
#SBATCH --partition=gpuMI100x8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --job-name=check_siglip_maxlen
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/check_siglip_maxlen_%A.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/check_siglip_maxlen_%A.err

set -euo pipefail

export HF_HOME=/tmp/michal5

source /sw/external/python/anaconda3/bin/activate
conda activate /projects/bcgp/michal5/envs/deepspeed_video

cd /work/hdd/bcgp/michal5/WFS-SB

echo "=== SigLIP max_length comparison (WFS-style vs ours) ==="
echo "Date: $(date) | Node: $(hostname)"

python scripts/check_siglip_maxlen.py
