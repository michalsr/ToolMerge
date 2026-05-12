#!/bin/bash
#SBATCH --account=bcgp-delta-cpu
#SBATCH --partition=cpu-preempt
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --job-name=recompute_ours_hit
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/recompute_ours_hit_%A.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/recompute_ours_hit_%A.err

set -euo pipefail

export HF_HOME=/tmp/michal5

source /sw/external/python/anaconda3/bin/activate
conda activate /projects/bcgp/michal5/envs/deepspeed_video

cd /work/hdd/bcgp/michal5/WFS-SB

echo "=== Recompute Ours hit@K for m2c_v2 ==="
echo "Date: $(date) | Node: $(hostname)"

python scripts/recompute_ours_hit_at_k.py
