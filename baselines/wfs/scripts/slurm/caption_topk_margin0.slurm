#!/bin/bash
#SBATCH --account=bcgp-delta-cpu
#SBATCH --partition=cpu-preempt
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --job-name=caption_topk_m0
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/caption_topk_margin0_%A.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/caption_topk_margin0_%A.err

set -euo pipefail
export HF_HOME=/tmp/michal5
source /sw/external/python/anaconda3/bin/activate
conda activate /projects/bcgp/michal5/envs/deepspeed_video
cd /work/hdd/bcgp/michal5/WFS-SB

echo "=== TopK-SigLIP caption hit@K at margin=0 ==="
echo "Date: $(date) | Node: $(hostname)"
python scripts/caption_topk_margin0.py
