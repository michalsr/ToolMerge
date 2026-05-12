#!/bin/bash
#SBATCH --account=bcgp-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --job-name=sig_mlvu_miss
#SBATCH --exclude=gpua039
#SBATCH --array=0-11
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/siglip_cache_mlvu_missing_%A_%a.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/siglip_cache_mlvu_missing_%A_%a.err

set -euo pipefail

export HF_HOME=/tmp/michal5
export PYTHONUNBUFFERED=1

source /sw/external/python/anaconda3/bin/activate
conda activate /projects/bcgp/michal5/envs/deepspeed_video

cd /work/hdd/bcgp/michal5/verify_video/multi_turn

python evidence_pipeline_v2/cache_scripts/build_caches_from_videos.py \
    --video_dir /work/nvme/bcgp/michal5/mlvu/videos_flat \
    --dataset_json /work/nvme/bcgp/michal5/mlvu/mlvu_missing_12.json \
    --tools siglip \
    --siglip_output_dir /work/hdd/bcgp/michal5/feature_cache/mlvu/qwen3vl_256 \
    --start_idx ${SLURM_ARRAY_TASK_ID} \
    --end_idx $((SLURM_ARRAY_TASK_ID + 1))
