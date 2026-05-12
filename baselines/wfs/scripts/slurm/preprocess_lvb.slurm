#!/bin/bash
#SBATCH --job-name=wfs_pp_lvb
#SBATCH --partition=gpuA100x4,gpuA100x8
#SBATCH --account=bcgp-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --exclude=gpua039
#SBATCH --time=01:00:00
#SBATCH --array=26,34,40
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/preprocess_lvb_%A_%a.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/preprocess_lvb_%A_%a.err

export HF_HOME=/tmp/michal5

source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/envs/wfs_sb

cd /work/hdd/bcgp/michal5/WFS-SB

TOTAL=1337
SHARDS=45
PER_SHARD=$(( (TOTAL + SHARDS - 1) / SHARDS ))
START=$(( SLURM_ARRAY_TASK_ID * PER_SHARD ))
END=$(( START + PER_SHARD ))

echo "Shard ${SLURM_ARRAY_TASK_ID}: items ${START}-${END}"

python -m preprocess.extract \
    --benchmark lvb \
    --feature_model blip2 \
    --dataset_root /work/nvme/bcgp/michal5/longvideobench \
    --json_file /work/hdd/bcgp/michal5/WFS-SB/datasets/longvideobench/lvb_val_local.json \
    --output_dir /work/hdd/bcgp/michal5/WFS-SB/features/lvb/blip2_2fps \
    --device cuda \
    --sample_fps 2.0 \
    --batch_size 256 \
    --start_index ${START} \
    --end_index ${END} \
    --skip_existing
