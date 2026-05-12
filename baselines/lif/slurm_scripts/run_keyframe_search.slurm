#!/bin/bash
#SBATCH --account=bcgp-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=128g
#SBATCH --time=04:00:00
#SBATCH --job-name=lif_kfs
#SBATCH --output=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/%j.out
#SBATCH --error=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/%j.err

export HF_HOME=/tmp/michal5
source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/conda/lif
export PYTHONPATH=$PYTHONPATH:/work/hdd/bcgp/michal5/Logic-in-Frames/YOLO-World

cd /work/hdd/bcgp/michal5/Logic-in-Frames

# Stage 2: Key frame search using T* with YOLO-World
# Input: grounding objects JSON from Stage 1
# Output: keyframe timestamps JSON
#
# OOM mitigation:
# - --mem=128g for CPU RAM
# - --search_budget 0.5 limits frames touched
# - --save_batch 5 saves progress frequently
#
# For A100 jobs, change partition to gpuA100x4

OBJ_PATH=${1:-"./runs/obj/obj_result.json"}
KFS_PATH=${2:-"./runs/kfs/kfs_result.json"}

python scripts/get_VSLS_key_frames.py \
    --obj_path "$OBJ_PATH" \
    --kfs_path "$KFS_PATH" \
    --config_path ./YOLO-World/configs/pretrain/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py \
    --checkpoint_path ./pretrained/YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth \
    --device cuda:0 \
    --search_nframes 8 \
    --grid_rows 4 \
    --grid_cols 4 \
    --confidence_threshold 0.7 \
    --search_budget 0.5 \
    --save_batch 5 \
    --num 100 \
    --update_method spline \
    --prompt_type cot
