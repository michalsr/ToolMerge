#!/bin/bash
#SBATCH --account=bcgp-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=128g
#SBATCH --time=06:00:00
#SBATCH --job-name=lif_kfs_pp
#SBATCH --output=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/%j.out
#SBATCH --error=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/%j.err

export HF_HOME=/tmp/michal5
source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/conda/lif
export PYTHONPATH=$PYTHONPATH:/work/hdd/bcgp/michal5/Logic-in-Frames/YOLO-World

cd /work/hdd/bcgp/michal5/Logic-in-Frames

# OOM-safe keyframe search: spawns a NEW process per video
# Each subprocess loads the model, processes one video, exits -> OS reclaims all RAM
# Resumes from where it left off (skips already-processed videos)

OBJ_PATH=${1:-"./runs/obj/obj_result.json"}
KFS_PATH=${2:-"./runs/kfs/kfs_result.json"}

python scripts/run_kfs_perprocess.py \
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
    --update_method spline
