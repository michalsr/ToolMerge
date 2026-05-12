#!/bin/bash
#SBATCH --account=bcgp-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=128g
#SBATCH --time=01:00:00
#SBATCH --job-name=lif_test
#SBATCH --output=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/%j.out
#SBATCH --error=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/%j.err

export HF_HOME=/tmp/michal5
source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/conda/lif
export PYTHONPATH=$PYTHONPATH:/work/hdd/bcgp/michal5/Logic-in-Frames/YOLO-World

cd /work/hdd/bcgp/michal5/Logic-in-Frames

# Quick smoke test: load YOLO model, run on a single video
# Uses a test video from LV-Bench
python -c "
import sys
sys.path.insert(0, '.')
sys.path.insert(0, './YOLO-World')
from VSLS.VSLSFramework import initialize_yolo
print('Loading YOLO-World XL model...')
yolo = initialize_yolo(
    config_path='./YOLO-World/configs/pretrain/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py',
    checkpoint_path='./pretrained/YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth',
    device='cuda:0'
)
print('YOLO-World loaded successfully!')

# Test detection on a single frame
import cv2
import numpy as np
cap = cv2.VideoCapture('/projects/bcgp/michal5/lv_bench_videos/-WnyRMZqV1U.mp4')
ret, frame = cap.read()
cap.release()
if ret:
    print(f'Frame shape: {frame.shape}')
    yolo.reparameterize_object_list(['person', 'car'], ['road', 'building'])
    detections = yolo.inference(frame, score_threshold=0.3)
    print(f'Detected {len(detections)} objects')
    print('SMOKE TEST PASSED')
else:
    print('ERROR: Could not read video frame')
"
