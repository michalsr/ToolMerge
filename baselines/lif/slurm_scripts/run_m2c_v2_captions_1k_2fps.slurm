#!/bin/bash
#SBATCH --account=bcgp-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --mem=128g
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --job-name=m2cv2_cap1k_2fps
#SBATCH --array=0-24
#SBATCH --output=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/lif_m2c_v2_captions_1k_%A_%a.out
#SBATCH --error=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/lif_m2c_v2_captions_1k_%A_%a.err
#
# LiF keyframe selection — CAPTION retrieval on group_v2 1k clip set, 2 FPS, Qwen3-VL
# Uses test_clip_captions_dedup_1k.json (1000 unique clips with merged_caption as "question").
# 25 chunks × ~40 items = 1000.

export HF_HOME=/tmp/michal5
source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/conda/lif
export PYTHONPATH=$PYTHONPATH:/work/hdd/bcgp/michal5/Logic-in-Frames/YOLO-World

cd /work/hdd/bcgp/michal5/Logic-in-Frames

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} started on $(hostname) at $(date)"
echo "Chunk ${SLURM_ARRAY_TASK_ID} of 25 (1000 unique clips, ~40/chunk)"
nvidia-smi

python -u run_clip_deduped_qwen3.py \
    --data_json /work/hdd/bcgp/michal5/verify_video/multi_turn/evidence_pipeline_v2/dataset_generation/group_v2/test_clip_captions_dedup_1k.json \
    --video_root /work/hdd/bcgp/michal5/molmo2_cap/videos \
    --yolo_device cuda:0 \
    --vlm_device cuda:1 \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --num_grounding_frames 16 \
    --search_budget 1.0 \
    --confidence_threshold 0.7 \
    --search_nframes 8 \
    --chunk_id ${SLURM_ARRAY_TASK_ID} \
    --num_chunks 25 \
    --output_dir ./runs/m2c_v2_captions_1k_qwen3_2fps

echo "Job finished at $(date)"
