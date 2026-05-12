#!/bin/bash
#SBATCH --account=bcgp-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --mem=128g
#SBATCH --time=04:00:00
#SBATCH --job-name=g1_2fps
#SBATCH --array=0-7
#SBATCH --output=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/lif_g1_0412_%A_%a.out
#SBATCH --error=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/lif_g1_0412_%A_%a.err

export HF_HOME=/tmp/michal5
source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/conda/lif
export PYTHONPATH=$PYTHONPATH:/work/hdd/bcgp/michal5/Logic-in-Frames/YOLO-World

cd /work/hdd/bcgp/michal5/Logic-in-Frames

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} started on $(hostname) at $(date)"
echo "Chunk ${SLURM_ARRAY_TASK_ID} of 8 (972 questions total, ~122 per chunk)"
nvidia-smi

python -u run_group1_0412_qwen3.py \
    --dataset_json ./Datasets/group1_0412_test_split.json \
    --yolo_device cuda:0 \
    --vlm_device cuda:1 \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --num_grounding_frames 16 \
    --search_budget 1.0 \
    --confidence_threshold 0.7 \
    --search_nframes 8 \
    --chunk_id ${SLURM_ARRAY_TASK_ID} \
    --num_chunks 8 \
    --output_dir ./runs/group1_0412_qwen3_2fps

echo "Job finished at $(date)"
