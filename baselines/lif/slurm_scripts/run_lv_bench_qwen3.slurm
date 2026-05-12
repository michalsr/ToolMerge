#!/bin/bash
#SBATCH --account=bcgp-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --mem=128g
#SBATCH --time=02:00:00
#SBATCH --job-name=lif_lvb_q3
#SBATCH --output=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/%j.out
#SBATCH --error=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/%j.err

# GPU layout: cuda:0 = YOLO-World (~2GB), cuda:1 = Qwen3-VL-8B (~16GB bf16)

export HF_HOME=/tmp/michal5
source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/conda/lif
export PYTHONPATH=$PYTHONPATH:/work/hdd/bcgp/michal5/Logic-in-Frames/YOLO-World

cd /work/hdd/bcgp/michal5/Logic-in-Frames

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
nvidia-smi

python -u run_lv_bench_qwen3.py \
    --yolo_device cuda:0 \
    --vlm_device cuda:1 \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --num_questions 10 \
    --num_grounding_frames 16 \
    --search_budget 0.5 \
    --confidence_threshold 0.7 \
    --search_nframes 8 \
    --output_dir ./runs/lv_bench_qwen3

echo "Job finished at $(date)"
