#!/bin/bash
#SBATCH --account=bcgp-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --mem=128g
#SBATCH --time=01:00:00
#SBATCH --job-name=lvb_1fps_m
#SBATCH --array=0-7
#SBATCH --output=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/%A_%a.out
#SBATCH --error=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/%A_%a.err

export HF_HOME=/tmp/michal5
source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/conda/lif
export PYTHONPATH=$PYTHONPATH:/work/hdd/bcgp/michal5/Logic-in-Frames/YOLO-World

cd /work/hdd/bcgp/michal5/Logic-in-Frames

# fps=1 is the default in interface_searcher.py
sed -i 's/self\.fps = 2/self.fps = 1/' VSLS/interface_searcher.py

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} started on $(hostname) at $(date)"
echo "Chunk ${SLURM_ARRAY_TASK_ID} of 8 (missing questions)"
nvidia-smi

python -u run_lvb_qwen3.py \
    --lvb_json ./runs/lvb_qwen3_t0_1fps/missing_v3.json \
    --yolo_device cuda:0 \
    --vlm_device cuda:1 \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --num_grounding_frames 16 \
    --search_budget 1.0 \
    --confidence_threshold 0.7 \
    --search_nframes 8 \
    --chunk_id ${SLURM_ARRAY_TASK_ID} \
    --num_chunks 8 \
    --output_dir ./runs/lvb_qwen3_t0_1fps/missing_v3

echo "Job finished at $(date)"
