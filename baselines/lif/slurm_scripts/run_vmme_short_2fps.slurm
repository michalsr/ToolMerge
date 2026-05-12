#!/bin/bash
#SBATCH --account=bcgp-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --mem=128g
#SBATCH --time=02:00:00
#SBATCH --job-name=lif_vmme_sh
#SBATCH --array=0-59
#SBATCH --output=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/%x_%A_%a.out
#SBATCH --error=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/%x_%A_%a.err

export HF_HOME=/tmp/michal5
source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/conda/lif
export PYTHONPATH=$PYTHONPATH:/work/hdd/bcgp/michal5/Logic-in-Frames/YOLO-World

cd /work/hdd/bcgp/michal5/Logic-in-Frames

# Set fps=2
sed -i 's/self\.fps = 1/self.fps = 2/' VSLS/interface_searcher.py

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} started on $(hostname) at $(date)"
echo "Chunk ${SLURM_ARRAY_TASK_ID} of 60"
nvidia-smi

python -u run_vmme_qwen3.py \
    --vmme_json /work/hdd/bcgp/michal5/verify_video/data/video_mme/video_mme_short.json \
    --yolo_device cuda:0 \
    --vlm_device cuda:1 \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --num_grounding_frames 16 \
    --search_budget 1.0 \
    --confidence_threshold 0.7 \
    --search_nframes 8 \
    --chunk_id ${SLURM_ARRAY_TASK_ID} \
    --num_chunks 60 \
    --output_dir ./runs/vmme_short_qwen3_t0_2fps

echo "Job finished at $(date)"
