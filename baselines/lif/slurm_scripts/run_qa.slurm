#!/bin/bash
#SBATCH --account=bcgp-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --job-name=lif_qa
#SBATCH --output=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/%j.out
#SBATCH --error=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/%j.err

export HF_HOME=/tmp/michal5
source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/conda/lif
export PYTHONPATH=$PYTHONPATH:/work/hdd/bcgp/michal5/Logic-in-Frames/YOLO-World

cd /work/hdd/bcgp/michal5/Logic-in-Frames

# Stage 3: VLM QA on keyframes
# Uses GPT-4o or a local VLM (Qwen3-VL via vLLM)
# Input: keyframes JSON from Stage 2
# Output: QA results JSON

# Uncomment and set your API key:
# export OPENAI_API_KEY="your-key-here"

KFS_PATH=${1:-"./runs/kfs/kfs_result.json"}
QA_PATH=${2:-"./runs/qa/qa_result.json"}

python scripts/get_qa_results.py \
    --backend gpt4 \
    --kfs_path "$KFS_PATH" \
    --qa_path "$QA_PATH" \
    --frame_num 4
