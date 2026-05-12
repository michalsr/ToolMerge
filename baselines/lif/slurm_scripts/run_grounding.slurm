#!/bin/bash
#SBATCH --account=bcgp-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --job-name=lif_ground
#SBATCH --output=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/%j.out
#SBATCH --error=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/%j.err

export HF_HOME=/tmp/michal5
source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/conda/lif
export PYTHONPATH=$PYTHONPATH:/work/hdd/bcgp/michal5/Logic-in-Frames/YOLO-World

cd /work/hdd/bcgp/michal5/Logic-in-Frames

# Stage 1: Object grounding via GPT-4o API
# Requires OPENAI_API_KEY to be set
# Input: dataset name or video root
# Output: grounding objects JSON

# Uncomment and set your API key:
# export OPENAI_API_KEY="your-key-here"

DATASET=${1:-"LongVideoBench"}
OBJ_PATH=${2:-"./runs/obj/obj_result.json"}

python scripts/get_VSLS_grounding_objects.py \
    --dataset "$DATASET" \
    --obj_path "$OBJ_PATH" \
    --backend gpt4 \
    --prompt_type cot \
    --upload_video 1 \
    --num 100 \
    --save_batch 10
