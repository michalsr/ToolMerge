#!/bin/bash
#SBATCH --account=bcgp-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=02:00:00
#SBATCH --job-name=gpt_eval
#SBATCH --array=0-26%2
#SBATCH --output=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/%A_%a.out
#SBATCH --error=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/%A_%a.err

export HF_HOME=/tmp/michal5
source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/conda/lif

cd /work/hdd/bcgp/michal5/Logic-in-Frames

# 1337 questions, 27 chunks of ~50 each
CHUNK_SIZE=50
START_IDX=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END_IDX=$(((SLURM_ARRAY_TASK_ID + 1) * CHUNK_SIZE))
if [ $END_IDX -gt 1337 ]; then
    END_IDX=1337
fi

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} started on $(hostname) at $(date)"
echo "Questions ${START_IDX}-${END_IDX}"

python -u gpt_eval_vsls.py \
    --start-idx ${START_IDX} \
    --end-idx ${END_IDX} \
    --output-dir ./runs/gpt_eval/chunk_${SLURM_ARRAY_TASK_ID}

echo "Job finished at $(date)"
