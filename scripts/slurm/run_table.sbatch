#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --job-name=tm_smoke
#SBATCH --output=scripts/slurm/logs/%x_%A_%a.out
#SBATCH --error=scripts/slurm/logs/%x_%A_%a.out
#SBATCH --array=0
#
# Generic smoke-test launcher. Override the account / partition lines above
# for your cluster, then run:
#     sbatch scripts/slurm/run_table.sbatch
# Customize the python command at the bottom for the table config you want.

set -euo pipefail

cd "$(dirname "$0")/../.."

# Activate your conda env (adapt path).
# source /path/to/miniconda3/bin/activate && conda activate toolmerge

# OpenAI / Azure credentials must be in the environment (or sourced from a
# user-provided .env). Both keys are optional — only the GPT-4o backend
# needs them; local Qwen3 runs without API access.
# export OPENAI_API_KEY=...
# export OPENAI_BASE_URL=...

python -m toolmerge.run \
    config=configs/tables/smoke_m2m_val.yaml \
    data.start_idx=0 \
    data.end_idx=2 \
    data.save_path="${TOOLMERGE_OUTPUT_DIR:-outputs}/smoke_m2m_val"
