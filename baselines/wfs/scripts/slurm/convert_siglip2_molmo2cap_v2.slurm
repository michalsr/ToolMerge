#!/bin/bash
#SBATCH --job-name=sig2_m2c_v2
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA100x8,gpuA100x4-preempt,gpuA40x4-preempt
#SBATCH --account=bcgp-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --signal=B:USR1@90
#SBATCH --requeue
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/%x_%j.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/%x_%j.out

# SigLIP2 question-frame similarity for molmo2cap group_v2 test (999 items).
# Shared by wavelet and top-k-question-similarity methods.

export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/michal5

source /sw/external/python/anaconda3/bin/activate
conda activate /projects/bcgp/michal5/envs/deepspeed_video

cd /work/hdd/bcgp/michal5/WFS-SB

TEST_JSON=/work/hdd/bcgp/michal5/verify_video/multi_turn/evidence_pipeline_v2/dataset_generation/group_v2/test.json
FEATURE_CACHE=/work/hdd/bcgp/michal5/feature_cache/molmo2_cap/qwen3vl_256
OUTPUT_DIR=/work/hdd/bcgp/michal5/WFS-SB/features/molmo2cap_v2/siglip_2fps

# Step 1: Compute SigLIP2 similarity scores per question
python scripts/compute_siglip_scores.py \
    --benchmark molmo2cap \
    --json-file "$TEST_JSON" \
    --feature-cache-dir "$FEATURE_CACHE" \
    --output-dir "$OUTPUT_DIR" \
    --skip-existing

# Step 2: Write WFS-compatible questions JSON (999 items w/ candidates list)
python -c "
import json
data = json.load(open('$TEST_JSON'))
for item in data:
    item['video_path'] = item['video_id'] + '.mp4'
    opts = item.get('options', {})
    item['candidates'] = [opts[k] for k in sorted(opts.keys())]
json.dump(data, open('$OUTPUT_DIR/questions_wfs.json', 'w'), indent=1)
print(f'Wrote {len(data)} items to questions_wfs.json')
"
