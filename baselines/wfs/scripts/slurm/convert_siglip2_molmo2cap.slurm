#!/bin/bash
#SBATCH --job-name=sig2_m2cap
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --account=bcgp-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/convert_siglip2_molmo2cap_%j.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/convert_siglip2_molmo2cap_%j.err

export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/michal5

source /sw/external/python/anaconda3/bin/activate
conda activate /projects/bcgp/michal5/envs/deepspeed_video

cd /work/hdd/bcgp/michal5/WFS-SB

# Step 1: Compute SigLIP2 similarity scores
python scripts/compute_siglip_scores.py \
    --benchmark molmo2cap \
    --json-file /work/hdd/bcgp/michal5/verify_video/multi_turn/evidence_pipeline_v2/dataset_generation/full_pipeline_final/group_1_0412/test_split.json \
    --feature-cache-dir /work/hdd/bcgp/michal5/feature_cache/molmo2_cap/qwen3vl_256 \
    --output-dir /work/hdd/bcgp/michal5/WFS-SB/features/molmo2cap/siglip_2fps \
    --skip-existing

# Step 2: Write WFS-compatible questions JSON
python -c "
import json
data = json.load(open('/work/hdd/bcgp/michal5/verify_video/multi_turn/evidence_pipeline_v2/dataset_generation/full_pipeline_final/group_1_0412/test_split.json'))
for item in data:
    item['video_path'] = item['video_id'] + '.mp4'
    opts = item.get('options', {})
    item['candidates'] = [opts[k] for k in sorted(opts.keys())]
json.dump(data, open('/work/hdd/bcgp/michal5/WFS-SB/features/molmo2cap/siglip_2fps/questions_wfs.json', 'w'), indent=1)
print(f'Wrote {len(data)} items to questions_wfs.json')
"
