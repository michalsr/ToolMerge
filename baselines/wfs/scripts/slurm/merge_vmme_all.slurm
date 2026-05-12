#!/bin/bash
#SBATCH --job-name=merge_vmme
#SBATCH --partition=cpu
#SBATCH --account=bcgp-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/merge_vmme_all_%j.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/merge_vmme_all_%j.err

source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/envs/wfs_sb

cd /work/hdd/bcgp/michal5/WFS-SB

OUTPUT_DIR=/work/hdd/bcgp/michal5/WFS-SB/output/vmme_all
FEATURES_DIR=/work/hdd/bcgp/michal5/WFS-SB/features/vmme_all/siglip2_2fps

# Merge 8f shards
python -c "
import json, glob, sys
shards = sorted(glob.glob('${OUTPUT_DIR}/shards_8f/shard_*.json'))
print(f'Merging {len(shards)} shards for 8f')
merged = []
for s in shards:
    merged.extend(json.load(open(s)))
print(f'Total records: {len(merged)}')
with open('${OUTPUT_DIR}/wfs_siglip2_8f.json', 'w') as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)
print('8f merge done')
"

# Merge 32f shards
python -c "
import json, glob, sys
shards = sorted(glob.glob('${OUTPUT_DIR}/shards_32f/shard_*.json'))
print(f'Merging {len(shards)} shards for 32f')
merged = []
for s in shards:
    merged.extend(json.load(open(s)))
print(f'Total records: {len(merged)}')
with open('${OUTPUT_DIR}/wfs_siglip2_32f.json', 'w') as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)
print('32f merge done')
"

# Convert 8f to reanswer format
python scripts/convert_to_reanswer.py \
    --benchmark videomme \
    --wfs-output ${OUTPUT_DIR}/wfs_siglip2_8f.json \
    --features-dir ${FEATURES_DIR} \
    --output-dir ${OUTPUT_DIR}/reanswer_source_8f

# Convert 32f to reanswer format
python scripts/convert_to_reanswer.py \
    --benchmark videomme \
    --wfs-output ${OUTPUT_DIR}/wfs_siglip2_32f.json \
    --features-dir ${FEATURES_DIR} \
    --output-dir ${OUTPUT_DIR}/reanswer_source_32f

echo "All done"
