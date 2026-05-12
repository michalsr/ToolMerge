#!/bin/bash
#SBATCH --job-name=wfs_vmme_a8f
#SBATCH --partition=cpu
#SBATCH --account=bcgp-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-8
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/wfs_vmme_all_8f_%A_%a.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/wfs_vmme_all_8f_%A_%a.err

source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/envs/wfs_sb

cd /work/hdd/bcgp/michal5/WFS-SB

TOTAL=2700
SHARD_SIZE=300
START=$((SLURM_ARRAY_TASK_ID * SHARD_SIZE))
END=$((START + SHARD_SIZE))
if [ $END -gt $TOTAL ]; then END=$TOTAL; fi

OUTPUT_DIR=/work/hdd/bcgp/michal5/WFS-SB/output/vmme_all
FEATURES_DIR=/work/hdd/bcgp/michal5/WFS-SB/features/vmme_all/siglip2_2fps

mkdir -p ${OUTPUT_DIR}/shards_8f

python -m wfs.pipeline \
    --benchmark videomme \
    --feature_model siglip2 \
    --max_frames 8 \
    --dataset_root /projects/bcgp/michal5/Video-MME/videos \
    --questions_file /work/hdd/bcgp/michal5/WFS-SB/datasets/videomme/videomme_json_file.json \
    --features_dir ${FEATURES_DIR} \
    --start_index ${START} \
    --end_index ${END} \
    --save_every 50 \
    --output_path ${OUTPUT_DIR}/shards_8f/shard_${SLURM_ARRAY_TASK_ID}.json \
    --save_every 50
