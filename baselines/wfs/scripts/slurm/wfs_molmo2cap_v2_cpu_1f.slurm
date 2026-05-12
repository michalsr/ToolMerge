#!/bin/bash
#SBATCH --job-name=wfs_m2cv2_1f
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --account=bcgp-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@90
#SBATCH --requeue
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/%x_%j.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/%x_%j.out

# Wavelet frame selection on precomputed SigLIP2 features — molmo2cap group_v2 test (999 items) — 1 frame.
# Mirrors wfs_molmo2cap_v2_cpu_4f.slurm; needed for question retrieval Hit@1 (Wavelet).

source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/envs/wfs_sb

cd /work/hdd/bcgp/michal5/WFS-SB

# Use the same snapshot the existing wfs_siglip_{2,4,8,16,32}f.json runs were
# generated against. The active siglip_2fps/ dir was rebalanced (Apr 21) after
# those outputs (Apr 19) were produced; its per-i ordering no longer matches.
# old_prebalance/ matches: wfs_siglip_4f.json[0].uid = "Bk_CFtf2vTE_15" =
# old_prebalance/questions_wfs.json[0].uid.
FEATURES_DIR=/work/hdd/bcgp/michal5/WFS-SB/features/molmo2cap_v2/siglip_2fps.old_prebalance
QUESTIONS=${FEATURES_DIR}/questions_wfs.json
WFS_OUTPUT=/work/hdd/bcgp/michal5/WFS-SB/output/molmo2cap_v2/wfs_siglip_1f.json

python -m wfs.pipeline \
    --benchmark lvb \
    --feature_model siglip2 \
    --max_frames 1 \
    --dataset_root /work/hdd/bcgp/michal5/molmo2_cap \
    --questions_file ${QUESTIONS} \
    --features_dir ${FEATURES_DIR} \
    --output_path ${WFS_OUTPUT}
