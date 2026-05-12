#!/bin/bash
#SBATCH --account=bcgp-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16g
#SBATCH --time=01:00:00
#SBATCH --job-name=lif_dl
#SBATCH --output=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/%j.out
#SBATCH --error=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/%j.err

cd /work/hdd/bcgp/michal5/Logic-in-Frames

# Download YOLO-World weights (with resume support)
mkdir -p pretrained/YOLO-World
wget -c --retry-connrefused --waitretry=5 --timeout=60 -t 10 \
    -O pretrained/YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth \
    "https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"

# Download LVIS data
mkdir -p data/coco/lvis data/texts
wget -c -O data/coco/lvis/lvis_v1_minival_inserted_image_name.json \
    "https://huggingface.co/GLIPModel/GLIP/resolve/main/lvis_v1_minival_inserted_image_name.json"
wget -c -O data/texts/lvis_v1_class_texts.json \
    "https://github.com/AILab-CVC/YOLO-World/raw/refs/heads/master/data/texts/lvis_v1_class_texts.json"

echo "Download complete"
ls -lh pretrained/YOLO-World/
ls -lh data/coco/lvis/
ls -lh data/texts/
