#!/bin/bash
#SBATCH --account=bcgp-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --job-name=mmcv_build
#SBATCH --output=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/%j.out
#SBATCH --error=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/%j.err

source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/conda/lif

export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6"
export MAX_JOBS=16

echo "Building mmcv on $(hostname) with GPU..."
nvidia-smi
which nvcc || echo "nvcc not in PATH"
echo "CUDA_HOME=$CUDA_HOME"

# Clear cached wheels so it actually recompiles
rm -rf /u/michal5/.cache/pip/wheels/*/mmcv*
pip cache remove mmcv 2>/dev/null

pip uninstall mmcv -y 2>&1
pip install mmcv==2.2.0 --no-cache-dir --no-binary mmcv --no-build-isolation 2>&1

pip install numpy==1.26.4 2>&1

echo "Checking .so size..."
ls -la /work/hdd/bcgp/michal5/conda/lif/lib/python3.11/site-packages/mmcv/_ext*.so

echo "Verifying NMS on CUDA..."
python -c "
import torch
print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
from mmcv.ops import nms
x = torch.tensor([[0,0,10,10],[1,1,11,11]], dtype=torch.float32).cuda()
s = torch.tensor([0.9, 0.8]).cuda()
result = nms(x, s, 0.5)
print('mmcv NMS CUDA OK, result:', result)
"

echo "Done at $(date)"
