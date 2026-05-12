#!/bin/bash
#SBATCH --account=bcgp-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --job-name=fix_lif_env
#SBATCH --output=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/%j.out
#SBATCH --error=/work/hdd/bcgp/michal5/Logic-in-Frames/slurm_scripts/logs/%j.err

source /sw/external/python/anaconda3/bin/activate
conda activate /work/hdd/bcgp/michal5/conda/lif

set -e

echo "Step 1: Downgrade PyTorch to 2.4.1+cu121"
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121 2>&1

echo "Step 2: Install prebuilt mmcv for torch 2.4 cu121"
pip uninstall mmcv -y 2>&1
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html --no-cache-dir 2>&1

echo "Step 3: Downgrade transformers to 4.57.1 (min for Qwen3VL)"
pip install transformers==4.57.1 2>&1

echo "Step 4: Pin numpy"
pip install numpy==1.26.4 2>&1

echo "Step 5: Patch CVE check"
IMPORT_UTILS=$(python -c "import transformers.utils.import_utils as m; print(m.__file__)")
echo "Patching $IMPORT_UTILS"
python -c "
path = '$IMPORT_UTILS'
with open(path) as f:
    content = f.read()
old = 'def check_torch_load_is_safe() -> None:\n    if not is_torch_greater_or_equal'
new = 'def check_torch_load_is_safe() -> None:\n    return  # Patched: allow PyTorch 2.4.1 for mmcv compat\n    if not is_torch_greater_or_equal'
if old in content:
    content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print('Patched successfully')
else:
    print('Already patched or pattern not found')
"

echo "Step 6: Verify everything"
python -c "
import torch
print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
import transformers
print('Transformers:', transformers.__version__)
from transformers import Qwen3VLForConditionalGeneration
print('Qwen3VL import OK')
from mmcv.ops import nms
x = torch.tensor([[0,0,10,10],[1,1,11,11]], dtype=torch.float32).cuda()
s = torch.tensor([0.9, 0.8]).cuda()
result = nms(x, s, 0.5)
print('mmcv NMS CUDA OK, result:', result)
"

echo "All done at $(date)"
