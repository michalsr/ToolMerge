# AKS — Adaptive Keyframe Sampling

CVPR 2025. Adaptive K-frame sampling on top of SigLIP-2 similarity scores.

**AKS code lives inside the vendored WFS-SB tree** (per the user's project
layout): the driver is ``baselines/wfs/scripts/run_aks.py`` and the per-table
SLURM scripts are under ``baselines/wfs/scripts/slurm/aks_*.slurm``.

## Run

```bash
cd toolmerge/  # repo root
python baselines/wfs/scripts/run_aks.py \
    --benchmark longvideobench \
    --feature_model siglip \
    --max_frames 8 \
    --output_dir $TOOLMERGE_OUTPUT_DIR/aks/table2_lvb_8
```

Then run the shared answerer on the keyframes:

```bash
python -m toolmerge.run reanswer \
    --keyframes $TOOLMERGE_OUTPUT_DIR/aks/table2_lvb_8/keyframes.json \
    --config configs/tables/table2_lvb_qwen3_8.yaml
```

## SLURM scripts

The full set of paper-row launchers lives under
``baselines/wfs/scripts/slurm/``:
``aks_{lvb,vmme_all}_{8,32}f.slurm`` plus ``caption_retrieval_1k_aks.slurm``.
Adjust the partition / mem / GPU lines for your cluster.
