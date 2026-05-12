# BOLT — Boost LVLM without Training

CVPR 2025. Inverse-transform sampling on top of precomputed SigLIP-2
similarity scores. The driver vendored here (``bolt_to_reanswer.py``) is the
one the paper actually uses — it reuses cached SigLIP-2 scores rather than
the upstream BOLT repo's CLIP-from-scratch encoding.

The standalone upstream ``/work/hdd/bcgp/michal5/BOLT/`` repo is **not**
vendored — the active code path is this single file.

## Run

```bash
python baselines/bolt/bolt_to_reanswer.py \
    --benchmark longvideobench \
    --feature-cache $TOOLMERGE_CACHE_DIR/siglip/longvideobench \
    --output-dir $TOOLMERGE_OUTPUT_DIR/bolt/table2_lvb_8 \
    --max-frames 8
```

Then run the shared answerer:

```bash
python -m toolmerge.run reanswer \
    --keyframes $TOOLMERGE_OUTPUT_DIR/bolt/table2_lvb_8/keyframes.json \
    --config configs/tables/table2_lvb_qwen3_8.yaml
```

The per-paper-row SLURM launchers are referenced in
``baselines/README.md`` — adjust their paths for your cluster.
