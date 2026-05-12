# LIF — Logic-in-Frames (vendored)

This directory is a near-verbatim vendoring of
[Logic-in-Frames](https://github.com/...). One subtree was intentionally
excluded — ``LLaVA-NeXT/`` (the upstream repo's LLaVA-NeXT clone) — because
the paper run does **not** use LLaVA-NeXT.

The pretrained YOLO-World weights ``pretrained/YOLO-World/*.pth`` were also
not vendored (too large for git). Download them per the upstream README:
``yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth``.

## Special conda environment required

LIF pins ``mmcv-full==1.7.0`` which doesn't build cleanly against newer
PyTorch releases. Create the dedicated env:

```bash
conda env create -f ../../env/lif.yaml
conda activate toolmerge-lif
```

Then apply the PyTorch source patch the upstream LIF authors describe in
their ``install.sh`` — search for ``torch/_C/__init__.pyi`` in that script.

## Run the paper's LIF baseline rows

```bash
conda activate toolmerge-lif
# Stage 1: grounding (LLM extracts object/relation tuples from the question)
python scripts/get_VSLS_grounding_objects.py --dataset LongVideoBench --obj_path obj.json
# Stage 2: keyframe search (YOLO-World scores frames; T*-style sampling)
python scripts/get_VSLS_key_frames.py --obj_path obj.json --kfs_path kfs.json
# Stage 3: emit keyframes JSON in our common format
# (LIF's stage 3 also runs QA itself; for the table parity, skip it and use
# our shared answerer instead — see baselines/README.md)
```

The original LIF ``README.md`` is in this directory; the ``install.sh``,
``environment.yml``, and run-* scripts work as in the upstream repo.
