# LIF — Logic-in-Frames (in-tree copy)

This directory is a near-verbatim in-tree copy of
[Logic-in-Frames](https://github.com/...). One subtree was intentionally
excluded — ``LLaVA-NeXT/`` (the upstream repo's LLaVA-NeXT clone) — because
the paper run does **not** use LLaVA-NeXT.

The pretrained YOLO-World weights ``pretrained/YOLO-World/*.pth`` are also
not included (too large for git). Download them per the upstream README:
``yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth``.

## Special conda environment required

LIF pins ``mmcv-full==1.7.0`` which doesn't build cleanly against newer
PyTorch releases. Create the dedicated env:

```bash
conda env create -f ../../env/lif.yaml
conda activate toolmerge-lif
```

LIF's mmcv pin forces ``torch==2.4.1`` (the ``install.sh`` step pulls torch 2.4.1
alongside YOLO-World). Newer ``transformers`` versions (≥4.50) added a guard
in ``transformers/utils/import_utils.py::check_torch_load_is_safe`` that
refuses to call ``torch.load`` when torch is older than 2.6 (CVE-2025-32434).
Because Qwen3-VL is loaded via that path, the guard fires before the model
weights ever get touched. To run Qwen3-VL inside this env we make
``check_torch_load_is_safe`` an early ``return``:

```python
# transformers/utils/import_utils.py
def check_torch_load_is_safe() -> None:
    return  # bypass: torch 2.4.1 pinned via mmcv; required for Qwen3-VL
    if not is_torch_greater_or_equal("2.6"):
        raise ValueError(...)
```

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

## Video paths in `Datasets/group1_0412_test_split.json`

The included split uses the placeholder ``${TOOLMERGE_DATA_DIR}/m2m/videos/<video_id>.mp4``
for every ``video_path`` field. LIF's loader reads ``video_path`` as a literal
string and passes it straight to ``cv2.VideoCapture``, so the placeholder must
be substituted before running. Either:

```bash
# One-shot in place, pointing at wherever your m2m videos live:
sed -i 's|\${TOOLMERGE_DATA_DIR}/m2m/videos|/abs/path/to/m2m/videos|g' \
    Datasets/group1_0412_test_split.json
```

or pipe through ``envsubst`` at load time after exporting
``TOOLMERGE_DATA_DIR``. The schema otherwise matches the M2M split
(see top-level ``README.md`` Datasets section).

The original LIF ``README.md`` is in this directory; the ``install.sh``,
``environment.yml``, and run-* scripts work as in the upstream repo.
