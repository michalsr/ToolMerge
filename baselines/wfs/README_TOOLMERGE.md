# WFS — Wavelet-based Frame Selection (vendored)

Near-verbatim vendoring of [WFS-SB](https://github.com/...) (CVPR 2026).

What was vendored:
  * ``wfs/`` — the WFS algorithm (wavelet denoising + MMR).
  * ``preprocess/`` — feature-extraction helpers.
  * ``configs/`` — default WFS settings.
  * ``scripts/`` — runner + per-paper-row SLURM launchers, including the
    co-located AKS driver (``scripts/run_aks.py``,
    ``scripts/slurm/aks_*.slurm``) and the BOLT driver
    (``scripts/bolt_to_reanswer.py``).
  * ``datasets/`` — dataset metadata JSONs.
  * ``lmms-eval-diff/`` — the lmms-eval integration patch.
  * ``README.md``, ``requirements.txt``.

Excluded:
  * ``features/`` — precomputed per-question similarity scores (multi-GB).
    Rebuild yourself with ``cache_build/`` or download from the HF Hub.
  * ``output/``, ``outputs/`` — research-tree experimental artifacts.
  * Logs under ``scripts/slurm/logs/``.

## Run

```bash
# Paper Table 2 (LVB) row, K=8, default SigLIP-2 features:
python -m wfs.pipeline \
    --benchmark longvideobench \
    --feature_model siglip \
    --max_frames 8 \
    --output_dir $TOOLMERGE_OUTPUT_DIR/wfs/table2_lvb_8
```

Then feed the resulting keyframes through ``toolmerge``'s shared answerer:

```bash
python -m toolmerge.cli reanswer \
    --keyframes $TOOLMERGE_OUTPUT_DIR/wfs/table2_lvb_8/keyframes.json \
    --config configs/tables/table2_lvb_qwen3_8.yaml
```

The upstream ``README.md`` (in this directory) covers the WFS algorithm
itself, the feature-cache build options, and the lmms-eval patch.
