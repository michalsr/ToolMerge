# Baselines

ToolMerge compares against four prior keyframe selectors and four simple
baselines. Every baseline emits a **keyframes JSON** in the same format; the
shared ``toolmerge/answerer.py`` then runs the answerer (Qwen3-VL or GPT-4o)
on those frames. Splitting selection from answering means table accuracies
depend only on which keyframes a baseline picked.

| Baseline | Source | Paper tables | Conda env |
|---|---|---|---|
| Blind Text | in-house | 2, 3 | toolmerge |
| Uniform | in-house | 2, 3, 4, 5 | toolmerge |
| Oracle (M2M only) | in-house | 3 | toolmerge |
| SigLIP-Q | in-house | 2, 3, 4, 5 | toolmerge |
| AKS (CVPR 2025) | vendored from WFS-SB | 2, 3, 4, 5 | toolmerge |
| BOLT (CVPR 2025) | vendored: WFS-SB/scripts/bolt_to_reanswer.py | 2, 3, 4, 5 | toolmerge |
| WFS (CVPR 2026) | vendored from upstream WFS-SB | 2, 3, 4, 5 | toolmerge |
| LIF (NeurIPS 2025) | vendored from Logic-in-Frames | 2, 3, 4, 5 | **toolmerge-lif** (separate; see env/lif.yaml) |

MDP3 and AIR (Table 11 of the paper) have no public code and are NOT
vendored here — those accuracies are cited from the AIR paper itself.

## Common keyframes JSON

Every baseline writes the same per-question record so the shared answerer
can be run uniformly:

```json
{
  "uid": "abc123_0",
  "video_id": "abc123",
  "question": "...",
  "options": {"A": "...", ...},
  "ground_truth": "C",
  "frames_used": [60, 81, 125, 152, 194, 267, 306, 326],
  "timestamps_used": [30.0, 40.5, 62.5, 76.0, 97.0, 133.5, 153.0, 163.0]
}
```

## Running

```bash
# 1. Pick keyframes with a baseline.
./baselines/wfs/run.sh table4_m2m_retrieval

# 2. Run the shared answerer on those keyframes.
python -m toolmerge.run reanswer \
    --keyframes outputs/wfs/table4_m2m_retrieval/keyframes.json \
    --config configs/tables/table4_m2m_retrieval.yaml
```

(LIF: switch to the ``toolmerge-lif`` conda env first; see ``env/lif.yaml``
+ ``baselines/lif/README.md`` for the required PyTorch source patch.)

## LIF's PyTorch source patch

The LIF baseline pins ``mmcv-full==1.7.0``, which doesn't build cleanly
against newer PyTorch releases. The LIF authors' workaround is a small edit
to ``torch/_C/__init__.pyi`` after ``pip install``. The exact patch is
documented in ``baselines/lif/README.md`` (carried over from the upstream
Logic-in-Frames repo).
