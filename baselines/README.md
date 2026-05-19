# Baselines

Frame-selection methods compared against ToolMerge. Every baseline writes
`keyframes.json` in the same schema; the shared answerer in `toolmerge.run`
then scores QA accuracy on the selected frames.

| Baseline       | Algorithm                                                    | Source                                              |
|----------------|--------------------------------------------------------------|-----------------------------------------------------|
| `blind_text/`  | no frames (text-only LLM)                                    | —                                                   |
| `uniform/`     | linspace over the full video                                 | —                                                   |
| `oracle/`      | linspace within the GT clip (M2M only)                       | —                                                   |
| `siglip_q/`    | SigLIP-2 cosine → greedy NMS                                 | —                                                   |
| `aks/`         | SigLIP-2 cosine → recursive split + top-k per segment        | CVPR 2025, https://github.com/ncTimTang/AKS         |
| `bolt/`        | SigLIP-2 cosine → inverse-transform sampling on the CDF      | CVPR 2025, https://github.com/sming256/BOLT         |
| `wfs/`         | SigLIP-2 cosine → wavelet event detection + MMR              | CVPR 2026, https://github.com/MAC-AutoML/WFS-SB     |
| `lif/`         | YOLO-World detection + T*-style search                       | NeurIPS 2025 (see `lif/README.md`, separate env)    |

## Common pipeline (siglip_q, aks, bolt, wfs)

These four share the first two steps and differ only in step 3:

1. **Query**: `"<question> <opt1> <opt2> ..."` — question + concatenated
   option values, sorted alphabetically, no letters.
2. **Relevance curve**: encode the query with SigLIP-2, take per-frame
   cosine similarity against the cached frame embeddings. Scores are used
   **raw** — no percentile normalization.
3. **Selector**: method-specific (see the corresponding `run.py` for the
   inlined upstream algorithm).

Each `run.py` is fully standalone — no imports from `toolmerge`. SigLIP-Q's
greedy NMS is copied inline; WFS / AKS / BOLT algorithm code is copied
verbatim from each upstream repo with a citation comment above the block.

## Output schema

```json
{
  "uid": "abc123_0",
  "video_id": "abc123",
  "question": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "ground_truth": "C",
  "frames_used":      [60, 81, 125, 152, 194, 267, 306, 326],
  "timestamps_used":  [30.0, 40.5, 62.5, 76.0, 97.0, 133.5, 153.0, 163.0]
}
```

## Cached SigLIP-2 features

Each baseline reads its frame embeddings from
`${TOOLMERGE_CACHE_DIR}/siglip/<dataset>/{video_id}.feature_cache_qwen3vl`.
Build the caches with `cache_build/build_caches.py --tools siglip ...`. See
the top-level README for the full build invocation.

## Run a baseline

```bash
python -m baselines.siglip_q.run config=configs/tables/table2_lvb_qwen3_8.yaml
python -m baselines.wfs.run      config=configs/tables/table2_lvb_qwen3_8.yaml
python -m baselines.aks.run      config=configs/tables/table2_lvb_qwen3_8.yaml
python -m baselines.bolt.run     config=configs/tables/table2_lvb_qwen3_8.yaml
```

Each writes `<cfg.data.save_path>/keyframes.json`. CLI overrides
(`data.save_path=...`, `max_final_k=32`, ...) work the same way as
`toolmerge.run`.

## Reanswer with the shared answerer

Once a baseline has produced `keyframes.json`, feed it to the shared
answerer by setting `data.source_dir` to the baseline's output directory:

```bash
python -m toolmerge.run \
    config=configs/tables/table2_lvb_qwen3_8.yaml \
    data.source_dir=<baseline_save_path> \
    data.save_path=<baseline_save_path>/reanswered
```

`data.source_dir` reads `frames_used` from `keyframes.json` (or
`pooled_candidates_K` from a prior `results.json`) and runs only the
answerer (Qwen3-VL or GPT-4o per the config), skipping planner/tools.

## LIF

LIF requires a separate conda env (`toolmerge-lif`) due to a
`mmcv-full==1.7.0` incompatibility with newer PyTorch; see
`lif/README.md` for the env setup and PyTorch source patch. Once LIF has
written `keyframes.json`, the same `data.source_dir=` reanswer flow above
applies.
