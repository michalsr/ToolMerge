# Cache build scripts

The inference pipeline reads precomputed caches from disk. Build them
locally: the SigLIP-2 image encoder, the T-REN region encoder, and EasyOCR
run over raw mp4s; the OCR-judge LLM then runs over the extracted OCR
strings. All four write to ``${TOOLMERGE_CACHE_DIR}``.

## Outputs

For each video ``{video_id}.mp4`` the build emits per-video files, plus the
per-question OCR-judge JSON (keyed by ``uid``):

```
${TOOLMERGE_CACHE_DIR}/
├── siglip/<dataset>/{video_id}.feature_cache_qwen3vl     # (T, D) SigLIP-2 features
├── tren/<dataset>/{video_id}.mp4.tren_pf_cache_qwen3vl   # T-REN per-frame tokens
├── ocr/<dataset>/{video_id}.ocr_cache                    # per-frame OCR text
└── ocr_judge/<dataset>/{uid}.json                        # LLM YES/NO per question
```

All caches are at ``target_fps=2`` (matching the paper).

## Building

One CLI dispatches all tools — pass the ones you need:

```bash
python -m cache_build.build_caches \
    --video_dir       /path/to/videos \
    --dataset_json    /path/to/dataset.json \
    --tools           siglip tren_per_frame ocr \
    --siglip_output_dir          ${TOOLMERGE_CACHE_DIR}/siglip/<dataset> \
    --tren_per_frame_output_dir  ${TOOLMERGE_CACHE_DIR}/tren/<dataset> \
    --ocr_output_dir             ${TOOLMERGE_CACHE_DIR}/ocr/<dataset> \
    --video_backend cv2
```

Per-tool notes:

- **siglip** — encodes frames with `SiglipClient` (paper default
  `google/siglip2-giant-opt-patch16-384`).
- **tren_per_frame** — encodes frames with `TrenClient` (DINOv3 +
  region encoder, no temporal aggregation).
- **tren** — same encoder but with cross-frame token tracking enabled.
  The paper uses **`tren_per_frame`**.
- **ocr** — EasyOCR (English-only, hardcoded).
- **ocr_judge** — see below; **must be built after `ocr`** since it reads
  `{video_id}.ocr_cache` files.

`--dataset_json` is optional for the per-video tools (used to filter which
videos to process) but **required for `ocr_judge`** because the judge cache
is per-question. Resume / chunking flags: `--start_idx`, `--end_idx`,
`--max_videos`, `--overwrite`.

## OCR-judge cache (run after the OCR cache)

The OCR judge is a small LLM (default ``gpt-4o-mini``) that decides, for
each (question, OCR snippet) pair, whether the snippet is relevant. Frames
whose text gets a YES land at rank 1 during runtime merging. The cache is
per-question (one ``{uid}.json`` per question), and the inference pipeline
**only reads it** — it never invokes the LLM at inference. A cache miss
means OCR contributes zero frames for that question.

Run it the same way as the other tools, with two extra requirements: the
matching `ocr/<dataset>/` cache must already exist, and `--dataset_json`
must be provided.

```bash
# 1. EasyOCR first
python -m cache_build.build_caches \
    --video_dir       /path/to/videos \
    --dataset_json    /path/to/dataset.json \
    --tools           ocr \
    --ocr_output_dir  ${TOOLMERGE_CACHE_DIR}/ocr/<dataset> \
    --video_backend   cv2

# 2. Then the LLM judge over the OCR strings (uses OPENAI_API_KEY).
python -m cache_build.build_caches \
    --video_dir              /path/to/videos \
    --dataset_json           /path/to/dataset.json \
    --tools                  ocr_judge \
    --ocr_judge_input_dir    ${TOOLMERGE_CACHE_DIR}/ocr/<dataset> \
    --ocr_judge_output_dir   ${TOOLMERGE_CACHE_DIR}/ocr_judge/<dataset> \
    --ocr_judge_backend      openai \
    --ocr_judge_model_name   gpt-4o-mini \
    --ocr_judge_batch_size   20
```

`--ocr_judge_input_dir` defaults to `--ocr_output_dir` when both tools run
in the same invocation, so you can also fuse the two steps:

```bash
python -m cache_build.build_caches \
    --video_dir              /path/to/videos \
    --dataset_json           /path/to/dataset.json \
    --tools                  ocr ocr_judge \
    --ocr_output_dir         ${TOOLMERGE_CACHE_DIR}/ocr/<dataset> \
    --ocr_judge_output_dir   ${TOOLMERGE_CACHE_DIR}/ocr_judge/<dataset> \
    --ocr_judge_backend      openai \
    --video_backend          cv2
```

Use `--ocr_judge_backend qwen3vl --ocr_judge_qwen_model_path <hf-or-local>`
to run the judge with a local Qwen3-VL model instead.

