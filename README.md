# ToolMerge

Code repo for **"Decomposing Queries into Tool Calls for Long-Video Keyframe Retrieval"**.

ToolMerge is a keyframe-retrieval method for long-video QA. A text-only LLM planner decomposes the query into independent tool calls (SigLIP-2 for scene similarity, T-REN for region-text alignment), combines them with AND/OR boolean operators over per-tool ranks, injects OCR-confirmed frames, and applies greedy NMS with a temporal gap. The selected top-K frames are passed to a downstream answerer VLM (Qwen3-VL-8B or GPT-4o).

The repo also releases two long-video evaluation sets:

- **Molmo-2 Moments** (M2C-v2) — long-video QA where every question is anchored to a specific time interval. 
- **Molmo-2 Captions** — 1000 caption + clip-interval pairs used for caption-retrieval evaluation.

See [Datasets](#datasets) below.

## Quick start

```bash
git clone https://github.com/michalsr/ToolMerge.git
cd ToolMerge
conda env create -f env/toolmerge.yaml
conda activate toolmerge
pip install -e .

# Optional: export TOOLMERGE_DATA_DIR / TOOLMERGE_CACHE_DIR / TOOLMERGE_OUTPUT_DIR
# (configs fall back to ./data, ./caches, ./outputs if unset)

# 1. Download precomputed caches (M2C-v2 / LVB / Video-MME) and T-REN weights
#    NOTE: HF Hub release pending — the `download_*` scripts are placeholders
#    pointing at `toolmerge/<repo>` slugs that will be filled in at release.
#    For now, contact the authors for the weight + cache bundle.
./scripts/download_caches.sh
./scripts/download_tren_weights.sh

# 2. Run one paper-row config on a single GPU
toolmerge config=configs/tables/table2_lvb_qwen3_8.yaml \
    data.start_idx=0 data.end_idx=10 \
    data.save_path=outputs/smoke_lvb
```

For larger runs, pass a `data.start_idx` / `data.end_idx` window per process and shard across GPUs yourself; nothing in the pipeline assumes a job scheduler.

## Layout

```
toolmerge/                # core method package
  run.py                  # end-to-end entry point + reanswer mode
  pipeline.py             # planner -> tools -> merge -> NMS -> answerer
  planner.py              # text-only Qwen3-VL planner + JSON parser
  merging.py              # AST parser + AND(min)/OR(max) merger + OCR injection
  selection.py            # greedy NMS (auto τ = min(D/(2K), 10))
  answerer.py             # build answerer prompt + run generation
  caches.py               # load precomputed SigLIP / T-REN / OCR caches
  config.py               # OmegaConf schema + load_config
  inputs.py               # plain JSON dataset I/O + resume-on-restart
  backends/               # qwen3vl (local) + openai (Azure/OpenAI)
  tools/                  # siglip / tren / ocr / scoring
  prompts/                # planner (v7_no_temporal) + answerer (lif, v1) + ocr_judge

tren/                     # T-REN model code (weights via download script)
cache_build/              # build SigLIP / T-REN / OCR caches from raw videos
baselines/                # comparison keyframe-selectors (see below)
training/                 # GRPO post-training (TRL)
configs/
  default.yaml            # parent config inherited by every per-table YAML
  tables/                 # per-row YAMLs (see "Configs" below)
scripts/                  # standalone helper scripts (caption retrieval eval, etc.)
tests/                    # 25 unit tests
```

## Configs

> **TODO:** rename / restructure the `configs/tables/` files — current names are tied to paper-table numbers (e.g. `table2_*`) which won't make sense once tables get reordered or split. Move to a flatter scheme like `configs/{dataset}/{answerer}_{K}.yaml` before public release.

All per-row configs currently live under `configs/tables/` and follow:

```
table{N}_{dataset}_{answerer}_{K}.yaml
```

- `dataset`: `lvb` (Long Video Bench), `vmme` (Video-MME long), `m2c_v2` (Molmo-2 Moments)
- `answerer`: `qwen3` (local Qwen3-VL-8B-Instruct) or `gpt4o` (Azure / OpenAI GPT-4o)
- `K`: `8` or `32` (final keyframes)

Every per-row YAML inherits from `configs/default.yaml`:

```yaml
defaults:
  - ../default
```

and overrides only what changes (paths, K, model backend, answerer prompt). Convention: Qwen3-VL at K=8 uses `prompt_template: v1`; every other cell uses `lif` (the default). All cache and data paths reference `${TOOLMERGE_DATA_DIR}` / `${TOOLMERGE_CACHE_DIR}` env vars (with sensible defaults).

## How it works: cache phase vs runtime

ToolMerge has two phases that touch different code paths:

```
┌──────────────────────────────── Cache phase (offline, GPU, one-time) ─────────────────────────────────┐
│                                                                                                       │
│   raw videos ──> cache_build/siglip.py    ──> {video_id}.feature_cache_qwen3vl                        │
│              ──> cache_build/tren.py      ──> {video_id}.mp4.tren_pf_cache_qwen3vl                    │
│              ──> cache_build/ocr.py       ──> {video_id}.ocr_cache  (EasyOCR text per frame)          │
│                                                                                                       │
│   Drivers in cache_build/ use the IMAGE-ENCODING sides of the same clients in toolmerge/tools/        │
│   (SiglipClient.encode_images, TrenClient.encode_video_per_frame, easyocr.Reader.readtext_batched).   │
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────── Runtime (per question, GPU only for the answerer) ──────────────────────────┐
│                                                                                                       │
│   planner LLM ──> {queries, combine_expr}                                                             │
│                       │                                                                               │
│                       ▼                                                                               │
│   For each query, the matching tool TEXT-ENCODES the query and dot-products against the cached       │
│   per-frame features (no image processing at runtime):                                                │
│       toolmerge/tools/siglip.py:SiglipClient.encode_texts   ─┐                                        │
│       toolmerge/tools/tren.py:TrenClient.encode_text         ├─> per-frame scores                     │
│       toolmerge/tools/ocr_judge.py:judge_ocr_relevance       ─┘   (LLM call over cached OCR strings)  │
│                                                                                                       │
│   AND/OR-merge ──> greedy NMS ──> K keyframes ──> answerer VLM (Qwen3-VL or GPT-4o)                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Cache phase:** the only stage that touches raw video pixels (other than the answerer extracting the K selected frames at the end). Run once per dataset; the outputs are reused by every subsequent inference run, every baseline, every reward call during training.

**Runtime:**  SigLIP-2 / T-REN models only encode the planner's short text queries; the answerer VLM is the only model that ever sees image pixels at runtime, and only for the K selected frames. Everything else is a dot product against precomputed features.

The same `toolmerge/tools/{siglip,tren,ocr_judge}.py` clients are imported by both phases — the cache builders use their image-encoding methods, the runtime pipeline uses their text-encoding (or LLM-judge) methods. Same is true at training reward time: [`training/frame_selection_backend.py`](training/frame_selection_backend.py) loads the same caches via `caches_for_video` and runs `gather_evidence`, identical to the inference path.

## Building the per-video caches

> **TODO:** publish the pre-built cache bundles on Hugging Face Hub. Until then, run the build locally.

On-disk layout consumed by the pipeline:

```
${TOOLMERGE_CACHE_DIR}/
├── siglip/<dataset>/{video_id}.feature_cache_qwen3vl
├── tren/<dataset>/{video_id}.mp4.tren_pf_cache_qwen3vl
├── ocr/<dataset>/{video_id}.ocr_cache
└── ocr_judge/<dataset>/{uid}.json          # see note below
```

Build all three per-video caches for one dataset in a single GPU job:

```bash
python -m cache_build.build_caches \
    --video_dir /path/to/longvideobench/videos \
    --dataset_json /path/to/lvb_val_std.json \
    --tools siglip tren_per_frame ocr \
    --siglip_output_dir ${TOOLMERGE_CACHE_DIR}/siglip/longvideobench \
    --tren_per_frame_output_dir ${TOOLMERGE_CACHE_DIR}/tren/longvideobench \
    --ocr_output_dir ${TOOLMERGE_CACHE_DIR}/ocr/longvideobench \
    --video_backend cv2
```

Each video takes ~2 min on an A100. LVB val (1337 videos, ~7 min avg) ≈ 12 GPU-hours. Per-tool commands and chunking flags are in `cache_build/README.md`.

### OCR judge cache (per-question)

`ocr_judge/<dataset>/{uid}.json` is per-question and must be **pre-built before inference**. The paper datasets ship with these caches in the bundle; if you bring your own dataset you need to build them yourself.

## Training (planner GRPO)

The released checkpoint is a GRPO-finetuned planner from `Qwen3-VL-8B-Instruct`. Training code is in `training/`:

- `training/train.py` — entry point (`torchrun -m training.train config=...`)
- `training/configs/m2m_grpo.yaml` — paper recipe (1 node × 4 GPUs, frames-in-GT + consistency reward)
- `training/configs/lvb_v7_no_temporal_t0_train.yaml` — the inference config the trainer uses for rollouts
- `training/configs/deepspeed_zero2.json` — ZeRO-2 config
- `training/data/train_correct_uniform_8f_clip_max1.json` — the filtered training subset (~50% of M2M train, items the answerer gets wrong on uniform 8f)

To launch:

```bash
torchrun --nnodes=1 --nproc_per_node=4 -m training.train \
    config=training/configs/m2m_grpo.yaml \
    trl.output_dir=outputs/grpo/m2m
```

The paper's released checkpoint is `global_step=50` of this run.

## Common CLI overrides

Every field in `configs/default.yaml` (or the per-table YAML) can be overridden at the
command line via OmegaConf dotted paths — no extra flags. Examples:

```bash
# Run only the first 50 items as a quick check
toolmerge config=configs/tables/table2_lvb_qwen3_8.yaml \
    data.start_idx=0 data.end_idx=50

# Override K and the answerer prompt
toolmerge config=configs/tables/table2_lvb_qwen3_8.yaml \
    max_final_k=16 answer_generator.prompt_template=lif

# Swap to a fine-tuned planner checkpoint
toolmerge config=configs/tables/table2_lvb_qwen3_8.yaml \
    model.base=/path/to/grpo-ckpt-step50

# Reanswer a baseline's `keyframes.json` with the toolmerge answerer
toolmerge config=configs/tables/table2_lvb_qwen3_8.yaml \
    data.source_dir=outputs/wfs/table2_lvb_qwen3_8

# Ablate a tool (drop OCR; keep SigLIP + T-REN)
toolmerge config=configs/tables/table2_lvb_qwen3_8.yaml \
    enabled_tools=[siglip,tren] ocr_cache_dir=""

# Resume a partial run — pick up where results.json left off (no flag needed)
toolmerge config=configs/tables/table2_lvb_qwen3_8.yaml \
    data.save_path=outputs/table2_lvb_qwen3_8     # same path as before
```

Per-field documentation lives inline in [`configs/default.yaml`](configs/default.yaml).

## End-to-end vs reanswer (`data.source_dir`)

A `toolmerge.run` invocation runs the full planner → tools → merge → NMS → answerer pipeline by default. To re-use a prior run's selections at a different K (or with a different answerer), set `data.source_dir=<prior-output>`:

```bash
# 1. Full pipeline at K=8 (saves trace with pooled_candidates_{8,16,32,64})
toolmerge config=configs/tables/table2_lvb_qwen3_8.yaml \
    data.save_path=outputs/table2_lvb_qwen3_8

# 2. Reanswer the same questions at K=32 — only the answerer runs
toolmerge config=configs/tables/table2_lvb_qwen3_32.yaml \
    data.source_dir=outputs/table2_lvb_qwen3_8 \
    data.save_path=outputs/table2_lvb_qwen3_32
```

`data.source_dir` also accepts any baseline's `keyframes.json` directory — every baseline emits the same shape (`uid, video_id, question, options, ground_truth, frames_used, timestamps_used`), so the same command runs the toolmerge answerer over those keyframes.

## Baselines

> **TODO:** the `baselines/wfs/` directory currently shares its name with the WFS baseline algorithm. Rename to disambiguate (e.g., split the wavelet algorithm into its own subdir; treat the rest as separate baselines).

Code to reproduce the comparison methods lives under `baselines/`. Each method writes `keyframes.json` in a shared shape; the toolmerge answerer consumes it via `data.source_dir`. Example:

```bash
# 1. Pick keyframes with one baseline (here: Uniform)
python -m baselines.uniform.run config=configs/tables/table2_lvb_qwen3_8.yaml \
    data.save_path=outputs/uniform_lvb_8

# 2. Run the toolmerge answerer over those keyframes
toolmerge config=configs/tables/table2_lvb_qwen3_8.yaml \
    data.source_dir=outputs/uniform_lvb_8 \
    data.save_path=outputs/uniform_lvb_8_answered
```

| Method | Where | Notes |
|---|---|---|
| Uniform | `baselines/uniform/` | |
| Oracle | `baselines/oracle/` | M2C-v2 only (needs `start`/`end`) |
| Blind Text | `baselines/blind_text/` | |
| SigLIP-Q | `baselines/siglip_q/` | |
| AKS | `baselines/aks/` | |
| BOLT | `baselines/bolt/` | |
| WFS | `baselines/wfs/` | |
| Logic-in-Frames | `baselines/lif/` | **needs `toolmerge-lif` env** — see `env/lif.yaml` + `baselines/lif/README.md` (it pins `mmcv-full==1.7.0` / `mmdet==2.28.x`) |

Per-baseline invocation details and the shared `keyframes.json` schema are in `baselines/README.md`.

## Datasets

Two evaluation sets ship with this repo, both anchored to per-question time intervals.

### Molmo-2 Moments (`m2c_v2`)

Long-video QA — every question has a `[start, end]` ground-truth clip. Used for Tables 2 / 3 and the GRPO training signal.

```
${TOOLMERGE_DATA_DIR}/m2c_v2/
  test.json                  # QA + clip intervals
  videos/                    # source mp4 files
```

Item schema:

```json
{
  "uid": "<video_id>_<q_idx>",
  "video_id": "abc123",
  "question": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "answer": "C",
  "start": 12.5,                       // seconds
  "end": 47.8
}
```

The Oracle baseline (`baselines/oracle/run.py`) uses the intervals as an upper-bound reference.

### Molmo-2 Captions

1000 caption + clip-interval pairs used for caption retrieval (Table 5). The "question" field carries the caption text and `options` is empty.

```
${TOOLMERGE_DATA_DIR}/m2c_v2/captions_1k.json
```

Item schema:

```json
{
  "uid": "<video_id>_<idx>",
  "video_id": "abc123",
  "question": "<caption text>",
  "options": {},
  "answer": "A",
  "start": "00:06:28.321",             // HH:MM:SS.mmm
  "end":   "00:08:19.832"
}
```

Caption retrieval scores hit@K over the planner's selected frames vs the GT interval — see `configs/tables/table5_caption_retrieval.yaml`.

## Bring your own dataset

To run ToolMerge on a new dataset:

1. **Produce a dataset JSON** with the LVB-style schema:
   ```json
   [
     {"uid": "...", "video_id": "...", "question": "...",
      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "answer": "C"}
   ]
   ```
   `m2c_v2`-style `start` / `end` fields are optional; only the Oracle baseline uses them.

2. **Build the per-video caches** (one-time, on a GPU):
   ```bash
   python cache_build/build_caches.py \
       --videos /path/to/your_videos/ \
       --output ${TOOLMERGE_CACHE_DIR}/{siglip,tren,ocr}/your_dataset/
   ```
   This emits `${video_id}.feature_cache_qwen3vl`, `${video_id}.tren_pf_cache_qwen3vl`, and `${video_id}.ocr_cache` for each video.

3. **Write a config** mirroring an existing one:
   ```yaml
   # configs/tables/table2_your_dataset_qwen3_8.yaml
   defaults:
     - ../default
   data:
     input_path: ${oc.env:TOOLMERGE_DATA_DIR,data}/your_dataset/test.json
     save_path: ${oc.env:TOOLMERGE_OUTPUT_DIR,outputs}/your_dataset_qwen3_8
     video_dir: ${oc.env:TOOLMERGE_DATA_DIR,data}/your_dataset/videos
   siglip_feature_cache_dir: ${oc.env:TOOLMERGE_CACHE_DIR,caches}/siglip/your_dataset
   tren_cache_dir: ${oc.env:TOOLMERGE_CACHE_DIR,caches}/tren/your_dataset
   ocr_cache_dir: ${oc.env:TOOLMERGE_CACHE_DIR,caches}/ocr/your_dataset
   max_final_k: 8
   model_backend: qwen3vl
   answer_generator:
     prompt_template: v1
   ```

4. **Run it**:
   ```bash
   toolmerge config=configs/tables/table2_your_dataset_qwen3_8.yaml
   ```

## License

Apache 2.0. See `LICENSE`.
