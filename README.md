# ToolMerge

Code repo for **"Decomposing Queries into Tool Calls for Long-Video Keyframe Retrieval"**.

ToolMerge is a keyframe-retrieval method for long-video QA. A text-only LLM planner decomposes the query into independent tool calls (SigLIP-2 for scene similarity, T-REN for region-text alignment), combines them with AND/OR boolean operators over per-tool ranks, injects OCR-confirmed frames, and applies greedy NMS with a temporal gap. The selected top-K frames are passed to a downstream answerer VLM (Qwen3-VL-8B or GPT-4o).

The repo also releases two long-video evaluation sets:

- **Molmo-2 Moments** (M2M) — long-video QA where every question is anchored to a specific time interval.
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

# 1. Download T-REN weights, then build the per-video caches once (see
#    cache_build/ for SigLIP / T-REN / OCR / OCR-judge build instructions).
./scripts/download_tren_weights.sh

# 2. Run one paper-row config on a single GPU
python -m toolmerge.run config=configs/lvb/qwen3_8.yaml \
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
  prompts/                # planner  + answerer + ocr_judge

tren/                     # T-REN model code (weights via download script)
cache_build/              # build SigLIP / T-REN / OCR / OCR Judge caches from raw videos
baselines/                # comparison keyframe-selectors (see below)
training/                 # GRPO post-training (TRL)
configs/
  default.yaml            # parent config inherited by every per-row YAML
  {lvb,m2m,vmme}/         # per-row YAMLs, one subdir per dataset (see "Configs")
  smoke.yaml              # tiny smoke run on M2M val
paper_keyframes/          # per-paper-row keyframe selections (see paper_keyframes/README.md)
scripts/                  # standalone helper scripts (caption retrieval eval, etc.)
tests/                    # unit tests (`pytest tests/`)
```

## Configs

Per-row configs are grouped by dataset:

```
configs/{lvb,m2m,vmme}/{qwen3,gpt4o}_{8,32}.yaml
configs/m2m/{retrieval,caption_retrieval}.yaml
```

- `dataset`: `lvb` (Long Video Bench), `vmme` (Video-MME), `m2m` (Molmo-2 Moments)
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
│       toolmerge/tools/ocr_judge.py:load_judge_cache          ─┘   (read pre-built {uid}.json; no LLM) │
│                                                                                                       │
│   AND/OR-merge ──> greedy NMS ──> K keyframes ──> answerer VLM (Qwen3-VL or GPT-4o)                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Cache phase:** the only stage that touches raw video pixels (other than the answerer extracting the K selected frames at the end). Run once per dataset; the outputs are reused by every subsequent inference run, every baseline, every reward call during training.

**Runtime:**  SigLIP-2 / T-REN models only encode the planner's short text queries; the answerer VLM is the only model that ever sees image pixels at runtime, and only for the K selected frames. Everything else is a dot product against precomputed features.

The same `toolmerge/tools/{siglip,tren,ocr_judge}.py` clients are imported by both phases — the cache builders use their image-encoding methods (and the OCR-judge LLM at build time); the runtime pipeline uses their text-encoding methods and reads the pre-built OCR-judge cache. Same is true at training reward time: [`training/frame_selection_backend.py`](training/frame_selection_backend.py) loads the same caches via `caches_for_video` and runs `gather_evidence`, identical to the inference path.

## Building the per-video caches

We do not redistribute the per-video caches — build them locally once with
`cache_build/`. On-disk layout consumed by the pipeline:

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

Per-tool commands and chunking flags are in `cache_build/README.md`.

### OCR judge cache (per-question)

`ocr_judge/<dataset>/{uid}.json` is per-question and **must be pre-built**:
the inference pipeline only reads it (a cache miss skips OCR for that
question). Build it with the same dispatcher after the per-frame OCR cache
exists:

```bash
python -m cache_build.build_caches \
    --video_dir              /path/to/longvideobench/videos \
    --dataset_json           /path/to/lvb_val_std.json \
    --tools                  ocr_judge \
    --ocr_judge_input_dir    ${TOOLMERGE_CACHE_DIR}/ocr/longvideobench \
    --ocr_judge_output_dir   ${TOOLMERGE_CACHE_DIR}/ocr_judge/longvideobench
```

`--dataset_json` is mandatory here (the judge is per-question). See
`cache_build/README.md` for backend / model-name / batch-size flags.

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

The paper's released checkpoint is `global_step=50` of this run. It is
available on the Hugging Face Hub at
[michalsr/toolmerge-planner-grpo](https://huggingface.co/michalsr/toolmerge-planner-grpo):

```bash
huggingface-cli download michalsr/toolmerge-planner-grpo \
    --local-dir checkpoints/grpo-step50

python -m toolmerge.run config=configs/m2m/qwen3_8.yaml \
    model.base=checkpoints/grpo-step50
```

## Common CLI overrides

Every field in `configs/default.yaml` (or a per-row YAML under
`configs/{lvb,m2m,vmme}/`) can be overridden at the command line via
OmegaConf dotted paths — no extra flags. Examples:

```bash
# Run only the first 50 items as a quick check
python -m toolmerge.run config=configs/lvb/qwen3_8.yaml \
    data.start_idx=0 data.end_idx=50

# Override K and the answerer prompt
python -m toolmerge.run config=configs/lvb/qwen3_8.yaml \
    max_final_k=16 answer_generator.prompt_template=lif

# Swap to a fine-tuned planner checkpoint
python -m toolmerge.run config=configs/lvb/qwen3_8.yaml \
    model.base=/path/to/grpo-ckpt-step50

# Reanswer a baseline's `keyframes.json` with the toolmerge answerer
python -m toolmerge.run config=configs/lvb/qwen3_8.yaml \
    data.source_dir=outputs/wfs_lvb_8

# Ablate a tool (drop OCR; keep SigLIP + T-REN)
python -m toolmerge.run config=configs/lvb/qwen3_8.yaml \
    enabled_tools=[siglip,tren] ocr_cache_dir=""

# Resume a partial run — pick up where results.json left off (no flag needed)
python -m toolmerge.run config=configs/lvb/qwen3_8.yaml \
    data.save_path=outputs/lvb_qwen3_8     # same path as before
```

Per-field documentation lives inline in [`configs/default.yaml`](configs/default.yaml).

## End-to-end vs reanswer (`data.source_dir`)

A `toolmerge.run` invocation runs the full planner → tools → merge → NMS → answerer pipeline by default. To re-use a prior run's selections at a different K (or with a different answerer), set `data.source_dir=<prior-output>`:

```bash
# 1. Full pipeline at K=8 (saves trace with pooled_candidates_{8,16,32,64})
python -m toolmerge.run config=configs/lvb/qwen3_8.yaml \
    data.save_path=outputs/lvb_qwen3_8

# 2. Reanswer the same questions at K=32 — only the answerer runs
python -m toolmerge.run config=configs/lvb/qwen3_32.yaml \
    data.source_dir=outputs/lvb_qwen3_8 \
    data.save_path=outputs/lvb_qwen3_32
```

`data.source_dir` also accepts any baseline's `keyframes.json` directory — every baseline emits the same shape (`uid, video_id, question, options, ground_truth, frames_used, timestamps_used`), so the same command runs the toolmerge answerer over those keyframes.

### Paper-row keyframes

[`paper_keyframes/`](paper_keyframes/) contains one `keyframes.json` per paper row (LVB/Video-MME/M2M × Qwen3/GPT-4o × 8/32). Point `data.source_dir` at any of those directories to reproduce the paper accuracy in answerer-only mode (no GPU planner / tools needed):

```bash
python -m toolmerge.run config=configs/lvb/qwen3_32.yaml \
    data.source_dir=paper_keyframes/lvb_qwen3_32 \
    data.save_path=outputs/lvb_qwen3_32_reanswer
```

See [paper_keyframes/README.md](paper_keyframes/README.md) for the full per-row mapping (item counts, reported accuracy, source dirs).

## Baselines

Code to reproduce the comparison methods lives under `baselines/`. Uniform, Oracle, and Blind Text are end-to-end — they pick frames and call the answerer themselves, writing `results.json` directly:

```bash
python -m baselines.uniform.run config=configs/lvb/qwen3_8.yaml \
    data.save_path=outputs/uniform_lvb_8
```

The scored baselines write `keyframes.json` and hand off to the toolmerge answerer via `data.source_dir`:

```bash
# 1. Pick keyframes with a scored baseline (here: SigLIP-Q)
python -m baselines.siglip_q.run config=configs/lvb/qwen3_8.yaml \
    data.save_path=outputs/siglipq_lvb_8

# 2. Run the toolmerge answerer over those keyframes
python -m toolmerge.run config=configs/lvb/qwen3_8.yaml \
    data.source_dir=outputs/siglipq_lvb_8 \
    data.save_path=outputs/siglipq_lvb_8_answered
```

| Method | Where | Notes |
|---|---|---|
| Uniform | `baselines/uniform/` | |
| Oracle | `baselines/oracle/` | M2M only (needs `start`/`end`) |
| Blind Text | `baselines/blind_text/` | |
| SigLIP-Q | `baselines/siglip_q/` | |
| AKS | `baselines/aks/` | |
| BOLT | `baselines/bolt/` | |
| WFS | `baselines/wfs/` | |
| Logic-in-Frames | `baselines/lif/` | **needs `toolmerge-lif` env** — see `env/lif.yaml` + `baselines/lif/README.md` (it pins `mmcv-full==1.7.0` / `mmdet==2.28.x`) |

Per-baseline invocation details and the shared `keyframes.json` schema are in `baselines/README.md`.

## Datasets

ToolMerge reads dataset JSON files directly from disk — no Hugging Face Datasets
dependency. Point ``data.input_path`` in your config at a local file.

Item schema:

```json
{
  "uid": "abc123_0",
  "video_id": "abc123",
  "question": "What does the woman in the red dress do after picking up the book?",
  "options": {"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."},
  "answer": "C",
  "start": 152.0,    // seconds — only present for M2M (ground-truth clip interval)
  "end":   168.0     // seconds — only present for M2M
}
```

### Where to put what

By default the configs reference ``${TOOLMERGE_DATA_DIR}``. With

```
TOOLMERGE_DATA_DIR=/your/path/datasets
```

the expected layout is:

```
${TOOLMERGE_DATA_DIR}/
├── m2m/
│   ├── test.json                           # 999 items (paper test set)
│   ├── val.json                            # 997 items (human-verified val)
│   ├── captions_1k.json                    # 1000 caption + clip-interval pairs
│   ├── video_durations.json
│   └── videos/                             # 1356 source mp4s (test ∪ val ∪ captions)
├── longvideobench/
│   ├── lvb_val_std.json                    # Long Video Bench val set
│   └── videos/                             # source mp4s
└── video_mme/
    ├── video_mme_short.json
    ├── video_mme_med.json
    ├── video_mme_long.json
    └── videos/
```

### Sources

- **M2M (Molmo-2 Moments)** — released alongside this repo on Hugging Face
  Hub at [michalsr/molmo2-moments](https://huggingface.co/datasets/michalsr/molmo2-moments).
  Built from the [Molmo-2 Captioning Dataset](https://huggingface.co/datasets/allenai/molmo2-captions);
  see paper Section 4 for the 8-step construction pipeline.

  ```bash
  huggingface-cli download michalsr/molmo2-moments --repo-type dataset \
      --local-dir $TOOLMERGE_DATA_DIR/m2m
  ```

  The HF dataset includes the JSONs **and** the 1356 source `.mp4` files
  under `videos/`. License: CC-BY-NC-SA-4.0. The Oracle baseline uses the
  `[start, end]` intervals as an upper-bound reference (Table 3).

- **Long Video Bench** — see
  [LongVideoBench/longvideobench](https://huggingface.co/datasets/longvideobench/LongVideoBench).
  Use the ``val_std`` split. Videos are NOT redistributed by this repo.

- **Video-MME** — see
  [lmms-lab/Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME).
  Paper uses no-subtitle mode (``video_mme_{short,med,long}.json`` formats
  match what the paper's runs consumed). Videos are NOT redistributed by
  this repo.

### Caption retrieval (M2M)

`${TOOLMERGE_DATA_DIR}/m2m/captions_1k.json` is 1000 caption + clip-interval
pairs used for caption retrieval. The `question` field carries the caption
text and `options` is empty:

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

Caption retrieval scores hit@K over the planner's selected frames vs. the GT
interval — see `configs/m2m/caption_retrieval.yaml`.

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
   `m2m`-style `start` / `end` fields are optional; only the Oracle baseline uses them.

2. **Build the per-video caches** (one-time, on a GPU):
   ```bash
   python -m cache_build.build_caches \
       --video_dir /path/to/your_videos/ \
       --dataset_json /path/to/your_dataset/test.json \
       --tools siglip tren_per_frame ocr \
       --siglip_output_dir ${TOOLMERGE_CACHE_DIR}/siglip/your_dataset \
       --tren_per_frame_output_dir ${TOOLMERGE_CACHE_DIR}/tren/your_dataset \
       --ocr_output_dir ${TOOLMERGE_CACHE_DIR}/ocr/your_dataset \
       --video_backend cv2
   ```
   This emits `${video_id}.feature_cache_qwen3vl`, `${video_id}.tren_pf_cache_qwen3vl`, and `${video_id}.ocr_cache` for each video.

3. **Write a config** mirroring an existing one:
   ```yaml
   # configs/your_dataset/qwen3_8.yaml
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
   ```

4. **Run it**:
   ```bash
   python -m toolmerge.run config=configs/your_dataset/qwen3_8.yaml
   ```

## Citation

If you use ToolMerge, the Molmo-2 Moments QA set, or the Molmo-2 Captions
retrieval set, please cite our paper:

```bibtex
@inproceedings{toolmerge2026,
  title     = {Decomposing Queries into Tool Calls for Long-Video Keyframe Retrieval},
  author    = {TODO: author list},
  booktitle = {TODO: venue},
  year      = {2026},
}
```

## Issues and contact

Bug reports, reproducibility questions, and feature requests are welcome on
the [GitHub issue tracker](https://github.com/michalsr/ToolMerge/issues).

## License

Apache 2.0. See `LICENSE`.
