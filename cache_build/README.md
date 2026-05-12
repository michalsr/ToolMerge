# Cache build scripts

The inference pipeline reads precomputed caches from disk. Two paths:

1. **Download the caches** — once we publish them on the Hugging Face Hub,
   ``scripts/download_caches.sh`` snapshot-downloads everything you need.
2. **Build the caches yourself** — these scripts run the SigLIP-2 image
   encoder, the T-REN region encoder, EasyOCR, and the frame extractor over
   raw mp4s and write the caches to ``${TOOLMERGE_CACHE_DIR}``.

## Outputs

For each video ``{video_id}.mp4`` the build emits:

```
${TOOLMERGE_CACHE_DIR}/
├── frames/<dataset>/{video_id}.frame_cache_qwen3vl       # (T, C, H, W) uint8
├── siglip/<dataset>/{video_id}.feature_cache_qwen3vl     # (T, D) SigLIP-2 features
├── tren/<dataset>/{video_id}.tren_pf_cache_qwen3vl       # T-REN per-frame tokens
└── ocr/<dataset>/{video_id}.ocr_cache                    # per-frame OCR text + bboxes
```

All caches are at ``target_fps=2`` (matching the paper).

## Scripts

```bash
python -m cache_build.build_frame_cache  --videos <dir> --out ${TOOLMERGE_CACHE_DIR}/frames/<dataset>
python -m cache_build.build_siglip_cache --videos <dir> --frames-cache <dir> --out ...
python -m cache_build.build_tren_cache   --videos <dir> --frames-cache <dir> --out ...
python -m cache_build.build_ocr_cache    --videos <dir> --out ...
```

Each script supports a ``--video-ids <ids.txt>`` filter so you can rebuild a
single video or restart a partial run. SLURM templates are in
``cache_build/slurm/``; mirror your cluster's partition / mem / GPU
constraints there.

## Compute budget (per the paper)

For a 10-minute video at 1 FPS on a single A100 40 GB (Table 7 of the paper):

| Stage     | Time   |
|-----------|--------|
| SigLIP-2  | 52 s   |
| T-REN     | 44 s   |
| OCR       | 30 s   |
| Captioning (DVD reference, NOT used here) | 428 s |
| **Total ToolMerge pre-process** | **~2 min** |

So caching the whole M2M test split (~750 videos at avg 19 min each) is
roughly 4 GPU-hours per cache type on a single A100.
