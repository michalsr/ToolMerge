"""Build EasyOCR per-video caches (text per frame, no bbox).

Output: ``{"ocr_results": [[{"text": str}, ...], ...], "fps", "num_frames", "video_id"}``
at ``{video_id}.ocr_cache``. The inference-time OCR judge
(``toolmerge/tools/ocr_judge.py``) reads this cache and runs an LLM over the
per-frame text strings.
"""

from __future__ import annotations

import gc
import os

import numpy as np
import torch

from cache_build.utils import (
    CV2Reader,
    LOAD_CHUNK_SIZE,
    TARGET_FPS,
    get_frame_indices,
    load_chunk,
)


OCR_DETECT_BATCH = 64    # frames per readtext_batched call (CRAFT batch); 128 OOMs on A40 48GB
OCR_RECOG_BATCH = 512    # boxes per recognizer forward pass


def ocr_process_batch(reader, frames_rgb):
    """Batched CRAFT detection + per-frame batched recognition via EasyOCR.

    Returns ``list[list[{"text": str}]]`` aligned with ``frames_rgb``.
    """
    batch_np = np.stack(frames_rgb, axis=0)  # (N, H, W, 3) uint8 RGB
    results_per_frame = reader.readtext_batched(
        batch_np, batch_size=OCR_RECOG_BATCH, detail=0,
    )
    out = []
    for frame_results in results_per_frame:
        out.append(
            [{"text": t.strip()} for t in frame_results
             if isinstance(t, str) and t.strip()]
        )
    return out


def build_ocr(video_path, video_id, output_dir, reader,
              max_nframes=None, backend="decord"):
    """Run EasyOCR over every sampled frame; save per-frame text strings."""
    out_path = os.path.join(output_dir, f"{video_id}.ocr_cache")
    frame_idx, nframes, vr = get_frame_indices(
        video_path, max_nframes=max_nframes, backend=backend,
    )

    all_results = []
    buf = []

    def flush():
        if not buf:
            return
        all_results.extend(ocr_process_batch(reader, buf))
        buf.clear()

    if isinstance(vr, CV2Reader):
        for frame_batch in vr.iter_chunks(frame_idx, chunk_size=LOAD_CHUNK_SIZE):
            for frame_np in frame_batch:
                buf.append(frame_np)
                if len(buf) >= OCR_DETECT_BATCH:
                    flush()
            del frame_batch
            gc.collect()
        flush()
    else:
        for start in range(0, nframes, LOAD_CHUNK_SIZE):
            end = min(start + LOAD_CHUNK_SIZE, nframes)
            chunk = load_chunk(vr, frame_idx[start:end])
            for i in range(chunk.shape[0]):
                frame_np = chunk[i].permute(1, 2, 0).numpy().astype(np.uint8)
                buf.append(frame_np)
                if len(buf) >= OCR_DETECT_BATCH:
                    flush()
            del chunk
            gc.collect()
        flush()

    cache_obj = {
        "ocr_results": all_results,
        "fps": TARGET_FPS,
        "num_frames": len(all_results),
        "video_id": video_id,
    }
    torch.save(cache_obj, out_path)
    n_det = sum(len(fr) for fr in all_results)
    del vr
    gc.collect()
    return n_det, nframes
