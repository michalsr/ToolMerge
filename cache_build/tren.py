"""Build T-REN per-video caches.

Two variants:
  - tracked (``build_tren``): T-REN with cross-frame tracking; output is a dict
    with ``track_text_aligned_tokens`` / ``track_pred_tokens`` / ``track_members``.
    File: ``{video_id}.mp4.tren_cache_qwen3vl``.
  - per-frame (``build_tren_per_frame``): each frame independent; output is a
    dict with ``per_frame_text_aligned_tokens`` (list of tensors).
    File: ``{video_id}.mp4.tren_pf_cache_qwen3vl``. **Used by the paper.**
"""

from __future__ import annotations

import gc
import os

import torch

from cache_build.utils import (
    CV2Reader,
    LOAD_CHUNK_SIZE,
    LazyFrameList,
    get_frame_indices,
)


def build_tren(video_path, video_id, output_dir, client,
               batch_size=32, max_nframes=None, backend="decord"):
    """T-REN with tracking. Returns (n_tracks, n_frames)."""
    out_path = os.path.join(output_dir, f"{video_id}.mp4.tren_cache_qwen3vl")
    frame_idx, nframes, vr = get_frame_indices(
        video_path, max_nframes=max_nframes, backend=backend,
    )

    if isinstance(vr, CV2Reader):
        all_frames = []
        for batch in vr.iter_chunks(frame_idx, chunk_size=LOAD_CHUNK_SIZE):
            all_frames.extend(batch)
        tren_cache = client.encode_video(all_frames, batch_size=batch_size)
        del all_frames
    else:
        frames = LazyFrameList(vr, frame_idx)
        tren_cache = client.encode_video(frames, batch_size=batch_size)

    torch.save(tren_cache, out_path)
    n_tracks = tren_cache["track_text_aligned_tokens"].shape[0]
    del vr, tren_cache
    gc.collect()
    return n_tracks, nframes


def build_tren_per_frame(video_path, video_id, output_dir, client,
                         batch_size=32, max_nframes=None, backend="decord"):
    """T-REN per-frame (no tracking). Returns (total_regions, n_frames)."""
    out_path = os.path.join(output_dir, f"{video_id}.mp4.tren_pf_cache_qwen3vl")
    frame_idx, nframes, vr = get_frame_indices(
        video_path, max_nframes=max_nframes, backend=backend,
    )

    if isinstance(vr, CV2Reader):
        all_frames = []
        for batch in vr.iter_chunks(frame_idx, chunk_size=LOAD_CHUNK_SIZE):
            all_frames.extend(batch)
        tren_cache = client.encode_video_per_frame(all_frames, batch_size=batch_size)
        del all_frames
    else:
        frames = LazyFrameList(vr, frame_idx)
        tren_cache = client.encode_video_per_frame(frames, batch_size=batch_size)

    torch.save(tren_cache, out_path)
    total_regions = sum(t.shape[0] for t in tren_cache["per_frame_text_aligned_tokens"])
    del vr, tren_cache
    gc.collect()
    return total_regions, nframes
