"""Build SigLIP-2 per-video feature caches.

Output: bare tensor (T, D) at ``{video_id}.feature_cache_qwen3vl``.
"""

from __future__ import annotations

import gc
import os

import torch

from cache_build.utils import (
    CV2Reader,
    LOAD_CHUNK_SIZE,
    frames_to_tensor,
    get_frame_indices,
    load_chunk,
)


def build_siglip(video_path, video_id, output_dir, client,
                 max_nframes=None, backend="decord"):
    """Encode every sampled frame with SigLIP-2 and save (T, D) features."""
    out_path = os.path.join(output_dir, f"{video_id}.feature_cache_qwen3vl")
    frame_idx, nframes, vr = get_frame_indices(
        video_path, max_nframes=max_nframes, backend=backend,
    )

    all_features = []
    if isinstance(vr, CV2Reader):
        for frame_batch in vr.iter_chunks(frame_idx, chunk_size=LOAD_CHUNK_SIZE):
            chunk = frames_to_tensor(frame_batch)
            feats = client.encode_images(chunk)
            all_features.append(feats.cpu())
            del chunk, frame_batch
            gc.collect()
    else:
        for start in range(0, nframes, LOAD_CHUNK_SIZE):
            end = min(start + LOAD_CHUNK_SIZE, nframes)
            chunk = load_chunk(vr, frame_idx[start:end])
            feats = client.encode_images(chunk)
            all_features.append(feats.cpu())
            del chunk
            gc.collect()

    features = torch.cat(all_features, dim=0)  # (T, D)
    torch.save(features, out_path)
    del all_features, vr
    gc.collect()
    return features.shape, nframes
