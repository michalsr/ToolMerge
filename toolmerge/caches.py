"""Cache I/O: load precomputed SigLIP / T-REN / OCR caches from disk.

The directory layout mirrors what the cache_build scripts produce:

    ${TOOLMERGE_CACHE_DIR}/siglip/<dataset>/{video_id}.feature_cache_qwen3vl
    ${TOOLMERGE_CACHE_DIR}/tren/<dataset>/{video_id}.tren_pf_cache_qwen3vl
    ${TOOLMERGE_CACHE_DIR}/ocr/<dataset>/{video_id}.ocr_cache

Both ``{video_id}.X`` and ``{video_id}.mp4.X`` are accepted to match files
emitted by the cache builders.

The paper runs never use a frame cache — pixel frames are extracted lazily
from the mp4 via ``toolmerge.run.extract_frames_by_index`` only when the
answerer needs them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch


SIGLIP_EXTS = [".feature_cache_qwen3vl", ".mp4.feature_cache_qwen3vl"]
TREN_EXTS = [
    ".tren_cache_qwen3vl",
    ".mp4.tren_cache_qwen3vl",
    ".tren_pf_cache_qwen3vl",
    ".mp4.tren_pf_cache_qwen3vl",
]
OCR_EXTS = [".ocr_cache", ".mp4.ocr_cache"]


def find_cache_file(cache_dir: str, video_id: str, exts) -> Optional[str]:
    """Try each extension in ``exts`` and return the first existing path."""
    if not cache_dir or not video_id:
        return None
    for ext in exts:
        p = os.path.join(cache_dir, f"{video_id}{ext}")
        if os.path.exists(p):
            return p
    return None


def load_siglip_cache(path: str) -> torch.Tensor:
    """Returns the ``(T, D)`` SigLIP-2 image embeddings tensor."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for key in ("embedding", "embeddings", "features", "frame_embeddings"):
            if key in obj and isinstance(obj[key], torch.Tensor):
                return obj[key]
        tensors = [v for v in obj.values() if isinstance(v, torch.Tensor)]
        if len(tensors) == 1:
            return tensors[0]
        raise ValueError(f"Unknown SigLIP cache keys: {list(obj.keys())}")
    raise TypeError(f"Unsupported SigLIP cache type: {type(obj)}")


def load_tren_cache(path: str) -> dict:
    """T-REN cache is a dict — see ``toolmerge.tools.tren.TrenClient``."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"T-REN cache must be a dict, got {type(obj)}")
    return obj


def load_ocr_cache(path: str) -> dict:
    """OCR cache: ``{"ocr_results": List[List[{text, confidence, bbox}]], "fps": float}``."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"OCR cache must be a dict, got {type(obj)}")
    return obj


def subsample_to_fps(tensor_or_list, source_fps: float, target_fps: Optional[float]):
    """Down-sample a per-frame cache from ``source_fps`` to ``target_fps``.

    ``target_fps=None`` is a no-op. Used to read a 2 fps cache at 1 fps.
    """
    if target_fps is None or target_fps >= source_fps:
        return tensor_or_list
    step = int(round(source_fps / target_fps))
    if step <= 1:
        return tensor_or_list
    if isinstance(tensor_or_list, torch.Tensor):
        return tensor_or_list[::step]
    if isinstance(tensor_or_list, list):
        return tensor_or_list[::step]
    return tensor_or_list


def caches_for_video(
    video_id: str,
    cfg,
    siglip_client=None,
    tren_client=None,
) -> dict:
    """Load all caches for one video into a single dict.

    Returns a dict consumed by ``toolmerge.pipeline.run_pipeline``:

        {
          "frames":            None  (pipeline extracts pixel frames lazily),
          "fps":               float,
          "num_frames":        int,
          "siglip_embeddings": (T, D) tensor or None,
          "siglip_client":     SiglipClient or None,
          "tren_cache":        dict or None,
          "tren_client":       TrenClient or None,
          "ocr_cache":         dict or None,
          "video_path":        str (path to mp4) or None,
        }
    """
    siglip_path = find_cache_file(getattr(cfg, "siglip_feature_cache_dir", ""), video_id, SIGLIP_EXTS)
    tren_path = find_cache_file(getattr(cfg, "tren_cache_dir", ""), video_id, TREN_EXTS)
    ocr_path = find_cache_file(getattr(cfg, "ocr_cache_dir", ""), video_id, OCR_EXTS)

    out = {
        "frames": None,
        "fps": 2.0,
        "num_frames": 0,
        "siglip_embeddings": None,
        "siglip_client": siglip_client,
        "tren_cache": None,
        "tren_client": tren_client,
        "ocr_cache": None,
        "video_path": None,
    }

    if siglip_path:
        out["siglip_embeddings"] = load_siglip_cache(siglip_path)
    if tren_path:
        out["tren_cache"] = load_tren_cache(tren_path)
    if ocr_path:
        out["ocr_cache"] = load_ocr_cache(ocr_path)

    # Derive num_frames + fps from whichever cache we have. The caches are all
    # produced at the same target FPS (2.0 by default) so any of them is fine.
    if out["siglip_embeddings"] is not None:
        out["num_frames"] = out["siglip_embeddings"].shape[0]
    elif out["ocr_cache"] is not None and "ocr_results" in out["ocr_cache"]:
        out["num_frames"] = len(out["ocr_cache"]["ocr_results"])
    elif out["tren_cache"] is not None and "num_frames" in out["tren_cache"]:
        out["num_frames"] = out["tren_cache"]["num_frames"]
    if out["ocr_cache"] is not None and "fps" in out["ocr_cache"]:
        out["fps"] = float(out["ocr_cache"]["fps"])

    # Find the raw video file. Match evidence_pipeline_v2/run.py::find_video_file
    # extension list (incl. ".avi" and the empty-string fallback for when the
    # dataset stores video_id with the extension already in it).
    video_dir = getattr(cfg.data, "video_dir", "")
    if video_dir:
        for ext in (".mp4", ".mkv", ".webm", ".avi", ""):
            cand = Path(video_dir) / f"{video_id}{ext}"
            if cand.exists():
                out["video_path"] = str(cand)
                break

    # FPS subsampling (cache native fps -> cfg.target_fps).
    target_fps = getattr(cfg, "target_fps", None)
    if target_fps and out["num_frames"] and target_fps < out["fps"]:
        step = int(round(out["fps"] / target_fps))
        if step > 1:
            if out["siglip_embeddings"] is not None:
                out["siglip_embeddings"] = out["siglip_embeddings"][::step]
            if out["ocr_cache"] is not None and "ocr_results" in out["ocr_cache"]:
                out["ocr_cache"]["ocr_results"] = out["ocr_cache"]["ocr_results"][::step]
            out["num_frames"] = (out["num_frames"] + step - 1) // step
            out["fps"] = float(target_fps)
            if out["ocr_cache"] is not None:
                out["ocr_cache"]["fps"] = out["fps"]

    return out
