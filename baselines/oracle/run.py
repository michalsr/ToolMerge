"""Oracle baseline (M2M only).

Samples K frames uniformly from inside the ground-truth clip interval
``[item["start"], item["end"]]`` (seconds). Reads the mp4 directly with cv2
(no feature caches needed) and writes indices in target-fps space so the
shared answerer's ``extract_cv2`` recovers the same pixel frames. Only
meaningful on benchmarks with clip-level supervision (Molmo-2 Moments) and
reported as the upper-bound reference in Table 3.

Usage:
    python -m baselines.oracle.run config=configs/m2m/qwen3_8.yaml
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

import cv2

from toolmerge.config import ToolMergeConfig, get_config_path_from_cli, load_config, save_config
from toolmerge.inputs import item_uid, load_dataset

logger = logging.getLogger(__name__)


def setup_logging():
    try:
        import coloredlogs
        coloredlogs.install(level="INFO", fmt="%(asctime)s %(name)s %(levelname)s: %(message)s")
    except ImportError:
        logging.basicConfig(level="INFO")


def find_video(video_dir: str, video_id: str) -> str:
    for ext in (".mp4", ".mkv", ".webm", ".avi", ""):
        p = Path(video_dir) / f"{video_id}{ext}"
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"No video for {video_id!r} under {video_dir}")


def video_nframes_at_fps(video_path: str, target_fps: float, frame_factor: int = 2) -> int:
    """Mirrors cache_build/utils.py:get_frame_indices nframes math: floor to a
    multiple of FRAME_FACTOR=2 (Qwen convention) so the uniform grid lines up
    with the SigLIP/T-REN/OCR cache grids if they ever get built."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if native_fps <= 0 or n_total <= 0:
        raise RuntimeError(f"Bad video metadata for {video_path}: fps={native_fps} total={n_total}")
    nframes = n_total / native_fps * target_fps
    nframes = min(nframes, n_total)
    nframes = (int(nframes) // frame_factor) * frame_factor
    return max(int(nframes), frame_factor)


def run_one(item: dict, cfg: ToolMergeConfig, target_fps: float) -> Dict[str, Any]:
    video_id = item["video_id"]
    uid = item_uid(item)
    if "start" not in item or "end" not in item:
        raise ValueError(
            f"Oracle baseline requires per-item 'start' / 'end' seconds "
            f"(only M2M provides these). uid={uid} is missing them."
        )

    video_path = find_video(cfg.data.video_dir, video_id)
    n = video_nframes_at_fps(video_path, target_fps)

    start_idx = max(0, int(round(item["start"] * target_fps)))
    end_idx = min(n - 1, int(round(item["end"] * target_fps)))

    k = cfg.max_final_k
    span = end_idx - start_idx + 1
    if span <= 1:
        indices = [start_idx]
    elif k >= span:
        indices = list(range(start_idx, end_idx + 1))
    else:
        indices = [start_idx + int(i * span / k) for i in range(k)]

    timestamps = [i / target_fps for i in indices]
    return {
        "uid": uid,
        "video_id": video_id,
        "question": item["question"],
        "options": item["options"],
        "ground_truth": item.get("answer"),
        "frames_used": indices,
        "timestamps_used": timestamps,
    }


def main():
    setup_logging()
    cfg = load_config(get_config_path_from_cli(), ToolMergeConfig)
    target_fps = float(cfg.target_fps or 2.0)
    save_dir = cfg.data.save_path
    os.makedirs(save_dir, exist_ok=True)
    save_config(cfg, os.path.join(save_dir, "config.yaml"))

    items = load_dataset(cfg.data.input_path, cfg.data.start_idx, cfg.data.end_idx)
    logger.info("Oracle on %d items (K=%d, target_fps=%.1f)", len(items), cfg.max_final_k, target_fps)

    results = []
    t_start = time.time()
    for item in items:
        try:
            results.append(run_one(item, cfg, target_fps))
        except Exception as e:  # noqa: BLE001
            logger.error("  Error on %s: %s", item.get("video_id"), e)
    out = os.path.join(save_dir, "keyframes.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Wrote %d keyframes to %s (%.0fs)", len(results), out, time.time() - t_start)


if __name__ == "__main__":
    main()
