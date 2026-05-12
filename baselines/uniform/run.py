"""Uniform sampling baseline.

Picks K frames evenly spaced over the full video (Tables 2-5).

Usage:
    python -m baselines.uniform.run config=configs/tables/table4_m2m_retrieval.yaml
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict

import torch

from toolmerge.caches import caches_for_video
from toolmerge.config import ToolMergeConfig, get_config_path_from_cli, load_config, save_config
from toolmerge.inputs import item_uid, load_dataset

logger = logging.getLogger(__name__)


def setup_logging():
    try:
        import coloredlogs
        coloredlogs.install(level="INFO", fmt="%(asctime)s %(name)s %(levelname)s: %(message)s")
    except ImportError:
        logging.basicConfig(level="INFO")


def run_one(item: dict, cfg: ToolMergeConfig) -> Dict[str, Any]:
    video_id = item["video_id"]
    uid = item_uid(item)
    caches = caches_for_video(video_id, cfg)
    fps = caches["fps"]

    num_frames = (
        caches["num_frames"]
        or (caches["siglip_embeddings"].shape[0] if caches.get("siglip_embeddings") is not None else 0)
    )
    if num_frames <= 0:
        raise ValueError(f"Cannot determine num_frames for {video_id}")

    k = cfg.max_final_k
    indices = torch.linspace(0, num_frames - 1, k).long().tolist()
    timestamps = [i / fps for i in indices]

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
    save_dir = cfg.data.save_path
    os.makedirs(save_dir, exist_ok=True)
    save_config(cfg, os.path.join(save_dir, "config.yaml"))

    items = load_dataset(cfg.data.input_path, cfg.data.start_idx, cfg.data.end_idx)
    logger.info("Uniform on %d items (K=%d)", len(items), cfg.max_final_k)

    results = []
    t_start = time.time()
    for i, item in enumerate(items):
        try:
            results.append(run_one(item, cfg))
        except Exception as e:  # noqa: BLE001
            logger.error("  Error on %s: %s", item.get("video_id"), e)
    out = os.path.join(save_dir, "keyframes.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Wrote %d keyframes to %s (%.0fs)", len(results), out, time.time() - t_start)


if __name__ == "__main__":
    main()
