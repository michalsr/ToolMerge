"""Oracle baseline (M2M only).

Samples K frames uniformly from inside the ground-truth clip interval
``[item["start"], item["end"]]`` — only meaningful on benchmarks with
clip-level supervision (i.e., Molmo-2 Moments). Used as the upper-bound
reference in Table 3.

Usage:
    python -m baselines.oracle.run config=configs/tables/table3_m2m_qa_qwen3_8.yaml
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
    if "start" not in item or "end" not in item:
        raise ValueError(
            f"Oracle baseline requires per-item 'start' / 'end' seconds "
            f"(only M2M provides these). Item uid={uid} is missing them."
        )

    caches = caches_for_video(video_id, cfg)
    fps = caches["fps"]

    start_idx = max(0, int(round(item["start"] * fps)))
    end_idx = int(round(item["end"] * fps))

    k = cfg.max_final_k
    if end_idx <= start_idx:
        # Degenerate clip; fall back to a single mid frame.
        indices = [start_idx]
    else:
        indices = torch.linspace(start_idx, end_idx - 1, min(k, end_idx - start_idx)).long().tolist()

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
    logger.info("Oracle on %d items (K=%d)", len(items), cfg.max_final_k)

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
