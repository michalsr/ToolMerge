"""SigLIP-Q baseline.

The paper's simple top-K SigLIP baseline (Tables 2-5). Encodes the
``question + concatenated answer choices`` as a single SigLIP-2 query, scores
every video frame, and applies greedy NMS with the same auto-τ formula as
ToolMerge. **No planner, no merging, no OCR.**

Usage:
    python -m baselines.siglip_q.run config=configs/tables/table4_m2m_retrieval.yaml

Outputs keyframes.json under ``cfg.data.save_path`` in the common format so
``toolmerge/answerer.py`` can score the QA accuracy on top.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from toolmerge.caches import caches_for_video
from toolmerge.config import ToolMergeConfig, get_config_path_from_cli, load_config, save_config
from toolmerge.inputs import item_uid, load_dataset
from toolmerge.selection import auto_tau_seconds, greedy_gap_select, ordered_by_time
from toolmerge.tools.scoring import score_siglip

logger = logging.getLogger(__name__)


def setup_logging():
    try:
        import coloredlogs
        coloredlogs.install(level="INFO", fmt="%(asctime)s %(name)s %(levelname)s: %(message)s")
    except ImportError:
        logging.basicConfig(level="INFO")


def build_query(question: str, options: dict) -> str:
    """Concatenate question + answer choices into one SigLIP query."""
    parts = [question.strip()]
    for letter, text in sorted(options.items()):
        parts.append(f"{letter}) {text}")
    return " ".join(parts)


def run_one(item: dict, cfg: ToolMergeConfig, siglip_client) -> Dict[str, Any]:
    video_id = item["video_id"]
    uid = item_uid(item)
    caches = caches_for_video(video_id, cfg, siglip_client=siglip_client, tren_client=None)

    if caches.get("siglip_embeddings") is None:
        raise FileNotFoundError(f"SigLIP cache missing for {video_id}")

    fps = caches["fps"]
    num_frames = caches["num_frames"] or caches["siglip_embeddings"].shape[0]

    query = build_query(item["question"], item["options"])
    scored = score_siglip(
        query,
        {"client": siglip_client, "embeddings": caches["siglip_embeddings"]},
        0, num_frames,
    )
    score_map = {idx: s for idx, s in scored}

    # Greedy NMS with the same auto-τ as ToolMerge.
    tau = auto_tau_seconds(num_frames, fps, cfg.max_final_k)
    gap_frames = int(tau * fps) if tau > 0 else 0
    selected = greedy_gap_select(score_map, cfg.max_final_k, gap_frames)
    indices = ordered_by_time(selected)
    timestamps = [idx / fps for idx in indices]

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

    from toolmerge.tools.siglip import SiglipClient
    siglip = SiglipClient(lazy_init=True)

    items = load_dataset(cfg.data.input_path, cfg.data.start_idx, cfg.data.end_idx)
    logger.info("SigLIP-Q on %d items", len(items))

    results = []
    t_start = time.time()
    for i, item in enumerate(items):
        try:
            r = run_one(item, cfg, siglip)
            results.append(r)
            logger.info(
                "[%d/%d] %s uid=%s -> %d frames", i + 1, len(items),
                item["video_id"], r["uid"], len(r["frames_used"]),
            )
        except Exception as e:  # noqa: BLE001
            logger.error("  Error on %s: %s", item.get("video_id"), e, exc_info=True)
            continue

    out = os.path.join(save_dir, "keyframes.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Wrote %d keyframes to %s (%.0fs)", len(results), out, time.time() - t_start)


if __name__ == "__main__":
    main()
