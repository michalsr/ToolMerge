"""Dataset I/O — plain JSON readers.

The pipeline reads a JSON list of items, each with ``video_id``, ``question``,
``options`` (dict ``A..E -> text``), ``answer`` (gold letter), and a unique
``uid`` (falls back to ``question_id`` for Video-MME-style files).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Iterable, List

logger = logging.getLogger(__name__)


def load_dataset(path: str, start_idx: int = 0, end_idx: int | None = None) -> List[dict]:
    """Read a dataset JSON file and slice it."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected a list of items in {path}, got {type(data).__name__}")
    return data[start_idx:end_idx] if end_idx is not None else data[start_idx:]


def item_uid(item: dict) -> str:
    """Extract the unique identifier for a question."""
    return item.get("uid") or item.get("question_id") or ""


def save_results(results: List[dict], path: str) -> None:
    """Write the per-question result list to JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved %d results to %s", len(results), path)


def load_prior_results(path: str) -> List[dict]:
    """Read an existing ``results.json`` for resume-on-restart support."""
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not load prior results %s: %s; starting fresh", path, e)
        return []
