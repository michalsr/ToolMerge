"""Build per-question OCR-judge caches.

For each (uid, question, options) in the dataset, runs the LLM judge
(``toolmerge.tools.ocr_judge.judge_ocr_relevance_batched``) over the OCR
strings extracted by ``cache_build/ocr.py`` for the same video. Output:
``<uid>.json`` files matching the format the inference pipeline produces
lazily, so a pre-built cache lets inference skip the LLM call entirely.

Same shape as the other ``cache_build/*.py`` builders: this module exposes
``build_ocr_judge`` to ``build_caches.py``; it is not a standalone script.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from toolmerge.tools.ocr_judge import judge_ocr_relevance_batched


@dataclass
class JudgeCfg:
    ocr_llm_max_tokens: int = 256


def options_from_item(item: dict) -> dict:
    """Accept options as dict ({'A': '...'}) or as a newline-separated string."""
    opts = item.get("options")
    if isinstance(opts, dict):
        return opts
    if isinstance(opts, str):
        out: dict = {}
        for line in opts.splitlines():
            line = line.strip()
            if not line:
                continue
            letter = line[0]
            rest = line.lstrip(letter).lstrip(").:- ").strip()
            out[letter] = rest
        return out
    return {}


def build_ocr_judge(
    video_id: str,
    items_for_video: List[dict],
    ocr_cache_dir: str,
    output_dir: str,
    backend: Any,
    batch_size: int = 20,
    overwrite: bool = False,
    judge_cfg: Optional[JudgeCfg] = None,
):
    """Build {uid}.json for every question in ``items_for_video``.

    Returns ``(n_built, n_skipped, n_missing_ocr)``.
    """
    judge_cfg = judge_cfg or JudgeCfg()

    ocr_path = os.path.join(ocr_cache_dir, f"{video_id}.ocr_cache")
    if not os.path.exists(ocr_path):
        return 0, 0, len(items_for_video)

    ocr_cache = torch.load(ocr_path, map_location="cpu", weights_only=False)
    num_frames = int(ocr_cache.get("num_frames", len(ocr_cache.get("ocr_results", []))))

    built = skipped = 0
    for item in items_for_video:
        uid = item["uid"]
        out_path = os.path.join(output_dir, f"{uid}.json")
        if os.path.exists(out_path) and not overwrite:
            skipped += 1
            continue

        judge_ocr_relevance_batched(
            question=item["question"],
            options=options_from_item(item),
            ocr_cache=ocr_cache,
            start_idx=0,
            end_idx=num_frames,
            backend=backend,
            cfg=judge_cfg,
            batch_size=batch_size,
            uid=uid,
            cache_dir=output_dir,
        )
        built += 1

    return built, skipped, 0
