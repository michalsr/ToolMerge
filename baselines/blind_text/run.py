"""Blind-text baseline.

Answer the question with the answerer alone — NO video frames. Reported in
Tables 2 and 3 of the paper as the floor (random chance is 25 % on 4-choice,
20 % on 5-choice MCQ; blind text usually does a few points better than
chance by exploiting world-knowledge regularities in the wrong distractors).

Usage:
    python -m baselines.blind_text.run config=configs/tables/table3_m2m_qa_qwen3_8.yaml
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict

from toolmerge.answerer import generate_answer
from toolmerge.backends import OpenAIBackend, Qwen3VLBackend
from toolmerge.config import ToolMergeConfig, get_config_path_from_cli, load_config, save_config
from toolmerge.inputs import item_uid, load_dataset

logger = logging.getLogger(__name__)


def setup_logging():
    try:
        import coloredlogs
        coloredlogs.install(level="INFO", fmt="%(asctime)s %(name)s %(levelname)s: %(message)s")
    except ImportError:
        logging.basicConfig(level="INFO")


def build_backend(cfg: ToolMergeConfig):
    if cfg.model_backend == "openai":
        return OpenAIBackend(
            model_name=cfg.openai.model_name,
            api_endpoint=cfg.openai.api_endpoint,
            use_azure=cfg.openai.use_azure,
            max_retries=cfg.openai.max_retries,
        )
    # Defer to the CLI helper for local Qwen3-VL loading.
    from toolmerge.run import load_qwen3_vl
    import torch as _torch
    model, processor = load_qwen3_vl(cfg)
    device = "cuda" if _torch.cuda.is_available() else "cpu"
    return Qwen3VLBackend(model, processor, device=device, qwen_version=cfg.qwen_version)


def main():
    setup_logging()
    cfg = load_config(get_config_path_from_cli(), ToolMergeConfig)
    save_dir = cfg.data.save_path
    os.makedirs(save_dir, exist_ok=True)
    save_config(cfg, os.path.join(save_dir, "config.yaml"))

    backend = build_backend(cfg)
    items = load_dataset(cfg.data.input_path, cfg.data.start_idx, cfg.data.end_idx)
    logger.info("Blind text on %d items", len(items))

    results = []
    correct = total = 0
    t_start = time.time()

    # Empty frames; the answerer detects this and skips vision processing.
    import torch as _torch
    empty = _torch.zeros(0, 3, 224, 224)
    empty_ts = []

    for i, item in enumerate(items):
        try:
            res = generate_answer(empty, empty_ts, item["question"], item["options"], backend, cfg.answer_generator)
        except Exception as e:  # noqa: BLE001
            logger.error("  Error on %s: %s", item.get("video_id"), e)
            continue
        gt = item.get("answer")
        is_correct = res["answer"] == gt
        total += 1
        if is_correct:
            correct += 1
        results.append({
            "uid": item_uid(item),
            "video_id": item["video_id"],
            "question": item["question"],
            "options": item["options"],
            "ground_truth": gt,
            "answer": res["answer"],
            "correct": is_correct,
            "answer_raw": res["raw_response"],
            "frames_used": [],
            "timestamps_used": [],
        })
        if (i + 1) % 50 == 0:
            logger.info("[%d/%d] acc=%.1f%% (%d/%d)", i + 1, len(items), 100 * correct / total, correct, total)

    out = os.path.join(save_dir, "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    acc = correct / total if total else 0.0
    logger.info("Done: %d items, acc=%.1f%%, elapsed=%.0fs", total, acc * 100, time.time() - t_start)
    with open(os.path.join(save_dir, "accuracy.json"), "w") as f:
        json.dump({"correct": correct, "total": total, "accuracy": round(acc, 4)}, f, indent=2)


if __name__ == "__main__":
    main()
