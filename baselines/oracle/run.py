"""Oracle baseline (M2M only, end-to-end).

Samples K frames uniformly with ``np.linspace`` from inside the ground-truth
clip interval ``[item["start"], item["end"]]`` (seconds), decodes them with
cv2, and calls the toolmerge answerer directly. No feature caches, no two-step
handoff — writes ``results.json`` and ``accuracy.json``. Only meaningful on
benchmarks with clip-level supervision (Molmo-2 Moments) and reported as the
upper-bound reference in Table 3.

Usage:
    python -m baselines.oracle.run config=configs/m2m/qwen3_8.yaml \
        data.save_path=outputs/oracle_m2m_8
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch

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


def find_video(video_dir: str, video_id: str) -> str:
    for ext in (".mp4", ".mkv", ".webm", ".avi", ""):
        p = Path(video_dir) / f"{video_id}{ext}"
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"No video for {video_id!r} under {video_dir}")


def build_backend(cfg: ToolMergeConfig):
    if cfg.model_backend == "openai":
        return OpenAIBackend(
            model_name=cfg.openai.model_name,
            api_endpoint=cfg.openai.api_endpoint,
            use_azure=cfg.openai.use_azure,
            max_retries=cfg.openai.max_retries,
        )
    from toolmerge.run import load_qwen3_vl
    model, processor = load_qwen3_vl(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return Qwen3VLBackend(model, processor, device=device)


def linspace_native_indices(start: int, end: int, k: int) -> List[int]:
    span = end - start + 1
    if span <= 1:
        return [start]
    if k >= span:
        return list(range(start, end + 1))
    return np.linspace(start, end, k).astype(int).tolist()


def decode_frames(video_path: str, native_indices: List[int]) -> torch.Tensor:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    for idx in native_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
        rgb = frame[:, :, ::-1].copy()
        frames.append(torch.from_numpy(rgb).permute(2, 0, 1))
    cap.release()
    return torch.stack(frames)


def video_metadata(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if native_fps <= 0 or n_total <= 0:
        raise RuntimeError(f"Bad video metadata for {video_path}: fps={native_fps} total={n_total}")
    return native_fps, n_total


def run_one(item: dict, cfg: ToolMergeConfig, backend) -> Dict[str, Any]:
    video_id = item["video_id"]
    uid = item_uid(item)
    if "start" not in item or "end" not in item:
        raise ValueError(
            f"Oracle baseline requires per-item 'start' / 'end' seconds "
            f"(only M2M provides these). uid={uid} is missing them."
        )

    video_path = find_video(cfg.data.video_dir, video_id)
    native_fps, n_total = video_metadata(video_path)

    start_idx = max(0, int(round(item["start"] * native_fps)))
    end_idx = min(n_total - 1, int(round(item["end"] * native_fps)))

    k = cfg.max_final_k
    indices = linspace_native_indices(start_idx, end_idx, k)
    timestamps = [idx / native_fps for idx in indices]

    frames = decode_frames(video_path, indices)
    res = generate_answer(
        frames, timestamps,
        question=item["question"], options=item["options"],
        backend=backend, cfg=cfg.answer_generator,
    )
    gt = item.get("answer")
    is_correct = res["answer"] == gt
    return {
        "uid": uid,
        "video_id": video_id,
        "question": item["question"],
        "options": item["options"],
        "ground_truth": gt,
        "answer": res["answer"],
        "correct": is_correct,
        "confidence": res["confidence"],
        "answer_raw": res["raw_response"],
        "frames_used": indices,
        "timestamps_used": timestamps,
    }


def main():
    setup_logging()
    cfg = load_config(get_config_path_from_cli(), ToolMergeConfig)
    save_dir = cfg.data.save_path
    os.makedirs(save_dir, exist_ok=True)
    save_config(cfg, os.path.join(save_dir, "config.yaml"))

    backend = build_backend(cfg)
    items = load_dataset(cfg.data.input_path, cfg.data.start_idx, cfg.data.end_idx)
    logger.info("Oracle on %d items (K=%d)", len(items), cfg.max_final_k)

    results: List[Dict[str, Any]] = []
    correct = total = 0
    t_start = time.time()
    for i, item in enumerate(items):
        try:
            r = run_one(item, cfg, backend)
        except Exception as e:  # noqa: BLE001
            logger.error("  Error on %s: %s", item.get("video_id"), e, exc_info=True)
            continue
        results.append(r)
        total += 1
        if r["correct"]:
            correct += 1
        if (i + 1) % 50 == 0:
            logger.info("[%d/%d] acc=%.1f%% (%d/%d)", i + 1, len(items),
                        100 * correct / max(total, 1), correct, total)

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    acc = correct / total if total else 0.0
    with open(os.path.join(save_dir, "accuracy.json"), "w") as f:
        json.dump({"correct": correct, "total": total, "accuracy": round(acc, 4)}, f, indent=2)
    logger.info("Done: %d items, acc=%.1f%%, elapsed=%.0fs", total, acc * 100, time.time() - t_start)


if __name__ == "__main__":
    main()
