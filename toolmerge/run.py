"""Command-line entry point.

Usage:
    python -m toolmerge.run config=configs/m2m/retrieval.yaml
    python -m toolmerge.run --config <path> data.save_path=...

The CLI parses an OmegaConf YAML plus dot-path overrides, loads backends
+ tool clients + per-video caches, and runs the pipeline on every item in
the dataset, writing ``results.json`` and ``accuracy.json``.

When ``data.source_dir`` is set, the pipeline runs in reanswer mode: it
reads per-question frame selections from the prior run's results / keyframes
JSON and only invokes the answerer (no planner / tools / OCR judge).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from toolmerge.answerer import generate_answer
from toolmerge.backends import OpenAIBackend, Qwen3VLBackend
from toolmerge.caches import caches_for_video
from toolmerge.config import ToolMergeConfig, get_config_path_from_cli, load_config, save_config
from toolmerge.inputs import item_uid, load_dataset, load_prior_results, save_results
from toolmerge.pipeline import run_pipeline

logger = logging.getLogger(__name__)


# --- Frame extraction (fallback when no frame cache) --------------------

def extract_frames_by_index(
    video_path: str, frame_indices: List[int], target_fps: float, backend: str = "opencv",
):
    """Extract specific frames from a video by their index at ``target_fps``.

    Used when no frame cache was loaded — the pipeline still has SigLIP /
    T-REN / OCR caches but needs pixel frames for the answerer.
    """
    if backend in ("opencv", "cv2"):
        return extract_cv2(video_path, frame_indices, target_fps)
    return extract_decord(video_path, frame_indices, target_fps)


def extract_cv2(video_path, frame_indices, target_fps):
    """Byte-parity with evidence_pipeline_v2/run.py::_extract_frames_cv2."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_indices = [
        min(int(idx / target_fps * native_fps), n_total - 1) for idx in frame_indices
    ]
    frames_list = []
    for ni in native_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, ni)
        ret, frame = cap.read()
        if ret:
            rgb = frame[:, :, ::-1].copy()       # BGR -> RGB
            frames_list.append(torch.from_numpy(rgb).permute(2, 0, 1))
        else:
            raise RuntimeError(f"Failed to read frame {ni} from {video_path}")
    cap.release()
    return torch.stack(frames_list)


def extract_decord(video_path, frame_indices, target_fps):
    import decord
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video_path, num_threads=1)
    native_fps = vr.get_avg_fps()
    step = native_fps / target_fps
    native_idxs = [int(round(i * step)) for i in frame_indices]
    native_idxs = [min(i, len(vr) - 1) for i in native_idxs]
    frames = vr.get_batch(native_idxs)        # (N, H, W, C)
    return frames.permute(0, 3, 1, 2)         # (N, C, H, W)


# --- Backend / tool client setup ----------------------------------------

def setup_main_backend(cfg: ToolMergeConfig):
    if cfg.model_backend == "openai":
        return OpenAIBackend(
            model_name=cfg.openai.model_name,
            api_endpoint=cfg.openai.api_endpoint,
            use_azure=cfg.openai.use_azure,
            max_retries=cfg.openai.max_retries,
        )
    if cfg.model_backend == "qwen3vl":
        model, processor = load_qwen3_vl(cfg)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return Qwen3VLBackend(model, processor, device=device)
    raise ValueError(f"Unknown model_backend: {cfg.model_backend!r}. Use 'qwen3vl' or 'openai'.")


def load_qwen3_vl(cfg: ToolMergeConfig):
    from transformers import AutoConfig, AutoProcessor

    model_path = cfg.model.base
    logger.info("Loading processor from %s", model_path)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(cfg.model.processor_base or model_path)
    logger.info("Processor loaded in %.1fs", time.time() - t0)

    logger.info("Loading model from %s", model_path)
    t0 = time.time()
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(hf_config, "model_type", "")
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg.model.attn_implementation or "flash_attention_2",
        device_map=cfg.model.device_map,
    )
    if model_type == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    elif model_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    else:
        raise ValueError(
            f"Unsupported model_type={model_type!r}. Expected qwen3_vl or qwen2_5_vl."
        )
    model.eval()
    logger.info("Model loaded in %.1fs (model_type=%s)", time.time() - t0, model_type)
    return model, processor


def setup_planner_backend(cfg: ToolMergeConfig, main_backend):
    if not cfg.planner_backend:
        return main_backend
    if cfg.planner_backend == "openai":
        return OpenAIBackend(
            model_name=cfg.planner_openai.model_name,
            api_endpoint=cfg.planner_openai.api_endpoint,
            use_azure=cfg.planner_openai.use_azure,
            max_retries=cfg.planner_openai.max_retries,
        )
    raise ValueError(f"Unknown planner_backend: {cfg.planner_backend!r}")


def setup_tool_clients(cfg: ToolMergeConfig):
    siglip_client = None
    tren_client = None
    if "siglip" in cfg.enabled_tools:
        from toolmerge.tools.siglip import SiglipClient
        siglip_client = SiglipClient(lazy_init=True)
        logger.info("SigLIP client initialized (lazy)")
    if "tren" in cfg.enabled_tools:
        from toolmerge.tools.tren import TrenClient
        tren_client = TrenClient(lazy_init=True)
        logger.info("T-REN client initialized (lazy)")
    return siglip_client, tren_client


def load_source_index(source_dir: str) -> Dict[str, Dict[str, Any]]:
    """Index a prior run's frame selections by uid for reanswer mode.

    Accepts either `results.json` (toolmerge end-to-end runs, has trace +
    pooled_candidates_K) or `keyframes.json` (every baseline, just frames_used).
    """
    for name in ("results.json", "keyframes.json"):
        path = os.path.join(source_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                rows = json.load(f)
            return {r["uid"]: r for r in rows if r.get("uid")}
    raise FileNotFoundError(
        f"source_dir {source_dir!r} has neither results.json nor keyframes.json"
    )


def select_frames_from_source(src: Dict[str, Any], k: int) -> List[int]:
    """Pick K frame indices from a source row.

    Prefers `trace.selection.pooled_candidates_K` (toolmerge end-to-end runs
    save pools at multiple K). Falls back to `frames_used` (every baseline's
    keyframes.json shape, plus toolmerge runs where the source K matches).
    """
    sel = (src.get("trace") or {}).get("selection") or {}
    pool = sel.get(f"pooled_candidates_{k}")
    if pool is not None:
        return [int(idx) for idx, _score in pool[:k]]
    frames_used = src.get("frames_used")
    if frames_used:
        return [int(i) for i in frames_used[:k]]
    avail = [key for key in sel if key.startswith("pooled_candidates_")]
    raise KeyError(
        f"Source has neither pooled_candidates_{k} (available: {avail}) "
        f"nor frames_used."
    )


def reanswer_one_item(item, uid, source_index, backend, cfg, extract_frames):
    """Read K frames from the source row, extract pixels, run only the answerer."""
    if uid not in source_index:
        raise KeyError(f"uid {uid!r} not in source index")
    k = int(cfg.max_final_k)
    indices = select_frames_from_source(source_index[uid], k)
    indices = sorted(indices)  # answerer expects frames in temporal order, not pool-rank order
    fps = 2.0  # paper caches are at 2 FPS; matches research-tree reanswer
    timestamps = [idx / fps for idx in indices]

    video_path = os.path.join(cfg.data.video_dir, f"{item['video_id']}.mp4")
    frames = extract_frames(video_path, indices, fps)

    answer = generate_answer(
        frames, timestamps,
        question=item["question"], options=item["options"],
        backend=backend, cfg=cfg.answer_generator,
    )
    return {
        "answer": answer["answer"],
        "confidence": answer["confidence"],
        "status": "answered",
        "trace": {"selection": {"selected_frames": indices,
                                 "selected_timestamps": timestamps,
                                 "source_uid": uid}},
        "answer_prompt": answer.get("prompt", ""),
        "answer_raw": answer["raw_response"],
        "frames_used": indices,
        "timestamps_used": timestamps,
    }


def set_seed(seed: int) -> None:
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(seed)
    except ImportError:
        pass


# --- Main loop -----------------------------------------------------------

def run(cfg: ToolMergeConfig) -> Dict[str, Any]:
    """Run inference for every item in ``cfg.data.input_path``.

    Returns a summary dict. Writes ``results.json`` + ``accuracy.json`` +
    ``config.yaml`` + ``manifest.json`` under ``cfg.data.save_path``.
    """
    save_dir = cfg.data.save_path
    os.makedirs(save_dir, exist_ok=True)
    results_path = os.path.join(save_dir, "results.json")
    config_save_path = os.path.join(save_dir, "config.yaml")
    save_config(cfg, config_save_path)
    logger.info("Saved config to %s", config_save_path)
    logger.info("Command: %s", " ".join(sys.argv))

    set_seed(cfg.seed)

    source_dir = (cfg.data.source_dir or "").strip()
    reanswer_mode = bool(source_dir)
    source_index: Dict[str, Dict[str, Any]] = {}
    if reanswer_mode:
        source_index = load_source_index(source_dir)
        logger.info(
            "Reanswer mode: %d source items from %s; reading pooled_candidates_%d",
            len(source_index), source_dir, cfg.max_final_k,
        )

    backend = setup_main_backend(cfg)
    if not reanswer_mode:
        siglip_client, tren_client = setup_tool_clients(cfg)
        planner_backend = setup_planner_backend(cfg, backend)
    else:
        siglip_client = tren_client = None
        planner_backend = None

    items = load_dataset(cfg.data.input_path, cfg.data.start_idx, cfg.data.end_idx)
    logger.info("Loaded %d items from %s", len(items), cfg.data.input_path)

    # Resume from prior partial results.
    prior_results = load_prior_results(results_path)
    done_uids = {r.get("uid") for r in prior_results if r.get("uid")}
    results: List[Dict[str, Any]] = list(prior_results)
    correct = sum(1 for r in prior_results if r.get("correct"))
    total = sum(1 for r in prior_results if "correct" in r)
    if prior_results:
        logger.info("Resume: %d completed (acc %d/%d)", len(prior_results), correct, total)

    t_start = time.time()

    def extract(video_path, indices, fps):
        return extract_frames_by_index(
            video_path, indices, fps, backend=cfg.data.video_backend,
        )

    for i, item in enumerate(items):
        video_id = item["video_id"]
        uid = item_uid(item)
        if uid and uid in done_uids:
            logger.info("[%d/%d] uid=%s SKIP (resume)", i + 1, len(items), uid)
            continue
        q_start = time.time()
        logger.info("[%d/%d] video=%s uid=%s", i + 1, len(items), video_id, uid)

        if reanswer_mode:
            try:
                result = reanswer_one_item(
                    item, uid, source_index, backend, cfg, extract,
                )
            except Exception as e:  # noqa: BLE001
                logger.error("  Error on %s (uid=%s): %s", video_id, uid, e, exc_info=True)
                continue
        else:
            try:
                video_caches = caches_for_video(video_id, cfg, siglip_client, tren_client)
            except FileNotFoundError as e:
                logger.warning("  Skipping %s: %s", video_id, e)
                continue

            try:
                result = run_pipeline(
                    question=item["question"],
                    options=item["options"],
                    video_caches=video_caches,
                    backend=backend,
                    cfg=cfg,
                    planner_backend=planner_backend,
                    uid=uid,
                    extract_frames=extract,
                )
            except Exception as e:  # noqa: BLE001
                logger.error("  Error on %s (uid=%s): %s", video_id, uid, e, exc_info=True)
                continue

        result["uid"] = uid
        result["video_id"] = video_id
        result["question"] = item["question"]
        result["options"] = item["options"]
        result["ground_truth"] = item.get("answer")

        gt = item.get("answer")
        is_correct = result["answer"] == gt
        result["correct"] = is_correct
        if is_correct:
            correct += 1
        total += 1
        acc = correct / total if total > 0 else 0.0

        logger.info(
            "  Answer: %s (GT: %s) %s | q_time=%.1fs | acc=%.1f%% (%d/%d)",
            result["answer"], gt, "OK" if is_correct else "WRONG",
            time.time() - q_start, acc * 100, correct, total,
        )

        results.append(result)
        if cfg.save_every_n and len(results) % cfg.save_every_n == 0:
            save_results(results, results_path)

    save_results(results, results_path)
    elapsed = time.time() - t_start
    acc = correct / total if total > 0 else 0.0
    logger.info("Done: %d questions, acc=%.1f%%, elapsed=%.0fs", total, acc * 100, elapsed)

    summary = {
        "correct": correct,
        "total": total,
        "accuracy": round(acc, 4),
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(os.path.join(save_dir, "accuracy.json"), "w") as f:
        json.dump(summary, f, indent=2)

    write_manifest(save_dir, cfg)
    return summary


def write_manifest(save_dir: str, cfg: ToolMergeConfig) -> None:
    """Emit a small JSON capturing config hash + env for reproducibility audits."""
    import hashlib
    import platform

    cfg_path = os.path.join(save_dir, "config.yaml")
    h = hashlib.sha256()
    if os.path.exists(cfg_path):
        with open(cfg_path, "rb") as f:
            h.update(f.read())
    manifest = {
        "config_sha256": h.hexdigest(),
        "seed": cfg.seed,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "command": " ".join(sys.argv),
    }
    with open(os.path.join(save_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


# --- Argparse shell -----------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    try:
        import coloredlogs
        coloredlogs.install(level=level, fmt="%(asctime)s %(name)s %(levelname)s: %(message)s")
    except ImportError:
        logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s: %(message)s")


def main() -> None:
    """Entry point: ``python -m toolmerge.run config=path.yaml [key=value ...]``."""
    setup_logging()
    cfg = load_config(get_config_path_from_cli(), ToolMergeConfig)
    if cfg.debug:
        logging.getLogger("toolmerge").setLevel(logging.DEBUG)
    run(cfg)


if __name__ == "__main__":
    main()
