"""SigLIP-Q baseline.

Encodes ``question + concatenated answer choices`` with SigLIP-2, scores every
video frame by raw cosine similarity, then applies greedy NMS with the paper's
auto-tau = min(D/(2K), 10s) rule.


Usage:
    python -m baselines.siglip_q.run config=configs/lvb/qwen3_8.yaml
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


# --------------------------- config loader (OmegaConf) ---------------------------

def load_config_from_cli() -> Any:
    """Read `config=path.yaml` plus dotted `key=value` overrides from sys.argv.

    Resolves an optional ``defaults: [- relative/path]`` block one level deep,
    matching the layout under ``configs/``.
    """
    config_path: Optional[str] = None
    overrides: List[str] = []
    for arg in sys.argv[1:]:
        if arg.startswith("config="):
            config_path = arg.split("=", 1)[1]
        elif "=" in arg:
            overrides.append(arg)
    if not config_path:
        raise SystemExit(
            "usage: python -m baselines.siglip_q.run config=<yaml> [k=v ...]"
        )

    def _load(p: str):
        cfg = OmegaConf.load(p)
        defaults = cfg.pop("defaults", None) if hasattr(cfg, "pop") else None
        if not defaults:
            return cfg
        merged = OmegaConf.create({})
        here = Path(p).resolve().parent
        for entry in defaults:
            parent = (here / f"{entry}.yaml").resolve()
            if parent.exists():
                merged = OmegaConf.merge(merged, _load(str(parent)))
        return OmegaConf.merge(merged, cfg)

    cfg = _load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


# --------------------------- SigLIP-2 text encoder ---------------------------

_TEXT_MODEL = None
_TEXT_PROCESSOR = None
_TEXT_DEVICE: Optional[str] = None


def encode_text(query: str, model_name: Optional[str] = None) -> torch.Tensor:
    """Returns a (D,) L2-normalized SigLIP-2 text embedding."""
    global _TEXT_MODEL, _TEXT_PROCESSOR, _TEXT_DEVICE
    if _TEXT_MODEL is None:
        from transformers import AutoModel, AutoProcessor
        name = model_name or os.environ.get(
            "SIGLIP_MODEL", "google/siglip2-giant-opt-patch16-384"
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading SigLIP-2 text encoder %s on %s", name, device)
        _TEXT_PROCESSOR = AutoProcessor.from_pretrained(name)
        attn = "sdpa" if device.startswith("cuda") else "eager"
        _TEXT_MODEL = AutoModel.from_pretrained(name, attn_implementation=attn).eval().to(device)
        _TEXT_DEVICE = device

    inputs = _TEXT_PROCESSOR(
        text=[query], return_tensors="pt",
        padding="max_length", truncation=True, max_length=64,
    )
    input_ids = inputs["input_ids"].to(_TEXT_DEVICE)
    with torch.no_grad():
        f = _TEXT_MODEL.get_text_features(input_ids=input_ids)
    f = f / f.norm(dim=-1, keepdim=True)
    return f.squeeze(0).cpu()


# --------------------------- SigLIP-2 frame cache I/O ---------------------------

_SIGLIP_EXTS = [".feature_cache_qwen3vl", ".mp4.feature_cache_qwen3vl"]


def find_siglip_cache(cache_dir: str, video_id: str) -> Optional[str]:
    for ext in _SIGLIP_EXTS:
        p = os.path.join(cache_dir, f"{video_id}{ext}")
        if os.path.exists(p):
            return p
    return None


def load_siglip_embeddings(path: str) -> torch.Tensor:
    """Returns the (T, D) SigLIP-2 image-embedding tensor at ``path``."""
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
    raise ValueError(f"Unrecognized SigLIP cache at {path}")


# --------------------------- dataset I/O ---------------------------

def load_dataset_items(path: str, start_idx: int = 0, end_idx: Optional[int] = None) -> List[dict]:
    with open(path) as f:
        data = json.load(f)
    return data[start_idx:end_idx] if end_idx is not None else data[start_idx:]


def item_uid(item: dict) -> str:
    return item.get("uid") or item.get("question_id") or ""


# --------------------------- query + raw cosine ---------------------------

def build_query(item: dict) -> str:
    """Question + concatenated option values, alphabetical, no letters."""
    opts = item.get("options") or {}
    if isinstance(opts, dict):
        opts_text = " ".join(opts[k] for k in sorted(opts.keys()))
    else:
        opts_text = " ".join(opts)
    return item["question"] + " " + opts_text


def siglip_cosine(query: str, embeddings: torch.Tensor) -> np.ndarray:
    """Per-frame raw cosine similarity (no percentile normalization)."""
    text_feat = encode_text(query)
    text_feat = F.normalize(text_feat.unsqueeze(0), p=2, dim=1).squeeze(0)
    emb = F.normalize(embeddings.float(), p=2, dim=1)
    return (emb @ text_feat).detach().cpu().numpy().astype(float)


# --------------------------- selector: greedy NMS ---------------------------
# Copied verbatim from toolmerge/selection.py so this file is standalone.

def auto_tau_seconds(num_frames: int, fps: float, max_k: int, cap: float = 10.0) -> float:
    """Paper's default tau = min(D/(2K), cap) in seconds."""
    if fps <= 0 or max_k <= 0:
        return 0.0
    duration = num_frames / fps
    return min(duration / (2 * max_k), cap)


def greedy_gap_select(scored: Dict[int, float], max_k: int, min_gap_frames: int) -> Dict[int, float]:
    """Greedy top-K with a per-frame temporal gap constraint."""
    ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
    selected: Dict[int, float] = {}
    for idx, score in ranked:
        if len(selected) >= max_k:
            break
        if min_gap_frames <= 0 or all(abs(idx - s) >= min_gap_frames for s in selected):
            selected[idx] = score
    return selected


def ordered_by_time(selected: Dict[int, float]) -> List[int]:
    return sorted(selected.keys())


# --------------------------- per-item driver ---------------------------

def run_one(item: dict, cache_dir: str, k: int, cap_seconds: float = 10.0) -> Dict[str, Any]:
    video_id = item["video_id"]
    uid = item_uid(item)
    cache_path = find_siglip_cache(cache_dir, video_id)
    if cache_path is None:
        raise FileNotFoundError(f"SigLIP cache missing for {video_id} in {cache_dir}")
    embeddings = load_siglip_embeddings(cache_path)
    num_frames = embeddings.shape[0]
    fps = 2.0  # cache native fps from cache_build/

    scores = siglip_cosine(build_query(item), embeddings)
    score_map = {i: float(scores[i]) for i in range(num_frames)}

    tau = auto_tau_seconds(num_frames, fps, k, cap_seconds)
    gap_frames = int(tau * fps) if tau > 0 else 0
    selected = greedy_gap_select(score_map, k, gap_frames)
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
    logging.basicConfig(level="INFO", format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    cfg = load_config_from_cli()
    save_dir = str(cfg.data.save_path)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    items = load_dataset_items(
        str(cfg.data.input_path),
        int(cfg.data.get("start_idx") or 0),
        cfg.data.get("end_idx"),
    )
    cache_dir = str(cfg.siglip_feature_cache_dir)
    k = int(cfg.max_final_k)
    cap_seconds = float(cfg.get("min_frame_gap_cap_seconds", 10.0))

    logger.info("SigLIP-Q on %d items, K=%d, cache=%s", len(items), k, cache_dir)
    results: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, item in enumerate(items):
        try:
            r = run_one(item, cache_dir, k, cap_seconds)
            results.append(r)
            logger.info(
                "[%d/%d] %s uid=%s -> %d frames",
                i + 1, len(items), item["video_id"], r["uid"], len(r["frames_used"]),
            )
        except Exception as e:  # noqa: BLE001
            logger.error("  Error on %s: %s", item.get("video_id"), e, exc_info=True)

    out_path = os.path.join(save_dir, "keyframes.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Wrote %d keyframes to %s (%.0fs)", len(results), out_path, time.time() - t0)


if __name__ == "__main__":
    main()
