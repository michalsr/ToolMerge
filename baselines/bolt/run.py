"""BOLT baseline.

BOLT (CVPR 2025) inverse-transform sampling on a per-frame relevance curve.
We compute the curve as raw SigLIP-2 cosine similarity between the
``question + concatenated options`` query and the cached frame embeddings,
then sample K indices from the relevance CDF.

Standalone: no imports from ``toolmerge``. The ``inverse_transform_sampling``
function is copied verbatim from
https://github.com/sming256/BOLT/blob/main/select_frames.py (lines 76-94 in
the local clone at /work/hdd/bcgp/michal5/BOLT/select_frames.py).

Usage:
    python -m baselines.bolt.run config=configs/lvb/qwen3_8.yaml
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


# --------------------------- config loader (OmegaConf) ---------------------------

def load_config_from_cli() -> Any:
    config_path: Optional[str] = None
    overrides: List[str] = []
    for arg in sys.argv[1:]:
        if arg.startswith("config="):
            config_path = arg.split("=", 1)[1]
        elif "=" in arg:
            overrides.append(arg)
    if not config_path:
        raise SystemExit("usage: python -m baselines.bolt.run config=<yaml> [k=v ...]")

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
    opts = item.get("options") or {}
    if isinstance(opts, dict):
        opts_text = " ".join(opts[k] for k in sorted(opts.keys()))
    else:
        opts_text = " ".join(opts)
    return item["question"] + " " + opts_text


def siglip_cosine(query: str, embeddings: torch.Tensor) -> np.ndarray:
    text_feat = encode_text(query)
    text_feat = F.normalize(text_feat.unsqueeze(0), p=2, dim=1).squeeze(0)
    emb = F.normalize(embeddings.float(), p=2, dim=1)
    return (emb @ text_feat).detach().cpu().numpy().astype(float)


# --------------------------- selector: BOLT inverse-transform sampling ---------------------------
# Copied verbatim from https://github.com/sming256/BOLT/blob/main/select_frames.py
# (local clone: /work/hdd/bcgp/michal5/BOLT/select_frames.py lines 76-94).

def inverse_transform_sampling(score, n, power=-1):
    # normalize the score to 0-1
    score = score - np.min(score)
    score = score / np.max(score)

    # power
    if power != -1:
        score = score**power

    # compute the cumulative distribution function (CDF)
    probabilities = score / np.sum(score)
    cdf = np.cumsum(probabilities)

    # generate uniform values between 0 and 1, exclude the 0 and 1 to avoid out of bounds
    uniform_sampling = np.linspace(1 / n, 1 - 1 / n, n)

    # use the inverse CDF to convert the uniform_sampling to indices
    sampled_indices = np.searchsorted(cdf, uniform_sampling)
    return sampled_indices


def bolt_select(scores: np.ndarray, k: int, power: float = -1.0) -> List[int]:
    """ITS with degenerate-input guard + budget padding."""
    if len(scores) <= k:
        return list(range(len(scores)))
    rng = float(scores.max()) - float(scores.min())
    if rng <= 0:
        return [int(i) for i in np.linspace(0, len(scores) - 1, k, dtype=int).tolist()]
    sampled = inverse_transform_sampling(scores.astype(float), k, power)
    sampled = np.clip(sampled, 0, len(scores) - 1)
    dedup = sorted(set(int(x) for x in sampled.tolist()))
    if len(dedup) < k:
        fill = np.linspace(0, len(scores) - 1, k, dtype=int).tolist()
        merged = sorted(set(dedup + [int(x) for x in fill]))
        dedup = merged[:k]
    return dedup[:k]


# --------------------------- per-item driver ---------------------------

def run_one(item: dict, cache_dir: str, k: int, power: float = -1.0) -> Dict[str, Any]:
    video_id = item["video_id"]
    uid = item_uid(item)
    cache_path = find_siglip_cache(cache_dir, video_id)
    if cache_path is None:
        raise FileNotFoundError(f"SigLIP cache missing for {video_id} in {cache_dir}")
    embeddings = load_siglip_embeddings(cache_path)
    num_frames = embeddings.shape[0]
    fps = 2.0

    scores = siglip_cosine(build_query(item), embeddings)
    indices = sorted(bolt_select(scores, k, power))
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
    power = float(cfg.get("bolt_power", -1.0))

    logger.info("BOLT on %d items, K=%d, power=%s, cache=%s", len(items), k, power, cache_dir)
    results: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, item in enumerate(items):
        try:
            r = run_one(item, cache_dir, k, power)
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
