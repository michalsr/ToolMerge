"""AKS baseline.

AKS (CVPR 2025) Adaptive Keyframe Sampling. We compute the per-frame
relevance curve as raw SigLIP-2 cosine similarity between the
``question + concatenated options`` query and the cached frame embeddings,
then recursively split the curve and take top-N per surviving segment.

Standalone: no imports from ``toolmerge``. The ``meanstd`` recursive
split + ``aks`` driver are copied verbatim from upstream AKS
(https://github.com/ncTimTang/AKS/blob/main/frame_select.py).

Hyperparameters (paper defaults):
    t1 = 0.8          # peakiness threshold (mean(top-n) - mean)
    t2 = -100         # std lower bound (effectively unconstrained)
    all_depth = min(5, floor(log2(K)))
        The canonical paper value is 5, which yields 2^5 = 32 segments. With
        a smaller budget (e.g. K=8) the paper rule allocates K / 2^depth =
        floor(8 / 32) = 0 frames per segment, falling back to uniform. We
        adapt all_depth = min(5, floor(log2(K))) so AKS remains
        score-driven at every K.

Usage:
    python -m baselines.aks.run config=configs/lvb/qwen3_8.yaml
"""

from __future__ import annotations

import heapq
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
        raise SystemExit("usage: python -m baselines.aks.run config=<yaml> [k=v ...]")

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


# --------------------------- selector: AKS recursive split ---------------------------
# Copied verbatim from https://github.com/ncTimTang/AKS/blob/main/frame_select.py
# Same as /work/hdd/bcgp/michal5/verify_video/aks.py.

def meanstd(len_scores, dic_scores, n, fns, t1, t2, all_depth):
    split_scores = []
    split_fn = []
    no_split_scores = []
    no_split_fn = []
    for dic_score, fn in zip(dic_scores, fns):
        score = dic_score['score']
        depth = dic_score['depth']
        mean = float(np.mean(score))
        std = float(np.std(score))
        top_n = heapq.nlargest(n, range(len(score)), score.__getitem__)
        top_score = [score[t] for t in top_n]
        mean_diff = float(np.mean(top_score)) - mean
        if mean_diff > t1 and std > t2:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
        elif depth < all_depth:
            score1 = score[:len(score) // 2]
            score2 = score[len(score) // 2:]
            fn1 = fn[:len(score) // 2]
            fn2 = fn[len(score) // 2:]
            split_scores.append(dict(score=score1, depth=depth + 1))
            split_scores.append(dict(score=score2, depth=depth + 1))
            split_fn.append(fn1)
            split_fn.append(fn2)
        else:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
    if split_scores:
        rec_scores, rec_fn = meanstd(len_scores, split_scores, n, split_fn, t1, t2, all_depth)
    else:
        rec_scores, rec_fn = [], []
    return no_split_scores + rec_scores, no_split_fn + rec_fn


def aks(scores: Sequence[float], frame_numbers: Sequence[int],
        ratio: int = 1, t1: float = 0.8, t2: float = -100.0,
        all_depth: int = 5, max_num_frames: int = 16) -> List[int]:
    """Run AKS on a 1-D relevance curve, return native frame indices."""
    score = list(scores)[::ratio]
    fn = list(frame_numbers)[::ratio]
    num = max_num_frames
    if len(score) < num:
        return list(fn)

    arr = np.asarray(score, dtype=float)
    rng = arr.max() - arr.min()
    if rng <= 0:
        normalized = np.zeros_like(arr)
    else:
        normalized = (arr - arr.min()) / rng

    segs, seg_fns = meanstd(
        len(score),
        [dict(score=normalized.tolist(), depth=0)],
        num, [fn], t1, t2, all_depth,
    )

    out: List[int] = []
    for seg, f in zip(segs, seg_fns):
        if not seg['score'] or not f:
            continue
        f_num = int(num / (2 ** seg['depth']))
        if f_num <= 0:
            continue
        topk = heapq.nlargest(f_num, range(len(seg['score'])), seg['score'].__getitem__)
        out.extend(int(f[t]) for t in topk)
    return sorted(set(out))


# --------------------------- per-item driver ---------------------------

def run_one(item: dict, cache_dir: str, k: int,
            t1: float = 0.8, t2: float = -100.0,
            all_depth: Optional[int] = None) -> Dict[str, Any]:
    video_id = item["video_id"]
    uid = item_uid(item)
    cache_path = find_siglip_cache(cache_dir, video_id)
    if cache_path is None:
        raise FileNotFoundError(f"SigLIP cache missing for {video_id} in {cache_dir}")
    embeddings = load_siglip_embeddings(cache_path)
    num_frames = embeddings.shape[0]
    fps = 2.0

    scores = siglip_cosine(build_query(item), embeddings)

    if all_depth is None:
        all_depth = min(5, max(1, int(math.log2(k))))
    indices = aks(
        scores=scores.tolist(),
        frame_numbers=list(range(num_frames)),
        ratio=1, t1=t1, t2=t2, all_depth=all_depth, max_num_frames=k,
    )

    # Pad with uniform fill if fewer than K were selected (small video or low all_depth).
    if len(indices) < k and num_frames > 0:
        fill = np.linspace(0, num_frames - 1, k, dtype=int).tolist()
        indices = sorted(set(indices + [int(x) for x in fill]))[:k]
    indices = sorted(indices)
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
    t1 = float(cfg.get("aks_t1", 0.8))
    t2 = float(cfg.get("aks_t2", -100.0))
    all_depth = cfg.get("aks_all_depth", None)
    if all_depth is not None:
        all_depth = int(all_depth)

    logger.info("AKS on %d items, K=%d, t1=%s, t2=%s, all_depth=%s, cache=%s",
                len(items), k, t1, t2, all_depth, cache_dir)
    results: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, item in enumerate(items):
        try:
            r = run_one(item, cache_dir, k, t1, t2, all_depth)
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
