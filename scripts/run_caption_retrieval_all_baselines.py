"""Caption retrieval (Table 5) for all 4 cleaned-up baselines, in one pass.

Encodes each caption query with SigLIP-2 ONCE and computes per-frame cosine
ONCE per clip, then applies SigLIP-Q / WFS / AKS / BOLT selectors at every
K in {1, 2, 4, 8, 16, 32}. Reports hit@K (fraction of clips where any
selected frame falls inside the GT clip interval ``[start, end]``).

This is the right shape for caption retrieval: cosine is identical across
methods and K's; only the selector changes. Doing 24 separate runs (4
methods x 6 K) wastes 24x the SigLIP encoding.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

# Import selectors + helpers from each baseline's standalone run.py.
from baselines.siglip_q.run import (
    auto_tau_seconds,
    greedy_gap_select,
)
from baselines.wfs.run import (
    WFS,
    WFSConfig,
    WFS_DEFAULTS,
    compute_dwt_level,
    compute_min_peak_distance,
)
from baselines.aks.run import aks as aks_select
from baselines.bolt.run import bolt_select


# --------------------------- SigLIP-2 text encoder ---------------------------

_TEXT_MODEL = None
_TEXT_PROCESSOR = None
_TEXT_DEVICE = None


def load_siglip(model_name: str = None):
    global _TEXT_MODEL, _TEXT_PROCESSOR, _TEXT_DEVICE
    if _TEXT_MODEL is not None:
        return
    from transformers import AutoModel, AutoProcessor
    name = model_name or os.environ.get(
        "SIGLIP_MODEL", "google/siglip2-giant-opt-patch16-384"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SigLIP-2 {name} on {device}")
    _TEXT_PROCESSOR = AutoProcessor.from_pretrained(name)
    attn = "sdpa" if device.startswith("cuda") else "eager"
    _TEXT_MODEL = AutoModel.from_pretrained(name, attn_implementation=attn).eval().to(device)
    _TEXT_DEVICE = device


def encode_text(query: str) -> torch.Tensor:
    inputs = _TEXT_PROCESSOR(
        text=[query], return_tensors="pt",
        padding="max_length", truncation=True, max_length=64,
    )
    input_ids = inputs["input_ids"].to(_TEXT_DEVICE)
    with torch.no_grad():
        f = _TEXT_MODEL.get_text_features(input_ids=input_ids)
    f = f / f.norm(dim=-1, keepdim=True)
    return f.squeeze(0).cpu()


# --------------------------- I/O ---------------------------

_SIGLIP_EXTS = [".feature_cache_qwen3vl", ".mp4.feature_cache_qwen3vl"]


def find_siglip_cache(cache_dir: str, video_id: str) -> str | None:
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


def parse_hms(s) -> float:
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    if ":" in s:
        h, m, sec = s.split(":")
        return 3600 * float(h) + 60 * float(m) + float(sec)
    return float(s)


# --------------------------- selectors ---------------------------

WFS_MODEL = WFS(WFSConfig(
    wavelet=WFS_DEFAULTS["wavelet"],
    lambda_param=WFS_DEFAULTS["lambda_param"],
    prominence_factor=WFS_DEFAULTS["prominence_factor"],
    height_factor=WFS_DEFAULTS["height_factor"],
    w_duration=WFS_DEFAULTS["w_duration"],
    w_mean=WFS_DEFAULTS["w_mean"],
    w_max=WFS_DEFAULTS["w_max"],
    w_var=WFS_DEFAULTS["w_var"],
    strictness_factor=WFS_DEFAULTS["strictness_factor"],
    temperature=WFS_DEFAULTS["temperature"],
))


def select_siglip_q(scores: np.ndarray, k: int, fps: float, num_frames: int) -> List[int]:
    score_map = {i: float(scores[i]) for i in range(num_frames)}
    tau = auto_tau_seconds(num_frames, fps, k)
    gap_frames = int(tau * fps) if tau > 0 else 0
    selected = greedy_gap_select(score_map, k, gap_frames)
    return sorted(selected.keys())


def select_wfs(scores: np.ndarray, k: int, embeddings_np: np.ndarray, num_frames: int) -> List[int]:
    dwt_level = compute_dwt_level(num_frames, wavelet=WFS_DEFAULTS["wavelet"],
                                  drift=WFS_DEFAULTS["drift_level"])
    min_peak_distance = compute_min_peak_distance(
        num_frames,
        ratio=WFS_DEFAULTS["min_distance_ratio"],
        absolute_min=WFS_DEFAULTS["min_distance_absolute"],
    )
    indices = WFS_MODEL.select_keyframes(
        relevance_scores=scores, num_frames=k,
        dwt_level=dwt_level, min_peak_distance=min_peak_distance,
        features=embeddings_np,
    )
    return sorted(int(i) for i in indices)


def select_aks(scores: np.ndarray, k: int, num_frames: int) -> List[int]:
    all_depth = min(5, max(1, int(math.log2(k))))
    indices = aks_select(
        scores=scores.tolist(),
        frame_numbers=list(range(num_frames)),
        ratio=1, t1=0.8, t2=-100.0, all_depth=all_depth, max_num_frames=k,
    )
    if len(indices) < k and num_frames > 0:
        fill = np.linspace(0, num_frames - 1, k, dtype=int).tolist()
        indices = sorted(set(indices + [int(x) for x in fill]))[:k]
    return sorted(indices)


def select_bolt(scores: np.ndarray, k: int) -> List[int]:
    return sorted(bolt_select(scores, k, power=-1.0))


SELECTORS = {
    "siglip_q": lambda scores, k, emb_np, n, fps: select_siglip_q(scores, k, fps, n),
    "wfs":      lambda scores, k, emb_np, n, fps: select_wfs(scores, k, emb_np, n),
    "aks":      lambda scores, k, emb_np, n, fps: select_aks(scores, k, n),
    "bolt":     lambda scores, k, emb_np, n, fps: select_bolt(scores, k),
}


# --------------------------- main loop ---------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/work/hdd/bcgp/michal5/molmo2_cap/test_clip_captions_dedup_1k.json")
    p.add_argument("--cache-dir", default="/work/hdd/bcgp/michal5/feature_cache/molmo2_cap/qwen3vl_256")
    p.add_argument("--output-root", default="/work/hdd/bcgp/michal5/toolmerge/outputs/caption_retrieval_1k")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--margin", type=float, default=0.0)
    p.add_argument("--methods", nargs="+", default=["siglip_q", "wfs", "aks", "bolt"])
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    args = p.parse_args()

    with open(args.dataset) as f:
        items = json.load(f)
    print(f"Loaded {len(items)} caption items")

    load_siglip()
    os.makedirs(args.output_root, exist_ok=True)

    # Per-method records: one entry per clip with all K's bundled.
    per_method_records: Dict[str, List[Dict[str, Any]]] = {m: [] for m in args.methods}
    hits_by_method_k: Dict[str, Dict[int, int]] = {m: {k: 0 for k in args.ks} for m in args.methods}
    n_processed = 0

    t0 = time.time()
    for i, item in enumerate(items):
        video_id = item["video_id"]
        uid = item["uid"]
        caption = item["question"]
        gt_start = parse_hms(item["start"])
        gt_end = parse_hms(item["end"])

        cache_path = find_siglip_cache(args.cache_dir, video_id)
        if cache_path is None:
            print(f"  [{i+1}/{len(items)}] missing cache for {video_id}")
            continue
        embeddings = load_siglip_embeddings(cache_path)
        num_frames = embeddings.shape[0]

        # 1x SigLIP encode + 1x cosine, shared across all (method, K).
        text_feat = encode_text(caption)
        text_feat = F.normalize(text_feat.unsqueeze(0), p=2, dim=1).squeeze(0)
        emb = F.normalize(embeddings.float(), p=2, dim=1)
        scores = (emb @ text_feat).detach().cpu().numpy().astype(float)
        embeddings_np = embeddings.detach().cpu().float().numpy()

        for method in args.methods:
            frames_by_k: Dict[str, List[int]] = {}
            hits_by_k: Dict[str, bool] = {}
            for k in args.ks:
                try:
                    indices = SELECTORS[method](scores, k, embeddings_np, num_frames, args.fps)
                except Exception as e:  # noqa: BLE001
                    print(f"  [{i+1}/{len(items)}] {method} K={k} error: {e}")
                    indices = []
                timestamps = [idx / args.fps for idx in indices]
                hit = any(gt_start - args.margin <= t <= gt_end + args.margin for t in timestamps)
                frames_by_k[str(k)] = indices
                hits_by_k[str(k)] = hit
                if hit:
                    hits_by_method_k[method][k] += 1
            per_method_records[method].append({
                "uid": uid,
                "video_id": video_id,
                "gt_start_s": gt_start,
                "gt_end_s": gt_end,
                "num_frames": num_frames,
                "frames_by_k": frames_by_k,
                "hits_by_k": hits_by_k,
            })
        n_processed += 1
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(items)}] elapsed={elapsed:.0f}s", flush=True)
            # Checkpoint: dump partial per-method records + running hit table.
            os.makedirs(args.output_root, exist_ok=True)
            for method in args.methods:
                with (Path(args.output_root) / f"{method}.json").open("w") as f:
                    json.dump(per_method_records[method], f, indent=2, default=str)
            partial = {
                "n_items": n_processed,
                "fps": args.fps, "margin": args.margin,
                "dataset": args.dataset, "cache_dir": args.cache_dir,
                "table": {
                    m: {str(k): round(hits_by_method_k[m][k] / n_processed, 4) for k in args.ks}
                    for m in args.methods
                },
            }
            with (Path(args.output_root) / "hit_at_k_table.json").open("w") as f:
                json.dump(partial, f, indent=2)

    # 4 per-method records files + 1 summary table.
    os.makedirs(args.output_root, exist_ok=True)
    for method in args.methods:
        path = Path(args.output_root) / f"{method}.json"
        with path.open("w") as f:
            json.dump(per_method_records[method], f, indent=2, default=str)

    table = {}
    print("\nhit@K table:")
    print(f"{'method':10s} " + " ".join(f"K={k:<6d}" for k in args.ks))
    for method in args.methods:
        row = {}
        rates_str = []
        for k in args.ks:
            rate = hits_by_method_k[method][k] / n_processed if n_processed else 0.0
            row[str(k)] = round(rate, 4)
            rates_str.append(f"{rate:.4f}")
        table[method] = row
        print(f"{method:10s} " + " ".join(f"{r:<8s}" for r in rates_str))

    with (Path(args.output_root) / "hit_at_k_table.json").open("w") as f:
        json.dump({
            "n_items": n_processed,
            "fps": args.fps,
            "margin": args.margin,
            "dataset": args.dataset,
            "cache_dir": args.cache_dir,
            "table": table,
        }, f, indent=2)
    print(f"\nDone in {time.time() - t0:.0f}s. Output at {args.output_root}")


if __name__ == "__main__":
    main()
