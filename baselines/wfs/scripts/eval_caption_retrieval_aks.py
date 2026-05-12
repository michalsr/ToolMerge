"""AKS caption retrieval eval (for comparison to WFS / TopK-SigLIP).

Reads per-clip similarity_scores.json produced by compute_caption_retrieval_scores*
and selects frames using AKS (Adaptive Keyframe Sampling) on the cached
text-image cosine similarities. Reports hit_rate / mean_precision / mean_mrr
with a configurable time margin around the GT clip window — identical eval
surface as eval_caption_retrieval_topk.py.

Usage:
    python scripts/eval_caption_retrieval_aks.py \
        --features-dir .../siglip2_2fps_group_v2 \
        --max-frames 8 \
        --margin 0 \
        --output-dir .../caption_retrieval_eval_group_v2_1k/aks_8f
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from run_aks import AKSConfig, aks_select  # noqa: E402


def compute_metrics(selected_frames, fps, gt_start_s, gt_end_s):
    if not selected_frames:
        return {"hit": False, "precision": 0.0,
                "num_selected": 0, "num_gt_frames": 0}
    selected_timestamps = [idx / fps for idx in selected_frames]
    in_gt = [gt_start_s <= t <= gt_end_s for t in selected_timestamps]
    n_in_gt = sum(in_gt)
    gt_start_idx = int(gt_start_s * fps)
    gt_end_idx = int(gt_end_s * fps)
    n_gt_frames = max(1, gt_end_idx - gt_start_idx + 1)
    return {
        "hit": n_in_gt > 0,
        "precision": round(n_in_gt / len(selected_frames), 4),
        "n_in_gt": n_in_gt,
        "num_selected": len(selected_frames),
        "num_gt_frames": n_gt_frames,
    }


def compute_mrr(similarities, frame_indices, fps, gt_start_s, gt_end_s):
    scored = sorted(zip(frame_indices, similarities),
                    key=lambda x: x[1], reverse=True)
    for rank, (idx, _) in enumerate(scored, 1):
        t = idx / fps
        if gt_start_s <= t <= gt_end_s:
            return round(1.0 / rank, 6)
    return 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", required=True)
    p.add_argument("--max-frames", type=int, default=8)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--margin", type=float, default=0.0)
    p.add_argument("--t1", type=float, default=0.8)
    p.add_argument("--t2", type=float, default=-100.0)
    p.add_argument("--ratio", type=int, default=1)
    p.add_argument("--all-depth", type=int, default=None,
                   help="AKS tree depth. Default adaptive: min(5, floor(log2(K))).")
    args = p.parse_args()

    if args.all_depth is None:
        all_depth = min(5, max(1, int(math.log2(max(2, args.max_frames)))))
    else:
        all_depth = args.all_depth

    cfg = AKSConfig(
        max_frames=args.max_frames,
        ratio=args.ratio,
        t1=args.t1,
        t2=args.t2,
        all_depth=all_depth,
    )

    fps = args.fps
    clip_dirs = sorted(d for d in os.listdir(args.features_dir)
                       if os.path.isdir(os.path.join(args.features_dir, d)))
    print(f"Found {len(clip_dirs)} clip feature dirs  (AKS all_depth={all_depth})")

    results = []
    for clip_id in clip_dirs:
        scores_path = os.path.join(args.features_dir, clip_id, "similarity_scores.json")
        if not os.path.exists(scores_path):
            continue
        with open(scores_path) as f:
            payload = json.load(f)

        sims = np.array(payload["siglip2_similarities"], dtype=float)
        frame_indices = payload["frame_indices"]
        gt_start_s = payload["gt_start_s"]
        gt_end_s = payload["gt_end_s"]

        k = min(args.max_frames, len(sims))
        if len(sims) <= args.max_frames:
            selected_original = sorted(int(i) for i in frame_indices)
        else:
            grid_selected = aks_select(sims, cfg)
            mapped = []
            for gi in grid_selected:
                if 0 <= gi < len(frame_indices):
                    mapped.append(int(frame_indices[gi]))
            mapped = sorted(set(mapped))
            if len(mapped) < k:
                order = np.argsort(-sims)
                for i in order:
                    fi = int(frame_indices[i])
                    if fi not in mapped:
                        mapped.append(fi)
                    if len(mapped) >= k:
                        break
                mapped = sorted(mapped)
            selected_original = mapped[:k]

        gt_s = gt_start_s - args.margin
        gt_e = gt_end_s + args.margin
        metrics = compute_metrics(selected_original, fps, gt_s, gt_e)
        mrr = compute_mrr(sims.tolist(), frame_indices, fps, gt_s, gt_e)
        results.append({
            "clip_id": clip_id,
            "video_id": payload["video_id"],
            "gt_start": payload["gt_start"],
            "gt_end": payload["gt_end"],
            "gt_start_s": gt_start_s,
            "gt_end_s": gt_end_s,
            "margin": args.margin,
            "selected_timestamps": [round(idx / fps, 2) for idx in selected_original],
            "mrr": mrr,
            "metrics": metrics,
        })

    ok = [r for r in results if r["metrics"]["num_selected"] > 0]
    hits = sum(1 for r in ok if r["metrics"]["hit"])
    precs = [r["metrics"]["precision"] for r in ok]
    mrrs = [r["mrr"] for r in ok]
    summary = {
        "method": "AKS-SigLIP",
        "max_frames": args.max_frames,
        "all_depth": all_depth,
        "t1": args.t1,
        "t2": args.t2,
        "total_clips": len(results),
        "hit_rate": round(hits / len(ok), 4) if ok else 0,
        "mean_precision": round(np.mean(precs), 4) if precs else 0,
        "mean_mrr": round(np.mean(mrrs), 4) if mrrs else 0,
    }

    print("\n" + "=" * 60)
    print("AKS-SIGLIP CAPTION RETRIEVAL EVALUATION")
    print(f"  Clips: {len(ok)}")
    print(f"  Hit rate:       {summary['hit_rate']:.1%}")
    print(f"  Mean precision: {summary['mean_precision']:.3f}")
    print(f"  Mean MRR:       {summary['mean_mrr']:.4f}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "retrieval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.output_dir, "retrieval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
