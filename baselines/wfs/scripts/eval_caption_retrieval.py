"""Evaluate WFS caption-based clip retrieval.

Runs WFS frame selection on pre-computed SigLIP2 caption scores and evaluates
whether selected frames fall within the ground-truth clip boundaries.

Usage:
    python scripts/eval_caption_retrieval.py \
        --features-dir /work/hdd/bcgp/michal5/WFS-SB/features/caption_retrieval/siglip2_2fps \
        --max-frames 8 \
        --output-dir /work/hdd/bcgp/michal5/WFS-SB/output/caption_retrieval_eval
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from wfs.core import WFS, WFSConfig, compute_dwt_level, compute_min_peak_distance


def timestamp_to_seconds(ts):
    parts = ts.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


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
    """MRR from raw SigLIP similarity scores (the full ranked list)."""
    scored = sorted(zip(frame_indices, similarities),
                    key=lambda x: x[1], reverse=True)
    for rank, (idx, _) in enumerate(scored, 1):
        t = idx / fps
        if gt_start_s <= t <= gt_end_s:
            return round(1.0 / rank, 6)
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", required=True,
                        help="Dir with per-clip WFS feature dirs")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    # WFS hyperparams
    parser.add_argument("--wavelet", default="db4")
    parser.add_argument("--lambda-param", type=float, default=0.5)
    parser.add_argument("--no-visual-features", action="store_true")
    parser.add_argument("--margin", type=float, default=0.0,
                        help="Seconds of tolerance around GT boundaries")
    args = parser.parse_args()

    fps = args.fps

    wfs = WFS(WFSConfig(
        wavelet=args.wavelet,
        lambda_param=args.lambda_param,
    ))

    # Find all clip dirs
    clip_dirs = sorted([
        d for d in os.listdir(args.features_dir)
        if os.path.isdir(os.path.join(args.features_dir, d))
    ])
    print(f"Found {len(clip_dirs)} clip feature dirs")

    results = []
    for clip_id in clip_dirs:
        scores_path = os.path.join(args.features_dir, clip_id, "similarity_scores.json")
        if not os.path.exists(scores_path):
            continue

        with open(scores_path) as f:
            payload = json.load(f)

        gt_start_s = payload["gt_start_s"]
        gt_end_s = payload["gt_end_s"]
        video_id = payload["video_id"]
        relevance_scores = np.array(payload["siglip2_similarities"])
        frame_indices = payload["frame_indices"]

        if len(relevance_scores) < args.max_frames:
            # Uniform fallback
            selected = np.linspace(0, len(relevance_scores) - 1,
                                   args.max_frames, dtype=int).tolist()
        else:
            dwt_level = compute_dwt_level(len(relevance_scores), wavelet=args.wavelet)
            min_peak_distance = compute_min_peak_distance(len(relevance_scores))

            features = None
            if not args.no_visual_features:
                feat_path = os.path.join(args.features_dir, clip_id,
                                         "siglip2_vision_features.pkl")
                if os.path.exists(feat_path):
                    with open(feat_path, "rb") as f:
                        features = pickle.load(f)

            selected = wfs.select_keyframes(
                relevance_scores=relevance_scores,
                num_frames=args.max_frames,
                dwt_level=dwt_level,
                min_peak_distance=min_peak_distance,
                features=features,
            )

        # Map back to original frame indices
        selected_original = []
        for s in selected:
            if s >= len(frame_indices):
                print(f"  WARNING: {clip_id} selected index {s} >= {len(frame_indices)} frames, clamping")
                s = len(frame_indices) - 1
            selected_original.append(frame_indices[s])

        gt_s = gt_start_s - args.margin
        gt_e = gt_end_s + args.margin
        metrics = compute_metrics(selected_original, fps, gt_s, gt_e)
        mrr = compute_mrr(relevance_scores.tolist(), frame_indices, fps, gt_s, gt_e)
        results.append({
            "clip_id": clip_id,
            "video_id": video_id,
            "gt_start": payload["gt_start"],
            "gt_end": payload["gt_end"],
            "gt_start_s": gt_start_s,
            "gt_end_s": gt_end_s,
            "margin": args.margin,
            "selected_timestamps": [round(idx / fps, 2) for idx in selected_original],
            "mrr": mrr,
            "metrics": metrics,
        })

    # Summary
    ok = [r for r in results if r["metrics"]["num_selected"] > 0]
    hits = sum(1 for r in ok if r["metrics"]["hit"])
    precs = [r["metrics"]["precision"] for r in ok]
    mrrs = [r["mrr"] for r in ok]

    summary = {
        "method": "WFS",
        "max_frames": args.max_frames,
        "total_clips": len(results),
        "hit_rate": round(hits / len(ok), 4) if ok else 0,
        "mean_precision": round(np.mean(precs), 4) if precs else 0,
        "mean_mrr": round(np.mean(mrrs), 4) if mrrs else 0,
    }

    print("\n" + "=" * 60)
    print("WFS CAPTION RETRIEVAL EVALUATION")
    print(f"  Clips: {len(ok)}")
    print(f"  Hit rate: {summary['hit_rate']:.1%}")
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
