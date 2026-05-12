"""Top-k SigLIP caption retrieval eval (for comparison to WFS).

Reads per-clip similarity_scores.json produced by compute_caption_retrieval_scores*
and selects the top-K frames by raw SigLIP text-image cosine similarity.
Reports hit_rate / mean_precision / mean_mrr with a configurable time margin
around the GT clip window — identical eval surface as eval_caption_retrieval.py.

Usage:
    python scripts/eval_caption_retrieval_topk.py \
        --features-dir .../siglip2_2fps_group_v2 \
        --max-frames 8 \
        --margin 5 \
        --output-dir .../caption_retrieval_eval_group_v2/topk_8f_margin5
"""

import argparse
import json
import os

import numpy as np


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


def greedy_gap_select(frame_indices, sims, k, min_gap_frames):
    """Same policy as evidence_gatherer._greedy_gap_select: highest-score first,
    reject any candidate within min_gap_frames of an already-selected frame."""
    order = np.argsort(-sims)
    selected = []
    for i in order:
        if len(selected) >= k:
            break
        idx = frame_indices[i]
        if min_gap_frames <= 0 or all(abs(idx - s) >= min_gap_frames for s in selected):
            selected.append(idx)
    return sorted(selected)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", required=True)
    p.add_argument("--max-frames", type=int, default=8)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--margin", type=float, default=0.0)
    p.add_argument("--apply-gap", action="store_true",
                   help="Apply adaptive temporal gap (min(duration/(K*2), 10s)) via greedy gap selection, matching 'ours' pipeline.")
    args = p.parse_args()

    fps = args.fps
    clip_dirs = sorted(d for d in os.listdir(args.features_dir)
                       if os.path.isdir(os.path.join(args.features_dir, d)))
    print(f"Found {len(clip_dirs)} clip feature dirs  (apply_gap={args.apply_gap})")

    results = []
    for clip_id in clip_dirs:
        scores_path = os.path.join(args.features_dir, clip_id, "similarity_scores.json")
        if not os.path.exists(scores_path):
            continue
        with open(scores_path) as f:
            payload = json.load(f)

        sims = np.array(payload["siglip2_similarities"])
        frame_indices = payload["frame_indices"]
        gt_start_s = payload["gt_start_s"]
        gt_end_s = payload["gt_end_s"]

        k = min(args.max_frames, len(sims))
        if args.apply_gap:
            duration_s = len(sims) / fps
            gap_s = min(duration_s / (args.max_frames * 2), 10.0)
            gap_frames = int(gap_s * fps)
            selected_original = greedy_gap_select(frame_indices, sims, k, gap_frames)
        else:
            top_idx = np.argsort(-sims)[:k]
            selected_original = sorted(frame_indices[i] for i in top_idx)

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
        "method": "TopK-SigLIP-Gap" if args.apply_gap else "TopK-SigLIP",
        "max_frames": args.max_frames,
        "apply_gap": args.apply_gap,
        "total_clips": len(results),
        "hit_rate": round(hits / len(ok), 4) if ok else 0,
        "mean_precision": round(np.mean(precs), 4) if precs else 0,
        "mean_mrr": round(np.mean(mrrs), 4) if mrrs else 0,
    }

    print("\n" + "=" * 60)
    print("TOPK-SIGLIP CAPTION RETRIEVAL EVALUATION")
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
