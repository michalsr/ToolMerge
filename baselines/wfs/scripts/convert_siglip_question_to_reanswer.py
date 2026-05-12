"""Convert SigLIP question-similarity scores to reanswer.py-compatible source format.

For each question, picks top-K frames by SigLIP text-image similarity score.
Similarity scores are already indexed on the 2-FPS grid, so selected indices
are directly usable as frames_used.

Output: {output_dir}/chunk_0/results.json in reanswer source format.

Usage:
  python scripts/convert_siglip_question_to_reanswer.py \
      --benchmark lvb \
      --features-dir features/lvb/siglip_2fps \
      --num-frames 8 \
      --output-dir /path/to/output/siglipq_lvb_8f
"""

import argparse
import json
import os
from pathlib import Path


def convert_options_lvb(item):
    candidates = item.get("candidates", [])
    labels = [chr(ord("A") + i) for i in range(len(candidates))]
    return dict(zip(labels, candidates))


def convert_options_vmme(item):
    opts = item.get("options", [])
    if isinstance(opts, dict):
        return opts
    result = {}
    for o in opts:
        parts = o.split(". ", 1)
        if len(parts) == 2:
            result[parts[0]] = parts[1]
    return result


def get_answer_lvb(item):
    idx = item.get("correct_choice", 0)
    return chr(ord("A") + idx)


def greedy_gap_select(sims, k, min_gap_grid):
    """Select up to k highest-scoring grid indices, each at least min_gap_grid apart.

    Matches evidence_gatherer._greedy_gap_select logic: iterate score-descending,
    accept if >= min_gap_grid from every already-selected index.
    """
    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    picked = []
    for idx, _score in ranked:
        if len(picked) >= k:
            break
        if min_gap_grid <= 0 or all(abs(idx - s) >= min_gap_grid for s in picked):
            picked.append(idx)
    return sorted(picked)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=["lvb", "videomme"])
    parser.add_argument("--features-dir", required=True, help="SigLIP features directory")
    parser.add_argument("--num-frames", type=int, required=True, help="Number of frames to select")
    parser.add_argument("--output-dir", required=True, help="Output directory for reanswer source")
    parser.add_argument("--fps", type=float, default=2.0, help="Source FPS")
    parser.add_argument("--duration-filter", default=None, help="Filter by duration (e.g. long, medium, short)")
    parser.add_argument("--min-frame-gap-seconds", type=float, default=-1.0,
                        help="Minimum gap (seconds) between selected frames. "
                             "-1 (default) = auto: min(duration/(k*2), 10.0), "
                             "matching evidence_gatherer.py. 0 = no gap constraint.")
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    feature_dirs = sorted(features_dir.iterdir())
    print(f"Found {len(feature_dirs)} feature directories in {features_dir}")

    results = []
    missing = 0

    for fdir in feature_dirs:
        scores_path = fdir / "similarity_scores.json"
        if not scores_path.exists():
            missing += 1
            continue

        with open(scores_path) as f:
            data = json.load(f)

        # Filter by duration if requested
        if args.duration_filter and data.get("duration", "") != args.duration_filter:
            continue

        # Find similarity scores (key varies: siglip_similarities or siglip2_similarities)
        sims = None
        for key in ("siglip2_similarities", "siglip_similarities"):
            if key in data:
                sims = data[key]
                break
        if sims is None:
            print(f"WARNING: no similarity scores in {scores_path}")
            missing += 1
            continue

        # Compute minimum gap in grid units (auto = match evidence_gatherer.py)
        n_grid = len(sims)
        duration_sec = n_grid / args.fps if args.fps > 0 else 0
        if args.min_frame_gap_seconds < 0:
            gap_sec = min(duration_sec / (args.num_frames * 2), 10.0) if args.num_frames > 0 else 0
        else:
            gap_sec = args.min_frame_gap_seconds
        min_gap_grid = int(gap_sec * args.fps) if gap_sec > 0 else 0

        # Greedy gap selection: top-scoring frames with >= gap separation
        frames_used = greedy_gap_select(sims, args.num_frames, min_gap_grid)
        timestamps_used = [idx / args.fps for idx in frames_used]

        if args.benchmark == "lvb":
            uid = data.get("question_id", data.get("id", fdir.name))
            video_id = data.get("video_id", "")
            question = data.get("question", data.get("query", ""))
            options = convert_options_lvb(data)
            ground_truth = get_answer_lvb(data)
        elif args.benchmark == "videomme":
            uid = data.get("question_id", fdir.name)
            video_id = data.get("video_id", data.get("videoID", ""))
            question = data.get("question", data.get("query", ""))
            options = convert_options_vmme(data)
            ground_truth = data.get("answer", "")

        result = {
            "uid": uid,
            "video_id": video_id,
            "question": question,
            "options": options,
            "ground_truth": ground_truth,
            "frames_used": frames_used,
            "timestamps_used": timestamps_used,
            "answer": None,
            "correct": False,
            "answer_raw": None,
            "source": "siglip-question",
        }
        results.append(result)

    # Write as chunk_0/results.json
    out_dir = Path(args.output_dir) / "chunk_0"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)

    print(f"Wrote {len(results)} items to {out_path}")
    if missing:
        print(f"WARNING: {missing} items had no similarity scores")

    # Save config
    config_path = Path(args.output_dir) / "wfs_config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "benchmark": args.benchmark,
                "features_dir": str(args.features_dir),
                "num_frames": args.num_frames,
                "fps": args.fps,
                "min_frame_gap_seconds": args.min_frame_gap_seconds,
                "num_items": len(results),
                "method": "siglip-question-similarity",
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
