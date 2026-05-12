"""Select top-k frames by SigLIP similarity score and write reanswer source format.

Usage:
  python scripts/topk_siglip_to_reanswer.py \
      --features-dir /work/hdd/bcgp/michal5/WFS-SB/features/molmo2cap/siglip_2fps \
      --k 8 \
      --output-dir /work/hdd/bcgp/michal5/WFS-SB/output/molmo2cap/reanswer_source_topk_siglip_8f
"""

import argparse
import json
import os
from pathlib import Path


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
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--questions-file", required=True,
                        help="questions_wfs.json with uid field")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--min-frame-gap-seconds", type=float, default=-1.0,
                        help="Minimum gap (seconds) between selected frames. "
                             "-1 (default) = auto: min(duration/(k*2), 10.0), "
                             "matching evidence_gatherer.py. 0 = no gap constraint.")
    args = parser.parse_args()

    features_dir = Path(args.features_dir)

    # Load questions to get UIDs
    with open(args.questions_file) as f:
        questions = json.load(f)
    uid_map = {i: q["uid"] for i, q in enumerate(questions)}
    print(f"Loaded {len(questions)} questions with UIDs")

    # Find all question directories (numbered 0, 1, 2, ...)
    question_dirs = sorted(
        [d for d in features_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    print(f"Found {len(question_dirs)} question directories")

    results = []
    for qdir in question_dirs:
        scores_path = qdir / "similarity_scores.json"
        if not scores_path.exists():
            print(f"  Skipping {qdir.name}: no similarity_scores.json")
            continue

        with open(scores_path) as f:
            data = json.load(f)

        sims = data["siglip2_similarities"]

        # Compute minimum gap in grid units (auto = match evidence_gatherer.py)
        n_grid = len(sims)
        duration_sec = n_grid / args.fps if args.fps > 0 else 0
        if args.min_frame_gap_seconds < 0:
            gap_sec = min(duration_sec / (args.k * 2), 10.0) if args.k > 0 else 0
        else:
            gap_sec = args.min_frame_gap_seconds
        min_gap_grid = int(gap_sec * args.fps) if gap_sec > 0 else 0

        # Greedy gap selection: top-scoring frames with >= gap separation
        topk_grid_indices = greedy_gap_select(sims, args.k, min_gap_grid)
        timestamps_used = [idx / args.fps for idx in topk_grid_indices]

        options = data.get("options", {})
        if isinstance(options, list):
            options = {chr(ord("A") + i): o for i, o in enumerate(options)}

        item_idx = int(qdir.name)
        uid = uid_map.get(item_idx, f"{data.get('video_id', '')}_{item_idx}")
        result = {
            "uid": uid,
            "video_id": data.get("video_id", ""),
            "question": data.get("question", ""),
            "options": options,
            "ground_truth": data.get("answer", ""),
            "frames_used": topk_grid_indices,
            "timestamps_used": timestamps_used,
            "answer": None,
            "correct": None,
            "answer_raw": None,
            "source": "topk_siglip",
        }
        results.append(result)

    # Write output
    out_dir = Path(args.output_dir) / "chunk_0"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Wrote {len(results)} items to {out_path}")


if __name__ == "__main__":
    main()
