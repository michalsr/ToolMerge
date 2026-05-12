"""Convert WFS-SB output to reanswer.py-compatible source format.

WFS outputs `keyframe_indices` as native video frame numbers.
reanswer.py expects `frames_used` as indices on the 2-FPS sampling grid.

This script:
1. Reads WFS output JSON
2. For each item, loads the preprocessing similarity_scores.json to get
   the frame_indices mapping (native frame numbers at each 2-FPS sample point)
3. Converts keyframe_indices (native) → 2-FPS grid indices
4. Maps to reanswer-compatible UIDs
5. Writes {output_dir}/chunk_0/results.json

Usage:
  python scripts/convert_to_reanswer.py \
      --benchmark lvb \
      --wfs-output output/lvb/wfs_blip2_16f.json \
      --features-dir features/lvb/blip2_2fps \
      --output-dir /path/to/reanswer_source/wfs_lvb_16f
"""

import argparse
import json
import os
from pathlib import Path


def native_to_2fps_index(native_frame, frame_indices):
    """Convert a native frame number to the closest 2-FPS grid index.

    frame_indices[i] = native frame number of the i-th sample at 2 FPS.
    Returns the index i such that frame_indices[i] is closest to native_frame.
    """
    best_i = 0
    best_dist = abs(frame_indices[0] - native_frame)
    for i, fi in enumerate(frame_indices):
        dist = abs(fi - native_frame)
        if dist < best_dist:
            best_dist = dist
            best_i = i
    return best_i


def get_feature_id(item, benchmark):
    """Get the feature directory ID for a WFS output item."""
    if benchmark == "lvb":
        # LVB uses the item's index in the annotation list, stored during WFS
        # pipeline as the record index. The WFS output preserves original fields.
        # The feature_id is the string index — we need to find it.
        # WFS pipeline stores feature_id = str(index) for LVB.
        return None  # handled separately
    elif benchmark == "videomme":
        return str(item.get("question_id", ""))
    elif benchmark == "mlvu":
        return str(item.get("question_id", ""))
    return None


def get_uid(item, benchmark, index=None):
    """Map a WFS output item to the UID used by reanswer.py."""
    if benchmark == "lvb":
        return item.get("id", "")  # e.g., "86CxyhFV9MI_0"
    elif benchmark == "videomme":
        return item.get("question_id", "")  # e.g., "601-1"
    elif benchmark == "mlvu":
        return item.get("question_id", "")  # e.g., "Q0"
    return ""


def get_video_id(item, benchmark):
    """Get the video_id for display/matching."""
    if benchmark == "lvb":
        return item.get("video_id", "")
    elif benchmark == "videomme":
        return item.get("videoID", item.get("video_id", ""))
    elif benchmark == "mlvu":
        return item.get("video_name", "").replace(".mp4", "")
    return ""


def convert_options(item, benchmark):
    """Convert options to dict {A: ..., B: ..., ...} for reanswer.py."""
    if benchmark == "lvb":
        candidates = item.get("candidates", [])
        labels = [chr(ord("A") + i) for i in range(len(candidates))]
        return dict(zip(labels, candidates))
    elif benchmark == "videomme":
        opts = item.get("options", [])
        if isinstance(opts, dict):
            return opts
        # List like ["A. Apples.", "B. Candles."] → dict
        result = {}
        for o in opts:
            parts = o.split(". ", 1)
            if len(parts) == 2:
                result[parts[0]] = parts[1]
        return result
    elif benchmark == "mlvu":
        candidates = item.get("candidates", [])
        labels = [chr(ord("A") + i) for i in range(len(candidates))]
        return dict(zip(labels, candidates))
    return {}


def get_answer(item, benchmark):
    """Get ground truth answer letter."""
    if benchmark == "lvb":
        idx = item.get("correct_choice", 0)
        return chr(ord("A") + idx)
    elif benchmark == "videomme":
        return item.get("answer", "")
    elif benchmark == "mlvu":
        # answer is the text, need to match to candidates
        ans_text = item.get("answer", "")
        candidates = item.get("candidates", [])
        for i, c in enumerate(candidates):
            if c == ans_text:
                return chr(ord("A") + i)
        return "A"
    return ""


def get_question_text(item, benchmark):
    """Get clean question text (without embedded options)."""
    q = item.get("question", "")
    if benchmark == "mlvu":
        # Remove embedded options from question text
        lines = q.split("\n")
        clean = []
        for line in lines:
            if line.strip().startswith("(") and len(line.strip()) > 2 and line.strip()[1] in "ABCDEFGH":
                break
            clean.append(line)
        return "\n".join(clean).strip()
    return q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=["lvb", "videomme", "mlvu"])
    parser.add_argument("--wfs-output", required=True, help="WFS pipeline output JSON")
    parser.add_argument("--features-dir", required=True, help="Preprocessing features directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for reanswer source")
    parser.add_argument("--fps", type=float, default=2.0, help="Source FPS used during preprocessing")
    args = parser.parse_args()

    features_dir = Path(args.features_dir)

    with open(args.wfs_output) as f:
        wfs_items = json.load(f)
    print(f"Loaded {len(wfs_items)} WFS items")

    # For LVB, feature_id = str(index in original annotation list).
    # The WFS output preserves all original fields, so we can use the position
    # in the output list IF the WFS ran on the full dataset without subsetting.
    # We also need the original annotation to find the index.
    if args.benchmark == "lvb":
        ann_path = Path(__file__).parent.parent / "datasets/longvideobench/lvb_val_local.json"
        with open(ann_path) as f:
            ann = json.load(f)
        # Build id → original_index map
        id_to_idx = {}
        for i, a in enumerate(ann):
            id_to_idx[a.get("id", "")] = i

    # For VideoMME, multiple questions share one video → one feature dir.
    # The feature_id is video_id (from the JSON), and scores are stored
    # per-question inside the similarity_scores.json.
    # For conversion we only need frame_indices which is per-video.

    results = []
    missing_features = 0

    for item_idx, item in enumerate(wfs_items):
        keyframe_indices = item.get("keyframe_indices", [])
        uid = get_uid(item, args.benchmark, item_idx)
        video_id = get_video_id(item, args.benchmark)

        # Find the preprocessing feature directory
        if args.benchmark == "lvb":
            item_id = item.get("id", "")
            feature_id = str(id_to_idx.get(item_id, item_idx))
        elif args.benchmark == "videomme":
            feature_id = str(item.get("question_id", ""))
        else:
            feature_id = str(item.get("question_id", ""))

        # Load frame_indices from preprocessing
        scores_path = features_dir / feature_id / "similarity_scores.json"
        if scores_path.exists():
            with open(scores_path) as f:
                scores_data = json.load(f)
            frame_indices = scores_data.get("frame_indices", [])
        else:
            frame_indices = []
            missing_features += 1

        # Convert native frame numbers → 2-FPS grid indices
        if frame_indices:
            # Build reverse map for O(1) lookup
            native_to_pos = {}
            for pos, native in enumerate(frame_indices):
                native_to_pos[native] = pos

            frames_used = []
            for kf in keyframe_indices:
                if kf in native_to_pos:
                    frames_used.append(native_to_pos[kf])
                else:
                    # Closest match
                    frames_used.append(native_to_2fps_index(kf, frame_indices))
            frames_used = sorted(set(frames_used))
        else:
            # No preprocessing data — use keyframe_indices as-is (will be wrong
            # scale but at least preserves relative ordering)
            frames_used = sorted(keyframe_indices)

        timestamps_used = [idx / args.fps for idx in frames_used]

        result = {
            "uid": uid,
            "video_id": video_id,
            "question": get_question_text(item, args.benchmark),
            "options": convert_options(item, args.benchmark),
            "ground_truth": get_answer(item, args.benchmark),
            "frames_used": frames_used,
            "timestamps_used": timestamps_used,
            "answer": None,
            "correct": False,
            "answer_raw": None,
            "source": "wfs-sb",
        }
        results.append(result)

    # Write as chunk_0/results.json
    out_dir = Path(args.output_dir) / "chunk_0"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)

    print(f"Wrote {len(results)} items to {out_path}")
    if missing_features:
        print(f"WARNING: {missing_features} items had no preprocessing features")

    # Also save the config for reference
    config_path = Path(args.output_dir) / "wfs_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "benchmark": args.benchmark,
            "wfs_output": args.wfs_output,
            "features_dir": args.features_dir,
            "fps": args.fps,
            "num_items": len(results),
        }, f, indent=2)


if __name__ == "__main__":
    main()
