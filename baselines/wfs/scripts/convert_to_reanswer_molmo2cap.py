"""Convert WFS output to reanswer-compatible format for molmo2cap test split.

Reads the WFS output JSON (which includes keyframe_indices as native video
frame numbers) and the preprocessing similarity_scores.json (which has
frame_indices mapping), converts native frames to 2-FPS grid indices, and
writes chunk_0/results.json in reanswer format.
"""

import argparse
import json
from pathlib import Path


def native_to_2fps_index(native_frame, frame_indices):
    best_i = 0
    best_dist = abs(frame_indices[0] - native_frame)
    for i, fi in enumerate(frame_indices):
        dist = abs(fi - native_frame)
        if dist < best_dist:
            best_dist = dist
            best_i = i
    return best_i


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wfs-output", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    args = parser.parse_args()

    features_dir = Path(args.features_dir)

    with open(args.wfs_output) as f:
        wfs_items = json.load(f)
    print(f"Loaded {len(wfs_items)} WFS items")

    results = []
    missing_features = 0

    for item_idx, item in enumerate(wfs_items):
        keyframe_indices = item.get("keyframe_indices", [])
        feature_id = str(item_idx)

        # Load frame_indices from preprocessing
        scores_path = features_dir / feature_id / "similarity_scores.json"
        if scores_path.exists():
            with open(scores_path) as f:
                scores_data = json.load(f)
            frame_indices = scores_data.get("frame_indices", [])
        else:
            frame_indices = []
            missing_features += 1

        # Convert native frame numbers to 2-FPS grid indices
        if frame_indices:
            native_to_pos = {native: pos for pos, native in enumerate(frame_indices)}
            frames_used = []
            for kf in keyframe_indices:
                if kf in native_to_pos:
                    frames_used.append(native_to_pos[kf])
                else:
                    frames_used.append(native_to_2fps_index(kf, frame_indices))
            frames_used = sorted(set(frames_used))
        else:
            frames_used = sorted(keyframe_indices)

        timestamps_used = [idx / args.fps for idx in frames_used]

        # Options: already a dict {A: ..., B: ...} in the test split format
        options = item.get("options", {})
        if isinstance(options, list):
            options = {chr(ord("A") + i): o for i, o in enumerate(options)}

        result = {
            "uid": item.get("uid", f"{item.get('video_id', '')}_{item_idx}"),
            "video_id": item.get("video_id", ""),
            "question": item.get("question", ""),
            "options": options,
            "ground_truth": item.get("answer", ""),
            "frames_used": frames_used,
            "timestamps_used": timestamps_used,
            "answer": None,
            "correct": False,
            "answer_raw": None,
            "source": "wfs-sb",
        }
        results.append(result)

    out_dir = Path(args.output_dir) / "chunk_0"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)

    print(f"Wrote {len(results)} items to {out_path}")
    if missing_features:
        print(f"WARNING: {missing_features} items had no preprocessing features")


if __name__ == "__main__":
    main()
