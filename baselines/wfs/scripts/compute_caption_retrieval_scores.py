"""Compute SigLIP2 text-frame similarity scores for caption-based retrieval.

For each unique clip in the test set, uses the clip's merged_caption as the
query text and computes cosine similarity against all frames in the full video.
Saves output in WFS-compatible format (one directory per clip).

Usage:
    python scripts/compute_caption_retrieval_scores.py \
        --test-split /work/hdd/bcgp/michal5/verify_video/multi_turn/evidence_pipeline_v2/dataset_generation/full_pipeline_final/group_1_0412/test_split.json \
        --annotations /work/hdd/bcgp/michal5/molmo2_cap/annotations_filtered.parquet \
        --feature-cache-dir /work/hdd/bcgp/michal5/feature_cache/molmo2_cap/qwen3vl_256 \
        --output-dir /work/hdd/bcgp/michal5/WFS-SB/features/caption_retrieval/siglip2_2fps \
        --skip-existing
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/work/hdd/bcgp/michal5/verify_video/multi_turn")
from time_r1.utils.clip_service import SiglipClient


def timestamp_to_seconds(ts):
    parts = ts.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-split", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    # Load test split and deduplicate to unique clips
    with open(args.test_split) as f:
        data = json.load(f)

    seen = set()
    clips = []
    for item in data:
        key = (item["video_id"], item["start"], item["end"])
        if key not in seen:
            seen.add(key)
            clips.append(key)
    print(f"Unique clips: {len(clips)}")

    # Load annotations
    ann_df = pd.read_parquet(args.annotations)
    ann_lookup = {}
    for _, row in ann_df.iterrows():
        k = (row["video_id"], row["video_start"], row["video_end"])
        ann_lookup[k] = row

    # Load SigLIP2 text encoder
    client = SiglipClient(device="cuda")

    feature_cache = {}

    for i, (video_id, start, end) in enumerate(clips):
        clip_id = f"{video_id}_{start}_{end}".replace(":", "-")
        out_dir = os.path.join(args.output_dir, clip_id)
        scores_path = os.path.join(out_dir, "similarity_scores.json")
        features_path = os.path.join(out_dir, "siglip2_vision_features.pkl")

        if args.skip_existing and os.path.exists(scores_path):
            continue

        # Get caption
        ann_row = ann_lookup.get((video_id, start, end))
        if ann_row is None:
            print(f"  [{i}] SKIP: no annotation for {video_id} {start}-{end}")
            continue

        caption = ann_row["merged_caption"]

        # Load vision features
        cache_file = os.path.join(args.feature_cache_dir, f"{video_id}.feature_cache_qwen3vl")
        if not os.path.exists(cache_file):
            print(f"  [{i}] SKIP: no feature cache for {video_id}")
            continue

        if video_id not in feature_cache:
            vision_feats = torch.load(cache_file, map_location="cpu", weights_only=False)
            feature_cache[video_id] = vision_feats
            if len(feature_cache) > 50:
                oldest = next(iter(feature_cache))
                del feature_cache[oldest]

        vision_feats = feature_cache[video_id]
        n_frames = vision_feats.shape[0]

        # Encode caption text
        text_feat = client.encode_texts(caption)

        # Cosine similarity
        sims = (vision_feats @ text_feat.T).squeeze(-1).numpy().tolist()
        frame_indices = list(range(n_frames))

        # Save
        os.makedirs(out_dir, exist_ok=True)

        scores_data = {
            "frame_indices": frame_indices,
            "num_frames": n_frames,
            "query": caption,
            "siglip2_similarities": sims,
            "video_id": video_id,
            "gt_start": start,
            "gt_end": end,
            "gt_start_s": timestamp_to_seconds(start),
            "gt_end_s": timestamp_to_seconds(end),
            "clip_id": clip_id,
        }

        with open(scores_path, "w") as f:
            json.dump(scores_data, f)

        with open(features_path, "wb") as f:
            pickle.dump(vision_feats.numpy(), f)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i}] {video_id}: {n_frames} frames, "
                  f"sim range [{min(sims):.3f}, {max(sims):.3f}]")

    print("Done.")


if __name__ == "__main__":
    main()
