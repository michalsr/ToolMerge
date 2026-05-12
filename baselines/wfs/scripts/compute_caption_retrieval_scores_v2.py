"""Compute SigLIP2 text-frame similarity scores using per-CLIP captions.

Reads a prebuilt dedup list (one entry per unique clip) where `question` holds
the per-clip caption (from annotations.clip_captions[i], NOT merged_caption).
Computes cosine similarity of the caption text embedding against all cached
SigLIP2 vision features for the video, writes one per-clip output dir.

Usage:
    python scripts/compute_caption_retrieval_scores_v2.py \
        --dedup-json /work/hdd/bcgp/michal5/molmo2_cap/test_clip_captions_dedup.json \
        --feature-cache-dir /work/hdd/bcgp/michal5/feature_cache/molmo2_cap/qwen3vl_256 \
        --output-dir /work/hdd/bcgp/michal5/WFS-SB/features/caption_retrieval/siglip2_2fps_group_v2 \
        --skip-existing
"""

import argparse
import json
import os
import pickle
import sys

import torch

sys.path.insert(0, "/work/hdd/bcgp/michal5/verify_video/multi_turn")
from time_r1.utils.clip_service import SiglipClient


def timestamp_to_seconds(ts):
    parts = ts.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dedup-json", required=True,
                        help="test_clip_captions_dedup.json (entries have video_id, start, end, question=caption)")
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    with open(args.dedup_json) as f:
        clips = json.load(f)
    print(f"Unique clips: {len(clips)}")

    client = SiglipClient(device="cuda")
    feature_cache = {}

    for i, item in enumerate(clips):
        video_id = item["video_id"]
        start = item["start"]
        end = item["end"]
        caption = item["question"]  # per-clip caption written during dedup build

        clip_id = f"{video_id}_{start}_{end}".replace(":", "-")
        out_dir = os.path.join(args.output_dir, clip_id)
        scores_path = os.path.join(out_dir, "similarity_scores.json")
        features_path = os.path.join(out_dir, "siglip2_vision_features.pkl")

        if args.skip_existing and os.path.exists(scores_path):
            continue

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

        text_feat = client.encode_texts(caption)
        sims = (vision_feats @ text_feat.T).squeeze(-1).numpy().tolist()
        frame_indices = list(range(n_frames))

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
            "uid": item.get("uid"),
        }

        with open(scores_path, "w") as f:
            json.dump(scores_data, f)
        with open(features_path, "wb") as f:
            pickle.dump(vision_feats.numpy(), f)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(clips)}] {video_id} done")

    print(f"Done: processed {len(clips)} clips.")


if __name__ == "__main__":
    main()
