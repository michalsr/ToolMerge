"""Compute SigLIP2 text-frame similarity scores from pre-cached vision features.

Uses existing SigLIP2 feature caches (torch tensors of shape (T, 1536))
and computes cosine similarity with text-encoded queries. Saves output
in the same format as WFS-SB's preprocess.extract for the WFS pipeline.

Requires the deepspeed_video conda environment (has SigLIP2 model deps).
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

# Add the evidence pipeline code to path for SiglipClient
sys.path.insert(0, "/work/hdd/bcgp/michal5/verify_video/multi_turn")
from time_r1.utils.clip_service import SiglipClient


def build_query_vmme(item):
    """Build query string for Video-MME: question + options."""
    q = item["question"]
    opts = " ".join(item["options"])
    return q + " " + opts


def build_query_mlvu(item):
    """Build query string for MLVU: question text (options already embedded)."""
    return item["question"]


def build_query_molmo2cap(item):
    """Build query string for molmo2cap: question + options dict values."""
    q = item["question"]
    opts = item.get("options", {})
    opts_text = " ".join(opts[k] for k in sorted(opts.keys()))
    return q + " " + opts_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=["vmme", "mlvu", "molmo2cap"])
    parser.add_argument("--json-file", required=True, help="Annotation JSON")
    parser.add_argument("--feature-cache-dir", required=True,
                        help="Dir with .feature_cache_qwen3vl files")
    parser.add_argument("--output-dir", required=True,
                        help="Output dir for WFS-compatible scores")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--query-mode", default="full",
                        choices=["full", "question_only"],
                        help="Text query for SigLIP. 'full' = question+options (default, "
                             "current behavior). 'question_only' = question text only "
                             "(molmo2cap ablation).")
    args = parser.parse_args()

    with open(args.json_file) as f:
        data = json.load(f)

    end_idx = args.end_index if args.end_index is not None else len(data)
    subset = data[args.start_index:end_idx]
    print(f"Processing items {args.start_index}-{end_idx} ({len(subset)} items)")

    # Load SigLIP2 text encoder
    client = SiglipClient(device="cuda")

    # Group by video for VMME (multiple questions per video)
    feature_cache = {}

    for i, item in enumerate(subset):
        global_idx = args.start_index + i

        if args.benchmark == "vmme":
            video_id = item["videoID"]
            feature_id = video_id  # unique per video
            cache_file = os.path.join(
                args.feature_cache_dir,
                f"{video_id}.feature_cache_qwen3vl"
            )
            query = build_query_vmme(item)
            out_id = item.get("question_id", str(global_idx))
        elif args.benchmark == "molmo2cap":
            video_id = item["video_id"]
            feature_id = video_id
            cache_file = os.path.join(
                args.feature_cache_dir,
                f"{video_id}.feature_cache_qwen3vl"
            )
            if args.query_mode == "question_only":
                query = item["question"]
            else:
                query = build_query_molmo2cap(item)
            out_id = str(global_idx)
        elif args.benchmark == "mlvu":
            video_name = item["video_name"]
            video_stem = os.path.splitext(video_name)[0]
            task_type = item.get("task_type", "")
            # Cache files use {task_num}_{task_name}__{video_stem} naming
            mlvu_task_prefix = {
                "plotQA": "1_plotQA",
                "findNeedle": "2_needle",
                "ego": "3_ego",
                "count": "4_count",
                "order": "5_order",
                "anomaly_reco": "6_anomaly_reco",
                "topic_reasoning": "7_topic_reasoning",
            }
            prefix = mlvu_task_prefix.get(task_type, "")
            cache_stem = f"{prefix}__{video_stem}"
            cache_file = os.path.join(
                args.feature_cache_dir,
                f"{cache_stem}.feature_cache_qwen3vl"
            )
            feature_id = cache_stem
            query = build_query_mlvu(item)
            out_id = item.get("question_id", str(global_idx))

        out_dir = os.path.join(args.output_dir, out_id)
        scores_path = os.path.join(out_dir, "similarity_scores.json")
        features_path = os.path.join(out_dir, "siglip2_vision_features.pkl")

        if args.skip_existing and os.path.exists(scores_path):
            continue

        if not os.path.exists(cache_file):
            print(f"  [{global_idx}] SKIP: no cache for {feature_id}")
            continue

        # Load vision features (cached)
        if feature_id not in feature_cache:
            vision_feats = torch.load(cache_file, map_location="cpu",
                                      weights_only=False)
            feature_cache[feature_id] = vision_feats
            # Keep cache bounded
            if len(feature_cache) > 50:
                oldest = next(iter(feature_cache))
                del feature_cache[oldest]

        vision_feats = feature_cache[feature_id]  # (T, 1536)
        n_frames = vision_feats.shape[0]

        # Encode query text
        text_feat = client.encode_texts(query)  # (1, 1536)

        # Cosine similarity (features are already L2-normalized)
        sims = (vision_feats @ text_feat.T).squeeze(-1).numpy().tolist()

        # Frame indices: 0, 1, 2, ... (at 2fps, these ARE the native 2fps grid)
        frame_indices = list(range(n_frames))

        # Save in WFS format
        os.makedirs(out_dir, exist_ok=True)

        scores_data = {
            "frame_indices": frame_indices,
            "num_frames": n_frames,
            "query": query,
            "siglip2_similarities": sims,
        }
        # Copy relevant metadata
        for k in ("video_name", "videoID", "question_id", "question",
                   "candidates", "answer", "task_type", "options",
                   "video_id", "duration"):
            if k in item:
                scores_data[k] = item[k]

        with open(scores_path, "w") as f:
            json.dump(scores_data, f)

        # Save vision features for MMR diversity
        with open(features_path, "wb") as f:
            pickle.dump(vision_feats.numpy(), f)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{global_idx}] {feature_id}: {n_frames} frames, "
                  f"sim range [{min(sims):.3f}, {max(sims):.3f}]")

    print("Done.")


if __name__ == "__main__":
    main()
