"""Convert pre-computed SigLIP2 image features to WFS-SB format for LongVideoBench.

Loads cached SigLIP2-giant image embeddings (2fps, L2-normalized),
encodes text queries with the same SigLIP2 text encoder on GPU,
computes cosine similarities, and writes per-question artifacts
in WFS-SB's expected format:
  - {output_dir}/{idx}/similarity_scores.json
  - {output_dir}/{idx}/siglip_vision_features.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

FEATURE_CACHE_DIR = "/work/hdd/bcgp/michal5/feature_cache/longvideobench/qwen3vl_256"
LVB_JSON = "/work/hdd/bcgp/michal5/WFS-SB/datasets/longvideobench/lvb_val.json"
OUTPUT_DIR = "/work/hdd/bcgp/michal5/WFS-SB/features/lvb/siglip_2fps"
MODEL_NAME = "google/siglip2-giant-opt-patch16-384"
SAMPLE_FPS = 2.0


def load_siglip2_text_encoder(model_name: str, device: str):
    """Load the SigLIP2 model and tokenizer for text encoding."""
    print(f"Loading SigLIP2 model: {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device).eval()
    return model, tokenizer


def encode_text(model, tokenizer, query: str, device: str) -> torch.Tensor:
    """Encode a single text query, return L2-normalized feature (1, D)."""
    inputs = tokenizer(
        [query],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=64,
    )
    input_ids = inputs["input_ids"].to(device)
    with torch.no_grad():
        text_features = model.get_text_features(input_ids=input_ids)
    text_features = text_features.float()
    text_features = F.normalize(text_features, p=2, dim=-1)
    return text_features  # (1, D)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature_cache_dir", default=FEATURE_CACHE_DIR)
    parser.add_argument("--lvb_json", default=LVB_JSON)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    with open(args.lvb_json) as f:
        data = json.load(f)
    print(f"LVB questions: {len(data)}")

    # Load text encoder
    model, tokenizer = load_siglip2_text_encoder(args.model_name, args.device)

    # Pre-load all feature caches into memory (keyed by video_id)
    video_ids = sorted(set(item["video_id"] for item in data))
    print(f"Unique videos: {len(video_ids)}")

    feat_cache = {}
    missing_videos = []
    for vid in tqdm(video_ids, desc="Loading feature caches"):
        cache_path = os.path.join(args.feature_cache_dir, f"{vid}.feature_cache_qwen3vl")
        if not os.path.exists(cache_path):
            missing_videos.append(vid)
            continue
        feat_cache[vid] = torch.load(cache_path, map_location="cpu", weights_only=False).float()

    if missing_videos:
        print(f"WARNING: {len(missing_videos)} videos missing from cache: {missing_videos[:5]}...")

    success = 0
    skipped = 0
    failed = 0

    for idx, item in enumerate(tqdm(data, desc="Processing LVB")):
        out_folder = output_dir / str(idx)
        score_path = out_folder / "similarity_scores.json"
        feat_path = out_folder / "siglip_vision_features.pkl"

        if args.skip_existing and score_path.exists() and feat_path.exists():
            skipped += 1
            continue

        vid = item["video_id"]
        if vid not in feat_cache:
            failed += 1
            continue

        image_features = feat_cache[vid]  # (T, D), L2-normalized
        num_frames = image_features.shape[0]

        # Build query: question + candidates
        query = item["question"] + " " + " ".join(item.get("candidates", []))

        # Encode text
        text_feat = encode_text(model, tokenizer, query, args.device)  # (1, D)

        # Cosine similarity (both already L2-normalized)
        sims = torch.matmul(image_features, text_feat.cpu().T).squeeze(-1)  # (T,)

        # Frame indices at 2fps — we don't know the native fps, so store
        # sequential indices (0, 1, 2, ...) which WFS maps back later
        frame_indices = list(range(num_frames))

        # Save similarity_scores.json
        payload = {
            "video_path": item["video_path"],
            "frame_indices": frame_indices,
            "num_frames": num_frames,
            "query": query,
            "siglip_similarities": sims.tolist(),
            "index": str(idx),
            "question_id": item.get("id", ""),
            "video_id": vid,
            "question_category": item.get("question_category", ""),
            "level": item.get("level", ""),
            "topic_category": item.get("topic_category", ""),
            "question": item.get("question", ""),
            "candidates": item.get("candidates", []),
            "answer": item.get("correct_choice", -1),
        }

        out_folder.mkdir(parents=True, exist_ok=True)
        with score_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        with feat_path.open("wb") as f:
            pickle.dump(image_features.numpy(), f)

        success += 1

    print(f"\nDone: {success} success, {skipped} skipped, {failed} failed (missing video)")


if __name__ == "__main__":
    main()
