#!/usr/bin/env python3
"""CLI dispatcher: build SigLIP / T-REN / OCR caches for a directory of videos.

The per-modality builders live in their own files:
  - ``cache_build/siglip.py`` -> SigLIP-2 frame features
  - ``cache_build/tren.py``   -> T-REN region tokens (tracked + per-frame)
  - ``cache_build/ocr.py``    -> EasyOCR text per frame (+ optional embedding pass)
  - ``cache_build/utils.py``  -> shared frame loaders (decord + cv2)

The inference-time OCR judge is separate (``toolmerge/tools/ocr_judge.py``);
it operates on the OCR cache produced here and is run from the main pipeline.

Usage:
    python -m cache_build.build_caches \\
        --video_dir /path/to/videos \\
        --dataset_json /path/to/dataset.json \\
        --tools siglip tren_per_frame ocr \\
        --siglip_output_dir          ${TOOLMERGE_CACHE_DIR}/siglip/<dataset> \\
        --tren_per_frame_output_dir  ${TOOLMERGE_CACHE_DIR}/tren/<dataset> \\
        --ocr_output_dir             ${TOOLMERGE_CACHE_DIR}/ocr/<dataset> \\
        --video_backend cv2
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch

from cache_build.ocr import build_ocr
from cache_build.siglip import build_siglip
from cache_build.tren import build_tren, build_tren_per_frame
from cache_build.utils import malloc_trim


TOOL_OUTPUT_FILES = {
    "siglip": lambda d, vid: os.path.join(d, f"{vid}.feature_cache_qwen3vl"),
    "tren":   lambda d, vid: os.path.join(d, f"{vid}.mp4.tren_cache_qwen3vl"),
    "tren_per_frame": lambda d, vid: os.path.join(d, f"{vid}.mp4.tren_pf_cache_qwen3vl"),
    "ocr":    lambda d, vid: os.path.join(d, f"{vid}.ocr_cache"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build per-video caches for ToolMerge")
    parser.add_argument("--video_dir", required=True,
                        help="Directory containing .mp4 videos")
    parser.add_argument("--dataset_json", default=None,
                        help="Optional JSON with `video_id` fields to filter "
                             "(e.g., lvb_val_std.json)")
    parser.add_argument("--tools", nargs="+", required=True,
                        choices=["siglip", "tren", "tren_per_frame", "ocr"],
                        help="Which caches to build")
    parser.add_argument("--siglip_output_dir", default=None)
    parser.add_argument("--tren_output_dir", default=None)
    parser.add_argument("--tren_per_frame_output_dir", default=None)
    parser.add_argument("--ocr_output_dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--max_nframes", type=int, default=None,
                        help="Cap on sampled frames per video")
    parser.add_argument("--tren_batch_size", type=int, default=32,
                        help="Batch size for T-REN DINOv3 feature extraction")
    parser.add_argument("--video_backend", type=str, default="decord",
                        choices=["decord", "cv2", "pyav"],
                        help="Video loading backend. Use cv2 on ARM/aarch64. "
                             "`pyav` is mapped to cv2.")
    return parser.parse_args()


def resolve_video_ids(args):
    video_dir = Path(args.video_dir)
    if args.dataset_json:
        with open(args.dataset_json) as f:
            data = json.load(f)
        video_ids = sorted(set(item["video_id"] for item in data))
    else:
        video_ids = sorted(p.stem for p in video_dir.glob("*.mp4"))

    video_ids = video_ids[args.start_idx:args.end_idx]
    if args.max_videos:
        video_ids = video_ids[:args.max_videos]
    return video_ids, video_dir


def all_outputs_exist(args, video_id):
    for tool in args.tools:
        out_dir = getattr(args, f"{tool}_output_dir")
        path = TOOL_OUTPUT_FILES[tool](out_dir, video_id)
        if not os.path.exists(path):
            return False
    return True


def initialize_clients(args):
    siglip_client = tren_client = ocr_reader = None

    if "siglip" in args.tools:
        from toolmerge.tools.siglip import SiglipClient
        print("Loading SigLIP model...")
        siglip_client = SiglipClient()
        print("SigLIP ready")

    if "tren" in args.tools or "tren_per_frame" in args.tools:
        from toolmerge.tools.tren import TrenClient
        os.environ["TREN_DEVICE"] = "cuda"
        print("Loading T-REN model...")
        tren_client = TrenClient(device="cuda")
        print("T-REN ready")

    if "ocr" in args.tools:
        import easyocr
        print("Loading EasyOCR...")
        ocr_reader = easyocr.Reader(["en"], gpu=True)
        print("EasyOCR ready")

    return siglip_client, tren_client, ocr_reader


def build_one_video(args, video_id, video_path, siglip_client, tren_client, ocr_reader):
    parts = []
    nframes = 0
    if "siglip" in args.tools:
        shape, nframes = build_siglip(
            str(video_path), video_id, args.siglip_output_dir, siglip_client,
            max_nframes=args.max_nframes, backend=args.video_backend,
        )
        parts.append(f"siglip={list(shape)}")
    if "tren" in args.tools:
        n_tracks, nframes = build_tren(
            str(video_path), video_id, args.tren_output_dir, tren_client,
            batch_size=args.tren_batch_size, max_nframes=args.max_nframes,
            backend=args.video_backend,
        )
        parts.append(f"tren={n_tracks}tracks")
    if "tren_per_frame" in args.tools:
        n_regions, nframes = build_tren_per_frame(
            str(video_path), video_id, args.tren_per_frame_output_dir, tren_client,
            batch_size=args.tren_batch_size, max_nframes=args.max_nframes,
            backend=args.video_backend,
        )
        parts.append(f"tren_pf={n_regions}regions")
    if "ocr" in args.tools:
        n_det, nframes = build_ocr(
            str(video_path), video_id, args.ocr_output_dir, ocr_reader,
            max_nframes=args.max_nframes, backend=args.video_backend,
        )
        parts.append(f"ocr={n_det}det")
    return parts, nframes


def main():
    args = parse_args()
    video_ids, video_dir = resolve_video_ids(args)

    print(f"Videos to process: {len(video_ids)}")
    print(f"Tools: {args.tools}")
    print(f"Video backend: {args.video_backend}")

    for tool in args.tools:
        out_dir = getattr(args, f"{tool}_output_dir")
        if out_dir is None:
            raise ValueError(f"--{tool}_output_dir is required when building {tool}")
        os.makedirs(out_dir, exist_ok=True)

    siglip_client, tren_client, ocr_reader = initialize_clients(args)

    processed = skipped = 0
    total_time = 0.0
    for video_id in video_ids:
        video_path = video_dir / f"{video_id}.mp4"
        if not video_path.exists():
            print(f"  MISSING: {video_path}")
            continue
        if not args.overwrite and all_outputs_exist(args, video_id):
            skipped += 1
            continue

        t0 = time.time()
        try:
            parts, nframes = build_one_video(
                args, video_id, video_path,
                siglip_client, tren_client, ocr_reader,
            )
        except Exception as e:
            print(f"  ERROR {video_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        malloc_trim()

        elapsed = time.time() - t0
        total_time += elapsed
        processed += 1
        rss_kb = os.popen(f"ps -o rss= -p {os.getpid()}").read().strip()
        rss_gb = int(rss_kb) / 1024 / 1024 if rss_kb else 0
        print(f"  [{processed}/{len(video_ids)}] {video_id}: "
              f"{nframes} frames, {', '.join(parts)}, {elapsed:.1f}s, RSS={rss_gb:.1f}G")

    print(f"\nDone: {processed} processed, {skipped} skipped, {total_time:.1f}s total")


if __name__ == "__main__":
    main()
