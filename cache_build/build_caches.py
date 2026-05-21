#!/usr/bin/env python3
"""CLI: build SigLIP / T-REN / OCR / OCR-judge caches for a directory of videos.

The per-modality builders live in their own files:
  - ``cache_build/siglip.py``     -> SigLIP-2 frame features
  - ``cache_build/tren.py``       -> T-REN region tokens (tracked + per-frame)
  - ``cache_build/ocr.py``        -> EasyOCR text per frame
  - ``cache_build/ocr_judge.py``  -> LLM YES/NO over OCR strings (per-question)
  - ``cache_build/utils.py``      -> shared frame loaders (decord + cv2)

Usage:
    python -m cache_build.build_caches \\
        --video_dir /path/to/videos \\
        --dataset_json /path/to/dataset.json \\
        --tools siglip tren_per_frame ocr ocr_judge \\
        --siglip_output_dir          ${TOOLMERGE_CACHE_DIR}/siglip/<dataset> \\
        --tren_per_frame_output_dir  ${TOOLMERGE_CACHE_DIR}/tren/<dataset> \\
        --ocr_output_dir             ${TOOLMERGE_CACHE_DIR}/ocr/<dataset> \\
        --ocr_judge_output_dir       ${TOOLMERGE_CACHE_DIR}/ocr_judge/<dataset> \\
        --video_backend cv2
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch

from cache_build.ocr import build_ocr
from cache_build.ocr_judge import build_ocr_judge
from cache_build.siglip import build_siglip
from cache_build.tren import build_tren, build_tren_per_frame
from cache_build.utils import malloc_trim


TOOL_OUTPUT_FILES = {
    "siglip": lambda d, vid: os.path.join(d, f"{vid}.feature_cache_qwen3vl"),
    "tren":   lambda d, vid: os.path.join(d, f"{vid}.mp4.tren_cache_qwen3vl"),
    "tren_per_frame": lambda d, vid: os.path.join(d, f"{vid}.mp4.tren_pf_cache_qwen3vl"),
    "ocr":    lambda d, vid: os.path.join(d, f"{vid}.ocr_cache"),
    # ocr_judge is per-question (per-uid), not per-video — handled separately.
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build per-video caches for ToolMerge")
    parser.add_argument("--video_dir", required=True,
                        help="Directory containing .mp4 videos")
    parser.add_argument("--dataset_json", default=None,
                        help="Optional JSON with `video_id` fields to filter "
                             "(e.g., lvb_val_std.json)")
    parser.add_argument("--tools", nargs="+", required=True,
                        choices=["siglip", "tren", "tren_per_frame", "ocr", "ocr_judge"],
                        help="Which caches to build")
    parser.add_argument("--siglip_output_dir", default=None)
    parser.add_argument("--tren_output_dir", default=None)
    parser.add_argument("--tren_per_frame_output_dir", default=None)
    parser.add_argument("--ocr_output_dir", default=None)
    parser.add_argument("--ocr_judge_output_dir", default=None,
                        help="Where to write per-question {uid}.json OCR-judge caches.")
    parser.add_argument("--ocr_judge_input_dir", default=None,
                        help="Directory containing {video_id}.ocr_cache files. "
                             "Defaults to --ocr_output_dir.")
    parser.add_argument("--ocr_judge_backend", default="openai",
                        choices=["openai", "qwen3vl"],
                        help="LLM backend for the OCR-judge step.")
    parser.add_argument("--ocr_judge_model_name", default=None,
                        help="OpenAI model name (default: env OPENAI_MODEL or gpt-4o-mini).")
    parser.add_argument("--ocr_judge_batch_size", type=int, default=20,
                        help="Snippets per LLM call; use 1 for one-at-a-time.")
    parser.add_argument("--ocr_judge_qwen_model_path", default=None,
                        help="HF path / local dir for the Qwen3-VL backend "
                             "(only when --ocr_judge_backend qwen3vl).")
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
    """Returns (video_ids, video_dir, items_by_video_id).

    ``items_by_video_id`` is populated when ``--dataset_json`` is supplied;
    it is required for the per-question OCR-judge cache.
    """
    video_dir = Path(args.video_dir)
    items_by_video_id: dict = defaultdict(list)
    if args.dataset_json:
        with open(args.dataset_json) as f:
            data = json.load(f)
        for item in data:
            items_by_video_id[item["video_id"]].append(item)
        video_ids = sorted(items_by_video_id.keys())
    else:
        video_ids = sorted(p.stem for p in video_dir.glob("*.mp4"))

    video_ids = video_ids[args.start_idx:args.end_idx]
    if args.max_videos:
        video_ids = video_ids[:args.max_videos]
    return video_ids, video_dir, items_by_video_id


def all_outputs_exist(args, video_id, items_for_video):
    for tool in args.tools:
        if tool == "ocr_judge":
            if not items_for_video:
                continue
            for item in items_for_video:
                path = os.path.join(args.ocr_judge_output_dir, f"{item['uid']}.json")
                if not os.path.exists(path):
                    return False
            continue
        out_dir = getattr(args, f"{tool}_output_dir")
        path = TOOL_OUTPUT_FILES[tool](out_dir, video_id)
        if not os.path.exists(path):
            return False
    return True


def initialize_clients(args):
    siglip_client = tren_client = ocr_reader = judge_backend = None

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

    if "ocr_judge" in args.tools:
        if args.ocr_judge_backend == "openai":
            from toolmerge.backends.openai import OpenAIBackend
            print("Loading OCR-judge backend: openai")
            judge_backend = OpenAIBackend(
                model_name=args.ocr_judge_model_name
                    or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            )
        else:
            if not args.ocr_judge_qwen_model_path:
                raise SystemExit("--ocr_judge_qwen_model_path is required when "
                                 "--ocr_judge_backend qwen3vl")
            from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
            from toolmerge.backends.qwen3_vl import Qwen3VLBackend
            print(f"Loading OCR-judge backend: qwen3vl from {args.ocr_judge_qwen_model_path}")
            AutoConfig.from_pretrained(args.ocr_judge_qwen_model_path, trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(args.ocr_judge_qwen_model_path)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                args.ocr_judge_qwen_model_path, torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2", device_map="auto",
            ).eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            judge_backend = Qwen3VLBackend(model, processor, device=device)
        print("OCR-judge backend ready")

    return siglip_client, tren_client, ocr_reader, judge_backend


def build_one_video(args, video_id, video_path, items_for_video,
                    siglip_client, tren_client, ocr_reader, judge_backend):
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
    if "ocr_judge" in args.tools:
        built, skipped, missing = build_ocr_judge(
            video_id=video_id,
            items_for_video=items_for_video,
            ocr_cache_dir=args.ocr_judge_input_dir or args.ocr_output_dir,
            output_dir=args.ocr_judge_output_dir,
            backend=judge_backend,
            batch_size=args.ocr_judge_batch_size,
            overwrite=args.overwrite,
        )
        if missing:
            parts.append(f"ocr_judge=missing_ocr({missing}q)")
        else:
            parts.append(f"ocr_judge={built}built/{skipped}skipped")
    return parts, nframes


def main():
    args = parse_args()
    video_ids, video_dir, items_by_video_id = resolve_video_ids(args)

    if "ocr_judge" in args.tools and not args.dataset_json:
        raise SystemExit("--dataset_json is required when building ocr_judge "
                         "(the judge is per-question, not per-video).")

    print(f"Videos to process: {len(video_ids)}")
    print(f"Tools: {args.tools}")
    print(f"Video backend: {args.video_backend}")

    for tool in args.tools:
        out_dir = getattr(args, f"{tool}_output_dir")
        if out_dir is None:
            raise ValueError(f"--{tool}_output_dir is required when building {tool}")
        os.makedirs(out_dir, exist_ok=True)

    siglip_client, tren_client, ocr_reader, judge_backend = initialize_clients(args)

    processed = skipped = 0
    total_time = 0.0
    for video_id in video_ids:
        video_path = video_dir / f"{video_id}.mp4"
        if not video_path.exists():
            print(f"  MISSING: {video_path}")
            continue
        items_for_video = items_by_video_id.get(video_id, [])
        if not args.overwrite and all_outputs_exist(args, video_id, items_for_video):
            skipped += 1
            continue

        t0 = time.time()
        try:
            parts, nframes = build_one_video(
                args, video_id, video_path, items_for_video,
                siglip_client, tren_client, ocr_reader, judge_backend,
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
