"""Downsample WFS-SB SigLIP feature caches from 2 FPS to 1 FPS.

Walks an input features directory layout
    <in_dir>/<sample_id>/<feat_filename>.pkl
    <in_dir>/<sample_id>/similarity_scores.json
and writes a parallel output with every other frame taken (step=2).

Only the fields known to be frame-indexed are downsampled:
  - raw feature array in the .pkl (first axis)
  - similarity_scores.json keys: frame_indices, num_frames, and any *_similarities / *_scores list

Usage:
  python scripts/downsample_features_2fps_to_1fps.py \\
      --in-dir features/lvb/siglip_2fps \\
      --out-dir features/lvb/siglip_1fps \\
      --feat-filename siglip_vision_features.pkl
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


FRAME_KEY_SUFFIXES = ("_similarities", "_scores")


def downsample_features_pkl(src: Path, dst: Path, step: int = 2) -> int:
    with src.open("rb") as f:
        arr = pickle.load(f)
    if not hasattr(arr, "shape"):
        raise TypeError(f"Unexpected pkl payload at {src}: {type(arr).__name__}")
    out = arr[::step]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        pickle.dump(out, f)
    return out.shape[0]


def downsample_scores_json(src: Path, dst: Path, step: int = 2) -> int:
    with src.open("r") as f:
        d = json.load(f)

    n_out = None

    if isinstance(d.get("frame_indices"), list):
        d["frame_indices"] = d["frame_indices"][::step]
        n_out = len(d["frame_indices"])

    for k, v in list(d.items()):
        if isinstance(v, list) and any(k.endswith(s) for s in FRAME_KEY_SUFFIXES):
            d[k] = v[::step]
            n_out = len(d[k])

    if n_out is not None:
        d["num_frames"] = n_out

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        json.dump(d, f)
    return n_out or 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--feat-filename", required=True,
                    help="e.g. siglip_vision_features.pkl or siglip2_vision_features.pkl")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir
    feat_name: str = args.feat_filename

    if not in_dir.is_dir():
        raise SystemExit(f"in-dir not found: {in_dir}")

    sample_dirs = sorted(p for p in in_dir.iterdir() if p.is_dir())
    print(f"Found {len(sample_dirs)} samples under {in_dir}")

    done = skipped = failed = 0
    for i, sdir in enumerate(sample_dirs):
        dst_sdir = out_dir / sdir.name
        feat_src = sdir / feat_name
        score_src = sdir / "similarity_scores.json"

        if not feat_src.exists() or not score_src.exists():
            failed += 1
            if failed <= 5:
                print(f"[skip missing] {sdir.name}: feat={feat_src.exists()} score={score_src.exists()}")
            continue

        feat_dst = dst_sdir / feat_name
        score_dst = dst_sdir / "similarity_scores.json"
        if args.skip_existing and feat_dst.exists() and score_dst.exists():
            skipped += 1
            continue

        try:
            n_feat = downsample_features_pkl(feat_src, feat_dst, step=args.step)
            n_score = downsample_scores_json(score_src, score_dst, step=args.step)
            if n_feat != n_score:
                print(f"[WARN] {sdir.name}: feat rows={n_feat} vs score len={n_score}")
            done += 1
            if done % 200 == 0:
                print(f"  progress {done}/{len(sample_dirs)}  last={sdir.name} n_out={n_feat}")
        except Exception as e:
            failed += 1
            print(f"[fail] {sdir.name}: {e}")

    print(f"Done. wrote={done} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
