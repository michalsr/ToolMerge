"""Compute hit@K for a baseline's keyframes.json on the 1K caption retrieval set.

Hit@K = fraction of clips where ANY selected frame falls within the GT
``[start, end]`` interval. Margin is 0 (paper default).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_hms(s: str) -> float:
    """Parse 'HH:MM:SS.ms' or numeric seconds string to float seconds."""
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    if ":" in s:
        parts = s.split(":")
        h, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
        return 3600 * h + 60 * m + sec
    return float(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--keyframes", required=True, help="path to keyframes.json")
    p.add_argument("--dataset", required=True, help="path to caption dataset JSON with uid/start/end")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--margin", type=float, default=0.0)
    p.add_argument("--out", default=None, help="optional path to write retrieval_summary.json")
    args = p.parse_args()

    with open(args.dataset) as f:
        items = json.load(f)
    gt = {it["uid"]: (parse_hms(it["start"]), parse_hms(it["end"])) for it in items}

    with open(args.keyframes) as f:
        keyframes = json.load(f)

    n = 0
    hits = 0
    misses = []
    for r in keyframes:
        uid = r["uid"]
        if uid not in gt:
            continue
        s, e = gt[uid]
        s -= args.margin
        e += args.margin
        timestamps = [idx / args.fps for idx in r["frames_used"]]
        hit = any(s <= t <= e for t in timestamps)
        n += 1
        if hit:
            hits += 1
        else:
            misses.append(uid)

    summary = {
        "keyframes_file": args.keyframes,
        "n_items": n,
        "n_hits": hits,
        "hit_rate": round(hits / n, 4) if n else 0.0,
        "fps": args.fps,
        "margin": args.margin,
        "n_misses": len(misses),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({**summary, "misses": misses}, f, indent=2)


if __name__ == "__main__":
    main()
