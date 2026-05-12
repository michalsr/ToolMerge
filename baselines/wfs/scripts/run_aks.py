"""Run AKS (Adaptive Keyframe Sampling) over cached SigLIP/SigLIP2 similarities.

Mirrors `wfs/pipeline.py` so the AKS output JSON matches the WFS schema and can
be consumed by the existing `convert_to_reanswer.py` + reanswer Slurm scripts.

AKS algorithm follows
https://github.com/ncTimTang/AKS/blob/main/frame_select.py:
- Min-max normalize per-frame scores
- Recursively split each segment in half until either:
    * (mean(top_n) - mean) > t1 AND std > t2  (peaky enough -> stop), or
    * depth >= all_depth (stop)
- For each surviving segment, take top (max_frames / 2**depth) by score
- Concatenate, sort by sampled-grid index, map back to native frame indices

Inputs come from `features/<bench>/<feature_model>_2fps/<feature_id>/similarity_scores.json`
which already stores `frame_indices` (native) and `<feature_model>_similarities`.
"""
from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

import sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from wfs.benchmarks import (  # noqa: E402
    BenchmarkRecord,
    benchmark_defaults,
    create_adapter,
)
from wfs.pipeline import (  # noqa: E402
    FeatureStore,
    parse_index_list,
    select_by_indices,
    uniform_sample_from_video,
)


@dataclass
class AKSConfig:
    max_frames: int = 32
    ratio: int = 1
    t1: float = 0.8
    t2: float = -100.0
    all_depth: int = 5


def _meanstd(
    dic_scores: List[Dict[str, Any]],
    fns: List[Sequence[int]],
    n: int,
    t1: float,
    t2: float,
    all_depth: int,
):
    """Recursive split-vs-keep used by AKS.

    Each entry of `dic_scores` is {"score": list[float], "depth": int}.
    `fns[i]` is the parallel list of (sampled-grid) indices for segment i.
    """
    split_scores: List[Dict[str, Any]] = []
    split_fn: List[Sequence[int]] = []
    no_split_scores: List[Dict[str, Any]] = []
    no_split_fn: List[Sequence[int]] = []

    for dic_score, fn in zip(dic_scores, fns):
        score = dic_score["score"]
        depth = dic_score["depth"]
        mean = float(np.mean(score))
        std = float(np.std(score))
        top_n = heapq.nlargest(n, range(len(score)), score.__getitem__)
        top_score = [score[t] for t in top_n]
        mean_diff = float(np.mean(top_score)) - mean
        if mean_diff > t1 and std > t2:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)
        elif depth < all_depth:
            half = len(score) // 2
            split_scores.append(dict(score=score[:half], depth=depth + 1))
            split_scores.append(dict(score=score[half:], depth=depth + 1))
            split_fn.append(fn[:half])
            split_fn.append(fn[half:])
        else:
            no_split_scores.append(dic_score)
            no_split_fn.append(fn)

    if split_scores:
        rec_scores, rec_fn = _meanstd(split_scores, split_fn, n, t1, t2, all_depth)
    else:
        rec_scores, rec_fn = [], []
    return no_split_scores + rec_scores, no_split_fn + rec_fn


def aks_select(
    scores: np.ndarray,
    cfg: AKSConfig,
) -> List[int]:
    """Run AKS on a 1-D score vector. Returns sampled-grid indices."""
    score_list = list(scores)
    fn_list = list(range(len(score_list)))

    if cfg.ratio > 1:
        score_list = score_list[:: cfg.ratio]
        fn_list = fn_list[:: cfg.ratio]

    n = cfg.max_frames
    if len(score_list) < n:
        return fn_list

    arr = np.asarray(score_list, dtype=float)
    rng = arr.max() - arr.min()
    if rng <= 0:
        normalized = np.zeros_like(arr)
    else:
        normalized = (arr - arr.min()) / rng

    segs, seg_fns = _meanstd(
        [dict(score=normalized.tolist(), depth=0)],
        [fn_list],
        n=n,
        t1=cfg.t1,
        t2=cfg.t2,
        all_depth=cfg.all_depth,
    )

    out: List[int] = []
    for seg, fn in zip(segs, seg_fns):
        f_num = int(n / (2 ** seg["depth"]))
        if f_num <= 0:
            continue
        topk = heapq.nlargest(f_num, range(len(seg["score"])), seg["score"].__getitem__)
        out.extend(int(fn[t]) for t in topk)

    return sorted(set(out))


def _map_to_original_indices(selected: List[int], frame_indices: List[int], budget: int) -> List[int]:
    """Map sampled-grid indices to native frame numbers; pad with uniform fill."""
    if not frame_indices:
        return list(range(budget))
    mapped: List[int] = []
    for idx in selected:
        if 0 <= idx < len(frame_indices):
            mapped.append(int(frame_indices[idx]))
    mapped = sorted(set(mapped))
    if len(mapped) < budget:
        fill_positions = np.linspace(0, len(frame_indices) - 1, budget, dtype=int).tolist()
        fill_frames = [int(frame_indices[p]) for p in fill_positions]
        merged = sorted(set(mapped + fill_frames))
        mapped = merged[:budget]
    if len(mapped) < budget and mapped:
        mapped = mapped + [mapped[-1]] * (budget - len(mapped))
    return mapped[:budget] if mapped else list(range(budget))


def run(args: argparse.Namespace) -> Dict[str, Any]:
    adapter = create_adapter(args.benchmark)
    dataset_root = Path(args.dataset_root)
    questions_file = Path(args.questions_file)
    features_dir = Path(args.features_dir)

    raw_items = adapter.load_raw(questions_file)
    selected_raw = select_by_indices(
        raw_items=raw_items,
        start=args.start_index,
        end=args.end_index,
        index_list=parse_index_list(args.index_list),
    )
    records: List[BenchmarkRecord] = [
        adapter.build_record(index=i, item=item, dataset_root=dataset_root)
        for i, item in selected_raw
    ]

    cfg = AKSConfig(
        max_frames=args.max_frames,
        ratio=args.ratio,
        t1=args.t1,
        t2=args.t2,
        all_depth=args.all_depth,
    )

    store = FeatureStore(features_dir=features_dir)
    results: List[Dict[str, Any]] = []
    missing = 0
    fallback = 0
    success = 0

    for record in tqdm(records, desc=f"AKS-{adapter.name}"):
        payload = store.load_scores(record.feature_id)
        if payload is None:
            missing += 1
            fallback += 1
            keyframes = uniform_sample_from_video(record.video_path, args.max_frames)
            results.append(adapter.to_output_item(record, keyframes))
            continue

        relevance = adapter.get_scores(payload, record, args.feature_model)
        frame_indices = payload.get("frame_indices", [])
        if relevance is None or len(relevance) == 0:
            fallback += 1
            keyframes = uniform_sample_from_video(record.video_path, args.max_frames)
            results.append(adapter.to_output_item(record, keyframes))
            continue
        if not isinstance(frame_indices, list) or len(frame_indices) == 0:
            frame_indices = list(range(len(relevance)))

        if len(relevance) < args.max_frames:
            fallback += 1
            keyframes = uniform_sample_from_video(record.video_path, args.max_frames)
            results.append(adapter.to_output_item(record, keyframes))
            continue

        selected = aks_select(np.asarray(relevance, dtype=float), cfg)
        keyframes = _map_to_original_indices(selected, frame_indices, args.max_frames)
        results.append(adapter.to_output_item(record, keyframes))
        success += 1

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return {
        "benchmark": adapter.name,
        "num_records": len(records),
        "success": success,
        "fallback": fallback,
        "missing_scores": missing,
        "output_path": str(output_path),
        "aks_config": cfg.__dict__,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run AKS frame selection on cached similarities")
    p.add_argument("--benchmark", required=True, choices=["videomme", "lvb", "mlvu"])
    p.add_argument("--dataset_root", default=None)
    p.add_argument("--questions_file", default=None)
    p.add_argument("--features_dir", default=None)
    p.add_argument("--feature_model", default="siglip", choices=["blip2", "blip1", "clip", "siglip", "siglip2"])
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--output_path", default=None)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--end_index", type=int, default=-1)
    p.add_argument("--index_list", default=None)
    p.add_argument("--ratio", type=int, default=1)
    p.add_argument("--t1", type=float, default=0.8)
    p.add_argument("--t2", type=float, default=-100.0)
    # all_depth default is None -> adaptive: min(5, floor(log2(max_frames))).
    # AKS partitions the video into 2^all_depth segments and picks
    # max_frames / 2^all_depth frames per segment. With the canonical paper
    # default of 5, any K < 32 yields 0 frames per segment -> empty selection
    # that falls back to uniform padding. Adaptive keeps AKS score-based at
    # every K. Pass --all_depth 5 explicitly to reproduce canonical behavior.
    p.add_argument("--all_depth", type=int, default=None)
    return p


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    d = benchmark_defaults(args.benchmark)
    if args.dataset_root is None:
        args.dataset_root = str(d["dataset_root"])
    if args.questions_file is None:
        args.questions_file = str(d["questions_file"])
    if args.features_dir is None:
        args.features_dir = str(d["features_dir"])
    if args.max_frames is None:
        args.max_frames = int(d["max_frames"])
    if args.all_depth is None:
        import math
        args.all_depth = min(5, max(1, int(math.log2(args.max_frames))))
    return args


def main() -> None:
    args = build_parser().parse_args()
    args = resolve_defaults(args)
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
