"""BOLT frame selection -> reanswer source format.

Implements BOLT (Burlington-style? — see https://github.com/sming256/BOLT)
inverse-transform-sampling frame selection on top of cached SigLIP-2
text-frame similarity scores.

Algorithm (from BOLT/select_frames.py):
    score' = (score - min) / max               # min-max normalize
    if power != -1: score' **= power
    cdf = cumsum(score' / sum(score'))
    u = linspace(1/n, 1 - 1/n, n)
    sampled = searchsorted(cdf, u)             # n grid indices

This script reuses the per-question SigLIP-2 similarity scores already
computed for AKS / Wavelet / TopK-SigLIP baselines (no re-encoding):
    LVB    -> features/lvb/siglip_2fps/{idx}/similarity_scores.json
              (key: "siglip_similarities", model = SigLIP-2)
    VMME-L -> features/vmme_long/siglip2_2fps/{question_id}/similarity_scores.json
              (key: "siglip2_similarities")
    m2c_v2 -> features/molmo2cap_v2/siglip_2fps.old_prebalance/{idx}/similarity_scores.json
              (key: "siglip2_similarities")  -- aligned to test.json order

Outputs reanswer source format at {output_dir}/chunk_0/results.json:
    [{ uid, video_id, question, options, ground_truth,
       frames_used (grid idx), timestamps_used (sec @ 2fps),
       answer=null, correct=null, answer_raw=null, source="bolt_siglip2" }]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

# Use BOLT's inverse_transform_sampling verbatim (cloned at /work/hdd/bcgp/michal5/BOLT).
# Cannot `import select_frames` cleanly because select_frames.py imports the
# extract_feature module (loads CLIP/SigLIP weights). Instead, we compile the
# specific function from its source file so the algorithm comes straight from
# upstream BOLT without our re-implementation.
_BOLT_SELECT_FRAMES = "/work/hdd/bcgp/michal5/BOLT/select_frames.py"


def _load_bolt_inverse_transform_sampling():
    src = Path(_BOLT_SELECT_FRAMES).read_text()
    ns: dict = {"np": np}
    # Locate the function definition and exec just that block.
    start = src.index("def inverse_transform_sampling(")
    # End at the next top-level def or "if __name__"
    after = src[start:]
    end_rel = min(
        (i for i in (
            after.find("\nif __name__"),
            after.find("\ndef ", 1),
        ) if i != -1),
        default=len(after),
    )
    block = after[:end_rel]
    exec(block, ns)
    return ns["inverse_transform_sampling"]


_bolt_its = _load_bolt_inverse_transform_sampling()


def inverse_transform_sampling(score, n: int, power: float = -1):
    """Wrapper around BOLT's inverse_transform_sampling with degenerate-input guards.

    BOLT's function (loaded verbatim from /work/hdd/bcgp/michal5/BOLT/select_frames.py)
    divides by max() and sum(); we fall back to uniform if the score range or
    sum is non-positive (constant-score videos).
    """
    score = np.asarray(score, dtype=float)
    rng = score.max() - score.min()
    if rng <= 0:
        return np.linspace(0, len(score) - 1, n).astype(int)
    idxs = _bolt_its(score, n, power=power)
    idxs = np.clip(np.asarray(idxs), 0, len(score) - 1)
    return idxs


def _pad_to_budget(grid_sel: List[int], n_grid: int, budget: int) -> List[int]:
    if len(grid_sel) >= budget:
        return sorted(set(grid_sel))[:budget]
    if n_grid <= 0:
        return sorted(set(grid_sel))
    fill = np.linspace(0, n_grid - 1, budget, dtype=int).tolist()
    merged = sorted(set([int(x) for x in grid_sel] + [int(x) for x in fill]))
    return merged[:budget]


# -------------------- benchmark adapters --------------------


def adapter_lvb(args):
    """LVB: 1337 items, scores at features/lvb/siglip_2fps/{i}/similarity_scores.json."""
    feat_root = Path(args.feat_dir or "/work/hdd/bcgp/michal5/WFS-SB/features/lvb/siglip_2fps")
    test_path = Path(args.dataset or "/work/nvme/bcgp/michal5/longvideobench/lvb_val_std.json")
    with test_path.open() as f:
        items = json.load(f)
    sims_key = "siglip_similarities"

    def lookup(i, item):
        scores_path = feat_root / str(i) / "similarity_scores.json"
        return scores_path, sims_key

    def to_record(i, item, frames_used, timestamps_used):
        return {
            "uid": item["uid"],
            "video_id": item["video_id"],
            "question": item["question"],
            "options": item["options"],
            "ground_truth": item.get("answer", ""),
            "frames_used": frames_used,
            "timestamps_used": timestamps_used,
            "answer": None,
            "correct": None,
            "answer_raw": None,
            "source": "bolt_siglip2",
        }

    return items, lookup, to_record


def adapter_vmme(args):
    """VMME-L: 900 items (long only), scores at features/vmme_long/siglip2_2fps/{qid}/."""
    feat_root = Path(args.feat_dir or "/work/hdd/bcgp/michal5/WFS-SB/features/vmme_long/siglip2_2fps")
    test_path = Path(args.dataset or "/work/hdd/bcgp/michal5/verify_video/data/video_mme/video_mme_long.json")
    with test_path.open() as f:
        items = json.load(f)
    sims_key = "siglip2_similarities"

    def lookup(i, item):
        qid = item["question_id"]
        scores_path = feat_root / qid / "similarity_scores.json"
        return scores_path, sims_key

    def to_record(i, item, frames_used, timestamps_used):
        opts = item["options"]
        if isinstance(opts, list):
            # Convert "A. text" -> {"A": "text"}
            d = {}
            for o in opts:
                parts = o.split(". ", 1)
                if len(parts) == 2:
                    d[parts[0]] = parts[1]
            opts = d
        return {
            "uid": item["question_id"],
            "video_id": item.get("videoID", item.get("video_id", "")),
            "question": item["question"],
            "options": opts,
            "ground_truth": item.get("answer", ""),
            "frames_used": frames_used,
            "timestamps_used": timestamps_used,
            "answer": None,
            "correct": None,
            "answer_raw": None,
            "source": "bolt_siglip2",
        }

    return items, lookup, to_record


def adapter_m2c_v2(args):
    """m2c_v2: 999 items, scores at features/molmo2cap_v2/siglip_2fps.old_prebalance/{i}/.

    The .old_prebalance dir is aligned to test.json order (same as AKS/topk_siglip).
    """
    feat_root = Path(args.feat_dir or "/work/hdd/bcgp/michal5/WFS-SB/features/molmo2cap_v2/siglip_2fps.old_prebalance")
    test_path = Path(args.dataset or "/work/hdd/bcgp/michal5/verify_video/multi_turn/evidence_pipeline_v2/dataset_generation/group_v2/test.json")
    with test_path.open() as f:
        items = json.load(f)
    sims_key = "siglip2_similarities"

    def lookup(i, item):
        scores_path = feat_root / str(i) / "similarity_scores.json"
        return scores_path, sims_key

    def to_record(i, item, frames_used, timestamps_used):
        opts = item.get("options") or {}
        if isinstance(opts, list):
            opts = {chr(ord("A") + j): o for j, o in enumerate(opts)}
        return {
            "uid": item["uid"],
            "video_id": item["video_id"],
            "question": item["question"],
            "options": opts,
            "ground_truth": item.get("ground_truth") or item.get("answer", ""),
            "frames_used": frames_used,
            "timestamps_used": timestamps_used,
            "answer": None,
            "correct": None,
            "answer_raw": None,
            "source": "bolt_siglip2",
        }

    return items, lookup, to_record


def adapter_vmme_all(args):
    """VMME (full, all durations): 2700 items, scores at features/vmme_all/siglip2_2fps/{qid}/."""
    feat_root = Path(args.feat_dir or "/work/hdd/bcgp/michal5/WFS-SB/features/vmme_all/siglip2_2fps")
    test_path = Path(args.dataset or "/work/hdd/bcgp/michal5/verify_video/data/video_mme/video_mme_all.json")
    with test_path.open() as f:
        items = json.load(f)
    sims_key = "siglip2_similarities"

    def lookup(i, item):
        qid = item["question_id"]
        scores_path = feat_root / qid / "similarity_scores.json"
        return scores_path, sims_key

    def to_record(i, item, frames_used, timestamps_used):
        opts = item["options"]
        if isinstance(opts, list):
            d = {}
            for o in opts:
                parts = o.split(". ", 1)
                if len(parts) == 2:
                    d[parts[0]] = parts[1]
            opts = d
        return {
            "uid": item["question_id"],
            "video_id": item.get("videoID", item.get("video_id", "")),
            "question": item["question"],
            "options": opts,
            "ground_truth": item.get("answer", ""),
            "frames_used": frames_used,
            "timestamps_used": timestamps_used,
            "answer": None,
            "correct": None,
            "answer_raw": None,
            "source": "bolt_siglip2",
        }

    return items, lookup, to_record


ADAPTERS = {
    "lvb": adapter_lvb,
    "vmme": adapter_vmme,
    "vmme_all": adapter_vmme_all,
    "m2c_v2": adapter_m2c_v2,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True, choices=list(ADAPTERS))
    p.add_argument("--k", type=int, required=True, help="Frame budget (8 or 32)")
    p.add_argument("--power", type=float, default=-1, help="BOLT inverse-transform power (default -1 = paper default, no exponent)")
    p.add_argument("--fps", type=float, default=2.0, help="Sampling fps used during feature extraction (default 2)")
    p.add_argument("--feat-dir", default=None, help="Override feature dir")
    p.add_argument("--dataset", default=None, help="Override dataset JSON")
    p.add_argument("--output-dir", required=True, help="Output reanswer source dir (chunk_0/results.json will be written inside)")
    args = p.parse_args()

    items, lookup, to_record = ADAPTERS[args.benchmark](args)
    print(f"BOLT {args.benchmark} K={args.k} power={args.power} -- {len(items)} items")

    out_records = []
    success = 0
    fallback = 0
    missing = 0

    for i, item in enumerate(items):
        scores_path, sims_key = lookup(i, item)
        if not scores_path.exists():
            missing += 1
            # Emit empty record so reanswer keeps alignment; reanswer will skip.
            out_records.append(to_record(i, item, [], []))
            continue
        with scores_path.open() as f:
            sc = json.load(f)
        sims = sc.get(sims_key)
        if sims is None:
            # Some files may use the alternate key naming
            for alt in ("siglip_similarities", "siglip2_similarities"):
                if alt in sc:
                    sims = sc[alt]
                    break
        frame_indices = sc.get("frame_indices") or list(range(len(sims) if sims else 0))
        n_grid = len(sims) if sims else 0

        if n_grid == 0:
            missing += 1
            out_records.append(to_record(i, item, [], []))
            continue

        if n_grid <= args.k:
            grid_sel = list(range(n_grid))
            grid_sel = _pad_to_budget(grid_sel, n_grid, args.k)
            fallback += 1
        else:
            arr = np.asarray(sims, dtype=float)
            sel = inverse_transform_sampling(arr, args.k, args.power)
            grid_sel = sorted(set(int(x) for x in sel.tolist()))
            grid_sel = _pad_to_budget(grid_sel, n_grid, args.k)
            success += 1

        timestamps_used = [g / args.fps for g in grid_sel]
        out_records.append(to_record(i, item, grid_sel, timestamps_used))

    out_dir = Path(args.output_dir) / "chunk_0"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with out_path.open("w") as f:
        json.dump(out_records, f, indent=1)
    print(f"wrote {len(out_records)} items to {out_path}")
    print(f"K={args.k} success={success} fallback={fallback} missing={missing}")


if __name__ == "__main__":
    main()
