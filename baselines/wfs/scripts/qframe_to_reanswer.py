"""Q-Frame (Xiaomi) frame selection -> reanswer source format.

Runs the Q-Frame text-image matching frame-selection algorithm from
xiaomi-research/q-frame using SigLIP-2 instead of Long-CLIP. The Q-Frame
algorithm consumes a similarity row `logits_per_text = text @ image.T`
between the question and the candidate frames, applies softmax with
temperature tau, then samples a ranking via the Gumbel-Max trick. The
cached SigLIP-2 cosine-similarity scores in WFS-SB/features are exactly
that quantity (L2-normalised SigLIP-2 features → text @ image.T), so the
encoding stage already used SigLIP-2 — we just feed the cached row into
Q-Frame's verbatim softmax+Gumbel-Max+argsort block.

Q-Frame qwen2vl uniformly samples max_num_frames=128 candidates from the
video before scoring; we mirror that by uniformly subsampling 128
candidates from the cached 2-FPS grid. tau=0.8 matches their qwen2vl run
(qwen2_vl_w_qframe.py:287).

Outputs the same reanswer-source schema used by bolt_to_reanswer.py /
aks_m2c_v2_to_reanswer.py: {output_dir}/chunk_0/results.json.

Adapters: lvb, vmme (full set, 2700 items), m2c_v2 (999 items).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch


# ---------------- Q-Frame scoring (vendored verbatim) ----------------
# Source: https://github.com/xiaomi-research/q-frame
#         lmms_eval/models/qwen2_vl_w_qframe.py, lines 60-66
#         (also identical at lmms_eval/models/gpt4o_w_qframe.py).
# Copied with no edits. `images` is the candidate-frame tensor in upstream
# (only its length matters here, as the Gumbel-noise length); we pass a 1-D
# placeholder of the same length so the line reads identically to upstream.
def _qframe_rank(logits_per_text, tau, images):
    probs = (logits_per_text / tau).softmax(dim=1)[0]

    probs = torch.log(probs) - torch.log(-torch.log(torch.rand(len(images), device=probs.device) + 1e-10) + 1e-10)  # gumble

    indices = np.argsort(-probs.cpu().detach().numpy())

    return indices


# SigLIP-2 giant model.logit_scale.exp() (verified by loading
# google/siglip2-giant-opt-patch16-384 — see commit notes). Long-CLIP's
# encode_*() return UN-normalised projections, so its `text @ image.T`
# is a raw matmul ~10-30 in magnitude and softmax(/tau=0.8) is sharply
# peaked. Our cached SigLIP-2 sims are L2-normalised cosine in [-1, 1],
# so we have to multiply by the model's logit_scale before applying
# Q-Frame's softmax to recover the same peakedness. This matches what
# SigLIP-2 itself uses pre-sigmoid (logit_scale * cos + logit_bias).
SIGLIP2_GIANT_LOGIT_SCALE = 108.33345794677734


def qframe_select(
    sims: np.ndarray,
    k: int,
    tau: float = 0.8,
    logit_scale: float = SIGLIP2_GIANT_LOGIT_SCALE,
    seed: int = None,
):
    """Apply Q-Frame's softmax+Gumbel-Max selection to a cached SigLIP-2
    similarity row at 2 FPS.

    The cached `sims` are L2-normalised SigLIP-2 cosine similarities. Q-Frame
    upstream feeds Long-CLIP's `text_features @ image_features.T` of
    UN-normalised projections (range ~10-30) into `softmax(/tau)`. To match
    that softmax peakedness with our cosine sims, we multiply by SigLIP-2's
    `logit_scale.exp() ≈ 108` before applying Q-Frame's vendored block.
    Candidate pool is the full 2-FPS grid (consistent with BOLT/AKS/WFS/
    TopK-SigLIP/Ours), not Q-Frame's 128-frame `np.linspace` subsample of
    the video, so frame budgets line up across methods.

    Args:
        sims: (T,) SigLIP-2 cosine sims at 2 FPS.
        k: frames to select.
        tau: softmax temperature; Q-Frame qwen2vl runs use 0.8
             (qwen2_vl_w_qframe.py:287).
        logit_scale: multiplier applied to cached cosine sims to match the
                     pre-softmax scale Q-Frame originally consumed
                     (defaults to SigLIP-2 giant's learned logit_scale.exp()).
        seed: optional torch RNG seed for the Gumbel noise.

    Returns:
        Sorted cached-grid indices, length min(k, T).
    """
    sims = np.asarray(sims, dtype=np.float64) * float(logit_scale)
    n = len(sims)
    if n == 0:
        return []

    logits_per_text = torch.from_numpy(sims).to(torch.float32).unsqueeze(0)
    images = torch.zeros(n)  # length-only stand-in for upstream `images`

    if seed is not None:
        torch.manual_seed(seed)
    order = _qframe_rank(logits_per_text, tau, images)

    pick = order[: min(k, n)]
    return sorted(int(x) for x in pick.tolist())


def _pad_to_budget(grid_sel: List[int], n_grid: int, budget: int) -> List[int]:
    if len(grid_sel) >= budget:
        return sorted(set(grid_sel))[:budget]
    if n_grid <= 0:
        return sorted(set(grid_sel))
    fill = np.linspace(0, n_grid - 1, budget, dtype=int).tolist()
    merged = sorted(set([int(x) for x in grid_sel] + [int(x) for x in fill]))
    return merged[:budget]


# ---------------- benchmark adapters ----------------

def adapter_lvb(args):
    """LVB: 1337 items, scores at features/lvb/siglip_2fps/{i}/."""
    feat_root = Path(args.feat_dir or
                     "/work/hdd/bcgp/michal5/WFS-SB/features/lvb/siglip_2fps")
    test_path = Path(args.dataset or
                     "/work/nvme/bcgp/michal5/longvideobench/lvb_val_std.json")
    with test_path.open() as f:
        items = json.load(f)
    sims_key = "siglip_similarities"

    def lookup(i, item):
        return feat_root / str(i) / "similarity_scores.json", sims_key

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
            "source": "qframe_siglip2",
        }

    return items, lookup, to_record


def adapter_vmme(args):
    """VMME (full 2700, video_mme_all.json), scores at features/vmme_all/siglip2_2fps/{question_id}/."""
    feat_root = Path(args.feat_dir or
                     "/work/hdd/bcgp/michal5/WFS-SB/features/vmme_all/siglip2_2fps")
    test_path = Path(args.dataset or
                     "/work/hdd/bcgp/michal5/verify_video/data/video_mme/video_mme_all.json")
    with test_path.open() as f:
        items = json.load(f)
    sims_key = "siglip2_similarities"

    def lookup(i, item):
        qid = item["question_id"]
        return feat_root / qid / "similarity_scores.json", sims_key

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
            "source": "qframe_siglip2",
        }

    return items, lookup, to_record


def adapter_m2c_v2(args):
    """m2c_v2: 999 items, scores at features/molmo2cap_v2/siglip_2fps.old_prebalance/{i}/.

    .old_prebalance is the dir aligned to test.json order (verified bit-exact
    by aks_m2c_v2_to_reanswer.py:131-138).
    """
    feat_root = Path(args.feat_dir or
                     "/work/hdd/bcgp/michal5/WFS-SB/features/molmo2cap_v2/"
                     "siglip_2fps.old_prebalance")
    test_path = Path(args.dataset or
                     "/work/hdd/bcgp/michal5/verify_video/multi_turn/"
                     "evidence_pipeline_v2/dataset_generation/group_v2/test.json")
    with test_path.open() as f:
        items = json.load(f)
    sims_key = "siglip2_similarities"

    def lookup(i, item):
        return feat_root / str(i) / "similarity_scores.json", sims_key

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
            "source": "qframe_siglip2",
        }

    return items, lookup, to_record


def adapter_m2c_v2_cap1k(args):
    """m2c_v2 caption retrieval 1k: 1000 captions used as queries against
    the full source video, scores at
    features/caption_retrieval/siglip2_2fps_group_v2/{clip_id}/.
    clip_id = video_id + start + end (with ':' replaced by '-').
    """
    feat_root = Path(args.feat_dir or
                     "/work/hdd/bcgp/michal5/WFS-SB/features/caption_retrieval/"
                     "siglip2_2fps_group_v2")
    test_path = Path(args.dataset or
                     "/work/hdd/bcgp/michal5/verify_video/multi_turn/"
                     "evidence_pipeline_v2/dataset_generation/group_v2/"
                     "test_clip_captions_dedup_1k.json")
    with test_path.open() as f:
        items = json.load(f)
    sims_key = "siglip2_similarities"

    def _clip_id(item):
        s = item["start"].replace(":", "-")
        e = item["end"].replace(":", "-")
        return f"{item['video_id']}_{s}_{e}"

    def lookup(i, item):
        return feat_root / _clip_id(item) / "similarity_scores.json", sims_key

    def to_record(i, item, frames_used, timestamps_used):
        return {
            "uid": item["uid"],
            "video_id": item["video_id"],
            "clip_id": _clip_id(item),
            "start": item["start"],
            "end": item["end"],
            "question": item["question"],
            "options": item.get("options") or {},
            "ground_truth": item.get("answer", ""),
            "frames_used": frames_used,
            "timestamps_used": timestamps_used,
            "answer": None,
            "correct": None,
            "answer_raw": None,
            "source": "qframe_siglip2",
        }

    return items, lookup, to_record


ADAPTERS = {
    "lvb": adapter_lvb,
    "vmme": adapter_vmme,
    "m2c_v2": adapter_m2c_v2,
    "m2c_v2_cap1k": adapter_m2c_v2_cap1k,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True, choices=list(ADAPTERS))
    p.add_argument("--k", type=int, required=True, help="Sample frame budget (8 or 32)")
    p.add_argument("--tau", type=float, default=0.8,
                   help="Softmax temperature; Q-Frame qwen2vl uses 0.8")
    p.add_argument("--logit-scale", type=float,
                   default=SIGLIP2_GIANT_LOGIT_SCALE,
                   help="Scale applied to cosine sims before softmax; "
                        "default = SigLIP-2 giant's logit_scale.exp() ~ 108")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the Gumbel noise (per-item seed = seed+i)")
    p.add_argument("--fps", type=float, default=2.0,
                   help="Sampling fps used during feature extraction (default 2)")
    p.add_argument("--feat-dir", default=None, help="Override feature dir")
    p.add_argument("--dataset", default=None, help="Override dataset JSON")
    p.add_argument("--output-dir", required=True,
                   help="Output reanswer source dir; chunk_0/results.json is written inside")
    args = p.parse_args()

    items, lookup, to_record = ADAPTERS[args.benchmark](args)
    print(f"Q-Frame {args.benchmark} K={args.k} tau={args.tau} "
          f"seed={args.seed} -- {len(items)} items")

    out_records = []
    success = 0
    fallback = 0
    missing = 0

    for i, item in enumerate(items):
        scores_path, sims_key = lookup(i, item)
        if not scores_path.exists():
            missing += 1
            out_records.append(to_record(i, item, [], []))
            continue
        with scores_path.open() as f:
            sc = json.load(f)
        sims = sc.get(sims_key)
        if sims is None:
            for alt in ("siglip_similarities", "siglip2_similarities"):
                if alt in sc:
                    sims = sc[alt]
                    break

        if sims is None or len(sims) == 0:
            missing += 1
            out_records.append(to_record(i, item, [], []))
            continue

        n_grid = len(sims)
        # Candidate pool is the full 2-FPS cached grid (matches all other
        # baselines in this repo). For very short videos (n_grid < k) we
        # emit n_grid unique frames; no uniform-fill fallback.
        grid_sel = qframe_select(
            np.asarray(sims, dtype=float),
            k=args.k,
            tau=args.tau,
            logit_scale=args.logit_scale,
            seed=args.seed + i,
        )
        if len(grid_sel) < args.k:
            fallback += 1
        else:
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
