"""One-shot verification of caption retrieval hit@K (margin=0) for all methods.

Methods:
  - TopK-SigLIP + gap  (reads features/caption_retrieval/siglip2_2fps_group_v2/)
  - Ours OCR-off       (reads output/retrieval_eval/caption_v7_qwen3vl_m2c_v2_no_ocr_v1/)
  - Ours OCR-on        (reads output/retrieval_eval/caption_v7_qwen3vl_m2c_v2/)

522 dedup clips. margin=0. fps=2.
"""

import json
import os

FPS = 2.0
Ks_pool = [8, 16, 32]
Ks_gap = [2, 4]
Ks = sorted(Ks_gap + Ks_pool)

DEDUP = "/work/hdd/bcgp/michal5/molmo2_cap/test_clip_captions_dedup.json"
FEAT_TOPK = "/work/hdd/bcgp/michal5/WFS-SB/features/caption_retrieval/siglip2_2fps_group_v2"
OURS_OFF = "/work/hdd/bcgp/michal5/verify_video/multi_turn/evidence_pipeline_v2/output/retrieval_eval/caption_v7_qwen3vl_m2c_v2_no_ocr_v1"
OURS_ON = "/work/hdd/bcgp/michal5/verify_video/multi_turn/evidence_pipeline_v2/output/retrieval_eval/caption_v7_qwen3vl_m2c_v2"


def ts_to_sec(s):
    h, m, rest = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def greedy_gap(idxs, sims, k, min_gap_frames):
    order = sorted(range(len(idxs)), key=lambda i: -sims[i])
    picked = []
    for oi in order:
        f = idxs[oi]
        if all(abs(f - pf) >= min_gap_frames for pf in picked):
            picked.append(f)
            if len(picked) == k:
                break
    return picked


def hit(frames, t0, t1):
    return any(t0 <= f / FPS <= t1 for f in frames)


def eval_topk_gap():
    hits = {k: 0 for k in Ks}
    data = json.load(open(DEDUP))
    N = 0
    for it in data:
        vid, s, e = it["video_id"], it["start"], it["end"]
        d = f"{vid}_{s.replace(':','-')}_{e.replace(':','-')}"
        p = os.path.join(FEAT_TOPK, d, "similarity_scores.json")
        if not os.path.exists(p):
            continue
        sc = json.load(open(p))
        sims, fidxs = sc["siglip2_similarities"], sc["frame_indices"]
        duration = (max(fidxs) + 1) / FPS
        t0, t1 = ts_to_sec(s), ts_to_sec(e)
        for k in Ks:
            gap = int(round(min(duration / (k * 2), 10.0) * FPS))
            if hit(greedy_gap(fidxs, sims, k, gap), t0, t1):
                hits[k] += 1
        N += 1
    return hits, N


def eval_ours(root):
    hits = {k: 0 for k in Ks}
    N = 0
    for d in sorted(os.listdir(root)):
        rp = os.path.join(root, d, "retrieval_results.json")
        if not os.path.exists(rp):
            continue
        items = json.load(open(rp))
        if not isinstance(items, list):
            items = items.get("results", [items])
        for it in items:
            t0, t1 = it.get("gt_start_s"), it.get("gt_end_s")
            if t0 is None or t1 is None:
                continue
            N += 1
            pools = it.get("pools") or {}
            for k in Ks_pool:
                cand = pools.get(str(k)) or pools.get(k)
                if not cand:
                    continue
                frames = [int(p[0]) for p in cand]
                if hit(frames, t0, t1):
                    hits[k] += 1
            all_scores = it.get("all_scores") or []
            if not all_scores:
                continue
            duration = (max(int(p[0]) for p in all_scores) + 1) / FPS
            for k in Ks_gap:
                gap = int(round(min(duration / (k * 2), 10.0) * FPS))
                picked = greedy_gap([int(p[0]) for p in all_scores],
                                    [float(p[1]) for p in all_scores],
                                    k, gap)
                if hit(picked, t0, t1):
                    hits[k] += 1
    return hits, N


def main():
    tk, n_tk = eval_topk_gap()
    off, n_off = eval_ours(OURS_OFF)
    on, n_on = eval_ours(OURS_ON)
    print(f"N topk={n_tk} ours_off={n_off} ours_on={n_on}")
    print(f"{'K':>4} {'TopK+gap':>10} {'Ours off':>10} {'Ours on':>10}")
    for k in Ks:
        print(f"{k:>4} {tk[k]/n_tk:>10.4f} {off[k]/n_off:>10.4f} {on[k]/n_on:>10.4f}")


if __name__ == "__main__":
    main()
