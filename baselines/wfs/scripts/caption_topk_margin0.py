"""TopK-SigLIP (caption) at margin=0 — hit@K for 522 dedup clips.

Reads features/caption_retrieval/siglip2_2fps_group_v2/{clip_id}/similarity_scores.json
and dedup mapping from test_clip_captions_dedup.json.

Clip dir format: {video_id}_{HH-MM-SS.mmm_start}_{HH-MM-SS.mmm_end}
"""

import json
import os

FEAT = "/work/hdd/bcgp/michal5/WFS-SB/features/caption_retrieval/siglip2_2fps_group_v2"
DEDUP = "/work/hdd/bcgp/michal5/molmo2_cap/test_clip_captions_dedup.json"
FPS = 2.0
Ks = [2, 4, 8, 16, 32]


def ts_to_sec(s):
    h, m, rest = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def clip_dir(vid, s, e):
    return f"{vid}_{s.replace(':','-')}_{e.replace(':','-')}"


def greedy_gap_select(idxs, sims, k, min_gap_frames):
    order = sorted(range(len(idxs)), key=lambda i: -sims[i])
    picked = []
    for oi in order:
        f = idxs[oi]
        if all(abs(f - pf) >= min_gap_frames for pf in picked):
            picked.append(f)
            if len(picked) == k:
                break
    return picked


def main():
    data = json.load(open(DEDUP))
    print("items:", len(data))

    hits_plain = {k: 0 for k in Ks}
    hits_gap = {k: 0 for k in Ks}
    miss = 0
    N = 0
    for it in data:
        vid = it["video_id"]
        s = it["start"]
        e = it["end"]
        cd = os.path.join(FEAT, clip_dir(vid, s, e))
        sp = os.path.join(cd, "similarity_scores.json")
        if not os.path.exists(sp):
            miss += 1
            continue
        sc = json.load(open(sp))
        sims = sc["siglip2_similarities"]
        fidxs = sc["frame_indices"]
        n = len(sims)
        t0 = ts_to_sec(s)
        t1 = ts_to_sec(e)
        duration = (max(fidxs) + 1) / FPS

        for K in Ks:
            top = sorted(range(n), key=lambda i: -sims[i])[:K]
            frames = [fidxs[i] for i in top]
            times = [f / FPS for f in frames]
            if any(t0 <= t <= t1 for t in times):
                hits_plain[K] += 1

            gap_s = min(duration / (K * 2), 10.0)
            gap_f = int(round(gap_s * FPS))
            picked = greedy_gap_select(fidxs, sims, K, gap_f)
            ptimes = [f / FPS for f in picked]
            if any(t0 <= t <= t1 for t in ptimes):
                hits_gap[K] += 1
        N += 1

    print(f"missing: {miss}  N: {N}")
    print(f"{'K':>4} {'plain':>8} {'+gap':>8}")
    for k in Ks:
        print(f"{k:>4} {hits_plain[k]/N:>8.4f} {hits_gap[k]/N:>8.4f}")


if __name__ == "__main__":
    main()
