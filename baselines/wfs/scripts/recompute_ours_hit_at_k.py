"""Recompute 'Ours' hit@K for m2c_v2 question retrieval from saved results.

Reads all chunk_*/rem*_chunk_*/results.json, uses:
- pooled_candidates_{8,16,32,64} for K in {8,16,32,64}
- all_scores + greedy gap select for K in {2,4}

GT segment from test.json start/end (HH:MM:SS.mmm). Margin = 0.
Compares against _hit_at_k_snapshot.json's 'ours' row.
"""

import glob
import json
import os

ROOT = "/work/hdd/bcgp/michal5/verify_video/multi_turn/output/evidence_pipeline_v2/m2c_v2_test_v7_t0_tmp_pool"
TEST = "/work/hdd/bcgp/michal5/verify_video/multi_turn/evidence_pipeline_v2/dataset_generation/group_v2/test.json"
SNAP = "/work/hdd/bcgp/michal5/verify_video/multi_turn/output/evidence_pipeline_v2/lif_prompt_m2c_v2/_hit_at_k_snapshot.json"
FPS = 2.0


def ts_to_sec(s):
    h, m, rest = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def greedy_gap_select(scored_frames, k, min_gap_frames):
    """scored_frames: list of [idx, score]. Already sorted desc by score (all_scores is idx-ascending, so resort)."""
    order = sorted(scored_frames, key=lambda x: -x[1])
    picked = []
    for idx, _score in order:
        if all(abs(idx - p) >= min_gap_frames for p in picked):
            picked.append(idx)
            if len(picked) == k:
                break
    return picked


def main():
    test_data = json.load(open(TEST))
    uid_to_item = {it["uid"]: it for it in test_data}

    # Collect all results keyed by uid
    by_uid = {}
    for rp in sorted(glob.glob(os.path.join(ROOT, "*", "results.json"))):
        d = json.load(open(rp))
        items = d if isinstance(d, list) else d.get("results", [d])
        for it in items:
            uid = it.get("uid")
            if uid and uid not in by_uid:
                by_uid[uid] = it

    print(f"unique items: {len(by_uid)} / test: {len(test_data)}")

    Ks_pool = [8, 16, 32, 64]
    Ks_gap = [2, 4]
    hits = {k: 0 for k in Ks_pool + Ks_gap}
    missing = {k: 0 for k in Ks_pool + Ks_gap}
    N = 0

    for uid, item in uid_to_item.items():
        res = by_uid.get(uid)
        if res is None:
            # count as miss for all Ks (same policy as snapshot)
            for k in Ks_pool + Ks_gap:
                missing[k] += 1
            N += 1
            continue
        g = res.get("trace", {}).get("gatherer", {}) or {}
        t0 = ts_to_sec(item["start"])
        t1 = ts_to_sec(item["end"])

        # K in {8,16,32,64} from pooled_candidates
        for k in Ks_pool:
            cand = g.get(f"pooled_candidates_{k}")
            if not cand:
                missing[k] += 1
                continue
            frames = [int(p[0]) for p in cand]
            times = [f / FPS for f in frames]
            if any(t0 <= t <= t1 for t in times):
                hits[k] += 1

        # K in {2,4} via greedy gap on all_scores
        all_scores = g.get("all_scores") or []
        if not all_scores:
            for k in Ks_gap:
                missing[k] += 1
        else:
            max_idx = max(int(p[0]) for p in all_scores)
            duration = (max_idx + 1) / FPS
            for k in Ks_gap:
                min_gap_sec = min(duration / (k * 2), 10.0)
                min_gap_frames = int(round(min_gap_sec * FPS))
                picked = greedy_gap_select(all_scores, k, min_gap_frames)
                times = [f / FPS for f in picked]
                if any(t0 <= t <= t1 for t in times):
                    hits[k] += 1

        N += 1

    snap = json.load(open(SNAP))
    ours_snap = snap["hit_at_k"]["ours"]

    print(f"\nN = {N}")
    print(f"{'K':>4} {'hits':>6} {'miss':>6} {'recompute':>10} {'snapshot':>10} {'delta':>8}")
    for k in Ks_gap + Ks_pool:
        recompute = hits[k] / N
        s = ours_snap.get(str(k))
        delta = recompute - s if s is not None else None
        delta_str = f"{delta:+.4f}" if delta is not None else "n/a"
        print(f"{k:>4} {hits[k]:>6} {missing[k]:>6} {recompute:>10.4f} "
              f"{s if s is not None else 'n/a':>10} {delta_str:>8}")


if __name__ == "__main__":
    main()
