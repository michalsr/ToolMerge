"""Recompute caption retrieval hit@K with consistent margin for both OCR-on and OCR-off runs.

Reads retrieval_results.json chunks from both dirs, uses `pools[K]` for K in {8,16,32}
and computes hit@K at margin=0 and margin=5s.

Each item has:
- gt_start_s, gt_end_s (seconds)
- pools: {str(K): [[frame_idx, score], ...]} — already pooled K frames
- all_scores: [[frame_idx, score], ...] for K=2,4 greedy gap
"""

import json
import os

FPS = 2.0
Ks_pool = [8, 16, 32]
Ks_gap = [2, 4]

RUNS = {
    "OCR_OFF": "/work/hdd/bcgp/michal5/verify_video/multi_turn/evidence_pipeline_v2/output/retrieval_eval/caption_v7_qwen3vl_m2c_v2_no_ocr_v1",
    "OCR_ON":  "/work/hdd/bcgp/michal5/verify_video/multi_turn/evidence_pipeline_v2/output/retrieval_eval/caption_v7_qwen3vl_m2c_v2",
}


def greedy_gap_select(scored, k, min_gap_frames):
    order = sorted(scored, key=lambda x: -x[1])
    picked = []
    for idx, _s in order:
        if all(abs(idx - p) >= min_gap_frames for p in picked):
            picked.append(idx)
            if len(picked) == k:
                break
    return picked


def hit(frames, t0, t1, margin):
    times = [f / FPS for f in frames]
    return any((t0 - margin) <= t <= (t1 + margin) for t in times)


def evaluate(root):
    hits = {m: {k: 0 for k in Ks_pool + Ks_gap} for m in [0, 5]}
    miss_pool = {k: 0 for k in Ks_pool}
    miss_gap = {k: 0 for k in Ks_gap}
    N = 0
    for d in sorted(os.listdir(root)):
        rp = os.path.join(root, d, "retrieval_results.json")
        if not os.path.exists(rp):
            continue
        data = json.load(open(rp))
        items = data if isinstance(data, list) else data.get("results", [data])
        for it in items:
            t0 = it.get("gt_start_s")
            t1 = it.get("gt_end_s")
            if t0 is None or t1 is None:
                continue
            N += 1
            pools = it.get("pools") or {}
            for k in Ks_pool:
                cand = pools.get(str(k)) or pools.get(k)
                if not cand:
                    miss_pool[k] += 1
                    continue
                frames = [int(p[0]) for p in cand]
                for m in (0, 5):
                    if hit(frames, t0, t1, m):
                        hits[m][k] += 1
            all_scores = it.get("all_scores") or []
            if not all_scores:
                for k in Ks_gap:
                    miss_gap[k] += 1
                continue
            max_idx = max(int(p[0]) for p in all_scores)
            duration = (max_idx + 1) / FPS
            for k in Ks_gap:
                gap_s = min(duration / (k * 2), 10.0)
                gap_frames = int(round(gap_s * FPS))
                picked = greedy_gap_select(all_scores, k, gap_frames)
                for m in (0, 5):
                    if hit(picked, t0, t1, m):
                        hits[m][k] += 1
    return hits, miss_pool, miss_gap, N


def main():
    results = {tag: evaluate(root) for tag, root in RUNS.items()}

    print(f"{'K':>4} {'OFF@0':>8} {'OFF@5':>8} {'ON@0':>8} {'ON@5':>8} {'Δ@0':>8} {'Δ@5':>8}")
    for k in sorted(Ks_gap + Ks_pool):
        off_h, _, _, off_N = results["OCR_OFF"]
        on_h, _, _, on_N = results["OCR_ON"]
        off0 = off_h[0][k] / off_N
        off5 = off_h[5][k] / off_N
        on0 = on_h[0][k] / on_N
        on5 = on_h[5][k] / on_N
        print(f"{k:>4} {off0:>8.4f} {off5:>8.4f} {on0:>8.4f} {on5:>8.4f} "
              f"{on0-off0:+8.4f} {on5-off5:+8.4f}")
    print()
    print("OFF N:", results["OCR_OFF"][3], " ON N:", results["OCR_ON"][3])


if __name__ == "__main__":
    main()
