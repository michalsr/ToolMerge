#!/bin/bash
#SBATCH --account=bcgp-delta-cpu
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --requeue
#SBATCH --job-name=topk_gap_grounding
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/eval_topk_gap_grounding_%A.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/eval_topk_gap_grounding_%A.err

set -euo pipefail
source /sw/external/python/anaconda3/bin/activate
conda activate /projects/bcgp/michal5/envs/deepspeed_video

python3 - <<'PY'
import os, json, numpy as np
from collections import defaultdict
from statistics import mean

FPS = 2.0
MARGIN = 0.0
Ks = [2, 4, 8, 16, 32]

test = json.load(open('/work/hdd/bcgp/michal5/verify_video/multi_turn/evidence_pipeline_v2/dataset_generation/group_v2/grounding/grounding_test_retrieval.json'))
fdir = '/work/hdd/bcgp/michal5/WFS-SB/features/grounding/siglip_2fps_stmt'

def hms(t): h,m,s=t.split(':'); return int(h)*3600+int(m)*60+float(s)

def greedy_gap(fi, sims, k, gap):
    order = np.argsort(-sims)
    sel = []
    for i in order:
        if len(sel) >= k: break
        idx = int(fi[i])
        if gap <= 0 or all(abs(idx - s) >= gap for s in sel):
            sel.append(idx)
    return sorted(sel)

# Split pos / neg
by_role = {"positive": [], "negative": []}
for i, item in enumerate(test):
    role = item.get("grounding_role", "positive")
    by_role[role].append((i, item))

print(f"Rows: pos={len(by_role['positive'])} neg={len(by_role['negative'])}")
print(f"Feature dir: {fdir}")

def eval_rows(rows):
    hits = {K: [] for K in Ks}
    uniform = {K: [] for K in Ks}
    missing = 0
    for i, item in rows:
        sp = os.path.join(fdir, str(i), 'similarity_scores.json')
        if not os.path.exists(sp):
            missing += 1
            continue
        d = json.load(open(sp))
        sims = np.array(d['siglip2_similarities'])
        fi = d['frame_indices']
        if len(sims) == 0: continue
        gs = hms(item['start']) - MARGIN
        ge = hms(item['end']) + MARGIN
        n = d.get('num_frames', int(max(fi)) + 1)
        dur = n / FPS
        for K in Ks:
            gap = int(round(min(dur / (K * 2), 10.0) * FPS))
            sel = greedy_gap(fi, sims, K, gap)
            hits[K].append(any(gs <= (idx/FPS) <= ge for idx in sel))
            uni = [int((j + 0.5) * n / K) for j in range(K)]
            uniform[K].append(any(gs <= (idx/FPS) <= ge for idx in uni))
    return hits, uniform, missing

for role in ("positive", "negative"):
    hits, uni, missing = eval_rows(by_role[role])
    n = len(hits[Ks[0]])
    print()
    print(f"=== {role}  (n={n}, missing={missing}) ===")
    print(f"{'K':>4} {'SigLIP-stmt+gap':>18} {'Uniform':>10}")
    for K in Ks:
        h = mean(hits[K]) if hits[K] else 0.0
        u = mean(uni[K]) if uni[K] else 0.0
        print(f"{K:>4} {h:>18.4f} {u:>10.4f}")

# Save JSON summary
summary = {}
for role in ("positive", "negative"):
    hits, uni, missing = eval_rows(by_role[role])
    summary[role] = {
        "n": len(hits[Ks[0]]),
        "missing": missing,
        "siglip_stmt_gap": {str(K): (mean(hits[K]) if hits[K] else 0.0) for K in Ks},
        "uniform": {str(K): (mean(uni[K]) if uni[K] else 0.0) for K in Ks},
    }
out_dir = '/work/hdd/bcgp/michal5/verify_video/multi_turn/evidence_pipeline_v2/dataset_generation/group_v2/grounding/retrieval_results'
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, 'hit_at_k_siglip_stmt_gap.json')
json.dump({
    "method": "SigLIP-stmt+gap (query=bare statement, SigLIP2 giant, 2fps, greedy gap min(dur/(K*2),10s))",
    "uniform_method": "evenly-spaced K frames at int((i+0.5)*N/K)",
    "margin_seconds": 0.0,
    "results": summary,
}, open(out, 'w'), indent=2)
print(f"\nSaved {out}")
PY
