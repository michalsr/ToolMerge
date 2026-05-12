#!/bin/bash
#SBATCH --account=bcgp-delta-cpu
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --requeue
#SBATCH --job-name=topk_gap_q_m2c_v2_hard250
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/eval_topk_gap_question_m2c_v2_hard250_%A.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/eval_topk_gap_question_m2c_v2_hard250_%A.err

set -euo pipefail
source /sw/external/python/anaconda3/bin/activate
conda activate /projects/bcgp/michal5/envs/deepspeed_video

python3 - <<'PY'
import os, json, numpy as np
from statistics import mean

FPS = 2.0
MARGIN = 0.0
DATA = '/work/hdd/bcgp/michal5/verify_video/multi_turn/evidence_pipeline_v2/dataset_generation/group_v2/hard_gpt52_wrong_250.json'
FDIR = '/work/hdd/bcgp/michal5/WFS-SB/features/molmo2cap_v2/siglip_2fps_hard250'
OUT  = '/work/hdd/bcgp/michal5/verify_video/multi_turn/output/evidence_pipeline_v2/lif_prompt_m2c_v2/_topk_question_hard250.json'

test = json.load(open(DATA))

def hms(t): h,m,s=t.split(':'); return int(h)*3600+int(m)*60+float(s)

def greedy_gap_select(frame_idxs, sims, k, gap):
    order = np.argsort(-sims)
    sel = []
    for i in order:
        if len(sel) >= k: break
        idx = int(frame_idxs[i])
        if gap <= 0 or all(abs(idx - s) >= gap for s in sel):
            sel.append(idx)
    return sorted(sel)

hits = {}
counts = {}
for K in [2, 4, 8, 16, 32]:
    arr = []
    for i, item in enumerate(test):
        sp = os.path.join(FDIR, str(i), 'similarity_scores.json')
        if not os.path.exists(sp): continue
        d = json.load(open(sp))
        sims = np.array(d['siglip2_similarities'])
        fi = d['frame_indices']
        if len(sims) == 0: continue
        gs = hms(item['start']) - MARGIN
        ge = hms(item['end']) + MARGIN
        dur = len(sims) / FPS
        gap_s = min(dur / (K * 2), 10.0)
        sel = greedy_gap_select(fi, sims, K, int(gap_s * FPS))
        arr.append(any(gs <= (idx/FPS) <= ge for idx in sel))
    hits[K] = mean(arr) if arr else 0.0
    counts[K] = len(arr)

print(f"TopK-SigLIP + gap on hard_gpt52_wrong_250")
for K in [2,4,8,16,32]:
    print(f"  K={K}: {hits[K]:.4f}  (n={counts[K]})")

out = {
    "dataset": "group_v2/hard_gpt52_wrong_250.json",
    "total_items": len(test),
    "method": "TopK-SigLIP + greedy gap (min(dur/(K*2), 10s))",
    "margin_seconds": MARGIN,
    "hit_at_k": {str(K): hits[K] for K in [2,4,8,16,32]},
    "counts": {str(K): counts[K] for K in [2,4,8,16,32]},
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(out, f, indent=2)
print(f"saved to {OUT}")
PY
