#!/bin/bash
#SBATCH --account=bcgp-delta-cpu
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --mem=100G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --requeue
#SBATCH --job-name=topk_gap_q_m2c_v2
#SBATCH --output=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/eval_topk_gap_question_m2c_v2_%A.out
#SBATCH --error=/work/hdd/bcgp/michal5/WFS-SB/scripts/slurm/logs/eval_topk_gap_question_m2c_v2_%A.err

set -euo pipefail
source /sw/external/python/anaconda3/bin/activate
conda activate /projects/bcgp/michal5/envs/deepspeed_video

python3 - <<'PY'
import os, json, numpy as np
from statistics import mean

FPS = 2.0
MARGIN = 0.0
test = json.load(open('/work/hdd/bcgp/michal5/verify_video/multi_turn/evidence_pipeline_v2/dataset_generation/group_v2/test.json'))
fdir = '/work/hdd/bcgp/michal5/WFS-SB/features/molmo2cap_v2/siglip_2fps'

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
for K in [2, 4, 8, 16, 32]:
    arr = []
    for i, item in enumerate(test):
        sp = os.path.join(fdir, str(i), 'similarity_scores.json')
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

print(f"TopK-SigLIP + gap, margin=0, rebalanced test")
for K in [2,4,8,16,32]:
    print(f"  K={K}: {hits[K]:.4f}")

out = {
    "dataset": "group_v2/test.json (rebalanced)",
    "total_items": len(test),
    "method": "TopK-SigLIP + greedy gap (min(dur/(K*2), 10s))",
    "margin_seconds": MARGIN,
    "hit_at_k": {str(K): hits[K] for K in [2,4,8,16,32]},
}
snapshot_dir = '/work/hdd/bcgp/michal5/verify_video/multi_turn/output/evidence_pipeline_v2/lif_prompt_m2c_v2'
os.makedirs(snapshot_dir, exist_ok=True)
with open(os.path.join(snapshot_dir, '_topk_question_rebalanced.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(f"saved to {snapshot_dir}/_topk_question_rebalanced.json")
PY
