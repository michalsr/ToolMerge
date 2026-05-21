# Paper-row keyframes

This directory ships the per-question keyframe selections used for the paper's
LVB / Video-MME / Molmo-2 Moments rows. Each `<row>/keyframes.json` is the
canonical "what frames did our pipeline pick" output for one paper cell, with
the full debug trace stripped — just the fields the toolmerge answerer needs
to reanswer.

These files exist so an external user can reproduce the paper's accuracy
numbers in **answerer-only mode** (no GPU planner / tools needed). Point
`data.source_dir` at any of the row directories and the toolmerge runtime
will read `frames_used` per question and run only the answerer VLM:

```bash
toolmerge config=configs/lvb/qwen3_32.yaml \
    data.source_dir=paper_keyframes/lvb_qwen3_32 \
    data.save_path=outputs/lvb_qwen3_32_reanswer
```

## Per-row contents

| Row | Items | Reported acc | Source |
|---|---:|---:|---|
| `lvb_qwen3_8`  | 1337 | 61.78% | `reanswer_lvb_8f_qwen` |
| `lvb_qwen3_32` | 1337 | 67.39% | `reanswer_lvb_32f_qwen_lif` |
| `lvb_gpt4o_8`  | 1337 | 61.33% | `reanswer_lvb_8f_gpt4o_lif` |
| `lvb_gpt4o_32` | 1337 | 65.37% | `reanswer_lvb_32f_gpt4o_lif` |
| `vmme_qwen3_8`  | 2694 | 64.74% | short+med+long triplet (`ra_vmme_*_8f_q3`) |
| `vmme_qwen3_32` | 2694 | 70.71% | short+med+long triplet (`ra_vmme_*_32f_q3`) |
| `vmme_gpt4o_8`  | 2694 | 71.20% | short+med+long triplet (`ra_vmme_*_8f_gpt4o` + `reanswer_vmme_8f_gpt4o_lif` for long) |
| `vmme_gpt4o_32` | 2694 | 73.42% | short+med+long triplet (same pattern) |
| `m2m_qwen3_8`   |  999 | 57.76% | direct pipeline (`v7_t0_tmp_pool`, deduped from 1614 → 999) |
| `m2m_qwen3_32`  |  994 | 58.15% | `ra_m2c_v2_32f_q3` |
| `m2m_gpt4o_8`   |  994 | 55.53% | `ra_m2c_v2_8f_gpt4o` |
| `m2m_gpt4o_32`  |  994 | 56.44% | `ra_m2c_v2_32f_gpt4o` |

Video-MME triplets concatenate three 897/898/899-item runs (short, medium,
long); their union is the full 2700-question Video-MME minus 6 dropped items
(`pred=None` in the source planner pass).

## Schema

Each `keyframes.json` is a list of items with:

```json
{
  "uid": "<id>",
  "video_id": "...",
  "question": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "ground_truth": "C",
  "frames_used": [123, 456, ...],         // selected native frame indices
  "timestamps_used": [0.5, 1.0, ...],     // seconds (target_fps grid)
  "answer": "C",                          // the answerer's prediction for that row
  "correct": true
}
```

Regenerated from the research-tree dirs by
`scripts/analysis/stage_paper_keyframes.py`; the picks live in
`outputs/hf_keyframes_picks.json`.
