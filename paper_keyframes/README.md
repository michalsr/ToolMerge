# Paper-row keyframes

This directory contains the per-question keyframe selections used for the paper's
LVB / Video-MME / Molmo-2 Moments rows. Each `<row>/keyframes.json` is the
canonical "what frames did our pipeline pick" output for one paper cell, with
the full debug trace stripped — just the fields the toolmerge answerer needs
to reanswer.

These files exist so an external user can reproduce the paper's accuracy
numbers in **answerer-only mode** . Point
`data.source_dir` at any of the row directories and the toolmerge runtime
will read `frames_used` per question and run only the answerer VLM:

```bash
toolmerge config=configs/lvb/qwen3_32.yaml \
    data.source_dir=paper_keyframes/lvb_qwen3_32 \
    data.save_path=outputs/lvb_qwen3_32_reanswer
```

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
