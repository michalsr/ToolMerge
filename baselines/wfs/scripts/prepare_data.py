"""Prepare benchmark annotations and video symlinks for WFS-SB.

Converts the user's local data formats to the format expected by the
WFS-SB preprocessing and pipeline code:

1. MLVU: Merges 9 per-task JSONs into one, adds video_name/question_id fields,
   embeds options in question text, creates flat video symlink directory.
2. Video-MME Long: Converts options dict→list format.
3. LongVideoBench: Already compatible — just copies to WFS-SB datasets dir.
"""

import json
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
MLVU_JSON_DIR = Path("/work/hdd/bcgp/michal5/mlvu/json")
MLVU_VIDEO_DIR = Path("/work/hdd/bcgp/michal5/mlvu/videos")
MLVU_OUT_JSON = Path("/work/hdd/bcgp/michal5/WFS-SB/datasets/mlvu/mlvu_dev_local.json")
MLVU_OUT_VIDEO = Path("/work/hdd/bcgp/michal5/WFS-SB/datasets/mlvu/video")

VMME_SRC_JSON = Path("/work/hdd/bcgp/michal5/verify_video/data/video_mme/video_mme_long.json")
VMME_OUT_JSON = Path("/work/hdd/bcgp/michal5/WFS-SB/datasets/videomme/videomme_long_local.json")

LVB_SRC_JSON = Path("/work/hdd/bcgp/michal5/longvideobench/lvb_val.json")
LVB_OUT_JSON = Path("/work/hdd/bcgp/michal5/WFS-SB/datasets/longvideobench/lvb_val_local.json")

MLVU_TASKS_MC = {
    1: "plotQA",
    2: "needle",
    3: "ego",
    4: "count",
    5: "order",
    6: "anomaly_reco",
    7: "topic_reasoning",
}


def prepare_mlvu():
    """Merge per-task MLVU JSONs and create flat video symlink directory."""
    merged = []
    qid_counter = 0

    for task_num, task_name in sorted(MLVU_TASKS_MC.items()):
        json_path = MLVU_JSON_DIR / f"{task_num}_{task_name}.json"
        items = json.load(open(json_path, "r", encoding="utf-8"))

        for item in items:
            # Embed options in question text (matching WFS-SB format)
            labels = ["A", "B", "C", "D"]
            options_text = "\n".join(
                f"({labels[i]}) {c}" for i, c in enumerate(item["candidates"])
            )
            question_with_options = item["question"] + "\n" + options_text + "\n"

            merged.append({
                "video_name": item["video"],
                "duration": item.get("duration", 0),
                "question": question_with_options,
                "candidates": item["candidates"],
                "answer": item["answer"],
                "task_type": item.get("question_type", task_name),
                "question_id": f"Q{qid_counter}",
            })
            qid_counter += 1

    # Write merged JSON
    MLVU_OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(MLVU_OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"MLVU: wrote {len(merged)} MC items to {MLVU_OUT_JSON}")

    # Create flat video symlink directory
    MLVU_OUT_VIDEO.mkdir(parents=True, exist_ok=True)
    created = 0
    for task_num, task_name in sorted(MLVU_TASKS_MC.items()):
        task_dir = MLVU_VIDEO_DIR / f"{task_num}_{task_name}"
        if not task_dir.exists():
            print(f"  WARNING: {task_dir} does not exist, skipping")
            continue
        for mp4 in task_dir.glob("*.mp4"):
            link = MLVU_OUT_VIDEO / mp4.name
            if not link.exists():
                os.symlink(mp4, link)
                created += 1
    print(f"MLVU: created {created} video symlinks in {MLVU_OUT_VIDEO}")


def prepare_videomme():
    """Convert Video-MME Long options from dict to list format."""
    data = json.load(open(VMME_SRC_JSON, "r", encoding="utf-8"))
    converted = []

    for item in data:
        item_out = dict(item)
        # Convert options dict {"A": "...", "B": "..."} → list ["A. ...", "B. ..."]
        if isinstance(item["options"], dict):
            opts_list = [f"{k}. {v}" for k, v in sorted(item["options"].items())]
            item_out["options"] = opts_list
        converted.append(item_out)

    VMME_OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(VMME_OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    print(f"Video-MME Long: wrote {len(converted)} items to {VMME_OUT_JSON}")


def prepare_lvb():
    """Copy LVB JSON (already compatible)."""
    import shutil
    LVB_OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LVB_SRC_JSON, LVB_OUT_JSON)
    data = json.load(open(LVB_SRC_JSON))
    print(f"LongVideoBench: copied {len(data)} items to {LVB_OUT_JSON}")


if __name__ == "__main__":
    prepare_mlvu()
    prepare_videomme()
    prepare_lvb()
    print("\nDone. All benchmark data prepared.")
