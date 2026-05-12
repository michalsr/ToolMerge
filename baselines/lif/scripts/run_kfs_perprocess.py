"""
OOM-safe keyframe search: spawns a NEW process per video.

Each subprocess loads the YOLO model, processes one video, then exits.
The OS reclaims all RAM between videos, preventing OOM from memory leaks.
Resumes from where it left off by checking existing results.
"""

import argparse
import json
import os
import subprocess
import sys
import time


def parse_arguments():
    parser = argparse.ArgumentParser(description="Per-process keyframe search wrapper")
    parser.add_argument('--obj_path', type=str, required=True)
    parser.add_argument('--kfs_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--search_nframes', type=int, default=8)
    parser.add_argument('--grid_rows', type=int, default=4)
    parser.add_argument('--grid_cols', type=int, default=4)
    parser.add_argument('--confidence_threshold', type=float, default=0.7)
    parser.add_argument('--search_budget', type=float, default=0.5)
    parser.add_argument('--update_method', type=str, default='spline')
    parser.add_argument('--relation_alpha', type=float, default=0.3)
    return parser.parse_args()


FILTER_TASK_TYPES = ['OCR Problems', 'Counting Problem', 'Temporal Perception',
                     'Information Synopsis', 'Temporal Reasoning']


def main():
    args = parse_arguments()

    with open(args.obj_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    os.makedirs(os.path.dirname(args.kfs_path), exist_ok=True)

    # Load existing results to resume
    results = []
    done_ids = set()
    if os.path.exists(args.kfs_path):
        with open(args.kfs_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            done_ids = {r.get('video_id', r.get('video_path', '')) for r in results
                        if 'error' not in r}
        print(f"Resuming: {len(done_ids)} videos already processed")

    total = len(dataset)
    for idx, data_item in enumerate(dataset):
        task_type = data_item.get('task_type', ' ')
        if task_type in FILTER_TASK_TYPES:
            continue

        vid = data_item.get('video_id', data_item.get('video_path', ''))
        if vid in done_ids:
            print(f"[{idx+1}/{total}] Skipping {vid} (already done)")
            continue

        print(f"[{idx+1}/{total}] Processing {vid}...")
        t0 = time.time()

        # Write this single item to a temp file
        tmp_input = args.kfs_path + f'.tmp_input_{idx}.json'
        tmp_output = args.kfs_path + f'.tmp_output_{idx}.json'
        with open(tmp_input, 'w') as f:
            json.dump([data_item], f)

        # Spawn a subprocess for this one video
        cmd = [
            sys.executable, 'scripts/get_VSLS_key_frames.py',
            '--obj_path', tmp_input,
            '--kfs_path', tmp_output,
            '--config_path', args.config_path,
            '--checkpoint_path', args.checkpoint_path,
            '--device', args.device,
            '--search_nframes', str(args.search_nframes),
            '--grid_rows', str(args.grid_rows),
            '--grid_cols', str(args.grid_cols),
            '--confidence_threshold', str(args.confidence_threshold),
            '--search_budget', str(args.search_budget),
            '--update_method', args.update_method,
            '--relation_alpha', str(args.relation_alpha),
            '--save_batch', '1',
            '--num', '1',
            '--prompt_type', 'cot',
        ]

        try:
            proc = subprocess.run(cmd, timeout=600, capture_output=True, text=True)
            elapsed = time.time() - t0

            if proc.returncode == 0 and os.path.exists(tmp_output):
                with open(tmp_output, 'r') as f:
                    video_results = json.load(f)
                if video_results:
                    results.append(video_results[0])
                    done_ids.add(vid)
                    print(f"  OK ({elapsed:.1f}s)")
            else:
                error_msg = proc.stderr[-500:] if proc.stderr else "Unknown error"
                print(f"  FAILED ({elapsed:.1f}s): {error_msg}")
                result = dict(data_item)
                result['error'] = error_msg
                results.append(result)

        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT (>600s)")
            result = dict(data_item)
            result['error'] = "Timeout (>600s)"
            results.append(result)

        # Clean up temp files
        for tmp in [tmp_input, tmp_output]:
            if os.path.exists(tmp):
                os.remove(tmp)

        # Save after each video
        with open(args.kfs_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nDone. {len(done_ids)} videos processed. Results: {args.kfs_path}")


if __name__ == '__main__':
    main()
