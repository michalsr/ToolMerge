"""
Run VSLS pipeline on Video-MME (Long) using Qwen3-VL-8B (native).
Supports chunked execution for parallel Slurm jobs.
Saves all intermediate artifacts (grounding, score distributions, selected frames).

GPU layout:
  cuda:0 — YOLO-World (~2GB)
  cuda:1 — Qwen3-VL-8B (~16GB bf16)

Usage:
  python run_vmme_qwen3.py --chunk_id 0 --num_chunks 10
"""

import gc
import json
import os
import sys
import argparse
import pickle
import numpy as np
import torch
from PIL import Image

sys.path.append('./')
sys.path.append('./YOLO-World/')

from VSLS.interface_llm import VSLSUniversalGrounder
from VSLS.VSLSFramework import VSLSFramework, initialize_yolo


def load_vmme(json_path, video_root, chunk_id=0, num_chunks=1):
    """Load Video-MME Long and return a chunk of questions."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    items = []
    for entry in data:
        video_path = os.path.join(video_root, f"{entry['videoID']}.mp4")
        if not os.path.exists(video_path):
            print(f"WARNING: missing video {video_path}")
            continue

        opts = entry['options']
        options_str = '\n'.join(f"{k}) {v}" for k, v in sorted(opts.items()))

        items.append({
            'id': entry['question_id'],
            'video_id': entry['video_id'],
            'video_path': video_path,
            'question': entry['question'],
            'options': options_str,
            'answer': entry['answer'],
            'duration': entry.get('duration', ''),
            'domain': entry.get('domain', ''),
            'sub_category': entry.get('sub_category', ''),
            'task_type': entry.get('task_type', ''),
        })

    # Sort by question_id for deterministic chunking
    items.sort(key=lambda x: x['id'])

    # Split into chunks
    chunk_size = len(items) // num_chunks
    remainder = len(items) % num_chunks
    start = chunk_id * chunk_size + min(chunk_id, remainder)
    end = start + chunk_size + (1 if chunk_id < remainder else 0)
    chunk = items[start:end]

    print(f"Total items with videos: {len(items)}")
    print(f"Chunk {chunk_id}/{num_chunks}: items {start}-{end-1} ({len(chunk)} questions)")

    return chunk


def save_frames(frames, timestamps, output_path):
    """Save selected frames as JPEG images."""
    os.makedirs(output_path, exist_ok=True)
    frame_paths = []
    for i, (frame, ts) in enumerate(zip(frames, timestamps)):
        fname = f"frame_{i:02d}_t{ts:.1f}.jpg"
        fpath = os.path.join(output_path, fname)
        if isinstance(frame, Image.Image):
            frame.save(fpath, quality=85)
        frame_paths.append(fpath)
    return frame_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vmme_json', type=str,
                        default='/work/hdd/bcgp/michal5/verify_video/data/video_mme/video_mme_long.json')
    parser.add_argument('--video_root', type=str,
                        default='/projects/bcgp/michal5/Video-MME/videos/data')
    parser.add_argument('--yolo_device', type=str, default='cuda:0')
    parser.add_argument('--vlm_device', type=str, default='cuda:1')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--output_dir', type=str, default='./runs/vmme_qwen3')
    parser.add_argument('--config_path', type=str,
                        default='./YOLO-World/configs/pretrain/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py')
    parser.add_argument('--checkpoint_path', type=str,
                        default='./pretrained/YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth')
    parser.add_argument('--search_budget', type=float, default=1.0)
    parser.add_argument('--confidence_threshold', type=float, default=0.7)
    parser.add_argument('--search_nframes', type=int, default=8)
    parser.add_argument('--num_grounding_frames', type=int, default=16)
    parser.add_argument('--chunk_id', type=int, default=0)
    parser.add_argument('--num_chunks', type=int, default=1)
    args = parser.parse_args()

    # Per-chunk output dir
    chunk_dir = os.path.join(args.output_dir, f"chunk_{args.chunk_id:03d}")
    os.makedirs(chunk_dir, exist_ok=True)

    # Save config
    with open(os.path.join(chunk_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load questions for this chunk
    questions = load_vmme(args.vmme_json, args.video_root,
                          chunk_id=args.chunk_id, num_chunks=args.num_chunks)

    with open(os.path.join(chunk_dir, 'questions.json'), 'w') as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    # Initialize grounder
    grounder = VSLSUniversalGrounder(
        backend="qwen3vl_native",
        model_name=args.model_name,
        num_frames=args.num_grounding_frames,
        vlm_device=args.vlm_device,
    )

    # Initialize YOLO
    yolo_interface = initialize_yolo(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.yolo_device,
    )

    # Run pipeline
    results = []
    correct = 0
    total = 0

    for idx, item in enumerate(questions):
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(questions)}] {item['id']} Video: {item['video_id']}")
        print(f"  Q: {item['question'][:120]}")
        print(f"  Options: {item['options'][:120]}")
        print(f"  GT: {item['answer']}")
        sys.stdout.flush()

        item_dir = os.path.join(chunk_dir, 'items', item['id'])
        os.makedirs(item_dir, exist_ok=True)

        try:
            framework = VSLSFramework(
                grounder=grounder,
                yolo_scorer=yolo_interface,
                video_path=item['video_path'],
                question=item['question'],
                options=item['options'],
                search_nframes=args.search_nframes,
                output_dir=chunk_dir,
                confidence_threshold=args.confidence_threshold,
                search_budget=args.search_budget,
                device=args.yolo_device,
            )

            # Step 1: Grounding
            target_objs, cue_objs, relations = framework.get_grounded_objects(
                prompt_type="cot", upload_video=1
            )
            print(f"  Target objects: {target_objs}")
            print(f"  Cue objects: {cue_objs}")
            sys.stdout.flush()

            # Save grounding results
            grounding_data = {
                'target_objects': target_objs,
                'cue_objects': cue_objs,
                'relations': relations,
            }
            with open(os.path.join(item_dir, 'grounding.json'), 'w') as f:
                json.dump(grounding_data, f, indent=2)

            # Step 2: Search
            searcher = framework.set_searching_targets(target_objs, cue_objs, relations)
            all_frames, timestamps, num_iters = searcher.search_with_visualization()
            print(f"  Found {len(all_frames)} frames in {num_iters} iters at timestamps: {timestamps}")
            sys.stdout.flush()

            # Save score distribution
            if hasattr(searcher, 'score_distribution'):
                np.save(os.path.join(item_dir, 'score_distribution.npy'),
                        np.array(searcher.score_distribution))
            if hasattr(searcher, 'Score_history'):
                np.save(os.path.join(item_dir, 'score_history.npy'),
                        np.array(searcher.Score_history))

            # Save selected frames
            frame_paths = save_frames(all_frames, timestamps, os.path.join(item_dir, 'frames'))

            # Step 3: QA
            answer = grounder.inference_qa(
                frames=all_frames,
                question=item['question'],
                options=item['options'],
            )
            print(f"  Predicted: {answer}")

            pred_clean = answer.strip().upper()[:1]
            gt_clean = item['answer'].strip().upper()[:1]
            is_correct = pred_clean == gt_clean
            if is_correct:
                correct += 1
            total += 1

            item['predicted'] = answer
            item['correct'] = is_correct
            item['target_objects'] = target_objs
            item['cue_objects'] = cue_objs
            item['relations'] = relations
            item['timestamps'] = [float(t) for t in timestamps]
            item['num_search_iters'] = num_iters
            item['frame_paths'] = frame_paths
            print(f"  {'CORRECT' if is_correct else 'WRONG'} (running: {correct}/{total})")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            item['predicted'] = f"ERROR: {e}"
            item['correct'] = False
            total += 1

        results.append(item)

        # Free memory
        try:
            del framework, searcher, all_frames
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()

        # Save intermediate results
        with open(os.path.join(chunk_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        sys.stdout.flush()

    # Final summary
    print(f"\n{'='*60}")
    print(f"CHUNK {args.chunk_id} ACCURACY: {correct}/{total} = {correct/max(total,1)*100:.1f}%")
    print(f"Results saved to {chunk_dir}/results.json")


if __name__ == "__main__":
    main()
