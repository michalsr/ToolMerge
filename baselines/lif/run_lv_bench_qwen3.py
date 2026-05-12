"""
Run VSLS pipeline on LV-Bench questions using Qwen3-VL-8B (native, no vLLM).

GPU layout:
  cuda:0 — YOLO-World (~2GB)
  cuda:1 — Qwen3-VL-8B (~16GB bf16)

Usage:
  python run_lv_bench_qwen3.py --yolo_device cuda:0 --vlm_device cuda:1
"""

import gc
import json
import os
import sys
import argparse
import numpy as np
import torch

sys.path.append('./')
sys.path.append('./YOLO-World/')

from VSLS.interface_llm import VSLSUniversalGrounder
from VSLS.VSLSFramework import VSLSFramework, initialize_yolo


def load_lv_bench(json_path, video_root, n=10, seed=42):
    """Load LV-Bench and pick n diverse questions."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Flatten all QA pairs with video paths
    all_items = []
    for video_entry in data:
        video_id = video_entry['key']
        video_path = os.path.join(video_root, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            continue
        for qa in video_entry['qa']:
            q_text = qa['question']
            # Split on the first option marker to separate question from options
            parts = q_text.split('\n(A)')
            if len(parts) == 2:
                question = parts[0].strip()
                options_raw = '(A)' + parts[1]
                options = (options_raw
                    .replace('(A)', 'A)')
                    .replace('(B)', 'B)')
                    .replace('(C)', 'C)')
                    .replace('(D)', 'D)')
                    .replace('(E)', 'E)'))
            else:
                question = q_text
                options = ""

            all_items.append({
                'video_id': video_id,
                'video_path': video_path,
                'uid': qa['uid'],
                'question': question,
                'options': options,
                'answer': qa['answer'],
                'question_type': qa.get('question_type', []),
                'time_reference': qa.get('time_reference', ''),
            })

    # Pick n items — diverse across question types and videos
    rng = np.random.RandomState(seed)
    rng.shuffle(all_items)

    by_type = {}
    for item in all_items:
        qt = item['question_type'][0] if item['question_type'] else 'other'
        by_type.setdefault(qt, []).append(item)

    selected = []
    type_iters = {k: iter(v) for k, v in by_type.items()}
    while len(selected) < n and type_iters:
        for qt in list(type_iters.keys()):
            if len(selected) >= n:
                break
            try:
                item = next(type_iters[qt])
                if len(selected) < n // 2 or item['video_id'] not in [s['video_id'] for s in selected]:
                    selected.append(item)
            except StopIteration:
                del type_iters[qt]

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lv_bench_json', type=str,
                        default='/work/hdd/bcgp/michal5/verify_video/data/lv_bench_original.json')
    parser.add_argument('--video_root', type=str,
                        default='/projects/bcgp/michal5/lv_bench_videos')
    parser.add_argument('--yolo_device', type=str, default='cuda:0')
    parser.add_argument('--vlm_device', type=str, default='cuda:1')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--num_questions', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='./runs/lv_bench_qwen3')
    parser.add_argument('--config_path', type=str,
                        default='./YOLO-World/configs/pretrain/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py')
    parser.add_argument('--checkpoint_path', type=str,
                        default='./pretrained/YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth')
    parser.add_argument('--search_budget', type=float, default=0.5)
    parser.add_argument('--confidence_threshold', type=float, default=0.7)
    parser.add_argument('--search_nframes', type=int, default=8)
    parser.add_argument('--num_grounding_frames', type=int, default=16,
                        help='Frames sent to VLM for grounding (reduce if OOM)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load questions
    questions = load_lv_bench(args.lv_bench_json, args.video_root, n=args.num_questions)
    print(f"\nSelected {len(questions)} questions:")
    for i, q in enumerate(questions):
        print(f"  {i+1}. [{q['question_type']}] {q['video_id']}: {q['question'][:80]}...")

    with open(os.path.join(args.output_dir, 'selected_questions.json'), 'w') as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    # Initialize grounder (Qwen3-VL native on vlm_device)
    grounder = VSLSUniversalGrounder(
        backend="qwen3vl_native",
        model_name=args.model_name,
        num_frames=args.num_grounding_frames,
        vlm_device=args.vlm_device,
    )

    # Initialize YOLO on yolo_device
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
        print(f"[{idx+1}/{len(questions)}] Video: {item['video_id']}")
        print(f"  Q: {item['question'][:100]}")
        print(f"  Options: {item['options'][:100]}")
        print(f"  GT: {item['answer']}")

        try:
            framework = VSLSFramework(
                grounder=grounder,
                yolo_scorer=yolo_interface,
                video_path=item['video_path'],
                question=item['question'],
                options=item['options'],
                search_nframes=args.search_nframes,
                output_dir=args.output_dir,
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

            # Step 2: Search
            searcher = framework.set_searching_targets(target_objs, cue_objs, relations)
            all_frames, timestamps, num_iters = searcher.search_with_visualization()
            print(f"  Found {len(all_frames)} frames at timestamps: {timestamps}")

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
            item['timestamps'] = [float(t) for t in timestamps]
            print(f"  {'CORRECT' if is_correct else 'WRONG'} (running: {correct}/{total})")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            item['predicted'] = f"ERROR: {e}"
            item['correct'] = False
            total += 1

        results.append(item)

        # Free memory between questions to avoid CPU OOM
        del framework
        try:
            del searcher, all_frames
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()

        # Save intermediate results
        with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL ACCURACY: {correct}/{total} = {correct/max(total,1)*100:.1f}%")
    print(f"Results saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
