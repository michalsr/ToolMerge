"""Evaluate GPT-5.2 on VSLS-selected frames + uniform frames on LongVideoBench.

For each question:
  1. uniform_16: 16 uniformly sampled frames → GPT-5.2
  2. combined: 8 VSLS frames + 8 uniform frames (deduplicated, sorted) → GPT-5.2

Reads VSLS results from runs/lvb_qwen3*/chunk_*/results.json to get frame timestamps.
Re-extracts frames from video since VSLS didn't save the JPEGs.

Usage:
  python gpt_eval_vsls.py --start-idx 0 --end-idx 50 --output-dir ./runs/gpt_eval/chunk_0
"""

import argparse
import base64
import json
import os
import re
import sys
import time
import glob
from io import BytesIO

import numpy as np
from PIL import Image

LVB_STD_PATH = "/work/hdd/bcgp/michal5/longvideobench/lvb_val_std.json"
VIDEO_DIR = "/work/hdd/bcgp/michal5/longvideobench/videos"
GPT_KEY_PATH = "/work/hdd/bcgp/michal5/molmo2_cap/gpt52.key"
GPT_BASE_URL = "https://micha-mlt2ioil-eastus2.cognitiveservices.azure.com/openai/v1/"
GPT_MODEL = "gpt-5.2"

# Directories to search for VSLS results
VSLS_RESULT_DIRS = [
    "./runs/lvb_qwen3_t0_2fps",
]

QA_PROMPT = (
    "Based on the video frames shown, answer the following question.\n\n"
    "Question: {question}\nOptions:\n{options}\n\n"
    "Select the best answer and respond with ONLY the letter ({option_letters})."
)

MAX_RETRIES = 5
RETRY_DELAY = 5


def get_uniform_indices(n_total, n_frames):
    return np.round(np.linspace(0, n_total - 1, n_frames)).astype(int).tolist()


def frame_to_base64(frame: Image.Image, max_size=768) -> str:
    w, h = frame.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        frame = frame.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    frame.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def build_messages(question, options, frames):
    option_letters = list(options.keys())
    options_str = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    prompt_text = QA_PROMPT.format(
        question=question,
        options=options_str,
        option_letters=", ".join(option_letters),
    )
    content = []
    for frame in frames:
        b64 = frame_to_base64(frame)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        })
    content.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": content}]


def extract_answer(text):
    text = text.strip()
    if text.upper() == "IDK":
        return "IDK"
    m = re.match(r"^([A-E])\b", text.upper())
    if m:
        return m.group(1)
    m = re.search(r"(?:answer|choice|option)\s*(?:is)?\s*:?\s*([A-E])\b", text, re.I)
    if m:
        return m.group(1).upper()
    return text[:10]


def call_gpt(client, messages, temperature=0):
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                messages=messages,
                max_completion_tokens=32,
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except Exception as e:
            err_str = str(e).lower()
            if "content_filter" in err_str or "content_management" in err_str:
                print(f"    Content filter hit (attempt {attempt+1}), skipping")
                return "FILTERED"
            print(f"    GPT call failed (attempt {attempt+1}): {e}")
            if "429" in err_str or "rate" in err_str:
                delay = min(2 ** (attempt + 2), 60)
                time.sleep(delay)
            elif attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return "ERROR"


def load_vsls_results(result_dirs):
    """Load all VSLS results, keyed by uid."""
    results = {}
    for rdir in result_dirs:
        for fpath in sorted(glob.glob(os.path.join(rdir, "chunk_*/results.json"))):
            try:
                with open(fpath) as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                # Try to recover partial JSON
                with open(fpath) as f:
                    content = f.read()
                # Find last complete entry
                last_bracket = content.rfind('}')
                if last_bracket > 0:
                    content = content[:last_bracket+1] + ']'
                    try:
                        data = json.loads(content)
                    except json.JSONDecodeError:
                        print(f"  Warning: could not parse {fpath}")
                        continue
                else:
                    continue
            for entry in data:
                uid = entry.get('id', '')
                if uid and entry.get('timestamps'):
                    results[uid] = entry
    print(f"Loaded VSLS results for {len(results)} questions")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--temperature", type=float, default=0)
    args = parser.parse_args()

    from openai import OpenAI
    api_key = open(GPT_KEY_PATH).read().strip()
    client = OpenAI(api_key=api_key, base_url=GPT_BASE_URL)

    with open(LVB_STD_PATH) as f:
        dataset = json.load(f)

    end_idx = args.end_idx or len(dataset)
    subset = dataset[args.start_idx:end_idx]
    print(f"Processing {len(subset)} questions (idx {args.start_idx}-{end_idx})")

    vsls_results = load_vsls_results(VSLS_RESULT_DIRS)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "results.json")

    existing = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        print(f"Resuming from {len(existing)} existing results")
    done_uids = {r["uid"] for r in existing}

    results = list(existing)

    for i, entry in enumerate(subset):
        global_idx = args.start_idx + i
        uid = entry["uid"]
        video_id = entry["video_id"]
        question = entry["question"]
        options = entry["options"]
        gt = entry["answer"]

        if uid in done_uids:
            continue

        video_path = os.path.join(VIDEO_DIR, entry["video_path"])
        if not os.path.exists(video_path):
            print(f"  [{global_idx}] {uid}: video not found, skipping")
            continue

        print(f"  [{global_idx}] {uid}: {question[:60]}...")
        sys.stdout.flush()

        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path, num_threads=1)
        n_total = len(vr)

        # ── Condition 1: uniform 16 frames ──
        uni16_idx = get_uniform_indices(n_total, 16)
        uni16_raw = vr.get_batch(uni16_idx).asnumpy()
        uni16_frames = [Image.fromarray(f) for f in uni16_raw]

        # ── Condition 2: VSLS 8 + uniform 8 combined ──
        uni8_idx = get_uniform_indices(n_total, 8)
        vsls_entry = vsls_results.get(uid)
        if vsls_entry and vsls_entry.get('timestamps'):
            # timestamps are frame indices from VSLS
            vsls_frame_idx = [int(t) for t in vsls_entry['timestamps']]
            vsls_frame_idx = [min(idx, n_total - 1) for idx in vsls_frame_idx]
            combined_idx = sorted(set(uni8_idx + vsls_frame_idx))
        else:
            combined_idx = None
            vsls_frame_idx = None

        if combined_idx:
            combined_raw = vr.get_batch(combined_idx).asnumpy()
            combined_frames = [Image.fromarray(f) for f in combined_raw]
        else:
            combined_frames = None

        del vr

        # ── Call GPT: uniform 16 ──
        uni16_msgs = build_messages(question, options, uni16_frames)
        uni16_raw_resp = call_gpt(client, uni16_msgs, args.temperature)
        uni16_answer = extract_answer(uni16_raw_resp)

        # ── Call GPT: combined ──
        if combined_frames:
            comb_msgs = build_messages(question, options, combined_frames)
            comb_raw_resp = call_gpt(client, comb_msgs, args.temperature)
            comb_answer = extract_answer(comb_raw_resp)
        else:
            comb_raw_resp = None
            comb_answer = None

        result = {
            "idx": global_idx,
            "uid": uid,
            "video_id": video_id,
            "question": question,
            "options": options,
            "ground_truth": gt,
            "duration_group": entry.get("duration_group", ""),
            # Uniform 16
            "uni16_frames": uni16_idx,
            "uni16_raw": uni16_raw_resp,
            "uni16_answer": uni16_answer,
            "uni16_correct": uni16_answer == gt,
            # Combined (VSLS 8 + uniform 8)
            "vsls_frames": vsls_frame_idx,
            "combined_frames": combined_idx,
            "combined_n_frames": len(combined_idx) if combined_idx else None,
            "combined_raw": comb_raw_resp,
            "combined_answer": comb_answer,
            "combined_correct": comb_answer == gt if comb_answer else None,
            # VSLS Qwen baseline
            "qwen_vsls_answer": vsls_entry.get("predicted") if vsls_entry else None,
            "qwen_vsls_correct": vsls_entry.get("correct") if vsls_entry else None,
        }
        results.append(result)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=1)

        status = f"uni16={'Y' if result['uni16_correct'] else 'N'}({uni16_answer})"
        if comb_answer:
            n_comb = len(combined_idx)
            status += f"  comb({n_comb}f)={'Y' if result['combined_correct'] else 'N'}({comb_answer})"
        if vsls_entry:
            status += f"  qwen_vsls={'Y' if vsls_entry.get('correct') else 'N'}"
        print(f"    gt={gt} {status}")
        sys.stdout.flush()

    # Summary
    n = len(results)
    if n == 0:
        return
    uni16_c = sum(1 for r in results if r["uni16_correct"])
    comb_valid = [r for r in results if r["combined_correct"] is not None]
    comb_c = sum(1 for r in comb_valid if r["combined_correct"])
    qwen_valid = [r for r in results if r.get("qwen_vsls_correct") is not None]
    qwen_c = sum(1 for r in qwen_valid if r["qwen_vsls_correct"])

    print(f"\n=== Final ({n} questions) ===")
    print(f"GPT uni16:      {uni16_c}/{n} = {100*uni16_c/n:.1f}%")
    if comb_valid:
        avg_f = np.mean([r["combined_n_frames"] for r in comb_valid])
        print(f"GPT combined:   {comb_c}/{len(comb_valid)} = {100*comb_c/len(comb_valid):.1f}% (avg {avg_f:.1f} frames)")
    if qwen_valid:
        print(f"Qwen VSLS:      {qwen_c}/{len(qwen_valid)} = {100*qwen_c/len(qwen_valid):.1f}%")

    for dg in sorted(set(r.get("duration_group", 0) for r in results)):
        dg_r = [r for r in results if r.get("duration_group") == dg]
        if not dg_r:
            continue
        u = sum(1 for r in dg_r if r["uni16_correct"])
        cv = [r for r in dg_r if r["combined_correct"] is not None]
        c = sum(1 for r in cv if r["combined_correct"])
        print(f"  dg={dg:5}: uni16 {u}/{len(dg_r)}={100*u/len(dg_r):.1f}%"
              + (f"  comb {c}/{len(cv)}={100*c/len(cv):.1f}%" if cv else ""))


if __name__ == "__main__":
    main()
