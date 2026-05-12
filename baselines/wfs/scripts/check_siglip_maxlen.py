"""Compare WFS-style vs Ours-style SigLIP tokenization.

WFS-style  (extract.py:351)  -> processor(text=q, truncation=True, padding="max_length")
Ours-style (clip_service.py) -> processor(text=q, truncation=True, padding="max_length", max_length=64)

Runs on CPU. Does NOT load a vision/text model - only the processor/tokenizer.
Reports input_ids shape and whether the non-padding portion is identical.
"""

import json

from transformers import AutoProcessor

MODELS = [
    "google/siglip-so400m-patch14-384",   # WFS default
    "google/siglip2-giant-opt-patch16-384",  # Our default
]

TEST_JSON = "/work/hdd/bcgp/michal5/verify_video/multi_turn/evidence_pipeline_v2/dataset_generation/group_v2/test.json"


def non_pad_prefix(ids, pad_id):
    return [int(t) for t in ids if int(t) != pad_id]


def main():
    items = json.load(open(TEST_JSON))

    # Sample queries: question-only, question+options (varying lengths)
    samples = []
    for item in items[:5]:
        q = item["question"]
        opts = item.get("options", {})
        ov = " ".join(opts[k] for k in sorted(opts.keys()))
        samples.append(("q_only", q))
        samples.append(("q_plus_opts", q + " " + ov))

    for model in MODELS:
        print(f"\n=== {model} ===")
        try:
            proc = AutoProcessor.from_pretrained(model)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        tok = proc.tokenizer
        print(f"  tokenizer.model_max_length = {tok.model_max_length}")
        print(f"  pad_token_id = {tok.pad_token_id}")

        diffs_prefix = 0
        diffs_len = 0
        for i, (tag, q) in enumerate(samples):
            wfs = tok(q, return_tensors="pt", truncation=True, padding="max_length")
            ours = tok(q, return_tensors="pt", truncation=True, padding="max_length", max_length=64)

            wfs_ids = wfs["input_ids"][0].tolist()
            ours_ids = ours["input_ids"][0].tolist()

            wfs_prefix = non_pad_prefix(wfs_ids, tok.pad_token_id)
            ours_prefix = non_pad_prefix(ours_ids, tok.pad_token_id)

            same_prefix = wfs_prefix == ours_prefix
            if not same_prefix:
                diffs_prefix += 1
            if len(wfs_ids) != len(ours_ids):
                diffs_len += 1

            print(
                f"  [{i}] {tag:12s} "
                f"wfs_shape={len(wfs_ids):4d} ours_shape={len(ours_ids):4d} "
                f"nonpad wfs={len(wfs_prefix):3d} ours={len(ours_prefix):3d} "
                f"same_content={same_prefix}"
            )

        print(f"  TOTAL: {len(samples)} samples, "
              f"shape diffs={diffs_len}, content diffs={diffs_prefix}")


if __name__ == "__main__":
    main()
