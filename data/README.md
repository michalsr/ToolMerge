# Datasets

ToolMerge reads dataset JSON files directly from disk. The pipeline does not
depend on Hugging Face Datasets — point ``data.input_path`` in your config at
a local file.

Each item has the shape:

```json
{
  "uid": "abc123_0",
  "video_id": "abc123",
  "question": "What does the woman in the red dress do after picking up the book?",
  "options": {"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."},
  "answer": "C",
  "start": 152.0,   // seconds — only present for M2M (ground-truth clip interval)
  "end":   168.0    // seconds — only present for M2M
}
```

## Where to put what

By default the configs reference ``${TOOLMERGE_DATA_DIR}``. With

```
TOOLMERGE_DATA_DIR=/your/path/datasets
```

the expected layout is:

```
${TOOLMERGE_DATA_DIR}/
├── m2m/
│   ├── test.json                           # 999 items (paper test set)
│   ├── val.json                            # 997 items (human-verified val)
│   ├── captions_1k.json                    # 1000 caption + clip-interval pairs
│   ├── video_durations.json
│   └── videos/                             # 1356 source mp4s (test ∪ val ∪ captions)
├── longvideobench/
│   ├── lvb_val_std.json                    # Long Video Bench val set
│   └── videos/                             # source mp4s
└── video_mme/
    ├── video_mme_short.json
    ├── video_mme_med.json
    ├── video_mme_long.json
    └── videos/
```

## Sources

- **M2M (Molmo-2 Moments)** — released alongside this repo on Hugging Face Hub
  at [michalsr/molmo2-moments](https://huggingface.co/datasets/michalsr/molmo2-moments).
  Built from the [Molmo-2 Captioning Dataset](https://huggingface.co/datasets/allenai/molmo2-captions);
  see the paper Section 4 for the 8-step construction pipeline.

  ```bash
  huggingface-cli download michalsr/molmo2-moments --repo-type dataset \
      --local-dir $TOOLMERGE_DATA_DIR/m2m
  ```

  The HF dataset ships the JSONs **and** the 1356 source `.mp4` files under
  `videos/`. License: CC-BY-NC-SA-4.0.

- **Long Video Bench** — see
  [LongVideoBench/longvideobench](https://huggingface.co/datasets/longvideobench/LongVideoBench).
  Use the ``val_std`` split.

- **Video-MME** — see
  [lmms-lab/Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME).
  The paper uses no-subtitle mode (``video_mme_{short,med,long}.json`` formats
  match what the paper's runs consumed).

## Source videos

- **M2M videos** are redistributed alongside the dataset on HF at
  [michalsr/molmo2-moments](https://huggingface.co/datasets/michalsr/molmo2-moments)
  under `videos/<video_id>.mp4` (1356 files, ~297 GB).
- **Long Video Bench / Video-MME videos** are NOT redistributed by this repo.
  Use the original dataset download instructions linked above.

The cache build scripts in `cache_build/` consume the mp4s and produce the
SigLIP / T-REN / OCR / frame caches the inference pipeline reads.
