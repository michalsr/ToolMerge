# T-REN

Vendored T-REN (text-aligned region tokens) package — code only. The model
weights (~4.6 GB total) live on the Hugging Face Hub and are pulled by
``scripts/download_tren_weights.sh``.

Source: Khosla et al., *T-REN: Learning Text-Aligned Region Tokens Improves
Dense Vision-Language Alignment and Scalability* (arXiv 2604.18573, 2026).

## Layout

```
tren/
├── model.py               # FeatureExtractor / RegionEncoder / TextEncoder
├── task_utils.py          # CenterPadding, upsample_features
├── video_query_search/
│   ├── models.py          # QuerySearch (wraps the three encoders)
│   └── config.yaml        # default architecture + similarity threshold
├── configs/
│   └── train_dinov3_vit16.yaml
└── weights/               # populated by scripts/download_tren_weights.sh
    ├── best_checkpoint.pth                                 (1.4 GB)
    ├── dinov3_vitl16_dinotxt_vision_head_and_text_encoder-…pth (2.1 GB)
    └── dinov3_vitl16_pretrain_lvd1689m-…pth                (1.2 GB)
```

## Use from Python

```python
from toolmerge.tools.tren import TrenClient
client = TrenClient(lazy_init=True)
cache = client.encode_video(frames)                       # one-time, cache to disk
scores = client.get_frame_scores(cache, "red car")        # per-query
```

## Why not pip-installable?

T-REN is a research codebase (no PyPI release as of the paper). It ships
with the repo for self-containment. If an upstream package is released
later, ``toolmerge.tools.tren.TrenClient`` can be retargeted at it without
changes to the rest of the repo.
