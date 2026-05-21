# T-REN

T-REN (text-aligned region tokens) model code packaged with this repo for
self-containment. The canonical T-REN source is
<https://github.com/savya08/T-REN>; the model weights are released on the
Hugging Face Hub at <https://huggingface.co/savyak2/T-REN>.

Source: Khosla et al., *T-REN: Learning Text-Aligned Region Tokens Improves
Dense Vision-Language Alignment and Scalability* (arXiv 2604.18573, 2026).

## Weight layout

The toolmerge runtime expects these files directly under `tren/`:

```
tren/
├── best_checkpoint.pth                                       # T-REN region encoder
├── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth              # DINOv3 ViT-L/16 backbone
├── dinov3_vitl16_dinotxt_vision_head_and_text_encoder-…pth   # DINOv3 + DINO-Txt heads
└── dinov3/                                                   # local clone of facebookresearch/dinov3 (torch.hub source='local')
```

Populate them in two steps:

1. **T-REN region encoder** — pulled from the canonical HF release:

   ```bash
   ./scripts/download_tren_weights.sh
   ```

   This downloads `tren_region_encoder.pth` from `savyak2/T-REN` into
   `tren/` and creates a `best_checkpoint.pth` symlink alongside it
   (the runtime config references the latter name).

2. **DINOv3 backbone + DINO-Txt heads** — not released by T-REN. Follow the
   instructions at <https://github.com/facebookresearch/dinov3> to download
   the two `.pth` files listed above, then place them under `tren/`. Also
   clone the dinov3 repo into `tren/dinov3/` (or point `DINOV3_REPO` at an
   existing clone); `tren/model.py` uses `torch.hub.load(..., source='local')`.

## Use from Python

```python
from toolmerge.tools.tren import TrenClient
client = TrenClient(lazy_init=True)
cache = client.encode_video(frames)                       # one-time, cache to disk
scores = client.get_frame_scores(cache, "red car")        # per-query
```

## Why include a copy?

T-REN is a research codebase (no PyPI release as of this paper). Keeping a
copy in-tree makes `toolmerge` self-contained. If an upstream package is
released, `toolmerge.tools.tren.TrenClient` can be retargeted at it without
changes elsewhere in the repo.
