"""SigLIP-2 client.

Encodes text queries and compares them against precomputed frame features
(loaded from `${TOOLMERGE_CACHE_DIR}/siglip/`). Model defaults to
``google/siglip2-giant-opt-patch16-384`` per the paper; override via
``SIGLIP_MODEL`` env var.
"""

from __future__ import annotations

import logging
import os
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "google/siglip2-giant-opt-patch16-384"

# Process-wide model cache so DataLoader workers can lazy-init.
_model_cache = None
_processor_cache = None
_device_cache: str | None = None


def tensor_to_pil(frame_tensor: torch.Tensor) -> Image.Image:
    """(C, H, W) uint8/float-in-[0, 255] -> PIL.Image(RGB)."""
    arr = frame_tensor.clamp(0, 255).byte().cpu().numpy()
    if arr.shape[0] == 3:  # CHW -> HWC
        arr = arr.transpose(1, 2, 0)
    return Image.fromarray(arr, mode="RGB")


class SiglipClient:
    """SigLIP-2 client using `transformers` directly (no separate service)."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        lazy_init: bool = False,
    ):
        global _model_cache, _processor_cache, _device_cache

        if model_name is None:
            model_name = os.environ.get("SIGLIP_MODEL", _DEFAULT_MODEL)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.device = device

        if lazy_init:
            self.model = None
            self.processor = None
        else:
            self.ensure_loaded()

    def ensure_loaded(self):
        global _model_cache, _processor_cache, _device_cache

        if getattr(self, "model", None) is not None and getattr(self, "processor", None) is not None:
            return

        if _model_cache is not None and _processor_cache is not None:
            self.model = _model_cache
            self.processor = _processor_cache
            if _device_cache is not None and self.device is None:
                self.device = _device_cache
            return

        logger.info("Loading SigLIP-2 model %s on %s", self.model_name, self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        attn_impl = "sdpa" if self.device.startswith("cuda") else "eager"
        self.model = AutoModel.from_pretrained(self.model_name, attn_implementation=attn_impl)
        self.model.eval().to(self.device)

        _model_cache = self.model
        _processor_cache = self.processor
        _device_cache = self.device

    def encode_texts(
        self,
        prompts: Union[str, List[str]],
        batch_size: int = 256,
    ) -> torch.Tensor:
        """Returns (N, D) normalized text features."""
        self.ensure_loaded()
        if isinstance(prompts, str):
            prompts = [prompts]
        feats: List[torch.Tensor] = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=64,
            )
            input_ids = inputs["input_ids"].to(self.device)
            with torch.no_grad():
                f = self.model.get_text_features(input_ids=input_ids)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
        return torch.cat(feats, dim=0)

    def encode_images(
        self,
        frames: Union[List[Image.Image], List[torch.Tensor], torch.Tensor],
        batch_size: int = 256,
    ) -> torch.Tensor:
        """Returns (T, D) normalized image features."""
        self.ensure_loaded()

        if isinstance(frames, torch.Tensor):
            frames = [tensor_to_pil(f) for f in frames]
        elif isinstance(frames[0], np.ndarray):
            frames = [
                Image.fromarray(f.transpose(1, 2, 0) if f.shape[0] == 3 else f)
                for f in frames
            ]
        elif isinstance(frames[0], torch.Tensor):
            frames = [tensor_to_pil(f) for f in frames]

        feats: List[torch.Tensor] = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with torch.no_grad():
                f = self.model.get_image_features(pixel_values=pixel_values)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
        return torch.cat(feats, dim=0)

    def move_to(self, device: str) -> None:
        if self.model is not None:
            self.model.to(device)
            self.device = device
