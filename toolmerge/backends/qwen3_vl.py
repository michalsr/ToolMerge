"""Local Qwen3-VL backend.

You load the model and processor once (e.g. in ``toolmerge/run.py``), wrap
them in this backend, and pass the backend through the pipeline. The
backend itself owns no global state.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from toolmerge.backends.base import ModelBackend

logger = logging.getLogger(__name__)


class Qwen3VLBackend(ModelBackend):
    """Wraps a HuggingFace Qwen3-VL model + processor."""

    def __init__(self, model, processor, device: str = "cuda", qwen_version: str = "qwen3"):
        self.model = model
        self.processor = processor
        self.device = device
        self.qwen_version = qwen_version

    def gen_kwargs(self, cfg: Any) -> Dict[str, Any]:
        do_sample = bool(getattr(cfg, "do_sample", False))
        temperature = float(getattr(cfg, "temperature", 0.0))
        if temperature == 0.0:
            do_sample = False
        out: Dict[str, Any] = {
            "max_new_tokens": int(getattr(cfg, "max_new_tokens", 128)),
            "do_sample": do_sample,
        }
        if do_sample:
            out["temperature"] = temperature
            out["top_p"] = float(getattr(cfg, "top_p", 0.8))
            out["top_k"] = int(getattr(cfg, "top_k", 20))
        return out

    def generate_text(self, messages: List[Dict], cfg: Any) -> str:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **self.gen_kwargs(cfg))
        trimmed = output_ids[0][inputs.input_ids.shape[1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True)

    def generate_vision(self, messages: List[Dict], cfg: Any) -> str:
        from toolmerge.qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True, qwen_version=self.qwen_version,
        )
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **self.gen_kwargs(cfg))
        trimmed = output_ids[0][inputs.input_ids.shape[1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True)
