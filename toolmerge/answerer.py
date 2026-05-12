"""Shared answerer.

Takes the K selected keyframes (in temporal order), the question, and the
options dict and returns a single letter. Used by ToolMerge AND by every
baseline — keeping a single answerer module means table accuracies depend
only on the keyframe-selection choice.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from toolmerge.prompts.answer_generator import ANSWER_TEMPLATES

logger = logging.getLogger(__name__)


# --- Format helpers ------------------------------------------------------

def format_options(options: dict, prompt_template: str = "lif") -> str:
    """The lif template uses ``A) text`` joined by newlines."""
    if prompt_template == "lif":
        return "\n".join(f"{k}) {v}" for k, v in sorted(options.items()))
    return "\n".join(f"  {k}: {v}" for k, v in options.items())


def parse_answer(response: str, options: dict, mode: str = "letter_only") -> Dict[str, Any]:
    """Extract the answer letter from a raw model response.

    Picks the first standalone letter from ``options`` that appears in the
    response; falls back to the first character if nothing matches.
    """
    response_stripped = response.strip()
    option_letters = list(options.keys())
    answer: str | None = None

    for letter in option_letters:
        if re.search(rf"\b{letter}\b", response_stripped):
            answer = letter
            break
    if answer is None and response_stripped and response_stripped[0].upper() in option_letters:
        answer = response_stripped[0].upper()

    confidence = 0.0
    if mode == "letter_and_confidence":
        m = re.search(r"confidence[:\s]*([0-5])(?:\s*/\s*5)?", response, re.IGNORECASE)
        if m:
            confidence = float(m.group(1))

    return {"answer": answer, "confidence": confidence, "raw_response": response}


# --- Frame helpers -------------------------------------------------------

def frame_to_pil(frame):
    """(3, H, W) uint8 tensor -> PIL.Image. Matches extract_frames_by_index output."""
    from PIL import Image

    arr = frame.detach().cpu().numpy()
    if arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    return Image.fromarray(arr, mode="RGB")


def build_frame_messages(frames, timestamps, prompt_text, include_timestamps: bool = True):
    """Standard pipeline message: ``{t:.1f}s`` text + image for each frame, then the prompt."""
    content: List[Dict[str, Any]] = []
    for t, frame in zip(timestamps, frames):
        if include_timestamps:
            content.append({"type": "text", "text": f"{t:.1f}s"})
        content.append({"type": "image", "image": frame})
    content.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": content}]


def generate_qwen_pil(
    backend, pil_frames, timestamps, prompt_text, cfg, include_timestamps: bool,
):
    """Byte-parity Qwen3-VL path: PIL images directly, no qwen_version override."""
    import torch
    from toolmerge.qwen_vl_utils import process_vision_info

    content: List[Dict[str, Any]] = []
    for t, pil in zip(timestamps, pil_frames):
        if include_timestamps:
            content.append({"type": "text", "text": f"{t:.1f}s"})
        content.append({"type": "image", "image": pil})
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]

    processor = backend.processor
    model = backend.model
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    temperature = float(getattr(cfg, "temperature", 0.0) or 0.0)
    gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(getattr(cfg, "max_new_tokens", 128))}
    if temperature == 0.0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = float(getattr(cfg, "top_p", 0.8))
        gen_kwargs["top_k"] = int(getattr(cfg, "top_k", 20))

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    trimmed = output_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip(), messages


# --- Public entry point --------------------------------------------------

def generate_answer(frames, timestamps, question, options, backend, cfg):
    """Answer one question.

    Returns ``{"answer", "confidence", "raw_response", "prompt"}``.
    """
    if len(frames) == 0:
        return {"answer": None, "confidence": 0.0, "raw_response": "", "prompt": ""}

    template = ANSWER_TEMPLATES.get(cfg.prompt_template)
    if template is None:
        raise KeyError(
            f"Unknown answerer prompt '{cfg.prompt_template}'. "
            f"Available: {list(ANSWER_TEMPLATES)}"
        )
    option_letters = ", ".join(options.keys())

    prompt_text = template.format(
        question=question,
        options=format_options(options, prompt_template=cfg.prompt_template),
        option_letters=option_letters,
    )

    include_timestamps = not bool(getattr(cfg, "no_timestamps", False))

    # Byte-parity with reanswer.generate_qwen when running locally on Qwen3-VL.
    from toolmerge.backends.qwen3_vl import Qwen3VLBackend  # local import to avoid cycle
    if isinstance(backend, Qwen3VLBackend):
        pil_frames = [frame_to_pil(f) for f in frames]
        response, _msgs = generate_qwen_pil(
            backend, pil_frames, timestamps, prompt_text, cfg,
            include_timestamps=include_timestamps,
        )
    else:
        messages = build_frame_messages(
            frames, timestamps, prompt_text, include_timestamps=include_timestamps,
        )
        response = backend.generate_vision(messages, cfg)

    logger.debug("Answer generator response: %s", response)
    result = parse_answer(response, options, mode=cfg.mode)
    result["prompt"] = prompt_text
    return result
