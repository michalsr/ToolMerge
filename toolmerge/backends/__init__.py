"""Model backends for the planner / OCR judge / answerer.

Two concrete backends:
    Qwen3VLBackend  -- local Qwen3-VL-8B (planner + answerer)
    OpenAIBackend   -- OpenAI or Azure OpenAI Chat Completions (OCR judge + answerer)

Both implement ``generate_text(messages, cfg)`` and ``generate_vision(messages, cfg)``.

Backend selection (``cfg.model_backend``):
    "qwen3vl" -- load the local Qwen3-VL model in your script and pass it in.
    "openai"  -- env-driven; works for both OpenAI and Azure OpenAI.
"""

from toolmerge.backends.base import ModelBackend
from toolmerge.backends.openai import OpenAIBackend
from toolmerge.backends.qwen3_vl import Qwen3VLBackend

__all__ = ["ModelBackend", "Qwen3VLBackend", "OpenAIBackend"]
