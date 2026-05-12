"""Backend ABC.

Concrete backends implement two methods that take a ``messages`` list (in the
Qwen3-VL multimodal format) and a small config dataclass and return a single
string. The pipeline doesn't care which backend produced the text.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ModelBackend(ABC):
    """Abstract base for VLM backends."""

    @abstractmethod
    def generate_text(self, messages: List[Dict], cfg: Any) -> str:
        """Generate text from text-only messages (planner, OCR judge)."""

    @abstractmethod
    def generate_vision(self, messages: List[Dict], cfg: Any) -> str:
        """Generate text from multimodal messages (text + image frames)."""
