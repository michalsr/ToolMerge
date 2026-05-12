"""Unified OpenAI / Azure OpenAI backend.

Auto-detects whether to use Azure based on ``AZURE_OPENAI_ENDPOINT`` or the
explicit ``OPENAI_USE_AZURE=1`` env var. All keys / endpoints / model
names come from env (with optional overrides in YAML).
"""

from __future__ import annotations

import base64
import io
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

import torch

from toolmerge.backends.base import ModelBackend

logger = logging.getLogger(__name__)


# --- Small helpers -------------------------------------------------------

def tensor_to_pil(img: torch.Tensor):
    """(3, H, W) uint8 tensor -> PIL.Image. Matches extract_frames_by_index output."""
    from PIL import Image

    arr = img.detach().cpu().numpy()
    if arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    return Image.fromarray(arr, mode="RGB")


def pil_to_b64_url(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def is_rate_limit(exc: Exception) -> bool:
    s = str(exc).lower()
    if "429" in s or "rate" in s or "quota" in s:
        return True
    if getattr(exc, "status_code", None) == 429:
        return True
    if "RateLimitError" in type(exc).__name__:
        return True
    return False


def retry(func, max_retries=5, base_delay=2.0, max_delay=120.0):
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:  # noqa: BLE001
            if not is_rate_limit(e):
                raise
            last_exc = e
            if attempt >= max_retries:
                logger.error("Rate limit exceeded after %d retries: %s", max_retries, e)
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.1)
            logger.warning(
                "Rate limit (attempt %d/%d); retrying in %.1fs: %s",
                attempt + 1, max_retries + 1, delay + jitter, e,
            )
            time.sleep(delay + jitter)
    raise last_exc  # pragma: no cover


def normalize_azure_v1_base_url(url: str) -> str:
    """Ensure the Azure base_url ends with the OpenAI-compatible /openai/v1/ path."""
    u = (url or "").strip().rstrip("/")
    if not u:
        return u
    if "/openai/v1" in u:
        return u + "/"
    return u + "/openai/v1/"


# --- Backend -------------------------------------------------------------

class OpenAIBackend(ModelBackend):
    """OpenAI Chat Completions backend (works for OpenAI and Azure OpenAI).

    Env vars (any of these can be passed via ``OpenAIBackendConfig`` instead):

    OpenAI:
        OPENAI_API_KEY        -- required.
        OPENAI_BASE_URL       -- optional.
        OPENAI_MODEL          -- default ``gpt-4o``.

    Azure OpenAI:
        AZURE_OPENAI_API_KEY  -- required.
        AZURE_OPENAI_ENDPOINT -- required.
        AZURE_OPENAI_DEPLOYMENT -- default ``gpt-4o``.
        AZURE_OPENAI_API_VERSION -- optional.

    Use Azure explicitly with ``OPENAI_USE_AZURE=1``; otherwise the backend
    auto-detects from ``AZURE_OPENAI_ENDPOINT``.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        use_azure: Optional[bool] = None,
        max_retries: int = 5,
        retry_base_delay: float = 2.0,
        retry_max_delay: float = 120.0,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAIBackend requires `openai`. Install: pip install openai"
            ) from e

        if use_azure is None:
            env_flag = os.environ.get("OPENAI_USE_AZURE", "").lower()
            if env_flag in ("1", "true", "yes"):
                use_azure = True
            elif env_flag in ("0", "false", "no"):
                use_azure = False
            else:
                use_azure = bool(os.environ.get("AZURE_OPENAI_ENDPOINT"))

        self.use_azure = use_azure
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay

        if self.use_azure:
            key = (
                api_key
                or os.environ.get("AZURE_OPENAI_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )
            if not key:
                raise ValueError(
                    "Azure backend requires AZURE_OPENAI_API_KEY (or OPENAI_API_KEY)."
                )
            base_url = (
                api_endpoint
                or os.environ.get("AZURE_OPENAI_ENDPOINT")
                or os.environ.get("OPENAI_BASE_URL")
            )
            if not base_url:
                raise ValueError(
                    "Azure backend requires AZURE_OPENAI_ENDPOINT."
                )
            base_url = normalize_azure_v1_base_url(base_url)
            self.client = OpenAI(api_key=key, base_url=base_url)
            self.model_name = (
                model_name
                or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
            )
            logger.info("Azure OpenAI: base_url=%s, deployment=%s", base_url, self.model_name)
        else:
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI backend.")
            base_url = api_endpoint or os.environ.get("OPENAI_BASE_URL")
            kwargs: Dict[str, Any] = {"api_key": key}
            if base_url:
                kwargs["base_url"] = base_url.rstrip("/") + "/"
            self.client = OpenAI(**kwargs)
            self.model_name = model_name or os.environ.get("OPENAI_MODEL", "gpt-4o")
            logger.info("OpenAI: model=%s", self.model_name)

    # --- Message conversion ---------------------------------------------

    def convert_messages(self, messages: List[Dict]) -> List[Dict]:
        """Pipeline message format -> OpenAI Chat Completions vision format."""
        out: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", [])

            if isinstance(content, str):
                out.append({"role": role, "content": content})
                continue

            parts: List[Dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                t = item.get("type")
                if t == "text":
                    text = item.get("text", "")
                    if text:
                        parts.append({"type": "text", "text": text})
                elif t == "image":
                    img = item.get("image")
                    if isinstance(img, torch.Tensor):
                        img = tensor_to_pil(img)
                    from PIL import Image  # noqa: WPS433
                    if isinstance(img, Image.Image):
                        parts.append({
                            "type": "image_url",
                            "image_url": {"url": pil_to_b64_url(img)},
                        })

            if not parts:
                continue
            if role == "system":
                texts = [p["text"] for p in parts if p.get("type") == "text"]
                out.append({"role": "system", "content": "\n".join(texts)})
            else:
                out.append({"role": role, "content": parts})
        return out

    # --- API call -------------------------------------------------------

    def call(self, messages: List[Dict], cfg: Any) -> str:
        openai_messages = self.convert_messages(messages)
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": openai_messages,
            "max_completion_tokens": int(getattr(cfg, "max_new_tokens", 128)),
        }
        do_sample = bool(getattr(cfg, "do_sample", False))
        temperature = float(getattr(cfg, "temperature", 0.0))
        if do_sample and temperature > 0:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = float(getattr(cfg, "top_p", 0.8))
        else:
            kwargs["temperature"] = 0.0

        def do_call() -> str:
            response = self.client.chat.completions.create(**kwargs)
            msg = response.choices[0].message
            text = msg.content or ""
            if not text:
                logger.warning(
                    "OpenAI API returned empty content (finish_reason=%s)",
                    response.choices[0].finish_reason,
                )
            return text

        return retry(
            do_call,
            max_retries=self.max_retries,
            base_delay=self.retry_base_delay,
            max_delay=self.retry_max_delay,
        )

    def generate_text(self, messages: List[Dict], cfg: Any) -> str:
        return self.call(messages, cfg)

    def generate_vision(self, messages: List[Dict], cfg: Any) -> str:
        return self.call(messages, cfg)
