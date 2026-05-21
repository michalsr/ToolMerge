"""Image-only Qwen3-VL processing for ToolMerge.


Usage:
    from toolmerge.qwen_vl_utils import process_vision_info
    images, videos, kwargs = process_vision_info(conversations, return_video_kwargs=True)
    # ``videos`` is always None and ``kwargs`` is a trivial dict — both are
    # returned so the call signature matches the upstream package.
"""

from __future__ import annotations

import base64
import copy
import logging
import math
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration (Qwen3-VL)
# ---------------------------------------------------------------------------

MAX_RATIO = 200
SPATIAL_MERGE_SIZE = 2
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384

QWEN3_DEFAULT_PATCH_SIZE = 16    # factor 32 with SPATIAL_MERGE_SIZE=2


# ---------------------------------------------------------------------------
# Resize math
# ---------------------------------------------------------------------------

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Tuple[int, int]:
    """Pick (h, w) divisible by ``factor`` with total pixels in [min, max]."""
    if min_pixels is None:
        min_pixels = IMAGE_MIN_TOKEN_NUM * factor ** 2
    if max_pixels is None:
        max_pixels = IMAGE_MAX_TOKEN_NUM * factor ** 2
    assert max_pixels >= min_pixels, "max_pixels must be >= min_pixels"

    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    return h_bar, w_bar


def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == "RGBA":
        bg = Image.new("RGB", pil_image.size, (255, 255, 255))
        bg.paste(pil_image, mask=pil_image.split()[3])
        return bg
    return pil_image.convert("RGB")


# ---------------------------------------------------------------------------
# Vision-info extraction
# ---------------------------------------------------------------------------

def fetch_image(
    ele: Dict[str, Union[str, Image.Image, torch.Tensor]],
    image_patch_size: Optional[int] = None,
    size_factor: Optional[int] = None,
) -> Union[Image.Image, torch.Tensor]:
    """Materialize ``ele['image']`` as a resized PIL.Image.

    ToolMerge always passes pre-decoded PIL inputs; the tensor / URL /
    file / data-URI branches are retained to match the upstream call site.
    """
    if size_factor is not None:
        patch_factor = size_factor
    elif image_patch_size is not None:
        patch_factor = image_patch_size * SPATIAL_MERGE_SIZE
    else:
        patch_factor = QWEN3_DEFAULT_PATCH_SIZE * SPATIAL_MERGE_SIZE

    image = ele["image"] if "image" in ele else ele["image_url"]
    image_obj: Optional[Image.Image] = None

    if isinstance(image, Image.Image):
        image_obj = image
    elif isinstance(image, torch.Tensor):
        return image.clone().cpu()
    elif image.startswith(("http://", "https://")):
        with requests.get(image, stream=True) as response:
            response.raise_for_status()
            with BytesIO(response.content) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, b64 = image.split("base64,", 1)
            with BytesIO(base64.b64decode(b64)) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    else:
        image_obj = Image.open(image)

    if image_obj is None:
        raise ValueError(f"Unrecognized image input: {image}")

    image = to_rgb(image_obj)

    if "resized_height" in ele and "resized_width" in ele:
        resized_h, resized_w = smart_resize(
            ele["resized_height"], ele["resized_width"], factor=patch_factor,
        )
    else:
        w, h = image.size
        min_pixels = ele.get("min_pixels", IMAGE_MIN_TOKEN_NUM * patch_factor ** 2)
        max_pixels = ele.get("max_pixels", IMAGE_MAX_TOKEN_NUM * patch_factor ** 2)
        resized_h, resized_w = smart_resize(
            h, w, factor=patch_factor,
            min_pixels=min_pixels, max_pixels=max_pixels,
        )

    return image.resize((resized_w, resized_h))


def extract_vision_info(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    """Flatten conversation messages into a list of image / image_url / video dicts."""
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    out: List[Dict[str, Any]] = []
    for conversation in conversations:
        for message in conversation:
            if isinstance(message.get("content"), list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele.get("type", "text") in ("image", "image_url", "video")
                    ):
                        out.append(ele)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def process_vision_info(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    return_video_kwargs: bool = False,
    image_patch_size: int = 14,
) -> Tuple[Optional[List[Image.Image]], Optional[List[Any]], Optional[Dict[str, Any]]]:
    """Image-only processing of ToolMerge vision messages.

    The ToolMerge answerer always builds messages from pre-decoded PIL frames,
    so this function handles images only — passing a ``"video"`` entry raises a
    ``ValueError``. ``video_inputs`` is always ``None`` and ``video_kwargs`` is
    a trivial dict; both are returned so the unpacking in ``Qwen3VLBackend``
    matches the upstream call site.

       """
    vision_infos = extract_vision_info(conversations)
    image_inputs: List[Image.Image] = []

    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(
                vision_info,
                image_patch_size=image_patch_size,
            ))
        elif "video" in vision_info:
            raise ValueError(
                "Video inputs are not supported in toolmerge.qwen_vl_utils. "
                "Pre-decode frames to PIL images and pass them as 'image' entries."
            )
        else:
            raise ValueError("image, image_url or video should be in content.")

    images = image_inputs if image_inputs else None
    if return_video_kwargs:
        return images, None, {"do_sample_frames": False, "fps": []}
    return images, None
