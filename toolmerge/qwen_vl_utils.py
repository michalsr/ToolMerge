"""Unified Qwen-VL utils supporting Qwen2.5-VL and Qwen3-VL.

Vendored from the research tree (``time_r1/utils/qwen_vl_utils.py``) to avoid
a dependency on the upstream ``qwen-vl-utils`` package (which has version
conflicts across Qwen2.5 / Qwen3 releases). Used by the local Qwen3-VL
backend and the answerer.

Usage:
    from toolmerge.qwen_vl_utils import process_vision_info
    images, videos, kwargs = process_vision_info(conversations, return_video_kwargs=True)

    # For Qwen2.5
    images, videos, kwargs = process_vision_info(conversations, return_video_kwargs=True, qwen_version="qwen2.5")

    # Or set global default
    from toolmerge.qwen_vl_utils import set_default_qwen_version
    set_default_qwen_version("qwen2.5")
"""

from __future__ import annotations

import base64
import copy
import logging
import math
import os
import sys
import time
import warnings
from enum import Enum
from functools import lru_cache
from io import BytesIO
from typing import Optional, Union, Tuple, List, Any, Dict, Literal
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
import torchvision
from packaging import version
from PIL import Image
import numpy as np
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode


logger = logging.getLogger(__name__)

# ============================================================================
# Version Configuration
# ============================================================================

QwenVersionType = Literal["qwen2.5", "qwen3"]

_DEFAULT_QWEN_VERSION: QwenVersionType = "qwen3"


def set_default_qwen_version(version: QwenVersionType) -> None:
    """Set the default Qwen version for all subsequent calls."""
    global _DEFAULT_QWEN_VERSION
    if version not in ("qwen2.5", "qwen3"):
        raise ValueError(f"Invalid Qwen version: {version}. Must be 'qwen2.5' or 'qwen3'")
    _DEFAULT_QWEN_VERSION = version
    logger.info(f"Default Qwen version set to: {version}")


def get_default_qwen_version() -> QwenVersionType:
    """Get the current default Qwen version."""
    return _DEFAULT_QWEN_VERSION


# ============================================================================
# Constants (matching TimeSearch-R / official qwen-vl-utils style)
# ============================================================================

# Shared constants
# get debug from environment variable
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
MAX_RATIO = 200
SPATIAL_MERGE_SIZE = 2
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
VIDEO_MIN_TOKEN_NUM = 128
VIDEO_MAX_TOKEN_NUM = 768

FPS = 2.0
FRAME_FACTOR = 2
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768
MAX_NUM_WORKERS_FETCH_VIDEO = 8

MODEL_SEQ_LEN = int(float(os.environ.get('MODEL_SEQ_LEN', 128000)))

# Version-specific defaults
QWEN25_DEFAULT_PATCH_SIZE = 14  # patch_size=14, spatial_merge=2 => factor 28
QWEN3_DEFAULT_PATCH_SIZE = 16   # patch_size=16, spatial_merge=2 => factor 32

QWEN25_IMAGE_FACTOR = QWEN25_DEFAULT_PATCH_SIZE * SPATIAL_MERGE_SIZE  # 28
QWEN3_IMAGE_FACTOR = QWEN3_DEFAULT_PATCH_SIZE * SPATIAL_MERGE_SIZE    # 32

QWEN25_FRAME_CACHE_EXTENSION = ".mp4.frame_cache"
QWEN3_FRAME_CACHE_EXTENSION = ".mp4.frame_cache_qwen3vl"

# Legacy alias for backwards compatibility
IMAGE_FACTOR = QWEN25_IMAGE_FACTOR  # 28


# ============================================================================
# Configuration Classes
# ============================================================================

class Qwen25Config:
    """Configuration constants for Qwen2.5-VL."""
    IMAGE_FACTOR = 28 # 28
    MIN_PIXELS = IMAGE_MIN_TOKEN_NUM * QWEN25_IMAGE_FACTOR ** 2  # 4 * 28 * 28
    MAX_PIXELS = IMAGE_MAX_TOKEN_NUM * QWEN25_IMAGE_FACTOR ** 2  # 16384 * 28 * 28
    VIDEO_MIN_PIXELS = VIDEO_MIN_TOKEN_NUM * QWEN25_IMAGE_FACTOR ** 2  # 128 * 28 * 28
    VIDEO_MAX_PIXELS = VIDEO_MAX_TOKEN_NUM * QWEN25_IMAGE_FACTOR ** 2  # 768 * 28 * 28
    FRAME_CACHE_EXTENSION = QWEN25_FRAME_CACHE_EXTENSION  # ".frame_cache"


class Qwen3Config:
    """Configuration constants for Qwen3-VL."""
    SPATIAL_MERGE_SIZE = SPATIAL_MERGE_SIZE  # 2
    DEFAULT_IMAGE_PATCH_SIZE = QWEN3_DEFAULT_PATCH_SIZE  # 16
    IMAGE_FACTOR = 32
    VIDEO_MIN_TOKEN_NUM = VIDEO_MIN_TOKEN_NUM  # 128
    VIDEO_MAX_TOKEN_NUM = VIDEO_MAX_TOKEN_NUM  # 768
    FRAME_CACHE_EXTENSION = QWEN3_FRAME_CACHE_EXTENSION  # ".frame_cache_qwen3vl"


def get_config_for_version(qwen_version: QwenVersionType) -> dict:
    """Get configuration constants for a specific Qwen version."""
    if qwen_version == "qwen2.5":
        factor = QWEN25_IMAGE_FACTOR
        return {
            "image_factor": factor,
            "min_pixels": VIDEO_MIN_TOKEN_NUM * factor * factor,
            "max_pixels": VIDEO_MAX_TOKEN_NUM * factor * factor,
            "frame_cache_extension": QWEN25_FRAME_CACHE_EXTENSION,
        }
    elif qwen_version == "qwen3":
        factor = QWEN3_IMAGE_FACTOR
        return {
            "image_factor": factor,
            "min_pixels": VIDEO_MIN_TOKEN_NUM * factor * factor,
            "max_pixels": VIDEO_MAX_TOKEN_NUM * factor * factor,
            "frame_cache_extension": QWEN3_FRAME_CACHE_EXTENSION,
        }
    else:
        raise ValueError(f"Unknown qwen_version: {qwen_version}. Must be 'qwen2.5' or 'qwen3'")

FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


# ============================================================================
# Utility Functions (shared)
# ============================================================================

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer >= 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer <= 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, 
    width: int, 
    factor: int,
    min_pixels: Optional[int] = None, 
    max_pixels: Optional[int] = None,
    qwen_version: Optional[QwenVersionType] = None
) -> Tuple[int, int]:
    """
    Rescales the image so that:
    1. Both dimensions are divisible by 'factor'.
    2. Total pixels is within [min_pixels, max_pixels].
    3. Aspect ratio is maintained.
    """
    qwen_version = qwen_version or _DEFAULT_QWEN_VERSION
    
    if qwen_version == "qwen2.5":
        default_min = IMAGE_MIN_TOKEN_NUM * QWEN25_IMAGE_FACTOR ** 2
        default_max = IMAGE_MAX_TOKEN_NUM * QWEN25_IMAGE_FACTOR ** 2
    else:
        default_min = IMAGE_MIN_TOKEN_NUM * factor ** 2
        default_max = IMAGE_MAX_TOKEN_NUM * factor ** 2
    
    min_pixels = min_pixels if min_pixels is not None else default_min
    max_pixels = max_pixels if max_pixels is not None else default_max
    
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
    """Convert image to RGB, handling RGBA with white background."""
    if pil_image.mode == 'RGBA':
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])
        return white_background
    else:
        return pil_image.convert("RGB")


# ============================================================================
# Video Reader Backend Detection
# ============================================================================

def is_decord_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("decord") is not None


def is_torchcodec_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("torchcodec") is not None


@lru_cache(maxsize=1)
def get_video_reader_backend(qwen_version: Optional[QwenVersionType] = None) -> str:
    """Get the best available video reader backend."""
    qwen_version = qwen_version or _DEFAULT_QWEN_VERSION
    
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER
    elif qwen_version == "qwen3" and is_torchcodec_available():
        video_reader_backend = "torchcodec"
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    
    print(f"qwen-vl-utils ({qwen_version}) using {video_reader_backend} to read video.", file=sys.stderr)
    return video_reader_backend


# ============================================================================
# Frame Calculation Functions
# ============================================================================

def smart_nframes(
    ele: Dict[str, Any],
    total_frames: int,
    video_fps: Union[int, float],
) -> int:
    """Calculate the number of frames for video used for model inputs."""
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    
    if not (FRAME_FACTOR <= nframes <= total_frames):
        raise ValueError(f"nframes should be in [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def calculate_video_frame_range(
    ele: Dict[str, Any],
    total_frames: int,
    video_fps: float,
) -> Tuple[int, int, int]:
    """
    Calculate start and end frame indices based on time range.
    Only used for Qwen3-VL which supports video_start/video_end in decord.
    """
    if video_fps <= 0:
        raise ValueError("video_fps must be a positive number")
    if total_frames <= 0:
        raise ValueError("total_frames must be a positive integer")

    video_start = ele.get("video_start", None)
    video_end = ele.get("video_end", None)
    
    if video_start is None and video_end is None:
        return 0, total_frames - 1, total_frames

    max_duration = total_frames / video_fps
    
    if video_start is not None:
        video_start_clamped = max(0.0, min(video_start, max_duration))
        start_frame = math.ceil(video_start_clamped * video_fps)
    else:
        start_frame = 0
    
    if video_end is not None:
        video_end_clamped = max(0.0, min(video_end, max_duration))
        end_frame = math.floor(video_end_clamped * video_fps)
        end_frame = min(end_frame, total_frames - 1)
    else:
        end_frame = total_frames - 1

    if start_frame >= end_frame:
        raise ValueError(
            f"Invalid time range: Start frame {start_frame} exceeds end frame {end_frame}. "
            f"Video duration: {max_duration:.2f}s ({total_frames} frames @ {video_fps}fps)"
        )

    logger.info(f"calculate video frame range: {start_frame=}, {end_frame=}, {total_frames=}")
    return start_frame, end_frame, end_frame - start_frame + 1


# ============================================================================
# Image Fetching
# ============================================================================

def fetch_image(
    ele: Dict[str, Union[str, Image.Image, torch.Tensor]],
    image_patch_size: Optional[int] = None,
    size_factor: Optional[int] = None,
    qwen_version: Optional[QwenVersionType] = None,
) -> Union[Image.Image, torch.Tensor]:
    """
    Fetch and process an image.
    
    Args:
        ele: Dict with 'image' or 'image_url' key
        image_patch_size: Patch size for Qwen3 (default 14)
        size_factor: Direct size factor (for Qwen2.5 compatibility, default 28)
        qwen_version: Model version to use
    """
    qwen_version = qwen_version or _DEFAULT_QWEN_VERSION
    
    # Determine the patch factor
    if size_factor is not None:
        patch_factor = size_factor
    elif image_patch_size is not None:
        patch_factor = image_patch_size * SPATIAL_MERGE_SIZE
    elif qwen_version == "qwen2.5":
        patch_factor = QWEN25_IMAGE_FACTOR
    else:
        patch_factor = QWEN3_DEFAULT_PATCH_SIZE * SPATIAL_MERGE_SIZE
    
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]

    image_obj = None
    
    if isinstance(image, Image.Image):
        image_obj = image
    elif isinstance(image, torch.Tensor):
        return image.clone().cpu()
    elif image.startswith("http://") or image.startswith("https://"):
        with requests.get(image, stream=True) as response:
            response.raise_for_status()
            with BytesIO(response.content) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            with BytesIO(data) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    else:
        image_obj = Image.open(image)
    
    if image_obj is None:
        raise ValueError(f"Unrecognized image input: {image}")
    
    image = to_rgb(image_obj)

    # Resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=patch_factor,
            qwen_version=qwen_version,
        )
    else:
        width, height = image.size
        if qwen_version == "qwen2.5":
            min_pixels = ele.get("min_pixels", IMAGE_MIN_TOKEN_NUM * QWEN25_IMAGE_FACTOR ** 2)
            max_pixels = ele.get("max_pixels", IMAGE_MAX_TOKEN_NUM * QWEN25_IMAGE_FACTOR ** 2)
        else:
            min_pixels = ele.get("min_pixels", IMAGE_MIN_TOKEN_NUM * patch_factor ** 2)
            max_pixels = ele.get("max_pixels", IMAGE_MAX_TOKEN_NUM * patch_factor ** 2)
        
        resized_height, resized_width = smart_resize(
            height, width,
            factor=patch_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            qwen_version=qwen_version,
        )
    
    image = image.resize((resized_width, resized_height))
    return image


# ============================================================================
# Video Reading Backends
# ============================================================================

def _read_video_torchvision_qwen25(ele: Dict[str, Any]) -> Tuple[torch.Tensor, float]:
    """Read video using torchvision for Qwen2.5."""
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn("torchvision < 0.19.0 does not support http/https video path")
        if "file://" in video_path:
            video_path = video_path[7:]
    
    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = video[idx]
    return video, sample_fps


def _read_video_torchvision_qwen3(ele: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any], float]:
    """Read video using torchvision for Qwen3."""
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn("torchvision < 0.19.0 does not support http/https video path")
        if "file://" in video_path:
            video_path = video_path[7:]
    
    st = time.time()
    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    logger.info(f"torchvision: {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = video[idx]

    video_metadata = dict(
        fps=video_fps,
        frames_indices=idx,
        total_num_frames=total_frames,
        video_backend="torchvision",
    )
    return video, video_metadata, sample_fps


def _read_video_decord_qwen25(ele: Dict[str, Any]) -> Tuple[torch.Tensor, float]:
    """Read video using decord for Qwen2.5."""
    import decord
    video_path = ele["video"]
    vr = decord.VideoReader(video_path)
    
    if 'video_start' in ele or 'video_end' in ele:
        raise NotImplementedError("video_start/video_end not supported in Qwen2.5 decord backend")
    
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    return video, sample_fps


def _read_video_decord_qwen3(ele: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any], float]:
    """Read video using decord for Qwen3 (supports video_start/video_end)."""
    import decord
    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele, total_frames, video_fps
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)
    logger.info(f"decord: {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps

    video_metadata = dict(
        fps=video_fps,
        frames_indices=idx,
        total_num_frames=total_frames,
        video_backend="decord",
    )
    return video, video_metadata, sample_fps


def _read_video_torchcodec(ele: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any], float]:
    """Read video using torchcodec (Qwen3 only)."""
    from torchcodec.decoders import VideoDecoder
    TORCHCODEC_NUM_THREADS = int(os.environ.get('TORCHCODEC_NUM_THREADS', 8))
    
    video_path = ele["video"]
    st = time.time()
    decoder = VideoDecoder(video_path, num_ffmpeg_threads=TORCHCODEC_NUM_THREADS)
    video_fps = decoder.metadata.average_fps
    total_frames = decoder.metadata.num_frames
    
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele, total_frames, video_fps
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = decoder.get_frames_at(indices=idx).data
    logger.info(f"torchcodec: {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    video_metadata = dict(
        fps=video_fps,
        frames_indices=idx,
        total_num_frames=total_frames,
        video_backend="torchcodec",
    )
    return video, video_metadata, sample_fps


def _read_video_tensor_qwen25(ele: Dict[str, Any]) -> Tuple[torch.Tensor, float]:
    """Read pre-decoded video frames for Qwen2.5."""
    if ele["video"].endswith(".frame_cache"):
        video_path = ele["video"]
    else:
        video_path = ele["video"] + ".frame_cache"
    
    frame_cache = torch.load(video_path, map_location="cpu")
    frame_tensor = frame_cache["frame_tensor"]
    video_fps = frame_cache["fps"]
    total_frames = len(frame_tensor)
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    video = frame_tensor[idx]
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    return video, sample_fps


def _read_video_tensor_qwen3(ele: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any], float]:
    """Read pre-decoded video frames for Qwen3."""
    video_path = ele["video"]
    if not isinstance(video_path, str):
        raise TypeError(f"tensor backend expects video path string, got {type(video_path)}")
    if not video_path.endswith(".frame_cache_qwen3vl"):
        video_path = video_path + ".frame_cache_qwen3vl"

    frame_cache = torch.load(video_path, map_location="cpu", weights_only=False)
    frame_tensor = frame_cache["frame_tensor"]
    video_fps = float(frame_cache["fps"])
    total_frames = int(frame_tensor.size(0))

    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    video = frame_tensor[idx]
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps

    video_metadata = dict(
        fps=video_fps,
        frames_indices=idx.tolist(),
        total_num_frames=total_frames,
        video_backend="tensor",
    )
    return video, video_metadata, sample_fps


# Backend registries
VIDEO_READER_BACKENDS_QWEN25 = {
    "decord": _read_video_decord_qwen25,
    "torchvision": _read_video_torchvision_qwen25,
    "tensor": _read_video_tensor_qwen25,
}

VIDEO_READER_BACKENDS_QWEN3 = {
    "decord": _read_video_decord_qwen3,
    "torchvision": _read_video_torchvision_qwen3,
    "torchcodec": _read_video_torchcodec,
    "tensor": _read_video_tensor_qwen3,
}


# ============================================================================
# Video Fetching
# ============================================================================

def fetch_video(
    ele: Dict[str, Any],
    image_patch_size: int = 14,
    image_factor: Optional[int] = None,
    return_video_sample_fps: bool = False,
    return_video_metadata: bool = False,
    frame_cache_path: Optional[str] = None,
    qwen_version: Optional[QwenVersionType] = None,
) -> Any:
    """
    Fetch and process video.
    
    Args:
        ele: Dict with 'video' key (path, tensor, or list of frames)
        image_patch_size: Patch size for Qwen3 (default 14)
        image_factor: Direct image factor (for Qwen2.5 compatibility)
        return_video_sample_fps: Whether to return the sample FPS
        return_video_metadata: Whether to return video metadata (Qwen3 only)
        frame_cache_path: Path to frame cache directory
        qwen_version: Model version to use
    """
    qwen_version = qwen_version or _DEFAULT_QWEN_VERSION
    
    # Determine factor
    if image_factor is not None:
        factor = image_factor
    elif qwen_version == "qwen2.5":
        factor = QWEN25_IMAGE_FACTOR
    elif qwen_version == "qwen3":
        factor = QWEN3_IMAGE_FACTOR
    else:
        raise ValueError(f"Unknown qwen_version: {qwen_version}")
    
    # Get pixel limits (calculated dynamically like TimeSearch-R)
    VIDEO_FRAME_MIN_PIXELS = VIDEO_MIN_TOKEN_NUM * factor * factor
    VIDEO_FRAME_MAX_PIXELS = VIDEO_MAX_TOKEN_NUM * factor * factor
    VIDEO_TOTAL_PIXELS = MODEL_SEQ_LEN * factor * factor * 0.9
    
    if qwen_version == "qwen2.5":
        frame_cache_ext = QWEN25_FRAME_CACHE_EXTENSION
        backends = VIDEO_READER_BACKENDS_QWEN25
    else:
        frame_cache_ext = QWEN3_FRAME_CACHE_EXTENSION
        backends = VIDEO_READER_BACKENDS_QWEN3
    
    if isinstance(ele["video"], str):
        video_reader_backend = get_video_reader_backend(qwen_version)
        try:
            video_path = ele["video"]

            # Check for frame cache
            frame_cache_candidate = None
            if frame_cache_path:
                frame_cache_candidate = os.path.join(
                    frame_cache_path, os.path.basename(video_path) + frame_cache_ext
                )
                if DEBUG:
                    print(f"Frame cache candidate: {frame_cache_candidate}")
                    print(f"Frame cache candidate exists: {os.path.exists(frame_cache_candidate)}")
            
            if frame_cache_candidate and os.path.exists(frame_cache_candidate):
                cache_ele = dict(ele)
                cache_ele["video"] = frame_cache_candidate
                result = backends["tensor"](cache_ele)
            elif os.path.exists(video_path + frame_cache_ext):
                if DEBUG:
                    print(f"Frame cache exists: {video_path + frame_cache_ext}")
                frame_cache = torch.load(video_path + frame_cache_ext, map_location="cpu")
                video = frame_cache["frame_tensor"]
                sample_fps = float(frame_cache["fps"])
                if qwen_version == "qwen3":
                    video_metadata = dict(
                        fps=sample_fps,
                        frames_indices=list(range(int(video.shape[0]))),
                        total_num_frames=int(video.shape[0]),
                        video_backend="tensor",
                    )
                    result = (video, video_metadata, sample_fps)
                else:
                    result = (video, sample_fps)
            else:
                result = backends[video_reader_backend](ele)
                
        except Exception as e:
            logger.warning(f"video_reader_backend {video_reader_backend} error, using torchvision: {e}")
            result = backends["torchvision"](ele)
        
        # Unpack result based on version
        if qwen_version == "qwen3":
            video, video_metadata, sample_fps = result
        else:
            video, sample_fps = result
            video_metadata = None
            
    elif isinstance(ele["video"], torch.Tensor):
        video = ele["video"]
        sample_fps = float(ele.get("fps", FPS))
        video_metadata = dict(
            fps=sample_fps,
            frames_indices=list(range(int(video.shape[0]))),
            total_num_frames=int(video.shape[0]),
            video_backend="tensor",
        ) if qwen_version == "qwen3" else None
        
        if qwen_version == "qwen3":
            final_video = (video, video_metadata) if return_video_metadata else video
        else:
            final_video = video
            
        if return_video_sample_fps:
            return final_video, sample_fps
        return final_video
    else:
        # List of frames
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        
        max_workers = min(MAX_NUM_WORKERS_FETCH_VIDEO, len(ele["video"]))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(fetch_image, {"image": video_element, **process_info}, 
                               image_patch_size=image_patch_size if qwen_version == "qwen3" else None,
                               size_factor=factor if qwen_version == "qwen2.5" else None,
                               qwen_version=qwen_version)
                for video_element in ele["video"]
            ]
            image_list = [future.result() for future in futures]

        nframes = ceil_by_factor(len(image_list), FRAME_FACTOR)
        if len(image_list) < nframes:
            image_list.extend([image_list[-1]] * (nframes - len(image_list)))

        sample_fps = ele.get("sample_fps", 2.0)
        video = torch.stack([
            torch.from_numpy(np.array(image).transpose(2, 0, 1))
            for image in image_list
        ])

        raw_fps = process_info.pop("raw_fps", sample_fps)
        video_metadata = dict(
            fps=raw_fps,
            frames_indices=[i for i in range(len(video))],
            total_num_frames=(nframes / sample_fps) * raw_fps,
        ) if qwen_version == "qwen3" else None

    # Resize video
    nframes, _, height, width = video.shape
    min_pixels = ele.get("min_pixels", VIDEO_FRAME_MIN_PIXELS)
    total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
    max_pixels = max(min(VIDEO_FRAME_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
    max_pixels_supposed = ele.get("max_pixels", max_pixels)
    if max_pixels_supposed > max_pixels:
        logger.warning(f"The given max_pixels[{max_pixels_supposed}] exceeds limit[{max_pixels}].")
    max_pixels = min(max_pixels_supposed, max_pixels)
    
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=factor,
            qwen_version=qwen_version,
        )
    else:
        resized_height, resized_width = smart_resize(
            height, width,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            qwen_version=qwen_version,
        )
    
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()

    # Return based on version and flags
    if qwen_version == "qwen3":
        final_video = (video, video_metadata) if return_video_metadata else video
    else:
        final_video = video
        
    if return_video_sample_fps:
        return final_video, sample_fps
    return final_video


# ============================================================================
# Vision Info Extraction and Processing
# ============================================================================

def extract_vision_info(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]
) -> List[Dict[str, Any]]:
    """Extract vision info from conversations."""
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele.get("type", "text") in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def replace_vision_info_with_placeholder(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]
) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """Replace in-memory tensors with string placeholders."""
    vision_infos = extract_vision_info(conversations)
    for vision_info in vision_infos:
        if "video" in vision_info and isinstance(vision_info["video"], torch.Tensor):
            vision_info["video"] = f"<video>, shape: {tuple(vision_info['video'].shape)}"
        elif "image" in vision_info and isinstance(vision_info["image"], torch.Tensor):
            vision_info["image"] = f"<image>, shape: {tuple(vision_info['image'].shape)}"
    return conversations


def tensor_to_base64(frame_tensor: torch.Tensor) -> str:
    """Convert (C, H, W) tensor to base64 data URL."""
    frame_uint8 = frame_tensor.clamp(0, 255).byte().cpu().numpy()
    if frame_uint8.ndim == 3 and frame_uint8.shape[0] == 3:
        frame_uint8 = frame_uint8.transpose(1, 2, 0)
    image = Image.fromarray(frame_uint8, mode="RGB")
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"


def replace_vision_info_with_base64(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]
) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """Convert in-memory tensors to base64-encoded jpegs."""
    vision_infos = extract_vision_info(conversations)
    for vision_info in vision_infos:
        if "video" in vision_info and isinstance(vision_info["video"], torch.Tensor):
            vision_info["video"] = [tensor_to_base64(frame) for frame in vision_info["video"]]
        elif "image" in vision_info and isinstance(vision_info["image"], torch.Tensor):
            vision_info["image"] = tensor_to_base64(vision_info["image"])
    return conversations


# ============================================================================
# Main Processing Function
# ============================================================================

def process_vision_info(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    return_video_kwargs: bool = False,
    return_video_metadata: bool = False,
    image_patch_size: int = 14,
    qwen_version: Optional[QwenVersionType] = None,
) -> Tuple[Optional[List[Image.Image]], Optional[List[Any]], Optional[Dict[str, Any]]]:
    """
    Process vision information from conversations.

    NOTE: image_patch_size defaults to 14 (Qwen2.5 value). For Qwen3-VL the
    correct patch size is 16 (factor=32), so the pre-resize in fetch_image
    uses factor=28 instead of 32. In practice this is harmless: the HF
    AutoProcessor re-resizes to factor=32 internally, so the final token
    count is identical. Verified empirically: a 420x280 (28-aligned) image
    and a 416x288 (32-aligned) image both produce grid [1,18,26] = 468 tokens.

    Args:
        conversations: List of conversation dicts or list of lists
        return_video_kwargs: Whether to return video kwargs
        return_video_metadata: Whether to return video metadata (Qwen3 only)
        image_patch_size: Patch size for Qwen3 (default 14)
        qwen_version: Model version ("qwen2.5" or "qwen3")
    
    Returns:
        Tuple of (image_inputs, video_inputs, video_kwargs)
    """
    qwen_version = qwen_version or _DEFAULT_QWEN_VERSION

    vision_infos = extract_vision_info(conversations)
    
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(
                vision_info, 
                image_patch_size=image_patch_size,
                qwen_version=qwen_version
            ))
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video(
                vision_info,
                return_video_sample_fps=True,
                image_patch_size=image_patch_size,
                return_video_metadata=return_video_metadata,
                qwen_version=qwen_version,
            )
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should be in content.")
    
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None

    # Build video_kwargs based on version
    if qwen_version == "qwen3":
        video_kwargs = {'do_sample_frames': False, 'fps': video_sample_fps_list}
    else:
        video_kwargs = {'fps': video_sample_fps_list}

    if return_video_kwargs:
        return image_inputs, video_inputs, video_kwargs
    return image_inputs, video_inputs
