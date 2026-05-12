"""Shared frame-loading helpers used by every per-modality cache builder."""

from __future__ import annotations

import ctypes
import logging
from typing import Iterator, List

import numpy as np
import torch


TARGET_FPS = 2.0
FRAME_FACTOR = 2          # Qwen convention: nframes must be divisible by 2
LOAD_CHUNK_SIZE = 256     # Decode at most this many frames at a time


# glibc malloc_trim: return free heap pages to OS (fights pymalloc arena fragmentation).
try:
    libc = ctypes.CDLL("libc.so.6")
    def malloc_trim():
        libc.malloc_trim(0)
except Exception:
    def malloc_trim():
        pass


def floor_by_factor(n, factor):
    return int(n // factor) * factor


class CV2Reader:
    """Minimal VideoReader-like wrapper around OpenCV (works on ARM/aarch64)."""

    def __init__(self, video_path: str):
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.path = video_path

    def __len__(self):
        return self.n_frames

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def get_batch(self, indices):
        """Load frames at the given indices. Returns list of (H, W, C) uint8 RGB numpy arrays."""
        cv2 = self.cv2
        cap = self.cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame {idx} from {self.path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frames

    def iter_chunks(self, indices, chunk_size=LOAD_CHUNK_SIZE) -> Iterator[List[np.ndarray]]:
        """Yield chunks of frames by reading the video sequentially once.

        Much faster than `get_batch` for cache building, where indices are sorted
        and we want all of them. Reads forward, keeping only requested frames.
        """
        cv2 = self.cv2
        cap = self.cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx_set = set(indices)
        max_idx = max(indices)
        buf = []
        collected = 0
        cur = 0
        while cur <= max_idx:
            ret = cap.grab()
            if not ret:
                if collected == 0:
                    raise RuntimeError(
                        f"Failed to read any frames from {self.path} (codec may be unsupported)"
                    )
                break
            if cur in idx_set:
                ret2, frame = cap.retrieve()
                if not ret2:
                    raise RuntimeError(f"Failed to retrieve frame {cur} from {self.path}")
                buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                collected += 1
                if len(buf) == chunk_size:
                    yield buf
                    buf = []
            cur += 1
        if buf:
            yield buf


def get_frame_indices(video_path: str, target_fps: float = TARGET_FPS,
                      max_nframes: int = None, backend: str = "decord"):
    """Choose which frame indices to sample. Returns (indices, n, reader)."""
    if backend == "pyav":
        backend = "cv2"

    if backend == "cv2":
        reader = CV2Reader(video_path)
    else:
        try:
            import decord
            reader = decord.VideoReader(video_path)
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"decord failed on {video_path}: {e}, falling back to cv2"
            )
            backend = "cv2"
            reader = CV2Reader(video_path)

    total_frames = len(reader)
    video_fps = reader.get_avg_fps() if backend == "decord" else reader.fps
    if total_frames == 0:
        raise ValueError(f"Video has no frames: {video_path}")

    nframes = total_frames / video_fps * target_fps
    nframes = min(nframes, total_frames)
    if max_nframes is not None:
        nframes = min(nframes, max_nframes)
    nframes = floor_by_factor(nframes, FRAME_FACTOR)
    nframes = max(int(nframes), FRAME_FACTOR)

    frame_idx = (
        torch.linspace(0, total_frames - 1, nframes)
        .round().long().clamp(0, total_frames - 1).tolist()
    )
    return frame_idx, nframes, reader


def load_chunk(vr, indices):
    """Load a chunk of frames from a decord VideoReader. Returns (T, C, H, W) float32."""
    batch = vr.get_batch(indices)
    if isinstance(batch, list):
        arr = np.stack(batch)
    elif hasattr(batch, "asnumpy"):
        arr = batch.asnumpy()
    else:
        arr = batch.numpy()
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float()


def frames_to_tensor(frames_list):
    """List of (H, W, C) uint8 RGB numpy arrays -> (N, C, H, W) float32 tensor."""
    arr = np.stack(frames_list)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float()


class LazyFrameList:
    """List-like wrapper that loads frames from a VideoReader on demand."""

    def __init__(self, vr, frame_idx):
        self.vr = vr
        self.frame_idx = frame_idx

    def __len__(self):
        return len(self.frame_idx)

    def __getitem__(self, key):
        if isinstance(key, slice):
            indices = self.frame_idx[key]
            batch = self.vr.get_batch(indices)
            if isinstance(batch, list):
                return batch
            arr = batch.asnumpy() if hasattr(batch, "asnumpy") else batch.numpy()
            return [arr[i] for i in range(arr.shape[0])]
        idx = self.frame_idx[key]
        if isinstance(self.vr, CV2Reader):
            return self.vr.get_batch([idx])[0]
        frame = self.vr[idx]
        return frame.asnumpy() if hasattr(frame, "asnumpy") else frame.numpy()
