"""Dataset for GRPO training of the evidence pipeline v2 planner.

Produces chat-formatted messages compatible with TRL's GRPOTrainer.
Extra columns (video_id, answer, options, etc.) are passed as kwargs
to reward functions.


"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

# Ensure project root is on path
PKG_PATH = str(Path(__file__).resolve().parent.parent.parent)
if PKG_PATH not in sys.path:
    sys.path.insert(0, PKG_PATH)

from toolmerge.pipeline import sample_uniform_frames
from toolmerge.planner import format_options
from toolmerge.prompts.planner import PLANNER_TEMPLATES, TEXT_ONLY_VERSIONS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colored log helpers
# ---------------------------------------------------------------------------

try:
    from rich.logging import RichHandler
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

_COLORS = {
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "cyan": "\033[96m",
    "reset": "\033[0m",
}


def color(text: str, color: str) -> str:
    return f"{_COLORS.get(color, '')}{text}{_COLORS['reset']}"


def setup_colored_logging(level: int = logging.INFO) -> None:
    """Set up colored logging. Uses rich if available, else ANSI codes."""
    if _RICH_AVAILABLE:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        )
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )


# ---------------------------------------------------------------------------
# Decord fallback
# ---------------------------------------------------------------------------

def parse_timestamp(ts) -> Optional[float]:
    """Parse 'HH:MM:SS(.mmm)' or numeric to seconds. Returns None on failure."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    if not isinstance(ts, str):
        return None
    parts = ts.strip().split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(ts)
    except (ValueError, TypeError):
        return None


def load_frames_cv2(video_path: str, fps: float = 2.0) -> Optional[torch.Tensor]:
    """Load video frames via OpenCV when frame cache is missing.

    Returns (N, C, H, W) uint8 tensor, or None on failure.
    """
    import cv2

    if not os.path.isfile(video_path):
        logger.warning(color(f"Video file not found: {video_path}", "yellow"))
        return None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(color(f"Cannot open video: {video_path}", "yellow"))
            return None

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0 or native_fps <= 0:
            logger.warning(color(f"Empty or invalid video: {video_path}", "yellow"))
            cap.release()
            return None

        # Compute indices at target fps
        duration = total_frames / native_fps
        n_sample = max(1, int(duration * fps))
        indices = torch.linspace(0, total_frames - 1, n_sample).round().long().tolist()

        frames_list = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # BGR -> RGB, then HWC -> CHW
                rgb = frame[:, :, ::-1].copy()
                frames_list.append(torch.from_numpy(rgb).permute(2, 0, 1))
        cap.release()

        if not frames_list:
            logger.warning(color(f"No frames read from {video_path}", "yellow"))
            return None

        logger.info(
            color(f"  Loaded {len(frames_list)} frames via cv2 from {Path(video_path).name}", "cyan")
        )
        return torch.stack(frames_list)
    except Exception as e:
        logger.error(color(f"Failed to load video via cv2: {video_path}: {e}", "red"))
        return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PlannerDataset(Dataset):
    """Dataset for planner GRPO training.

    Each item returns a dict with:
        - prompt: list of message dicts (chat format, possibly with images)
        - idx: original index
        - video_id: str
        - answer: str (ground truth)
        - options: dict
        - question: str
        - clip_start / clip_end: optional float
        - video_path: str (for decord fallback in reward)
    """

    def __init__(
        self,
        data_path: str,
        prompt_template: str = "v5",
        num_overview_frames: int = 16,
        frame_cache_dir: Optional[str] = None,
        video_dir: Optional[str] = None,
        fps: float = 2.0,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ):
        """
        Args:
            data_path: Path to JSON file with list of question dicts.
            prompt_template: Planner prompt version (v3, v4, v5, ...).
            num_overview_frames: Number of uniform frames for the planner.
                Must be 0 for text-only prompts (v4, v5).
            frame_cache_dir: Directory with pre-built frame caches.
            video_dir: Directory with raw video files (decord fallback).
            fps: Target FPS for frame caches / decord loading.
            start_idx: Start index in the data.
            end_idx: End index in the data (exclusive). None = all.
        """
        # Validate prompt / frames config
        is_text_only = prompt_template in TEXT_ONLY_VERSIONS
        if is_text_only and num_overview_frames > 0:
            raise ValueError(
                f"Prompt template '{prompt_template}' is text-only but "
                f"num_overview_frames={num_overview_frames} > 0. "
                f"Set num_overview_frames=0 for text-only prompts."
            )
        if not is_text_only and num_overview_frames == 0:
            logger.warning(
                color(
                    f"Prompt template '{prompt_template}' expects frames but "
                    f"num_overview_frames=0 — planner will receive no visual input.",
                    "yellow",
                )
            )

        self.prompt_template = prompt_template
        self.template_text = PLANNER_TEMPLATES[prompt_template]
        self.num_overview_frames = num_overview_frames
        self.is_text_only = is_text_only
        self.frame_cache_dir = frame_cache_dir
        self.video_dir = video_dir
        self.fps = fps

        # Load data
        with open(data_path, "r") as f:
            all_items = json.load(f)

        # Handle both list and dict-with-list formats
        if isinstance(all_items, dict):
            # Try common keys
            for key in ("data", "items", "questions", "train"):
                if key in all_items:
                    all_items = all_items[key]
                    break
            else:
                raise ValueError(
                    f"JSON is a dict with keys {list(all_items.keys())}, "
                    f"but none of ['data', 'items', 'questions', 'train'] found"
                )

        self.items = all_items[start_idx:end_idx]
        logger.info(
            color(
                f"PlannerDataset: {len(self.items)} items loaded "
                f"(prompt={prompt_template}, frames={num_overview_frames})",
                "green",
            )
        )

    def __len__(self) -> int:
        return len(self.items)

    def load_overview_frames(self, video_id: str) -> Optional[torch.Tensor]:
        """Load all video frames (for later uniform sampling) by decoding the mp4."""
        video_path = self.resolve_video_path(video_id)
        if video_path:
            return load_frames_cv2(video_path, fps=self.fps)
        logger.warning(
            color(f"  No frames available for {video_id} (no video file)", "red")
        )
        return None

    def resolve_video_path(self, video_id: str) -> Optional[str]:
        """Resolve video file path from video_dir + video_id."""
        if not self.video_dir:
            return None
        # Try with and without .mp4 extension
        for ext in [".mp4", ""]:
            path = os.path.join(self.video_dir, f"{video_id}{ext}")
            if os.path.isfile(path):
                return path
        return None

    def build_prompt_messages(
        self, question: str, options: dict, duration: str,
        frames: Optional[torch.Tensor] = None,
        timestamps: Optional[List[float]] = None,
    ) -> List[dict]:
        """Build chat messages for the planner, matching planner.plan_evidence format."""
        prompt_text = self.template_text.replace(
            "{question}", question,
        ).replace(
            "{options}", format_options(options),
        ).replace(
            "{duration}", str(duration),
        ).replace(
            "{fps}", str(int(self.fps) if self.fps == int(self.fps) else self.fps),
        )

        content = []
        if frames is not None and len(frames) > 0 and timestamps is not None:
            for t, frame in zip(timestamps, frames):
                content.append({"type": "text", "text": f"{t:.1f}s"})
                content.append({"type": "image", "image": frame})

        content.append({"type": "text", "text": prompt_text})
        return [{"role": "user", "content": content}]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.items[index]
        idx = item.get("idx", index)
        video_id = item["video_id"]
        question = item["question"]
        # Support both "choices" and "options" keys
        options = item.get("choices") or item.get("options") or {}
        answer = item.get("answer", "")

        # Load and sample overview frames if needed
        planner_frames = None
        planner_ts = None
        duration_str = "unknown"

        if not self.is_text_only and self.num_overview_frames > 0:
            all_frames = self.load_overview_frames(video_id)
            if all_frames is not None:
                planner_frames, planner_ts = sample_uniform_frames(
                    all_frames, self.fps, self.num_overview_frames
                )
                duration_str = f"{len(all_frames) / self.fps:.0f}s"
            else:
                logger.warning(
                    color(f"  [{idx}] No frames for {video_id}, planner gets text-only input", "yellow")
                )
                duration_str = item.get("duration", "unknown")
        else:
            # Text-only prompt — try to get duration from item metadata
            duration_str = item.get("duration", "unknown")

        messages = self.build_prompt_messages(
            question, options, duration_str,
            frames=planner_frames, timestamps=planner_ts,
        )

        video_path = self.resolve_video_path(video_id) or ""

        return {
            "prompt": messages,
            "idx": idx,
            "video_id": video_id,
            "answer": answer,
            "options": options,
            "question": question,
            "uid": item.get("uid") or item.get("question_id", ""),
            "clip_start": item.get("clip_start") if item.get("clip_start") is not None
                          else parse_timestamp(item.get("start")),
            "clip_end": item.get("clip_end") if item.get("clip_end") is not None
                        else parse_timestamp(item.get("end")),
            "video_path": video_path,
        }
