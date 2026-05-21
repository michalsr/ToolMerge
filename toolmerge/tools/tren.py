"""T-REN client wrapping the ``tren`` package's ``QuerySearch`` model.

Splits ``QuerySearch.forward()`` into:

1. ``encode_video(frames)`` — DINOv3 + RegionEncoder over every frame.
   Expensive; cache the output per (video, fps) tuple to disk.
2. ``get_frame_scores(cache, query)`` — text encode + similarity over the
   cached track tokens. Cheap; runs per query.

Per-frame scores have shape ``(T,)``, matching SigLIP-2, so downstream code
treats both tools the same. T-REN model code lives in the top-level
``tren/`` package; weights are downloaded separately via
``scripts/download_tren_weights.sh``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.nn.functional as F
import yaml

logger = logging.getLogger(__name__)

# Process-wide model cache.
_query_search_cache: Optional[dict] = None


def default_config_path() -> str:
    # Co-located with the top-level ``tren`` package.
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "tren", "video_query_search", "config.yaml"))


class TrenClient:
    """Drop-in replacement for ``SiglipClient`` with two-phase scoring."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        lazy_init: bool = False,
    ):
        self.config_path = os.path.abspath(config_path or default_config_path())
        self.checkpoint_override = checkpoint_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._query_search = None
        self._config = None
        self._similarity_threshold: Optional[float] = None
        self._current_device: Optional[str] = None

        if not lazy_init:
            self.ensure_loaded()

    def load_config(self):
        if self._config is None:
            with open(self.config_path) as f:
                self._config = yaml.safe_load(f)
        return self._config

    def ensure_loaded(self):
        global _query_search_cache

        if self._query_search is not None:
            return

        if _query_search_cache is not None:
            self._query_search = _query_search_cache["query_search"]
            self._config = _query_search_cache["config"]
            self._similarity_threshold = self._config["parameters"]["similarity_threshold"]
            return

        config = self.load_config()
        if self.checkpoint_override:
            config["tren"]["logging"]["save_dir"] = os.path.dirname(
                os.path.dirname(self.checkpoint_override)
            )
            config["tren"]["logging"]["exp_name"] = os.path.basename(
                os.path.dirname(self.checkpoint_override)
            )

        # Top-level T-REN package.
        from tren.video_query_search.models import QuerySearch  # noqa: E501  pylint: disable=import-outside-toplevel

        logger.info("Loading T-REN QuerySearch from config: %s", self.config_path)
        self._query_search = QuerySearch(config)
        self._query_search.eval()

        # Default to CUDA (matches paper-run launchers). Override with TREN_DEVICE=cpu.
        tren_device = os.environ.get("TREN_DEVICE", "cuda").lower()
        if tren_device == "cpu":
            self._query_search.cpu()
            for mod in self._query_search.modules():
                if hasattr(mod, "device") and isinstance(mod.device, str):
                    mod.device = "cpu"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._current_device = "cpu"
        else:
            self._query_search.to(tren_device)
            for mod in self._query_search.modules():
                if hasattr(mod, "device") and isinstance(mod.device, str):
                    mod.device = tren_device
            self._current_device = tren_device

        self._similarity_threshold = config["parameters"]["similarity_threshold"]
        _query_search_cache = {"query_search": self._query_search, "config": config}
        logger.info(
            "T-REN loaded (similarity_threshold=%.2f, device=%s)",
            self._similarity_threshold, self._current_device,
        )

    def move_to(self, device: str) -> None:
        device = device.lower()
        if self._query_search is None or self._current_device == device:
            return
        if device == "cpu":
            self._query_search.cpu()
            for mod in self._query_search.modules():
                if hasattr(mod, "device") and isinstance(mod.device, str):
                    mod.device = "cpu"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            self._query_search.to(device)
            for mod in self._query_search.modules():
                if hasattr(mod, "device") and isinstance(mod.device, str):
                    mod.device = device
        self._current_device = device

    def encode_video(self, frames, batch_size: int = 32) -> dict:
        """Pre-compute step: returns a dict of cacheable track tokens."""
        self.ensure_loaded()
        if isinstance(frames, torch.Tensor):
            frames_np = frames.permute(0, 2, 3, 1).byte().cpu().numpy()
            frames_list = [frames_np[i] for i in range(len(frames_np))]
        else:
            frames_list = frames

        with torch.inference_mode():
            track_results, compression = self._query_search.video_ren(frames_list, batch_size=batch_size)

        n_tracks = track_results["track_text_aligned_tokens"].shape[0]
        logger.info(
            "T-REN: %d tracks from %d frames (patches %.1fx, regions %.1fx)",
            n_tracks, len(frames_list),
            compression["from_patches"], compression["from_regions"],
        )

        return {
            "track_text_aligned_tokens": track_results["track_text_aligned_tokens"].cpu().half(),
            "track_pred_tokens": track_results["track_pred_tokens"].cpu().half(),
            "track_members": track_results["track_members"],
            "num_frames": len(frames_list),
        }

    def encode_video_per_frame(self, frames, batch_size: int = 32) -> dict:
        """Per-frame variant: one set of region tokens per frame (no aggregation)."""
        self.ensure_loaded()
        if isinstance(frames, torch.Tensor):
            frames_np = frames.permute(0, 2, 3, 1).byte().cpu().numpy()
            frames_list = [frames_np[i] for i in range(len(frames_np))]
        else:
            frames_list = frames

        video_ren = self._query_search.video_ren
        T = len(frames_list)
        per_frame_tokens = []
        total_regions = 0

        with torch.inference_mode():
            for start in range(0, T, batch_size):
                end = min(T, start + batch_size)
                frame_batch = torch.stack(
                    [video_ren.transform(f) for f in frames_list[start:end]]
                ).to(next(video_ren.parameters()).device)

                feature_maps = video_ren.tren_image_encoder(frame_batch)["feature_maps"]
                grid_points = [video_ren.grid_points for _ in range(frame_batch.shape[0])]
                tren_outputs = video_ren.tren_region_encoder(
                    feature_maps, grid_points, aggregate_tokens=True
                )
                text_aligned_tokens = tren_outputs["text_aligned_tokens"]

                for frame_idx in range(frame_batch.shape[0]):
                    tokens = text_aligned_tokens[frame_idx].cpu().half()
                    per_frame_tokens.append(tokens)
                    total_regions += tokens.shape[0]

                del frame_batch, feature_maps, tren_outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        logger.info(
            "T-REN per-frame: %d regions from %d frames (avg %.1f/frame)",
            total_regions, T, total_regions / max(T, 1),
        )

        return {
            "per_frame_text_aligned_tokens": per_frame_tokens,
            "num_frames": T,
            "cache_mode": "per_frame",
        }

    def get_frame_scores(self, tren_cache: dict, query: str) -> torch.Tensor:
        """Query-time scoring: ``(T,)`` similarity tensor for ``query``."""
        self.ensure_loaded()

        if tren_cache.get("cache_mode") == "per_frame" or "per_frame_text_aligned_tokens" in tren_cache:
            return self.scores_per_frame(tren_cache, query)

        return self.scores_tracked(tren_cache, query)

    def scores_tracked(self, tren_cache: dict, query: str) -> torch.Tensor:
        track_text_aligned = tren_cache["track_text_aligned_tokens"].float()
        track_members = tren_cache["track_members"]
        num_frames = tren_cache["num_frames"]

        with torch.inference_mode():
            text_query_tokens = self._query_search.text_query_encoder(query)
            similarity = self._query_search.find_similarity(
                track_text_aligned.to(text_query_tokens.device),
                text_query_tokens,
            )

        similar_track_idxs = torch.where(similarity >= self._similarity_threshold)[0]
        if len(similar_track_idxs) == 0:
            # Fallback: most-similar single track.
            flat_idx = similarity.argmax()
            most_similar_track_idx = flat_idx // similarity.shape[1]
            similar_track_idxs = [most_similar_track_idx]

        track_sims = similarity.max(dim=-1).values

        frame_scores = torch.zeros(num_frames, dtype=torch.float32)
        for idx in similar_track_idxs:
            idx_v = idx.item() if isinstance(idx, torch.Tensor) else idx
            score = track_sims[idx_v].item()
            for frame_id, _ in track_members[idx_v]:
                if frame_id < num_frames:
                    frame_scores[frame_id] = max(frame_scores[frame_id].item(), score)
        return frame_scores

    def scores_per_frame(self, tren_cache: dict, query: str) -> torch.Tensor:
        per_frame_tokens = tren_cache["per_frame_text_aligned_tokens"]
        num_frames = tren_cache["num_frames"]

        with torch.inference_mode():
            text_query_tokens = self._query_search.text_query_encoder(query)
            query_norm = F.normalize(text_query_tokens, p=2, dim=-1)

        frame_scores = torch.zeros(num_frames, dtype=torch.float32)
        for i, tokens in enumerate(per_frame_tokens):
            if tokens.numel() == 0:
                continue
            tokens_dev = tokens.float().to(query_norm.device)
            tokens_norm = F.normalize(tokens_dev, p=2, dim=-1)
            sim = torch.matmul(tokens_norm, query_norm.reshape(-1))
            frame_scores[i] = sim.max().item()
        return frame_scores
