"""WFS baseline.

Wavelet-based Frame Selection (CVPR 2026). We compute the per-frame relevance
curve as raw SigLIP-2 cosine similarity between the
``question + concatenated options`` query and the cached frame embeddings,
then run the upstream WFS algorithm: wavelet event detection + budget
allocation + MMR selection within each detected segment.

Standalone: no imports from ``toolmerge``. The
``WFSEventDetector`` / ``WFSBudgetAllocator`` / ``WFSFrameSelector`` /
``WFS.select_keyframes`` classes are copied verbatim from upstream WFS-SB
(https://github.com/MAC-AutoML/WFS-SB) at ``wfs/core.py``.

Usage:
    python -m baselines.wfs.run config=configs/lvb/qwen3_8.yaml
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pywt
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.signal import find_peaks
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# Paper defaults from WFS-SB/configs/wfs_defaults.yaml.
WFS_DEFAULTS = dict(
    wavelet="db4",
    drift_level=3,
    height_factor=0.5,
    prominence_factor=0.05,
    min_distance_ratio=0.02,
    min_distance_absolute=5,
    w_duration=0.4,
    w_mean=0.2,
    w_max=0.3,
    w_var=0.1,
    strictness_factor=1.2,
    temperature=1.0,
    lambda_param=0.5,
)


# --------------------------- config loader (OmegaConf) ---------------------------

def load_config_from_cli() -> Any:
    config_path: Optional[str] = None
    overrides: List[str] = []
    for arg in sys.argv[1:]:
        if arg.startswith("config="):
            config_path = arg.split("=", 1)[1]
        elif "=" in arg:
            overrides.append(arg)
    if not config_path:
        raise SystemExit("usage: python -m baselines.wfs.run config=<yaml> [k=v ...]")

    def _load(p: str):
        cfg = OmegaConf.load(p)
        defaults = cfg.pop("defaults", None) if hasattr(cfg, "pop") else None
        if not defaults:
            return cfg
        merged = OmegaConf.create({})
        here = Path(p).resolve().parent
        for entry in defaults:
            parent = (here / f"{entry}.yaml").resolve()
            if parent.exists():
                merged = OmegaConf.merge(merged, _load(str(parent)))
        return OmegaConf.merge(merged, cfg)

    cfg = _load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


# --------------------------- SigLIP-2 text encoder ---------------------------

_TEXT_MODEL = None
_TEXT_PROCESSOR = None
_TEXT_DEVICE: Optional[str] = None


def encode_text(query: str, model_name: Optional[str] = None) -> torch.Tensor:
    global _TEXT_MODEL, _TEXT_PROCESSOR, _TEXT_DEVICE
    if _TEXT_MODEL is None:
        from transformers import AutoModel, AutoProcessor
        name = model_name or os.environ.get(
            "SIGLIP_MODEL", "google/siglip2-giant-opt-patch16-384"
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading SigLIP-2 text encoder %s on %s", name, device)
        _TEXT_PROCESSOR = AutoProcessor.from_pretrained(name)
        attn = "sdpa" if device.startswith("cuda") else "eager"
        _TEXT_MODEL = AutoModel.from_pretrained(name, attn_implementation=attn).eval().to(device)
        _TEXT_DEVICE = device

    inputs = _TEXT_PROCESSOR(
        text=[query], return_tensors="pt",
        padding="max_length", truncation=True, max_length=64,
    )
    input_ids = inputs["input_ids"].to(_TEXT_DEVICE)
    with torch.no_grad():
        f = _TEXT_MODEL.get_text_features(input_ids=input_ids)
    f = f / f.norm(dim=-1, keepdim=True)
    return f.squeeze(0).cpu()


# --------------------------- SigLIP-2 frame cache I/O ---------------------------

_SIGLIP_EXTS = [".feature_cache_qwen3vl", ".mp4.feature_cache_qwen3vl"]


def find_siglip_cache(cache_dir: str, video_id: str) -> Optional[str]:
    for ext in _SIGLIP_EXTS:
        p = os.path.join(cache_dir, f"{video_id}{ext}")
        if os.path.exists(p):
            return p
    return None


def load_siglip_embeddings(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for key in ("embedding", "embeddings", "features", "frame_embeddings"):
            if key in obj and isinstance(obj[key], torch.Tensor):
                return obj[key]
        tensors = [v for v in obj.values() if isinstance(v, torch.Tensor)]
        if len(tensors) == 1:
            return tensors[0]
    raise ValueError(f"Unrecognized SigLIP cache at {path}")


# --------------------------- dataset I/O ---------------------------

def load_dataset_items(path: str, start_idx: int = 0, end_idx: Optional[int] = None) -> List[dict]:
    with open(path) as f:
        data = json.load(f)
    return data[start_idx:end_idx] if end_idx is not None else data[start_idx:]


def item_uid(item: dict) -> str:
    return item.get("uid") or item.get("question_id") or ""


# --------------------------- query + raw cosine ---------------------------

def build_query(item: dict) -> str:
    opts = item.get("options") or {}
    if isinstance(opts, dict):
        opts_text = " ".join(opts[k] for k in sorted(opts.keys()))
    else:
        opts_text = " ".join(opts)
    return item["question"] + " " + opts_text


def siglip_cosine(query: str, embeddings: torch.Tensor) -> np.ndarray:
    text_feat = encode_text(query)
    text_feat = F.normalize(text_feat.unsqueeze(0), p=2, dim=1).squeeze(0)
    emb = F.normalize(embeddings.float(), p=2, dim=1)
    return (emb @ text_feat).detach().cpu().numpy().astype(float)


# --------------------------- selector: WFS wavelet + MMR ---------------------------
# Copied verbatim from https://github.com/MAC-AutoML/WFS-SB/blob/main/wfs/core.py.

class WFSEventDetector:
    def __init__(self, wavelet: str = "db4", mode: str = "symmetric",
                 height_factor: float = 0.5, prominence_factor: float = 0.05) -> None:
        self.wavelet = wavelet
        self.mode = mode
        self.height_factor = height_factor
        self.prominence_factor = prominence_factor

    def decompose(self, relevance_scores: np.ndarray, level: int) -> List[np.ndarray]:
        return pywt.wavedec(relevance_scores, self.wavelet, level=level, mode=self.mode)

    def reconstruct_detail(self, coeffs: Sequence[np.ndarray], target_length: int,
                           detail_level: int = 1) -> np.ndarray:
        coeffs_zero = [np.zeros_like(c) for c in coeffs]
        detail_level = int(np.clip(detail_level, 1, len(coeffs_zero) - 1))
        coeffs_zero[detail_level] = coeffs[detail_level]
        detail_signal = pywt.waverec(coeffs_zero, self.wavelet, mode=self.mode)
        return detail_signal[:target_length]

    def detect_peaks(self, detail_signal: np.ndarray, min_distance: int) -> np.ndarray:
        abs_detail = np.abs(detail_signal)
        if len(abs_detail) == 0:
            return np.array([], dtype=int)
        mean_val = np.mean(abs_detail)
        std_val = np.std(abs_detail)
        height_threshold = mean_val + self.height_factor * std_val
        signal_range = np.max(abs_detail) - np.min(abs_detail)
        prominence_threshold = self.prominence_factor * signal_range
        peaks, _ = find_peaks(
            abs_detail,
            height=height_threshold,
            distance=max(1, int(min_distance)),
            prominence=prominence_threshold,
        )
        return peaks

    @staticmethod
    def create_segments(peaks: Sequence[int], total_frames: int) -> List[Tuple[int, int]]:
        boundaries = [0] + list(peaks) + [total_frames]
        return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


class WFSBudgetAllocator:
    def __init__(self, w_duration: float = 0.3, w_mean: float = 0.4,
                 w_max: float = 0.2, w_var: float = 0.1,
                 strictness_factor: float = 1.5, temperature: float = 1.0,
                 min_importance: float = 0.05) -> None:
        self.w_duration = w_duration
        self.w_mean = w_mean
        self.w_max = w_max
        self.w_var = w_var
        self.strictness_factor = strictness_factor
        self.temperature = temperature
        self.min_importance = min_importance

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x / max(self.temperature, 1e-8)
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def compute_importance(self, segment: Tuple[int, int],
                           relevance_scores: np.ndarray, total_frames: int) -> float:
        start, end = segment
        segment_scores = relevance_scores[start:end]
        if len(segment_scores) == 0:
            return 0.0
        duration_norm = (end - start) / max(total_frames, 1)
        mean_score = float(np.mean(segment_scores))
        max_score = float(np.max(segment_scores))
        var_segment = float(np.var(segment_scores))
        var_global = float(np.var(relevance_scores))
        var_norm = var_segment / (var_global + 1e-8)
        return (
            self.w_duration * duration_norm
            + self.w_mean * mean_score
            + self.w_max * max_score
            + self.w_var * var_norm
        )

    def filter_segments(self, segments: Sequence[Tuple[int, int]],
                        importance_scores: Sequence[float]) -> Tuple[List[Tuple[int, int]], List[float]]:
        if len(segments) <= 3:
            return list(segments), list(importance_scores)
        mean_imp = float(np.mean(importance_scores))
        std_imp = float(np.std(importance_scores))
        adaptive_threshold = mean_imp - self.strictness_factor * std_imp
        threshold = max(self.min_importance, adaptive_threshold)
        kept_segments: List[Tuple[int, int]] = []
        kept_scores: List[float] = []
        for segment, score in zip(segments, importance_scores):
            if score >= threshold:
                kept_segments.append(segment)
                kept_scores.append(score)
        if not kept_segments:
            return list(segments), list(importance_scores)
        return kept_segments, kept_scores

    def allocate_budget(self, importance_scores: Sequence[float], total_budget: int) -> Dict[int, int]:
        if total_budget <= 0 or len(importance_scores) == 0:
            return {}
        proportions = self._softmax(np.asarray(importance_scores, dtype=float))
        raw_alloc = proportions * total_budget
        int_alloc = np.floor(raw_alloc).astype(int)
        allocation = {i: int(int_alloc[i]) for i in range(len(int_alloc))}
        remaining = int(total_budget - sum(allocation.values()))
        fractions = raw_alloc - int_alloc
        sorted_indices = np.argsort(fractions)[::-1]
        for i in range(remaining):
            allocation[int(sorted_indices[i])] += 1
        return allocation


class WFSFrameSelector:
    def __init__(self, lambda_param: float = 0.5) -> None:
        self.lambda_param = lambda_param

    def select_from_segment(self, segment: Tuple[int, int], n_frames: int,
                            relevance_scores: np.ndarray,
                            features: Optional[np.ndarray] = None) -> List[int]:
        start, end = segment
        if end <= start or n_frames <= 0:
            return []
        segment_scores = relevance_scores[start:end]
        if len(segment_scores) == 0:
            return []
        anchor_local = int(np.argmax(segment_scores))
        anchor_global = start + anchor_local
        selected = [anchor_global]
        candidates = [start + i for i in range(len(segment_scores)) if i != anchor_local]
        while len(selected) < n_frames and candidates:
            mmr_scores = []
            for idx in candidates:
                relevance = relevance_scores[idx]
                if features is not None:
                    frame_feat = features[idx: idx + 1]
                    selected_feats = features[selected]
                    max_sim = float(np.max(cosine_similarity(frame_feat, selected_feats)[0]))
                else:
                    min_dist = min(abs(idx - s) for s in selected)
                    max_sim = float(np.exp(-min_dist / 10.0))
                mmr = self.lambda_param * relevance - (1.0 - self.lambda_param) * max_sim
                mmr_scores.append((idx, mmr))
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected.append(best_idx)
            candidates.remove(best_idx)
        return selected

    def adjust_to_budget(self, selected_frames: Sequence[int], target_budget: int,
                         relevance_scores: np.ndarray,
                         features: Optional[np.ndarray] = None) -> List[int]:
        frames = sorted(set(int(i) for i in selected_frames))
        if target_budget <= 0:
            return []
        if len(frames) == target_budget:
            return frames
        if len(frames) < target_budget:
            candidates = [i for i in range(len(relevance_scores)) if i not in frames]
            while len(frames) < target_budget and candidates:
                mmr_scores = []
                for idx in candidates:
                    relevance = relevance_scores[idx]
                    if features is not None and len(frames) > 0:
                        frame_feat = features[idx: idx + 1]
                        selected_feats = features[frames]
                        max_sim = float(np.max(cosine_similarity(frame_feat, selected_feats)[0]))
                    elif len(frames) > 0:
                        min_dist = min(abs(idx - f) for f in frames)
                        max_sim = float(np.exp(-min_dist / 10.0))
                    else:
                        max_sim = 0.0
                    mmr = self.lambda_param * relevance - (1.0 - self.lambda_param) * max_sim
                    mmr_scores.append((idx, mmr))
                best_idx = max(mmr_scores, key=lambda x: x[1])[0]
                frames.append(best_idx)
                frames = sorted(frames)
                candidates.remove(best_idx)
        else:
            scored = [(f, relevance_scores[f]) for f in frames]
            scored.sort(key=lambda x: x[1], reverse=True)
            frames = sorted([f for f, _ in scored[:target_budget]])
        return frames[:target_budget]


@dataclass
class WFSConfig:
    wavelet: str = "db4"
    lambda_param: float = 0.5
    prominence_factor: float = 0.05
    height_factor: float = 0.5
    w_duration: float = 0.3
    w_mean: float = 0.4
    w_max: float = 0.2
    w_var: float = 0.1
    strictness_factor: float = 1.5
    temperature: float = 1.0


class WFS:
    def __init__(self, config: Optional[WFSConfig] = None) -> None:
        self.config = config or WFSConfig()
        self.event_detector = WFSEventDetector(
            wavelet=self.config.wavelet,
            height_factor=self.config.height_factor,
            prominence_factor=self.config.prominence_factor,
        )
        self.budget_allocator = WFSBudgetAllocator(
            w_duration=self.config.w_duration,
            w_mean=self.config.w_mean,
            w_max=self.config.w_max,
            w_var=self.config.w_var,
            strictness_factor=self.config.strictness_factor,
            temperature=self.config.temperature,
        )
        self.frame_selector = WFSFrameSelector(lambda_param=self.config.lambda_param)

    def select_keyframes(self, relevance_scores: np.ndarray, num_frames: int,
                         dwt_level: int, min_peak_distance: int,
                         features: Optional[np.ndarray] = None) -> List[int]:
        total_frames = len(relevance_scores)
        if total_frames == 0:
            return []
        if total_frames < num_frames:
            return list(range(total_frames))

        coeffs = self.event_detector.decompose(relevance_scores, dwt_level)
        detail_signal = self.event_detector.reconstruct_detail(coeffs, total_frames)
        peaks = self.event_detector.detect_peaks(detail_signal, min_peak_distance)
        segments = self.event_detector.create_segments(peaks, total_frames)

        # WAKS-style fallback when no peaks are detected.
        if len(segments) <= 1 or len(peaks) == 0:
            n_uniform = num_frames // 2
            n_top = num_frames - n_uniform
            uniform_indices = np.linspace(0, total_frames - 1, n_uniform, dtype=int)
            top_indices = np.argsort(relevance_scores)[::-1]
            top_selected: List[int] = []
            for idx in top_indices:
                if idx not in uniform_indices:
                    top_selected.append(int(idx))
                    if len(top_selected) >= n_top:
                        break
            merged = sorted(list(uniform_indices) + top_selected)
            return [int(i) for i in merged[:num_frames]]

        importance_scores = [
            self.budget_allocator.compute_importance(seg, relevance_scores, total_frames)
            for seg in segments
        ]
        valid_segments, valid_scores = self.budget_allocator.filter_segments(segments, importance_scores)
        allocation = self.budget_allocator.allocate_budget(valid_scores, num_frames)

        # Map scores from [0, 1] to [-1, 1] before MMR so negative relevance penalizes.
        normalized_scores = 2.0 * relevance_scores - 1.0
        selected: List[int] = []
        for seg_idx, n_alloc in allocation.items():
            seg = valid_segments[seg_idx]
            selected.extend(
                self.frame_selector.select_from_segment(
                    segment=seg, n_frames=n_alloc,
                    relevance_scores=normalized_scores, features=features,
                )
            )
        return self.frame_selector.adjust_to_budget(
            selected_frames=selected, target_budget=num_frames,
            relevance_scores=normalized_scores, features=features,
        )


def compute_dwt_level(num_frames: int, wavelet: str = "db4", drift: int = 3) -> int:
    if num_frames <= 1:
        return 1
    wavelet_obj = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(num_frames, wavelet_obj.dec_len)
    if max_level <= 1:
        return 1
    raw_level = int(np.floor(np.log2(num_frames) - drift))
    return int(np.clip(raw_level, 1, max_level))


def compute_min_peak_distance(num_frames: int, ratio: float = 0.02, absolute_min: int = 5) -> int:
    return max(int(absolute_min), int(num_frames * ratio))


# --------------------------- per-item driver ---------------------------

def run_one(item: dict, cache_dir: str, k: int, wfs_model: WFS) -> Dict[str, Any]:
    video_id = item["video_id"]
    uid = item_uid(item)
    cache_path = find_siglip_cache(cache_dir, video_id)
    if cache_path is None:
        raise FileNotFoundError(f"SigLIP cache missing for {video_id} in {cache_dir}")
    embeddings = load_siglip_embeddings(cache_path)
    num_frames = embeddings.shape[0]
    fps = 2.0

    relevance = siglip_cosine(build_query(item), embeddings)
    if relevance.shape[0] > num_frames:
        relevance = relevance[:num_frames]
    elif relevance.shape[0] < num_frames:
        pad = np.full(num_frames - relevance.shape[0], float(relevance.min()))
        relevance = np.concatenate([relevance, pad])

    dwt_level = compute_dwt_level(
        num_frames, wavelet=WFS_DEFAULTS["wavelet"], drift=WFS_DEFAULTS["drift_level"],
    )
    min_peak_distance = compute_min_peak_distance(
        num_frames,
        ratio=WFS_DEFAULTS["min_distance_ratio"],
        absolute_min=WFS_DEFAULTS["min_distance_absolute"],
    )
    features = embeddings.detach().cpu().float().numpy()

    indices = wfs_model.select_keyframes(
        relevance_scores=relevance, num_frames=k,
        dwt_level=dwt_level, min_peak_distance=min_peak_distance,
        features=features,
    )
    indices = sorted(int(i) for i in indices)
    timestamps = [idx / fps for idx in indices]

    return {
        "uid": uid,
        "video_id": video_id,
        "question": item["question"],
        "options": item["options"],
        "ground_truth": item.get("answer"),
        "frames_used": indices,
        "timestamps_used": timestamps,
    }


def main():
    logging.basicConfig(level="INFO", format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    cfg = load_config_from_cli()
    save_dir = str(cfg.data.save_path)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    items = load_dataset_items(
        str(cfg.data.input_path),
        int(cfg.data.get("start_idx") or 0),
        cfg.data.get("end_idx"),
    )
    cache_dir = str(cfg.siglip_feature_cache_dir)
    k = int(cfg.max_final_k)

    wfs_cfg = WFSConfig(
        wavelet=WFS_DEFAULTS["wavelet"],
        lambda_param=WFS_DEFAULTS["lambda_param"],
        prominence_factor=WFS_DEFAULTS["prominence_factor"],
        height_factor=WFS_DEFAULTS["height_factor"],
        w_duration=WFS_DEFAULTS["w_duration"],
        w_mean=WFS_DEFAULTS["w_mean"],
        w_max=WFS_DEFAULTS["w_max"],
        w_var=WFS_DEFAULTS["w_var"],
        strictness_factor=WFS_DEFAULTS["strictness_factor"],
        temperature=WFS_DEFAULTS["temperature"],
    )
    wfs_model = WFS(wfs_cfg)

    logger.info("WFS on %d items, K=%d, cache=%s", len(items), k, cache_dir)
    results: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, item in enumerate(items):
        try:
            r = run_one(item, cache_dir, k, wfs_model)
            results.append(r)
            logger.info(
                "[%d/%d] %s uid=%s -> %d frames",
                i + 1, len(items), item["video_id"], r["uid"], len(r["frames_used"]),
            )
        except Exception as e:  # noqa: BLE001
            logger.error("  Error on %s: %s", item.get("video_id"), e, exc_info=True)

    out_path = os.path.join(save_dir, "keyframes.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Wrote %d keyframes to %s (%.0fs)", len(results), out_path, time.time() - t0)


if __name__ == "__main__":
    main()
