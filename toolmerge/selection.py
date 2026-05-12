"""Greedy NMS selection.

Picks up to ``max_k`` frames in descending score order, accepting each only
when it is at least ``τ`` seconds from every already-selected frame. ``τ``
defaults to ``min(D / (2 K), 10)`` per the paper.
"""

from __future__ import annotations

from typing import Dict, List


def auto_tau_seconds(num_frames: int, fps: float, max_k: int, cap: float = 10.0) -> float:
    """The paper's default ``τ = min(D / (2 K), cap)`` in seconds."""
    if fps <= 0 or max_k <= 0:
        return 0.0
    duration = num_frames / fps
    return min(duration / (2 * max_k), cap)


def greedy_gap_select(
    scored: Dict[int, float],
    max_k: int,
    min_gap_frames: int,
) -> Dict[int, float]:
    """Greedy top-K with a per-frame temporal gap constraint.

    Args:
        scored: ``{frame_idx: score}`` (higher = better).
        max_k: maximum number of frames to keep.
        min_gap_frames: minimum frame-index distance between any two
            selected frames. ``<= 0`` disables the gap constraint.

    Returns a dict in insertion (rank) order — best score first.
    """
    ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
    selected: Dict[int, float] = {}
    for idx, score in ranked:
        if len(selected) >= max_k:
            break
        if min_gap_frames <= 0 or all(abs(idx - s) >= min_gap_frames for s in selected):
            selected[idx] = score
    return selected


def select_pool(
    scored: Dict[int, float],
    pool_k: int,
    fps: float,
    num_frames: int,
    min_gap_seconds: float,
    gap_cap_seconds: float = 10.0,
) -> Dict[int, float]:
    """Greedy-gap select with the per-pool auto-τ rule.

    When ``min_gap_seconds < 0`` we recompute τ for *this* pool's K
    (``min(D / (2 pool_k), cap)``) — this matches the gatherer's
    pool-time behavior so smaller pools use a wider temporal spacing.
    """
    if min_gap_seconds < 0:
        gap_seconds = auto_tau_seconds(num_frames, fps, pool_k, gap_cap_seconds)
    else:
        gap_seconds = min_gap_seconds
    gap_frames = int(gap_seconds * fps) if gap_seconds > 0 else 0
    if len(scored) > pool_k:
        return greedy_gap_select(scored, pool_k, gap_frames)
    return dict(scored)


def ordered_by_time(selected: Dict[int, float]) -> List[int]:
    """Return selected frame indices sorted in temporal order for the answerer."""
    return sorted(selected.keys())
