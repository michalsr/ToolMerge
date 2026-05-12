"""Greedy NMS smoke tests."""

import pytest

from toolmerge.selection import auto_tau_seconds, greedy_gap_select, ordered_by_time


def test_auto_tau_clamps_at_cap():
    # 1 hour video, K=8 -> D/(2K) = 3600/16 = 225 s -> clamped to cap=10.
    tau = auto_tau_seconds(num_frames=3600 * 2, fps=2.0, max_k=8, cap=10.0)
    assert tau == 10.0


def test_auto_tau_short_video():
    # 30 s video, K=8 -> D/(2K) = 30/16 = 1.875 s (under cap).
    tau = auto_tau_seconds(num_frames=60, fps=2.0, max_k=8, cap=10.0)
    assert tau == pytest.approx(1.875)


def test_auto_tau_zero_when_fps_or_k_zero():
    assert auto_tau_seconds(num_frames=100, fps=0, max_k=8) == 0.0
    assert auto_tau_seconds(num_frames=100, fps=2.0, max_k=0) == 0.0


def test_greedy_gap_select_respects_gap():
    # Frames 0, 1, 2, 3, 4 with descending scores; gap=2 forces every-other selection.
    scored = {0: 1.0, 1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6}
    selected = greedy_gap_select(scored, max_k=3, min_gap_frames=2)
    # Best-first: 0 picked, 1 rejected (gap), 2 picked, 3 rejected, 4 picked.
    assert set(selected.keys()) == {0, 2, 4}


def test_greedy_gap_select_no_gap():
    scored = {0: 1.0, 1: 0.9, 2: 0.8, 3: 0.7}
    selected = greedy_gap_select(scored, max_k=2, min_gap_frames=0)
    # Top-2 by score, in insertion order.
    assert list(selected.keys()) == [0, 1]


def test_greedy_gap_select_caps_at_max_k():
    scored = {i: 1.0 - i * 0.01 for i in range(100)}
    selected = greedy_gap_select(scored, max_k=4, min_gap_frames=0)
    assert len(selected) == 4


def test_ordered_by_time_sorts_indices():
    out = ordered_by_time({100: 0.3, 5: 0.9, 42: 0.5})
    assert out == [5, 42, 100]
