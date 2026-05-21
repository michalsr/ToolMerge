"""Pipeline orchestration for ToolMerge.

Flow per question:
    1. Run the planner (text-only) to get tool calls + AND/OR combine.
    2. Score every frame with each tool, normalize to percentile ranks.
    3. Read OCR-judge frames from the per-question cache
       (built offline by ``cache_build/ocr_judge.py``; a miss skips OCR).
    4. Combine per-query scores via the AND/OR expression; inject OCR frames
       at rank 1; per-pool greedy NMS with τ = min(D/(2K), 10).
    5. Pass the top-K frames (in temporal order) to the answerer.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from toolmerge.merging import (
    combine_or_all,
    evaluate_combine_scores,
    inject_ocr_frames,
    parse_combine_expr,
)
from toolmerge.planner import plan_evidence
from toolmerge.selection import greedy_gap_select, ordered_by_time, select_pool, auto_tau_seconds
from toolmerge.tools.ocr_judge import load_judge_cache
from toolmerge.tools.scoring import score_siglip, score_tren
from toolmerge.answerer import generate_answer

logger = logging.getLogger(__name__)


# --- Shared helpers (also used by training/) -----------------------------

def sample_uniform_frames(frames_all, fps, n_frames):
    """Uniformly sample ``n_frames`` from ``frames_all`` (or ``None`` if no frames).

    Returns ``(frames_tensor, timestamps_list)``.
    """
    if n_frames <= 0 or frames_all is None or len(frames_all) == 0:
        return None, None
    total = len(frames_all)
    indices = torch.linspace(0, total - 1, min(n_frames, total)).long().tolist()
    return frames_all[indices], [idx / fps for idx in indices]


def gather_evidence(
    queries_with_ids,
    combine_expr,
    video_caches,
    cfg,
    ocr_frames=None,
):
    """Score queries → combine → inject OCR → select frames.

    Returns ``(frames_tensor or None, timestamps, debug)``.
    """
    fps = video_caches["fps"]
    frames_all = video_caches.get("frames")
    num_frames = video_caches.get("num_frames") or (
        frames_all.shape[0] if frames_all is not None else 0
    )

    per_query, per_query_debug = score_queries(queries_with_ids, video_caches, num_frames)
    debug: dict = {"per_query": per_query_debug, "combine_expr": combine_expr}

    if not per_query:
        debug["selected_frames"] = []
        debug["selected_timestamps"] = []
        return None, [], debug

    combined = combine(per_query, combine_expr)
    if ocr_frames:
        combined = inject_ocr_frames(
            combined, ocr_frames, fps, num_frames,
            cfg.max_final_k, getattr(cfg, "ocr_pool_seconds", -1.0),
            getattr(cfg, "min_frame_gap_cap_seconds", 10.0),
        )

    indices, timestamps, sel_debug = select_frames(combined, fps, num_frames, cfg)
    debug.update(sel_debug)

    if frames_all is not None and indices:
        return frames_all[indices], timestamps, debug
    return None, timestamps, debug


# --- Helpers -------------------------------------------------------------

def check_available_tools(video_caches: dict, cfg: Any) -> List[str]:
    """Return the subset of enabled tools that actually have a cache loaded."""
    available: List[str] = []
    if "siglip" in cfg.enabled_tools and video_caches.get("siglip_embeddings") is not None:
        available.append("siglip")
    if "tren" in cfg.enabled_tools and video_caches.get("tren_cache") is not None:
        available.append("tren")
    return available


# --- Per-question pipeline ----------------------------------------------

def score_queries(
    queries: List[dict], video_caches: dict, num_frames: int,
) -> Tuple[Dict[str, Dict[int, float]], List[dict]]:
    """Run each query through its tool and collect percentile score maps."""
    per_query_scores: Dict[str, Dict[int, float]] = {}
    per_query_debug: List[dict] = []

    for q in queries:
        tool = q["tool"]
        text = q["query"]
        qid = q["id"]
        logger.info("  Scoring %s: tool=%s, query=%r", qid, tool, text[:60])

        results = []
        if tool == "siglip" and video_caches.get("siglip_embeddings") is not None:
            results = score_siglip(
                text,
                {"client": video_caches["siglip_client"],
                 "embeddings": video_caches["siglip_embeddings"]},
                0, num_frames,
            )
        elif tool == "tren" and video_caches.get("tren_cache") is not None:
            results = score_tren(
                text, video_caches["tren_cache"], 0, num_frames,
                video_caches["tren_client"],
            )
        else:
            logger.warning("  Tool %r unavailable for %s, skipping", tool, qid)
            continue

        per_query_scores[qid] = {idx: score for idx, score in results}
        per_query_debug.append({
            "query_id": qid,
            "tool": tool,
            "query": text,
            "total_scored": len(results),
            "top_frames": [(idx, round(s, 4)) for idx, s in results[:10]],
        })

    return per_query_scores, per_query_debug


def combine(
    per_query: Dict[str, Dict[int, float]],
    combine_expr: str,
) -> Dict[int, float]:
    """Resolve the combine expression into a single per-frame score map."""
    if not per_query:
        return {}
    if len(per_query) == 1:
        return dict(next(iter(per_query.values())))
    if not combine_expr or not combine_expr.strip():
        return combine_or_all(per_query)
    try:
        ast = parse_combine_expr(combine_expr)
        return evaluate_combine_scores(ast, per_query)
    except Exception as e:  # noqa: BLE001
        logger.warning("Combine parse failed (%r): falling back to OR-all", e)
        return combine_or_all(per_query)


def select_frames(
    combined_scores: Dict[int, float],
    fps: float,
    num_frames: int,
    cfg: Any,
) -> Tuple[List[int], List[float], Dict[str, Any]]:
    """Apply per-pool greedy NMS and return the top-K final frames."""
    selection: Dict[str, Any] = {}

    selection["all_scores"] = [
        (idx, round(s, 4)) for idx, s in sorted(combined_scores.items())
    ]

    if not combined_scores:
        return [], [], selection

    pool_k_values = sorted(getattr(cfg, "pool_k_values", [8, 16, 32, 64]))
    min_gap_sec = getattr(cfg, "min_frame_gap_seconds", -1.0)
    gap_cap = getattr(cfg, "min_frame_gap_cap_seconds", 10.0)
    for pk in pool_k_values:
        pooled = select_pool(combined_scores, pk, fps, num_frames, min_gap_sec, gap_cap)
        ranked = sorted(pooled.items(), key=lambda x: x[1], reverse=True)
        selection[f"pooled_candidates_{pk}"] = [(idx, round(s, 4)) for idx, s in ranked]

    final_k = cfg.max_final_k
    final_key = f"pooled_candidates_{final_k}"
    if final_key in selection:
        selected = {idx: s for idx, s in selection[final_key]}
    else:
        if min_gap_sec < 0:
            gap_sec = auto_tau_seconds(num_frames, fps, final_k, gap_cap)
        else:
            gap_sec = min_gap_sec
        gap_frames = int(gap_sec * fps) if gap_sec > 0 else 0
        if gap_frames > 0 and len(combined_scores) > final_k:
            selected = greedy_gap_select(combined_scores, final_k, gap_frames)
        else:
            top_k = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:final_k]
            selected = dict(top_k)

    indices = ordered_by_time(selected)
    timestamps = [idx / fps for idx in indices]
    selection["selected_frames"] = indices
    selection["selected_timestamps"] = timestamps
    selection["selected_scores"] = [round(selected[idx], 4) for idx in indices]
    return indices, timestamps, selection


def run_pipeline(
    question: str,
    options: dict,
    video_caches: dict,
    backend: Any,
    cfg: Any,
    planner_backend: Optional[Any] = None,
    uid: str = "",
    extract_frames=None,
) -> Dict[str, Any]:
    """Run ToolMerge on one question.

    Args:
        question, options: the question text and options dict (``A..E``).
        video_caches: ``toolmerge.caches.caches_for_video`` output.
        backend: answerer backend.
        cfg: ToolMergeConfig.
        planner_backend: backend for the planner. ``None`` = use ``backend``.
        uid: question UID for OCR-judge cache lookup.
        extract_frames: callable ``(video_path, indices, fps) -> tensor`` used
            when no pre-loaded frame cache is available. Provided by the CLI.

    Returns: ``{answer, confidence, status, trace, frames_used, timestamps_used,
              answer_prompt, answer_raw}``.

    OCR-judge frames are read from the pre-built cache pointed to by
    ``cfg.ocr_judge_cache_dir``; the pipeline never invokes an LLM for OCR
    judging at inference time. Build the cache with
    ``python -m cache_build.build_caches --tools ocr_judge ...``.
    """
    fps = video_caches["fps"]
    frames_all = video_caches.get("frames")
    num_frames = video_caches.get("num_frames") or (
        frames_all.shape[0] if frames_all is not None else 0
    )
    if num_frames <= 0:
        raise ValueError(
            f"Cannot determine num_frames (uid={uid!r}); ensure SigLIP/T-REN/OCR caches exist for this video."
        )
    duration = f"{num_frames / fps:.0f}"

    available_tools = check_available_tools(video_caches, cfg)
    logger.info(
        "  Video: %d frames, %.1f fps, duration=%s, tools=%s",
        num_frames, fps, duration, available_tools,
    )

    cfg._planner_fps = fps  # used by the planner prompt formatting

    # 1. Planner
    plan_be = planner_backend or backend
    queries, combine_expr, planner_debug = plan_evidence(
        question, options, available_tools, duration, plan_be, cfg,
    )
    logger.info(
        "  Planner: %d queries, combine=%r", len(queries), combine_expr,
    )
    for q in queries:
        logger.info("    %s: [%s] %s", q["id"], q["tool"], q["query"][:80])

    if not queries:
        return {
            "answer": None,
            "confidence": 0.0,
            "status": "no_queries",
            "trace": {"planner": planner_debug},
            "frames_used": [],
            "timestamps_used": [],
        }

    # 2. OCR judge — read pre-built cache only (no LLM call at inference).
    ocr_frames: List[int] = []
    ocr_debug: Dict[str, Any] = {}
    ocr_cache = video_caches.get("ocr_cache")
    ocr_judge_cache_dir = getattr(cfg, "ocr_judge_cache_dir", "")
    if ocr_cache is not None and ocr_judge_cache_dir and uid:
        current_fps = float(ocr_cache.get("fps", fps)) if isinstance(ocr_cache, dict) else fps
        cached = load_judge_cache(ocr_judge_cache_dir, uid, current_fps=current_fps)
        if cached is None:
            logger.info("  OCR judge: no cache for %s (skipping)", uid)
        else:
            ocr_frames = cached
            ocr_debug = {"ocr_frames": ocr_frames, "num_ocr_frames": len(ocr_frames)}
            logger.info("  OCR judge: %d relevant frames (cache hit)", len(ocr_frames))

    # 3. Score each query through its tool.
    per_query_scores, per_query_debug = score_queries(queries, video_caches, num_frames)

    # 4. Combine + inject OCR + select.
    combined = combine(per_query_scores, combine_expr)
    if ocr_frames:
        combined = inject_ocr_frames(
            combined, ocr_frames, fps, num_frames,
            cfg.max_final_k, getattr(cfg, "ocr_pool_seconds", -1.0),
            getattr(cfg, "min_frame_gap_cap_seconds", 10.0),
        )

    if not combined:
        return {
            "answer": None,
            "confidence": 0.0,
            "status": "no_scores",
            "trace": {"planner": planner_debug, "ocr": ocr_debug, "per_query": per_query_debug},
            "frames_used": [],
            "timestamps_used": [],
        }

    indices, timestamps, sel_debug = select_frames(combined, fps, num_frames, cfg)
    logger.info(
        "  Selected %d frames, ts=%s",
        len(indices), [f"{t:.1f}s" for t in timestamps],
    )

    if not indices:
        return {
            "answer": None,
            "confidence": 0.0,
            "status": "no_frames",
            "trace": {"planner": planner_debug, "ocr": ocr_debug,
                      "per_query": per_query_debug, "selection": sel_debug},
            "frames_used": [],
            "timestamps_used": [],
        }

    # Retrieval-only mode: no answer choices => skip pixel extraction + answerer.
    if not options:
        return {
            "answer": None,
            "confidence": 0.0,
            "status": "retrieval_only",
            "trace": {"planner": planner_debug, "ocr": ocr_debug,
                      "per_query": per_query_debug, "selection": sel_debug},
            "frames_used": indices,
            "timestamps_used": timestamps,
        }

    # 5. Get the actual pixel frames for the answerer.
    if frames_all is not None:
        frames = frames_all[indices]
    elif extract_frames is not None and video_caches.get("video_path"):
        frames = extract_frames(video_caches["video_path"], indices, fps)
    else:
        raise RuntimeError(
            "Selected frames but no pixel data available: provide a frame cache or "
            "pass `extract_frames` to run_pipeline."
        )

    answer_result = generate_answer(frames, timestamps, question, options, backend, cfg.answer_generator)

    trace = {
        "planner": planner_debug,
        "ocr": ocr_debug,
        "per_query": per_query_debug,
        "selection": sel_debug,
        "answerer_num_frames": len(timestamps),
        "answerer_timestamps": list(timestamps),
    }
    return {
        "answer": answer_result["answer"],
        "confidence": answer_result["confidence"],
        "status": "answered",
        "trace": trace,
        "answer_prompt": answer_result.get("prompt", ""),
        "answer_raw": answer_result["raw_response"],
        "frames_used": indices,
        "timestamps_used": timestamps,
    }
