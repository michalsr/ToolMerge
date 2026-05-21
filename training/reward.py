"""Reward functions for GRPO training of the evidence pipeline v2 planner.

Reward functions compatible with TRL's GRPOTrainer:
    consistency_reward      — parseable JSON + combine IDs match query IDs
    accuracy_reward         — full pipeline (tools + frozen answerer) → P(correct)
    hit_at_k_reward         — 1.0 if any selected frame inside [clip_start, clip_end]
    frames_in_gt_reward     — count of selected timestamps inside [clip_start, clip_end]
                              (does NOT use the answerer VLM — only the frame-selection
                              backend; pair with FrameSelectionBackend in training)

Each follows TRL's callable signature:
    def reward_fn(prompts, completions, **kwargs) -> list[float]
"""

import logging
from functools import lru_cache
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_COLORS = {
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "cyan": "\033[96m",
    "magenta": "\033[95m",
    "reset": "\033[0m",
}


def color(text: str, color: str) -> str:
    return f"{_COLORS.get(color, '')}{text}{_COLORS['reset']}"


# ---------------------------------------------------------------------------
# 1. Consistency reward (parseable + combine IDs match query IDs)
# ---------------------------------------------------------------------------

def extract_combine_ids(combine_expr: str) -> set:
    """Extract all query IDs referenced in a combine expression (e.g. Q1, Q2)."""
    import re
    return set(re.findall(r'\bQ\d+\b', combine_expr))


@lru_cache(maxsize=2048)
def parse_completion_cached(text: str):
    """Memoise parse_planner_response so all three reward fns share one parse
    per unique completion text. LRU eviction keeps memory bounded."""
    from toolmerge.planner import parse_planner_response
    return parse_planner_response(text)


def is_malformed_plan(text: str) -> bool:
    """True if the planner emitted any query that got dropped during parsing
    (missing 'id' field, wrong structure, etc.). When True, all three reward
    functions return 0 / -1 — no partial credit for partially-valid plans.
    """
    raw = count_raw_queries_cached(text)
    valid, _, _ = parse_completion_cached(text)
    return raw > 0 and len(valid) < raw


@lru_cache(maxsize=2048)
def count_raw_queries_cached(text: str) -> int:
    """Return the raw length of the 'queries' list in the planner JSON, BEFORE
    filtering to valid entries. Compared against the filtered count to detect
    dropped queries (e.g. missing 'id' field) so consistency_reward can penalise
    malformed output that would otherwise silently shrink the plan.
    Returns 0 if the completion is unparseable — caller should use the normal
    parseability check for that case.
    """
    import re
    import json as _json
    json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not brace_match:
            return 0
        json_str = brace_match.group(0)
    try:
        data = _json.loads(json_str)
    except Exception:
        return 0
    qs = data.get("queries", []) or []
    return len(qs) if isinstance(qs, list) else 0


def consistency_reward(
    prompts: list,
    completions: list,
    **kwargs,
) -> list[float]:
    """Check planner output is parseable, all emitted queries are well-formed,
    and combine references only real query IDs.

    Returns:
       -1.0  — not parseable as JSON at all / no valid queries / any emitted
              query was dropped for missing id or bad structure (strict)
        0.0  — parseable but combine references hallucinated query IDs
        1.0  — parseable, every emitted query valid, combine consistent
    """
    rewards = []
    for i, completion in enumerate(completions):
        text = extract_text(completion)
        queries, combine_expr = parse_completion_cached(text)

        if not queries and not combine_expr:
            rewards.append(-1.0)
            logger.debug(color(f"  [consistency] #{i}: -1.0 (unparseable)", "red"))
            continue

        if not queries:
            rewards.append(-1.0)
            logger.debug(color(f"  [consistency] #{i}: -1.0 (no valid queries)", "red"))
            continue

        if is_malformed_plan(text):
            rewards.append(-1.0)
            raw = count_raw_queries_cached(text)
            logger.debug(color(
                f"  [consistency] #{i}: -1.0 (dropped {raw - len(queries)}/{raw} queries: missing id or bad structure)",
                "red",
            ))
            continue

        # Check combine references
        query_ids = {q.get("id", "") for q in queries}
        combine_ids = extract_combine_ids(combine_expr) if combine_expr else set()

        hallucinated = combine_ids - query_ids
        if hallucinated:
            rewards.append(0.0)
            logger.debug(
                color(
                    f"  [consistency] #{i}: 0.0 (hallucinated IDs in combine: "
                    f"{hallucinated}, have: {query_ids})",
                    "yellow",
                )
            )
        else:
            rewards.append(1.0)
            logger.debug(
                color(
                    f"  [consistency] #{i}: 1.0 ({len(queries)} queries, "
                    f"combine='{combine_expr}')",
                    "green",
                )
            )

    return rewards


# ---------------------------------------------------------------------------
# 5. Frames-in-GT reward (recall of selected timestamps inside [clip_start, clip_end])
# ---------------------------------------------------------------------------

# Cap from gather_evidence's auto-gap formula: min(duration / (max_final_k*2), 10.0).
# Any video > max_final_k * 2 * 10 = 160 s falls back to 10 s, which covers the
# entire group_v2 dataset. Used as fallback when the inference cfg sets
# min_frame_gap_seconds to -1 (auto).
_DEFAULT_GAP_CAP = 10.0
_DEFAULT_MAX_FINAL_K = 8


def resolve_gap_and_max_k(frame_backend: Any) -> tuple:
    """Read (min_frame_gap_seconds, max_final_k) off the frame backend's
    inference config, falling back to the gather_evidence cap defaults.
    Returns (gap_seconds: float, max_k: int).
    """
    cfg = getattr(frame_backend, "_inference_cfg", None)
    raw_gap = float(getattr(cfg, "min_frame_gap_seconds", -1)) if cfg is not None else -1.0
    gap = raw_gap if raw_gap > 0 else _DEFAULT_GAP_CAP
    max_k = int(getattr(cfg, "max_final_k", _DEFAULT_MAX_FINAL_K)) if cfg is not None else _DEFAULT_MAX_FINAL_K
    return gap, max_k


def make_frames_in_gt_reward(frame_backend: Any) -> Callable:
    """Recall reward — fraction of GT-clip frames the planner managed to land.

    Reward = count_inside / max_possible_inside, where:
        count_inside        = #(selected timestamps t with cs <= t <= ce)
        max_possible_inside = min(max_final_k, floor((ce - cs) / gap) + 1)

    The denominator accounts for the min_frame_gap constraint imposed by
    gather_evidence: with a fixed min spacing g, only floor(L / g) + 1
    timestamps can fit inside an interval of length L. Without this
    normalisation, short GT clips (10–20 s vs 10 s gap) cap at reward 1–2 and
    the within-group reward variance collapses to 0 → no GRPO gradient.

    Recall is bounded [0, 1] and removes the clip-length bias from the count
    formulation. Does NOT use the answerer VLM. Pass a ``FrameSelectionBackend``
    as ``frame_backend``; it only needs ``evaluate_plan(...) -> {"timestamps": [...]}``.

    Returns 0.0 for malformed plans, missing GT segments, or pipeline failures.
    """

    gap, max_k = resolve_gap_and_max_k(frame_backend)
    logger.info(color(
        f"[frames_in_gt] recall reward configured: gap={gap:.1f}s, max_final_k={max_k}",
        "cyan",
    ))

    def frames_in_gt_reward(
        prompts: list,
        completions: list,
        video_id: Optional[list] = None,
        uid: Optional[list] = None,
        clip_start: Optional[list] = None,
        clip_end: Optional[list] = None,
        **kwargs,
    ) -> list[float]:
        if video_id is None or clip_start is None or clip_end is None:
            logger.warning(color(
                "[frames_in_gt] Missing video_id / clip_start / clip_end", "red"
            ))
            return [0.0] * len(completions)

        if hasattr(frame_backend, "begin_tren_gpu_session"):
            frame_backend.begin_tren_gpu_session()

        rewards = []
        try:
            for i, completion in enumerate(completions):
                vid = video_id[i]
                cs = clip_start[i]
                ce = clip_end[i]
                item_uid = uid[i] if uid else ""

                if cs is None or ce is None:
                    rewards.append(0.0)
                    logger.debug(color(
                        f"  [frames_in_gt] #{i} ({vid}): 0.0 (no GT segment)", "yellow"
                    ))
                    continue

                text = extract_text(completion)
                queries, combine_expr = parse_completion_cached(text)
                if not queries:
                    rewards.append(0.0)
                    logger.debug(color(
                        f"  [frames_in_gt] #{i} ({vid}): 0.0 (no queries)", "yellow"
                    ))
                    continue

                if is_malformed_plan(text):
                    rewards.append(0.0)
                    logger.debug(color(
                        f"  [frames_in_gt] #{i} ({vid}): 0.0 (malformed plan)", "yellow"
                    ))
                    continue

                try:
                    result = frame_backend.evaluate_plan(
                        queries=queries,
                        combine_expr=combine_expr,
                        video_id=vid,
                        uid=item_uid,
                    )
                except Exception as e:
                    logger.warning(color(
                        f"  [frames_in_gt] #{i} ({vid}): error: {e}", "red"
                    ))
                    rewards.append(0.0)
                    continue

                timestamps = (result or {}).get("timestamps") or []
                count = sum(1 for t in timestamps if cs <= float(t) <= ce)

                clip_len = max(0.0, float(ce) - float(cs))
                # floor(L/g) + 1 timestamps fit inside an interval of length L
                # under the min-gap constraint; cap at max_final_k since the
                # planner can't emit more than that anyway. max(1, ...) guards
                # against a 0-length clip (count==0 → reward 0 either way).
                max_possible = max(1, min(max_k, int(clip_len / gap) + 1))
                # min(1.0, ...) guards against off-by-one slop from frame
                # timestamp quantization (e.g., 2-fps frame placement).
                recall = min(1.0, count / max_possible)
                rewards.append(float(recall))
                logger.debug(color(
                    f"  [frames_in_gt] #{i} ({vid}): {recall:.3f} = {count}/{max_possible} "
                    f"({len(timestamps)} picks, GT [{cs:.1f},{ce:.1f}] L={clip_len:.1f}s)",
                    "green" if recall > 0 else "magenta",
                ))
        finally:
            if hasattr(frame_backend, "end_tren_gpu_session"):
                frame_backend.end_tren_gpu_session()

        return rewards

    return frames_in_gt_reward


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def extract_text(completion) -> str:
    """Extract text from a completion (handles both str and message-list formats)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # TRL passes completions as list of message dicts
        parts = []
        for msg in completion:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            parts.append(c.get("text", ""))
        return "\n".join(parts)
    return str(completion)
