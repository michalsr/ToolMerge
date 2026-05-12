"""LLM planner for ToolMerge.

Reads the question + answer choices + available tools + video duration and
emits independent SigLIP / T-REN tool calls combined under AND/OR. The
planner is text-only (no input frames). Ships one prompt template
(``v7_no_temporal``).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch

from toolmerge.prompts.planner import PLANNER_TEMPLATES

logger = logging.getLogger(__name__)


@dataclass
class PlannerCfg:
    """Lightweight config passed to the backend for a single planner call."""
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.8
    top_k: int = 20
    do_sample: bool = False


def format_options(options: dict) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in options.items())


def parse_planner_response(response: str) -> Tuple[List[dict], str]:
    """Extract the JSON block (queries + combine expression) from a planner reply.

    Returns ``(queries, combine_expr)``. Both are empty on parse failure.
    """
    json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        brace_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not brace_match:
            logger.warning("Could not find JSON in planner response: %s", response[:200])
            return [], ""
        json_str = brace_match.group(0)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse planner JSON: %s", e)
        return [], ""

    queries = data.get("queries", [])
    combine_expr = data.get("combine", "")

    valid: List[dict] = []
    for q in queries:
        if isinstance(q, dict) and "tool" in q and "query" in q and "id" in q:
            valid.append({"tool": q["tool"], "query": q["query"], "id": q["id"]})
        else:
            logger.warning("Skipping invalid query: %s", q)

    return valid, combine_expr


def plan_evidence(
    question: str,
    options: dict,
    available_tools: List[str],
    duration: str,
    backend: Any,
    cfg: Any,
) -> Tuple[List[dict], str, dict]:
    """Run the planner LLM. Returns ``(queries, combine_expr, debug_info)``."""
    template = PLANNER_TEMPLATES.get(cfg.planner_prompt)
    if template is None:
        raise KeyError(
            f"Unknown planner prompt '{cfg.planner_prompt}'. "
            f"Available: {list(PLANNER_TEMPLATES)}"
        )

    fps = getattr(cfg, "_planner_fps", 2.0)
    prompt_text = (
        template.replace("{question}", question)
        .replace("{options}", format_options(options))
        .replace("{duration}", str(duration))
        .replace("{fps}", str(int(fps) if fps == int(fps) else fps))
    )

    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
    ]

    planner_cfg = PlannerCfg(
        max_new_tokens=cfg.planner_max_new_tokens,
        temperature=cfg.planner_temperature,
        top_p=cfg.planner_top_p,
        top_k=cfg.planner_top_k,
        do_sample=cfg.planner_do_sample,
    )

    response = backend.generate_text(messages, planner_cfg)
    queries, combine_expr = parse_planner_response(response)

    # Filter to tools we can actually run on this video.
    allowed = set(available_tools)
    filtered = [q for q in queries if q["tool"] in allowed]
    if len(filtered) < len(queries):
        dropped = [q for q in queries if q["tool"] not in allowed]
        logger.warning("Dropped queries for unavailable tools: %s", dropped)

    debug = {
        "prompt": prompt_text,
        "raw_response": response,
        "parsed_queries": queries,
        "combine_expr": combine_expr,
        "filtered_queries": filtered,
    }
    return filtered, combine_expr, debug
