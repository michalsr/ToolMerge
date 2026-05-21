"""Rank-based merging of tool-scored frames.

Each tool call assigns a percentile score in [0, 1] to every frame (higher =
better). Frames from different tool calls are merged into a single per-frame
score using boolean operators:

    AND = min(scores)   # worst-ranked tool wins; frame must rank well on ALL
    OR  = max(scores)   # best-ranked tool wins; frame must rank well on ANY

The combine expression supports `Q1`, `Q2`, ... query IDs, the AND/OR
operators, and parentheses. AND binds tighter than OR.
"""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


# --- AST -----------------------------------------------------------------

class Node:
    pass


class Leaf(Node):
    __slots__ = ("query_id",)

    def __init__(self, query_id: str):
        self.query_id = query_id

    def __repr__(self) -> str:
        return self.query_id


class BinOp(Node):
    __slots__ = ("op", "left", "right")

    def __init__(self, op: str, left: Node, right: Node):
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


def parse_combine_expr(expr: str) -> Node:
    """Parse 'Q1 AND (Q2 OR Q3)' style strings into an AST.

    AND binds tighter than OR. Parentheses group sub-expressions. Tokens
    other than AND / OR / ( / ) are treated as query IDs.
    """
    tokens = tokenize(expr)
    pos = [0]
    return parse_or(tokens, pos)


def tokenize(expr: str) -> List[str]:
    tokens: List[str] = []
    s = expr.strip()
    i = 0
    while i < len(s):
        c = s[i]
        if c in ("(", ")"):
            tokens.append(c)
            i += 1
        elif c.isspace():
            i += 1
        else:
            j = i
            while j < len(s) and not s[j].isspace() and s[j] not in ("(", ")"):
                j += 1
            tokens.append(s[i:j])
            i = j
    return tokens


def parse_or(tokens: List[str], pos: List[int]) -> Node:
    left = parse_and(tokens, pos)
    while pos[0] < len(tokens) and tokens[pos[0]].upper() == "OR":
        pos[0] += 1
        right = parse_and(tokens, pos)
        left = BinOp("OR", left, right)
    return left


def parse_and(tokens: List[str], pos: List[int]) -> Node:
    left = parse_atom(tokens, pos)
    while pos[0] < len(tokens) and tokens[pos[0]].upper() == "AND":
        pos[0] += 1
        right = parse_atom(tokens, pos)
        left = BinOp("AND", left, right)
    return left


def parse_atom(tokens: List[str], pos: List[int]) -> Node:
    if pos[0] >= len(tokens):
        raise ValueError("Unexpected end of combine expression")
    tok = tokens[pos[0]]
    if tok == "(":
        pos[0] += 1
        node = parse_or(tokens, pos)
        if pos[0] < len(tokens) and tokens[pos[0]] == ")":
            pos[0] += 1
        return node
    pos[0] += 1
    return Leaf(tok)


# --- Score evaluator -----------------------------------------------------

def evaluate_combine_scores(
    node: Node,
    query_score_maps: Dict[str, Dict[int, float]],
) -> Dict[int, float]:
    """Walk the AST and combine per-query percentile scores.

    Frames present in only one branch of an operator get 0.0 on the missing
    side (worst rank), so AND naturally filters them out and OR keeps them.
    """
    if isinstance(node, Leaf):
        return dict(query_score_maps.get(node.query_id, {}))
    if isinstance(node, BinOp):
        left = evaluate_combine_scores(node.left, query_score_maps)
        right = evaluate_combine_scores(node.right, query_score_maps)
        all_indices = set(left) | set(right)
        out: Dict[int, float] = {}
        for idx in all_indices:
            ls = left.get(idx, 0.0)
            rs = right.get(idx, 0.0)
            if node.op == "AND":
                out[idx] = min(ls, rs)
            elif node.op == "OR":
                out[idx] = max(ls, rs)
        return out
    return {}


def combine_or_all(query_score_maps: Dict[str, Dict[int, float]]) -> Dict[int, float]:
    """OR-fallback over every query when no combine expression is provided."""
    out: Dict[int, float] = {}
    for sm in query_score_maps.values():
        for idx, s in sm.items():
            out[idx] = s if idx not in out else max(out[idx], s)
    return out


# --- OCR injection -------------------------------------------------------

def inject_ocr_frames(
    combined_scores: Dict[int, float],
    ocr_frames: List[int],
    fps: float,
    num_frames: int,
    max_final_k: int,
    ocr_pool_seconds: float,
    ocr_pool_cap_seconds: float = 10.0,
) -> Dict[int, float]:
    """Cluster OCR-confirmed frames and insert one per cluster at rank 1.0.

    OCR frames carry the highest possible score (1.0) so they're always
    kept by the downstream NMS — the LLM judge has already confirmed
    query relevance, so they're given priority over visual-similarity matches.

    Args:
        combined_scores: per-frame scores in [0, 1] from the tool merge.
        ocr_frames: frame indices whose extracted text passed the LLM judge.
        fps: video frame rate (target fps used during scoring).
        num_frames: total frames in the video at ``fps``.
        max_final_k: caller's final K (used for the auto pool window).
        ocr_pool_seconds: temporal pool window between OCR clusters. ``< 0`` =
            use ``min(duration / (2 * max_final_k), ocr_pool_cap_seconds)``.
        ocr_pool_cap_seconds: cap on the auto formula.

    Returns: the updated score map (mutated copy).
    """
    if not ocr_frames:
        return combined_scores

    if ocr_pool_seconds < 0:
        duration = num_frames / fps if fps > 0 else 0
        ocr_pool_seconds = min(duration / max(1, 2 * max_final_k), ocr_pool_cap_seconds)

    pool_gap = int(ocr_pool_seconds * fps)
    sorted_ocr = sorted(ocr_frames)

    clusters: List[List[int]] = [[sorted_ocr[0]]]
    for f in sorted_ocr[1:]:
        if f - clusters[-1][-1] <= pool_gap:
            clusters[-1].append(f)
        else:
            clusters.append([f])

    representatives = [c[len(c) // 2] for c in clusters]
    out = dict(combined_scores)
    for idx in representatives:
        out[idx] = 1.0

    logger.info(
        "  OCR: %d frames -> %d clusters (pool=%.1fs) -> %d total",
        len(ocr_frames),
        len(representatives),
        ocr_pool_seconds,
        len(out),
    )
    return out
