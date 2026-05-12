"""Smoke tests for the AND/OR rank merging.

These pin the behavior described in Section 3 ("Merging") and Figure 2 of
the paper.
"""

from toolmerge.merging import (
    combine_or_all,
    evaluate_combine_scores,
    inject_ocr_frames,
    parse_combine_expr,
)


def rank_to_score(rank: int, n: int = 8) -> float:
    """Linear inverse: rank 1 -> 1.0, rank n -> 0.0."""
    return (n - rank) / (n - 1)


def test_parse_simple_leaf():
    node = parse_combine_expr("Q1")
    assert str(node) == "Q1"


def test_parse_and_or_precedence():
    # AND binds tighter than OR.
    node = parse_combine_expr("Q1 OR Q2 AND Q3")
    # Expect: Q1 OR (Q2 AND Q3) -> (Q1 OR (Q2 AND Q3))
    assert str(node) == "(Q1 OR (Q2 AND Q3))"


def test_parse_parens():
    node = parse_combine_expr("(Q1 AND Q2) OR Q3")
    assert str(node) == "((Q1 AND Q2) OR Q3)"


def test_evaluate_and_min_or_max():
    """AND = min, OR = max — matches the paper's "worst rank wins / best rank wins"."""
    maps = {
        "Q1": {0: 0.9, 1: 0.1, 2: 0.5},
        "Q2": {0: 0.2, 1: 0.8, 2: 0.6},
    }
    and_node = parse_combine_expr("Q1 AND Q2")
    or_node = parse_combine_expr("Q1 OR Q2")
    assert evaluate_combine_scores(and_node, maps) == {0: 0.2, 1: 0.1, 2: 0.5}
    assert evaluate_combine_scores(or_node, maps) == {0: 0.9, 1: 0.8, 2: 0.6}


def test_evaluate_compound_expression():
    maps = {
        "Q1": {0: 1.0, 1: 0.0},
        "Q2": {0: 0.0, 1: 1.0},
        "Q3": {0: 0.5, 1: 0.5},
    }
    # Q1 AND (Q2 OR Q3):
    #   inner OR: max(Q2, Q3) -> {0: 0.5, 1: 1.0}
    #   outer AND with Q1:    -> {0: min(1.0, 0.5)=0.5, 1: min(0.0, 1.0)=0.0}
    expected = {0: 0.5, 1: 0.0}
    assert evaluate_combine_scores(parse_combine_expr("Q1 AND (Q2 OR Q3)"), maps) == expected


def test_figure2_example():
    """Paper Figure 2: SigLIP "ziplining over river" + T-REN "man" + T-REN "canyon".

    With raw ranks (lower = better in the figure), AND picks the worst rank
    per frame. In our percentile-score space, this is min — and the frame
    rankings (best to worst by combined score) must match the figure.
    """
    # Figure 2 ranks for f1, f2, f3 in 8-frame pool:
    #   SigLIP "ziplining over river":   7, 8, 2
    #   T-REN "man":                      3, 1, 4
    #   T-REN "canyon":                   1, 2, 5
    maps = {
        "Q1": {0: rank_to_score(7), 1: rank_to_score(8), 2: rank_to_score(2)},
        "Q2": {0: rank_to_score(3), 1: rank_to_score(1), 2: rank_to_score(4)},
        "Q3": {0: rank_to_score(1), 1: rank_to_score(2), 2: rank_to_score(5)},
    }
    out = evaluate_combine_scores(parse_combine_expr("Q1 AND Q2 AND Q3"), maps)
    # Expected per-frame worst ranks: f1=7, f2=8, f3=5
    # In percentile space (rank -> (n-r)/(n-1)) that's:
    expected_worst_ranks = {0: 7, 1: 8, 2: 5}
    for idx, expected_rank in expected_worst_ranks.items():
        assert abs(out[idx] - rank_to_score(expected_rank)) < 1e-6, (
            f"frame {idx}: expected rank {expected_rank}, got score {out[idx]}"
        )

    # And OR: best rank per frame is 1, 1, 2 -> percentile 1.0, 1.0, ~0.857
    out_or = evaluate_combine_scores(parse_combine_expr("Q1 OR Q2 OR Q3"), maps)
    assert abs(out_or[0] - 1.0) < 1e-6
    assert abs(out_or[1] - 1.0) < 1e-6
    assert abs(out_or[2] - rank_to_score(2)) < 1e-6


def test_combine_or_all_fallback():
    maps = {"Q1": {0: 0.3, 1: 0.4}, "Q2": {0: 0.7}}
    out = combine_or_all(maps)
    assert out == {0: 0.7, 1: 0.4}


def test_different_combine_expressions_yield_different_rankings():
    """Smoke check that the parser + evaluator actually distinguish AND vs OR
    vs compound expressions — they shouldn't collapse to the same output.

    Constructs three queries with disjoint top-scorers and walks through the
    common combine expressions; each one should pick a different best frame.
    """
    maps = {
        # Q1's best frame is 0; Q2's best is 1; Q3's best is 2. Cross-query
        # scores are intentionally non-zero so AND has something to work with.
        "Q1": {0: 1.0, 1: 0.4, 2: 0.3},
        "Q2": {0: 0.3, 1: 1.0, 2: 0.5},
        "Q3": {0: 0.2, 1: 0.5, 2: 1.0},
    }

    def best(expr):
        out = evaluate_combine_scores(parse_combine_expr(expr), maps)
        return max(out.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    # Top-scoring frame must depend on which queries are in the expression.
    assert best("Q1") == 0
    assert best("Q2") == 1
    assert best("Q3") == 2

    # AND vs OR over the same queries must NOT collapse to the same result.
    and_all = evaluate_combine_scores(parse_combine_expr("Q1 AND Q2 AND Q3"), maps)
    or_all = evaluate_combine_scores(parse_combine_expr("Q1 OR Q2 OR Q3"), maps)
    assert and_all != or_all, "AND and OR collapsed to identical scores"

    # AND scores <= every constituent's score (worst tool dominates).
    for idx, s in and_all.items():
        assert s <= maps["Q1"][idx]
        assert s <= maps["Q2"][idx]
        assert s <= maps["Q3"][idx]

    # OR scores >= every constituent's score (best tool dominates).
    for idx, s in or_all.items():
        assert s >= maps["Q1"][idx]
        assert s >= maps["Q2"][idx]
        assert s >= maps["Q3"][idx]

    # Compound expressions must be distinguishable from each other.
    a = evaluate_combine_scores(parse_combine_expr("Q1 AND (Q2 OR Q3)"), maps)
    b = evaluate_combine_scores(parse_combine_expr("(Q1 AND Q2) OR Q3"), maps)
    c = evaluate_combine_scores(parse_combine_expr("Q1 OR (Q2 AND Q3)"), maps)
    assert not (a == b == c), "all three compound forms collapsed to the same result"


def test_combine_is_commutative():
    """A AND B == B AND A; A OR B == B OR A."""
    maps = {
        "Q1": {0: 0.5, 1: 0.7},
        "Q2": {0: 0.9, 1: 0.2},
    }
    assert evaluate_combine_scores(parse_combine_expr("Q1 AND Q2"), maps) \
        == evaluate_combine_scores(parse_combine_expr("Q2 AND Q1"), maps)
    assert evaluate_combine_scores(parse_combine_expr("Q1 OR Q2"), maps) \
        == evaluate_combine_scores(parse_combine_expr("Q2 OR Q1"), maps)


def test_paren_grouping_changes_result():
    """Parentheses must actually re-precedence the expression."""
    maps = {
        "Q1": {0: 0.9, 1: 0.1},
        "Q2": {0: 0.1, 1: 0.9},
        "Q3": {0: 0.5, 1: 0.5},
    }
    # Default precedence (AND tighter than OR): "Q1 AND Q2 OR Q3" == "(Q1 AND Q2) OR Q3".
    #   inner AND -> {0:0.1, 1:0.1}; then OR with Q3 -> {0:0.5, 1:0.5}
    default = evaluate_combine_scores(parse_combine_expr("Q1 AND Q2 OR Q3"), maps)
    # Re-grouped: "Q1 AND (Q2 OR Q3)":
    #   inner OR  -> {0:0.5, 1:0.9}; then AND with Q1 -> {0:0.5, 1:0.1}
    paren = evaluate_combine_scores(parse_combine_expr("Q1 AND (Q2 OR Q3)"), maps)
    assert default != paren, "parens didn't re-group the expression"
    assert default == {0: 0.5, 1: 0.5}
    assert paren == {0: 0.5, 1: 0.1}


def test_inject_ocr_frames_clusters_and_promotes_to_one():
    # 3 OCR frames clustered within τ; should collapse to one representative at rank 1.
    combined = {10: 0.5, 100: 0.5}
    ocr_frames = [50, 51, 52]   # all within a few frames of each other
    fps = 2.0
    num_frames = 200
    out = inject_ocr_frames(
        combined, ocr_frames, fps=fps, num_frames=num_frames,
        max_final_k=8, ocr_pool_seconds=10.0,
    )
    # The cluster's median frame (51) should be in the output at score 1.0.
    assert 51 in out
    assert out[51] == 1.0
    # Original scores preserved.
    assert out[10] == 0.5
    assert out[100] == 0.5
