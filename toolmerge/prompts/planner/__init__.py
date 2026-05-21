"""Planner prompts.

Only one prompt is included: ``v7_no_temporal``. This is the prompt printed in
the paper's Appendix F and used for **all** inference (including caption
retrieval). The other v* / planner_* variants from the research tree are not
included.
"""

from toolmerge.prompts.planner.v7_no_temporal import PLANNER_V7_NO_TEMPORAL

PLANNER_TEMPLATES = {
    "v7_no_temporal": PLANNER_V7_NO_TEMPORAL,
}

# Prompts that don't consume input frames (planner runs text-only).
# Used by the GRPO trainer to decide whether to feed overview frames.
TEXT_ONLY_VERSIONS = {"v7_no_temporal"}

__all__ = ["PLANNER_TEMPLATES", "PLANNER_V7_NO_TEMPORAL", "TEXT_ONLY_VERSIONS"]
