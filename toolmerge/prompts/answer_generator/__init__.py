"""Answerer prompts.

Two templates are shipped:
- ``lif`` — default; used by every paper row EXCEPT Qwen3-VL 8f rows.
- ``v1``  — used by Qwen3-VL 8f rows only (Table 2 LVB Qwen3 8f, Table 3 M2M QA Qwen3 8f).
"""

from toolmerge.prompts.answer_generator.lif import ANSWERER_LIF
from toolmerge.prompts.answer_generator.v1 import ANSWERER_V1

ANSWER_TEMPLATES = {
    "lif": ANSWERER_LIF,
    "v1": ANSWERER_V1,
}

__all__ = ["ANSWER_TEMPLATES", "ANSWERER_LIF", "ANSWERER_V1"]
