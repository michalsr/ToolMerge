"""GRPO post-training for the ToolMerge planner.

Layout:
    train.py              -- entry point (``python -m training.train``)
    grpo_trainer.py       -- TRL GRPO wrapper
    reward.py             -- frames_in_gt + consistency reward functions
    planner_dataset.py    -- training dataset wrapping the M2M JSON
    frame_selection_backend.py -- runs ``gather_evidence`` at reward time
    config_schema.py      -- PlannerTrainingConfig dataclass
    data/                 -- GRPO training subset JSON
    configs/              -- m2m_grpo.yaml + reward-time inference YAML
"""
