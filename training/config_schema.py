"""Training-only config dataclasses (TRL GRPO trainer).

Kept separate from ``toolmerge.config`` so users who only run inference don't
pay the TRL import cost. Mirrors ``PlannerTrainingConfig`` from the research
tree (``time_r1/config/base.py``), trimmed to the fields used by
``training/train_planner.py``.
"""

from dataclasses import dataclass, field
from typing import Any, Dict

from toolmerge.config import ModelConfig


@dataclass
class PlannerConfig:
    """Which prompt the planner emits during rollouts."""
    prompt_template: str = "v7_no_temporal"
    num_overview_frames: int = 0  # 0 = text-only prompt (paper default)


@dataclass
class RewardWeightsConfig:
    """Reward combination weights — paper run uses frames_in_gt + consistency only."""
    consistency_weight: float = 1.0
    frames_in_gt_weight: float = 1.0


@dataclass
class PlannerTrainingConfig:
    """Top-level config for ``training/train_planner.py``.

    The reward pipeline is controlled entirely by ``inference_config``, which
    points to a real ``ToolMergeConfig`` YAML. To change reward behavior
    (answerer template, cache dirs, thresholds, etc.), modify that YAML or
    override its fields via CLI.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    reward: RewardWeightsConfig = field(default_factory=RewardWeightsConfig)
    inference_config: str = ""

    # Training data
    train_data_path: str = ""
    start_idx: int = 0
    end_idx: int = -1                # -1 = use all data
    val_data_path: str = ""
    val_start_idx: int = 0
    val_end_idx: int = -1
    early_stopping_patience: int = 0  # 0 = no early-stopping callback

    # TRL GRPOConfig overrides (free-form, applied via dict-merge in train_planner.py).
    trl: Dict[str, Any] = field(default_factory=dict)
