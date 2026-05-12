"""Thin GRPOTrainer subclass for planner GRPO training.

Adds:
- Accelerate env var pre-configuration for DeepSpeed (same fix as
  time_r1/trainer/grpo_trainer_env_verl_safe_v3_accelerate_fix.py)
- Optional vision encoder freezing
- NaN debug callback
- BestValSaverCallback: saves weights to `best_val/` whenever eval_reward
  improves. Reads the merged metrics out of state.log_history (GRPOTrainer
  writes reward keys there via log()), so it works around TRL 0.27.2 not
  supporting load_best_model_at_end with GRPO reward. Survives chained jobs
  by rebuilding best-so-far from log_history on train begin.
"""

import json
import logging
import os
from typing import Any, Optional

import torch
from transformers import TrainerCallback
from trl import GRPOTrainer, GRPOConfig

logger = logging.getLogger(__name__)

_COLORS = {
    "green": "\033[92m",
    "yellow": "\033[93m",
    "cyan": "\033[96m",
    "reset": "\033[0m",
}


def color(text: str, color: str) -> str:
    return f"{_COLORS.get(color, '')}{text}{_COLORS['reset']}"


def rank0_print(msg: str):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# NaN debug callback
# ---------------------------------------------------------------------------

class NaNDebugCallback(TrainerCallback):
    """Check for NaN/Inf in params and gradients after each step."""

    def __init__(self, check_every_n_steps: int = 1):
        self.check_every_n_steps = check_every_n_steps
        self._step_count = 0

    def on_step_end(self, args, state, control, model=None, **kwargs):
        self._step_count += 1
        if self._step_count % self.check_every_n_steps != 0:
            return
        if model is None:
            return

        nan_params, inf_params = [], []
        nan_grads, inf_grads = [], []
        for name, param in model.named_parameters():
            if param.data is not None:
                if torch.isnan(param.data).any():
                    nan_params.append(name)
                if torch.isinf(param.data).any():
                    inf_params.append(name)
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_grads.append(name)
                if torch.isinf(param.grad).any():
                    inf_grads.append(name)

        if nan_params or inf_params or nan_grads or inf_grads:
            rank0_print(
                color(f"\n[NaN DEBUG] Step {state.global_step}:", "yellow")
            )
            if nan_params:
                rank0_print(f"  NaN in {len(nan_params)} params: {nan_params[:3]}...")
            if inf_params:
                rank0_print(f"  Inf in {len(inf_params)} params: {inf_params[:3]}...")
            if nan_grads:
                rank0_print(f"  NaN in {len(nan_grads)} grads: {nan_grads[:3]}...")
            if inf_grads:
                rank0_print(f"  Inf in {len(inf_grads)} grads: {inf_grads[:3]}...")


# ---------------------------------------------------------------------------
# BestValSaverCallback
# ---------------------------------------------------------------------------

class BestValSaverCallback(TrainerCallback):
    """Saves trainer weights to ``<output_dir>/best_val`` when the tracked
    eval metric improves. Designed as a drop-in replacement for
    ``load_best_model_at_end=True`` with GRPO, which TRL 0.27.2 does not
    support (the reward keys are merged into logs by GRPOTrainer.log() and
    never reach Trainer._determine_best_metric, causing KeyError).

    Reads metrics from ``state.log_history`` rather than the ``metrics``
    kwarg because ``on_evaluate`` fires AFTER ``log()`` has already merged
    the reward keys into log_history (but not into the dict Trainer returns
    from evaluate()). Also rebuilds best-so-far from log_history on
    ``on_train_begin`` so chained jobs resume the correct best tracker.
    """

    def __init__(
        self,
        trainer: Any,
        output_dir: str,
        metric: str = "eval_reward",
        greater_is_better: bool = True,
    ):
        self.trainer = trainer
        self.output_dir = output_dir
        self.metric = metric if metric.startswith("eval_") else f"eval_{metric}"
        self.greater_is_better = greater_is_better
        self.best: Optional[float] = None
        self.best_step: Optional[int] = None
        self.best_dir = os.path.join(output_dir, "best_val")

    def is_better(self, v: float) -> bool:
        if self.best is None:
            return True
        return v > self.best if self.greater_is_better else v < self.best

    def on_train_begin(self, args, state, control, **kw):
        # Rebuild best-so-far from prior log history so chained jobs continue
        # the same best-tracker rather than starting over.
        try:
            for entry in state.log_history:
                if self.metric in entry:
                    v = float(entry[self.metric])
                    if self.is_better(v):
                        self.best = v
                        self.best_step = int(entry.get("step", 0))
            if self.best is not None and int(os.environ.get("LOCAL_RANK", 0)) == 0:
                logger.info(
                    f"[BestValSaver] Restored best {self.metric}={self.best:.4f} "
                    f"@ step {self.best_step} from log_history"
                )
        except Exception as e:
            logger.warning(f"[BestValSaver] Failed to restore best from log_history: {e}")

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        try:
            # Pull the latest value from log_history (GRPOTrainer.log() writes
            # the merged eval_* keys there). metrics kwarg may be missing it.
            val = None
            if metrics and self.metric in metrics:
                val = float(metrics[self.metric])
            else:
                for entry in reversed(state.log_history):
                    if self.metric in entry:
                        val = float(entry[self.metric])
                        break
            if val is None:
                if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                    logger.warning(
                        f"[BestValSaver] {self.metric} not found in metrics "
                        f"or log_history; skipping best-val save this eval"
                    )
                return
            if not self.is_better(val):
                return
            prev = self.best
            self.best = val
            self.best_step = int(state.global_step)
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                msg = f"{val:.4f}" if prev is None else f"{val:.4f} (prev {prev:.4f})"
                logger.info(
                    f"[BestValSaver] New best {self.metric}={msg} @ step "
                    f"{self.best_step} → saving to {self.best_dir}"
                )
            # All ranks must participate in save_model (DeepSpeed shards state).
            os.makedirs(self.best_dir, exist_ok=True)
            self.trainer.save_model(self.best_dir)
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                with open(os.path.join(self.best_dir, "best_metric.json"), "w") as f:
                    json.dump(
                        {
                            "metric": self.metric,
                            "value": self.best,
                            "step": self.best_step,
                            "greater_is_better": self.greater_is_better,
                        },
                        f,
                        indent=2,
                    )
        except Exception as e:
            # Never crash training from this callback — log and carry on.
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                logger.warning(f"[BestValSaver] on_evaluate failed: {e}")


# ---------------------------------------------------------------------------
# PlannerGRPOTrainer
# ---------------------------------------------------------------------------

class PlannerGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with accelerate fix and optional vision encoder freezing.

    This is a thin subclass. All multimodal generation, GRPO loss,
    DeepSpeed, and reward aggregation come from TRL's GRPOTrainer.
    """

    def __init__(
        self,
        model: Any = None,
        args: Optional[GRPOConfig] = None,
        freeze_vision_encoder: bool = False,
        **kwargs,
    ):
        # --- Accelerate env var pre-configuration ---
        # Must happen BEFORE super().__init__() creates Accelerator
        self.configure_accelerate_env(args)

        super().__init__(model=model, args=args, **kwargs)

        # --- Freeze vision encoder if requested ---
        if freeze_vision_encoder:
            self.freeze_vision_encoder()

        # --- Add NaN debug callback ---
        self.add_callback(NaNDebugCallback(check_every_n_steps=1))

        # Log param counts
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        rank0_print(
            color(
                f"[PlannerGRPO] Total params: {total_params:,} | "
                f"Trainable: {trainable_params:,} | "
                f"Frozen: {total_params - trainable_params:,}",
                "cyan",
            )
        )

        # Debug: verify generation config
        gc = getattr(self, "generation_config", None)
        if gc is not None:
            rank0_print(
                color(
                    f"[PlannerGRPO] GenerationConfig: do_sample={gc.do_sample}, "
                    f"temperature={gc.temperature}, top_p={gc.top_p}, "
                    f"top_k={gc.top_k}, max_new_tokens={gc.max_new_tokens}",
                    "cyan",
                )
            )

    def configure_accelerate_env(self, args: Optional[GRPOConfig]):
        """Pre-set accelerate env vars from DeepSpeed config.

        Replicates the fix from grpo_trainer_env_verl_safe_v3_accelerate_fix.py.
        """
        if args is None:
            return

        ds_cfg = getattr(args, "deepspeed", None)
        if not ds_cfg:
            return

        os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"

        if isinstance(ds_cfg, str):
            os.environ["ACCELERATE_DEEPSPEED_CONFIG_FILE"] = ds_cfg
            try:
                with open(ds_cfg, "r") as f:
                    ds_config_dict = json.load(f)

                zero_opt = ds_config_dict.get("zero_optimization", {})
                zero_stage = zero_opt.get("stage", 2)

                offload_opt = zero_opt.get("offload_optimizer", {})
                offload_opt_device = (
                    offload_opt.get("device", "none")
                    if isinstance(offload_opt, dict)
                    else "none"
                )

                offload_param = zero_opt.get("offload_param", {})
                offload_param_device = (
                    offload_param.get("device", "none")
                    if isinstance(offload_param, dict)
                    else "none"
                )

                os.environ["ACCELERATE_DEEPSPEED_ZERO_STAGE"] = str(zero_stage)
                os.environ["ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE"] = offload_opt_device
                os.environ["ACCELERATE_DEEPSPEED_OFFLOAD_PARAM_DEVICE"] = offload_param_device

                rank0_print(color("[Accelerate Fix] Parsed DeepSpeed config:", "cyan"))
                rank0_print(f"  ZeRO stage: {zero_stage}")
                rank0_print(f"  Offload optimizer: {offload_opt_device}")
                rank0_print(f"  Offload param: {offload_param_device}")
            except Exception as e:
                rank0_print(
                    color(f"[Accelerate Fix] WARNING: Failed to parse DS config: {e}", "yellow")
                )

        # Mixed precision
        if getattr(args, "bf16", False):
            os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"
        elif getattr(args, "fp16", False):
            os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16"

        # Gradient accumulation
        os.environ["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"] = str(
            args.gradient_accumulation_steps
        )

        # Debug: print accelerate env state
        rank0_print(color("[Accelerate Fix] Env vars before Trainer init:", "cyan"))
        for key in sorted(os.environ.keys()):
            if "ACCELERATE" in key or "DEEPSPEED" in key:
                rank0_print(f"  {key}={os.environ[key]}")

    def freeze_vision_encoder(self):
        """Freeze vision encoder parameters."""
        frozen_count = 0
        model = self.model
        # Handle DeepSpeed-wrapped models
        if hasattr(model, "module"):
            model = model.module

        # Try common VLM vision encoder attribute names
        vision_modules = []
        for attr in ["visual", "vision_model", "vision_tower", "vit"]:
            if hasattr(model, attr):
                vision_modules.append(getattr(model, attr))

        if not vision_modules:
            rank0_print(
                color(
                    "[PlannerGRPO] WARNING: No vision encoder found to freeze. "
                    "Checked: visual, vision_model, vision_tower, vit",
                    "yellow",
                )
            )
            return

        for vm in vision_modules:
            for param in vm.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_count += param.numel()

        rank0_print(
            color(
                f"[PlannerGRPO] Froze vision encoder: {frozen_count:,} params",
                "green",
            )
        )
