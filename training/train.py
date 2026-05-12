"""
Training script for evidence pipeline v2 planner GRPO.

Config-only (OmegaConf), following the time_r1/train.py pattern:
    torchrun --standalone --nproc_per_node=4 \
        training/train_planner.py \
        config=training/config/default_planner_training.yaml \
        inference_config=configs/training/lvb_v7_no_temporal_t0_idk_dtai.yaml
"""

import logging
import os
import sys
from typing import Optional

# Ensure project root is on path
sys.path.insert(0, "/work/hdd/bcgp/michal5/verify_video/multi_turn")

import torch
# Disable torch.compile/dynamo to avoid symbolic shape errors with variable-length sequences
torch._dynamo.config.disable = True
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

from trl import GRPOConfig

from training.planner_dataset import PlannerDataset, setup_colored_logging
from training.reward import (
    consistency_reward,
    make_frames_in_gt_reward,
)
from training.frame_selection_backend import FrameSelectionBackend
from training.grpo_trainer import (
    BestValSaverCallback,
    PlannerGRPOTrainer,
)

logger = logging.getLogger(__name__)


def rank0_print(msg: str):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(msg, flush=True)


def apply_overrides(obj, overrides: dict, *, ignore_unknown: bool = True) -> None:
    """Apply a dict of key/value pairs onto an object via setattr."""
    for k, v in (overrides or {}).items():
        if hasattr(obj, k):
            setattr(obj, k, v)
        elif not ignore_unknown:
            raise AttributeError(f"Unknown override key for {type(obj).__name__}: {k}")


def main(cfg, inference_cfg) -> None:
    """Run planner GRPO training.

    Args:
        cfg: PlannerTrainingConfig (training-specific settings).
        inference_cfg: EvidencePipelineV2Config (pipeline + cache settings).
    """
    setup_colored_logging()

    rank0_print("=" * 60)
    rank0_print("  Evidence Pipeline v2 — Planner GRPO Training")
    rank0_print("=" * 60)

    # ----------------------------------------------------------------
    # 1. Build GRPOConfig from cfg.trl section
    # ----------------------------------------------------------------
    training_args = GRPOConfig(output_dir=cfg.trl.get("output_dir", "output/planner_grpo"))
    apply_overrides(training_args, cfg.trl, ignore_unknown=True)

    # Model init kwargs
    model_init_kwargs = getattr(training_args, "model_init_kwargs", None) or {}
    attn_impl = getattr(cfg.model, "attn_implementation", None) or "flash_attention_2"
    model_init_kwargs["attn_implementation"] = attn_impl
    _dtype_str = getattr(cfg.model, "torch_dtype", "bfloat16")
    _dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    model_init_kwargs["dtype"] = _dtype_map.get(_dtype_str, torch.bfloat16)
    model_init_kwargs.pop("use_cache", None)
    training_args.model_init_kwargs = model_init_kwargs

    # ----------------------------------------------------------------
    # 2. Save resolved config
    # ----------------------------------------------------------------
    try:
        from toolmerge.config import save_config
        from omegaconf import OmegaConf

        os.makedirs(training_args.output_dir, exist_ok=True)
        resolved_path = os.path.join(training_args.output_dir, "resolved_training_config.yaml")
        resolved_inf_path = os.path.join(training_args.output_dir, "resolved_inference_config.yaml")
        save_config(cfg, resolved_path)
        save_config(inference_cfg, resolved_inf_path)
        rank0_print(f"[train] Resolved training config:\n{OmegaConf.to_yaml(OmegaConf.structured(cfg))}")
        rank0_print(f"[train] Resolved inference config:\n{OmegaConf.to_yaml(OmegaConf.structured(inference_cfg))}")
        rank0_print(f"[train] Saved configs to: {resolved_path}, {resolved_inf_path}")
    except Exception as e:
        rank0_print(f"[train] WARNING: failed to save resolved config: {e}")

    # ----------------------------------------------------------------
    # 3. Build dataset
    # ----------------------------------------------------------------
    rank0_print("[train] Building dataset...")

    # Dataset uses inference_cfg for video_dir and data paths
    video_dir = getattr(inference_cfg.data, "video_dir", None) or getattr(inference_cfg, "video_dir", None)
    dataset = PlannerDataset(
        data_path=cfg.train_data_path,
        prompt_template=cfg.planner.prompt_template,
        num_overview_frames=cfg.planner.num_overview_frames,
        frame_cache_dir=getattr(inference_cfg.data, "frame_cache_dir", None),
        video_dir=video_dir,
        fps=getattr(inference_cfg.data, "fps", 2.0) if hasattr(inference_cfg.data, "fps") else 2.0,
        start_idx=cfg.start_idx,
        end_idx=cfg.end_idx if cfg.end_idx >= 0 else None,
    )
    rank0_print(f"[train] Dataset: {len(dataset)} items")

    eval_dataset = None
    if getattr(cfg, "val_data_path", ""):
        eval_dataset = PlannerDataset(
            data_path=cfg.val_data_path,
            prompt_template=cfg.planner.prompt_template,
            num_overview_frames=cfg.planner.num_overview_frames,
            frame_cache_dir=getattr(inference_cfg.data, "frame_cache_dir", None),
            video_dir=video_dir,
            fps=getattr(inference_cfg.data, "fps", 2.0) if hasattr(inference_cfg.data, "fps") else 2.0,
            start_idx=cfg.val_start_idx,
            end_idx=cfg.val_end_idx if cfg.val_end_idx >= 0 else None,
        )
        rank0_print(f"[train] Eval dataset: {len(eval_dataset)} items")

    # ----------------------------------------------------------------
    # 4. Build reward backends + reward functions
    # ----------------------------------------------------------------
    # FrameSelectionBackend (no answerer VLM; gather_evidence only) is needed
    # whenever the frames_in_gt reward is on.
    cons_w = float(getattr(cfg.reward, "consistency_weight", 0.0) or 0.0)
    fig_w = float(getattr(cfg.reward, "frames_in_gt_weight", 0.0) or 0.0)

    frame_backend = None
    if fig_w > 0:
        frame_backend = FrameSelectionBackend(inference_cfg=inference_cfg)
        rank0_print("[train] FrameSelectionBackend: enabled (frames_in_gt reward)")

    reward_fns = []
    reward_weights = []
    names = []

    if cons_w != 0:
        reward_fns.append(consistency_reward)
        reward_weights.append(cons_w)
        names.append("consistency")
    if fig_w != 0:
        reward_fns.append(make_frames_in_gt_reward(frame_backend=frame_backend))
        reward_weights.append(fig_w)
        names.append("frames_in_gt")

    if not reward_fns:
        raise ValueError(
            "No reward functions configured — set at least one of "
            "consistency_weight / frames_in_gt_weight"
        )

    training_args.reward_weights = reward_weights

    rank0_print(
        "[train] Reward functions: "
        + ", ".join(f"{n}({w})" for n, w in zip(names, reward_weights))
    )

    # ----------------------------------------------------------------
    # 6. DeepSpeed check
    # ----------------------------------------------------------------
    ds_config = getattr(training_args, "deepspeed", None)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if ds_config:
        if isinstance(ds_config, str) and not os.path.exists(ds_config):
            raise FileNotFoundError(f"DeepSpeed config not found: {ds_config}")
        rank0_print(f"[train] DeepSpeed ENABLED: {ds_config}")
    elif world_size > 1:
        rank0_print(
            f"[train] WARNING: Multi-node (WORLD_SIZE={world_size}) without DeepSpeed!"
        )

    # ----------------------------------------------------------------
    # 7. Create trainer and train
    # ----------------------------------------------------------------
    freeze_vision = getattr(cfg.model, "freeze_vision_encoder", False)
    rank0_print(f"[train] Freeze vision encoder: {freeze_vision}")
    rank0_print(f"[train] Model: {cfg.model.base}")

    trainer = PlannerGRPOTrainer(
        model=cfg.model.base,
        args=training_args,
        reward_funcs=reward_fns,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        freeze_vision_encoder=freeze_vision,
    )

    patience = int(getattr(cfg, "early_stopping_patience", 0) or 0)
    if patience > 0 and eval_dataset is not None:
        from transformers import TrainerCallback

        class PersistentEarlyStoppingCallback(TrainerCallback):
            """Self-contained early-stopping. Reads the tracked metric directly
            from state.log_history (where GRPOTrainer.log() merges reward keys
            after each eval) and keeps its own best/patience state.

            Independent of HF's metric_for_best_model / state.best_metric
            machinery — that path doesn't work for GRPO and we deliberately
            don't engage it. Rebuilds counter from log_history on
            on_train_begin so chained jobs share one patience window. Every
            handler is try/except-wrapped so a callback bug never kills
            training."""

            def __init__(
                self,
                patience: int,
                metric: str = "eval_reward",
                greater_is_better: bool = True,
                threshold: float = 0.0,
            ):
                self.patience = int(patience)
                self.metric = metric if metric.startswith("eval_") else f"eval_{metric}"
                self.greater_is_better = bool(greater_is_better)
                self.threshold = float(threshold)
                self.best_value: Optional[float] = None
                self.patience_counter: int = 0

            def improved(self, v: float) -> bool:
                if self.best_value is None:
                    return True
                return (
                    (v > self.best_value + self.threshold)
                    if self.greater_is_better
                    else (v < self.best_value - self.threshold)
                )

            def on_train_begin(self, args, state, control, **kw):
                try:
                    self.best_value = None
                    self.patience_counter = 0
                    seen = 0
                    for entry in state.log_history or []:
                        if self.metric not in entry:
                            continue
                        seen += 1
                        v = float(entry[self.metric])
                        if self.improved(v):
                            self.best_value = v
                            self.patience_counter = 0
                        else:
                            self.patience_counter += 1
                    rank0_print(
                        f"[EarlyStop] Restored patience_counter="
                        f"{self.patience_counter}/{self.patience} from {seen} logged evals "
                        f"(metric={self.metric}, best={self.best_value})"
                    )
                except Exception as e:
                    logger.warning(f"[EarlyStop] on_train_begin failed (training continues): {e}")

            def on_evaluate(self, args, state, control, metrics=None, **kw):
                try:
                    val = None
                    if metrics and self.metric in metrics:
                        val = float(metrics[self.metric])
                    else:
                        for entry in reversed(state.log_history or []):
                            if self.metric in entry:
                                val = float(entry[self.metric])
                                break
                    if val is None:
                        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                            logger.warning(
                                f"[EarlyStop] {self.metric} not found in metrics "
                                f"or log_history — skipping this eval"
                            )
                        return control
                    if self.improved(val):
                        self.best_value = val
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                            logger.info(
                                f"[EarlyStop] patience exhausted "
                                f"({self.patience_counter}/{self.patience}, "
                                f"best {self.metric}={self.best_value:.4f}) — stopping"
                            )
                        control.should_training_stop = True
                except Exception as e:
                    logger.warning(f"[EarlyStop] on_evaluate failed (training continues): {e}")
                return control

        trainer.add_callback(
            PersistentEarlyStoppingCallback(
                patience=patience,
                metric="eval_reward",
                greater_is_better=True,
            )
        )
        rank0_print(
            f"[train] PersistentEarlyStoppingCallback added "
            f"(patience={patience}, metric=eval_reward)"
        )

    # --- Local completion logger: append every step's completions to JSONL ---
    # Uses TRL's own _logs buffer (same data used for Rich table + wandb Table).
    # Rank-0 only; file lives under the training output dir for easy packaging.
    import json as _json_mod
    from transformers import TrainerCallback

    class CompletionJsonlCallback(TrainerCallback):
        """Appends every step's prompt/completion/rewards/advantages to a JSONL
        file on rank 0. Safe to run alongside log_completions — that one prints
        + wandb-Tables, this one persists everything locally."""

        def __init__(self, trainer, output_dir: str):
            self._trainer = trainer
            self.path = os.path.join(output_dir, "completions.jsonl")
            os.makedirs(output_dir, exist_ok=True)

        def on_log(self, args, state, control, **kw):
            if int(os.environ.get("LOCAL_RANK", 0)) != 0:
                return
            logs = getattr(self._trainer, "_logs", None)
            if not logs or not logs.get("prompt"):
                return
            prompts = logs["prompt"]
            completions = logs["completion"]
            rewards = logs.get("rewards", {})
            advantages = logs.get("advantages", [None] * len(prompts))
            try:
                with open(self.path, "a") as f:
                    for i in range(len(prompts)):
                        _json_mod.dump({
                            "step": int(state.global_step),
                            "prompt": prompts[i],
                            "completion": completions[i],
                            "rewards": {k: (v[i] if i < len(v) else None) for k, v in rewards.items()},
                            "advantage": advantages[i] if i < len(advantages) else None,
                        }, f, default=str)
                        f.write("\n")
            except Exception as e:
                logger.warning(f"[CompletionJsonl] write failed: {e}")

    trainer.add_callback(CompletionJsonlCallback(trainer, training_args.output_dir))
    rank0_print(f"[train] CompletionJsonlCallback → {training_args.output_dir}/completions.jsonl")

    # Best-val saver — replaces load_best_model_at_end=True (unsupported on TRL
    # 0.27.2 GRPO because reward keys never reach _determine_best_metric).
    # Failures inside the callback are swallowed with a warning so they never
    # crash training.
    if eval_dataset is not None:
        try:
            trainer.add_callback(
                BestValSaverCallback(
                    trainer=trainer,
                    output_dir=training_args.output_dir,
                    metric="eval_reward",
                    greater_is_better=True,
                )
            )
            rank0_print(
                f"[train] BestValSaverCallback → {training_args.output_dir}/best_val "
                f"(metric=eval_reward, greater_is_better=True)"
            )
        except Exception as e:
            rank0_print(f"[train] WARNING: failed to add BestValSaverCallback: {e}")

    from transformers.trainer_utils import get_last_checkpoint
    resume_ckpt = None
    if os.path.isdir(training_args.output_dir):
        resume_ckpt = get_last_checkpoint(training_args.output_dir)
    if resume_ckpt:
        rank0_print(f"[train] Resuming from {resume_ckpt}")
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        rank0_print("[train] Starting fresh training run")
        trainer.train()
    rank0_print("[train] Training complete.")

    # After load_best_model_at_end=True, the best weights are already in
    # trainer.model. Write a weights-only copy for downstream eval so the
    # full checkpoint dirs (optimizer/scheduler/DS shards) can be deleted
    # without losing the trained model.
    final_dir = os.path.join(training_args.output_dir, "best_weights")
    rank0_print(f"[train] Saving weights-only best model → {final_dir}")
    trainer.save_model(final_dir)
    rank0_print("[train] Done.")


if __name__ == "__main__":
    from toolmerge.config import load_config, get_config_path_from_cli
    from training.config_schema import PlannerTrainingConfig
    from toolmerge.config import ToolMergeConfig as EvidencePipelineV2Config

    default_cfg_path = os.path.join(
        os.path.dirname(__file__), "config", "default_planner_training.yaml"
    )
    cfg_path = get_config_path_from_cli() or default_cfg_path
    cfg = load_config(cfg_path, PlannerTrainingConfig)

    # Load inference config from YAML only (no CLI args — those belong to PlannerTrainingConfig).
    # To change reward behavior, edit the inference YAML directly.
    if not cfg.inference_config:
        raise ValueError(
            "inference_config is required — point it to an EvidencePipelineV2Config YAML "
            "(e.g. configs/training/lvb_v7_no_temporal_t0_idk_dtai.yaml)"
        )
    from omegaconf import OmegaConf
    _inf_schema = OmegaConf.structured(EvidencePipelineV2Config)
    _inf_file = OmegaConf.load(cfg.inference_config)
    inference_cfg = OmegaConf.to_object(OmegaConf.merge(_inf_schema, _inf_file))

    main(cfg, inference_cfg)
