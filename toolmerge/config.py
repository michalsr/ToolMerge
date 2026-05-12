"""ToolMerge configuration schema and OmegaConf loader.

Single dataclass tree mirrored after `EvidencePipelineV2Config` in the research
tree, trimmed to just the fields used by the public method. Every numeric
default matches the paper's verified run configs.

Use `load_config(path)` to merge a YAML file with CLI overrides (OmegaConf
`from_cli` style: `key=value`, dotted paths allowed).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Type, TypeVar, Tuple, Any
import sys

from omegaconf import OmegaConf, MISSING


# --- Submodules --------------------------------------------------------------

@dataclass
class ModelConfig:
    """Local Qwen3-VL model loading options."""
    base: str = "Qwen/Qwen3-VL-8B-Instruct"
    processor_base: Optional[str] = None
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    device_map: str = "auto"
    freeze_vision_encoder: bool = False


@dataclass
class DataConfig:
    """Dataset + raw-video paths."""
    input_path: str = MISSING            # path to dataset JSON (M2M / LVB / Video-MME)
    save_path: str = MISSING             # output directory (mandatory)
    video_dir: str = ""                  # directory of raw .mp4 files
    video_backend: str = "opencv"        # cv2 backend per workflow rule
    start_idx: int = 0
    end_idx: Optional[int] = None
    # When set, run.py reads pooled_candidates_K from this dir's results.json
    # and runs ONLY the answerer (no planner/tool/merge). K = max_final_k.
    source_dir: str = ""


@dataclass
class AnswerGeneratorConfig:
    prompt_template: str = "lif"         # the only template used by the paper
    max_new_tokens: int = 128
    temperature: float = 0.0             # verified from paper run configs
    top_p: float = 0.8
    top_k: int = 20
    do_sample: bool = False
    mode: str = "letter_only"
    no_timestamps: bool = False


@dataclass
class OpenAIBackendConfig:
    """OpenAI / Azure OpenAI API access.

    All fields fall back to env vars when omitted:
        OPENAI_API_KEY / AZURE_OPENAI_API_KEY
        OPENAI_BASE_URL / AZURE_OPENAI_ENDPOINT
        OPENAI_MODEL / AZURE_OPENAI_DEPLOYMENT
        OPENAI_USE_AZURE
    """
    model_name: Optional[str] = None
    api_endpoint: Optional[str] = None
    use_azure: Optional[bool] = None     # None = auto-detect from env
    max_retries: int = 5


# --- Top-level ---------------------------------------------------------------

@dataclass
class ToolMergeConfig:
    """Single root config for inference.

    Covers the paper-run fields: planner v7_no_temporal, SigLIP + T-REN + OCR
    tools, rank-merge AND/OR, greedy NMS, lif/v1 answerers.
    """
    # Models / backends
    model: ModelConfig = field(default_factory=ModelConfig)
    model_backend: str = "qwen3vl"       # "qwen3vl" | "openai"
    qwen_version: str = "qwen3"
    openai: OpenAIBackendConfig = field(default_factory=OpenAIBackendConfig)

    # Separate planner backend (empty = use model_backend for everything)
    planner_backend: str = ""
    planner_openai: OpenAIBackendConfig = field(default_factory=OpenAIBackendConfig)

    # Data
    data: DataConfig = field(default_factory=DataConfig)

    # Planner
    planner_prompt: str = "v7_no_temporal"
    planner_max_new_tokens: int = 512
    planner_temperature: float = 0.0     # deterministic for paper runs
    planner_top_p: float = 0.8
    planner_top_k: int = 20
    planner_do_sample: bool = False
    planner_num_frames: int = 0          # text-only planner (per paper)

    # Tools / caches
    enabled_tools: List[str] = field(default_factory=lambda: ["siglip", "tren"])
    siglip_feature_cache_dir: str = ""
    tren_cache_dir: str = ""
    ocr_cache_dir: str = ""

    # Thresholding / pooling (defaults from default.yaml)
    score_threshold_mode: str = "percentile"
    score_threshold_value: float = 0.0   # 0 keeps everything; ablation uses higher
    min_frames_per_query: int = 16
    max_final_k: int = 8                 # paper uses K=8 or K=32 via override
    pool_k_values: List[int] = field(default_factory=lambda: [8, 16, 32, 64])

    # Greedy NMS (paper uses min_frame_gap_seconds=-1 → auto τ = min(D/(2K), 10))
    min_frame_gap_seconds: float = -1.0
    min_frame_gap_cap_seconds: float = 10.0

    # OCR judge
    ocr_llm_model: str = "openai:gpt-4o-mini"   # routed through OpenAIBackend
    ocr_llm_max_tokens: int = 256
    ocr_batch_size: int = 20
    ocr_pool_seconds: float = -1.0       # auto: match min_frame_gap formula
    ocr_judge_cache_dir: str = ""

    # FPS subsampling (null = use cache native fps, typically 2.0)
    target_fps: Optional[float] = None

    # Answerer
    answer_generator: AnswerGeneratorConfig = field(default_factory=AnswerGeneratorConfig)

    # General
    save_trace: bool = True
    save_every_n: int = 5
    seed: int = 42
    debug: bool = False


# --- Loader / saver ---------------------------------------------------------

T = TypeVar("T")


def load_with_defaults(path: str):
    """Load a YAML, resolving an optional ``defaults:`` block first.

    Supports Hydra-style inheritance for the per-table configs:

        # configs/tables/table4.yaml
        defaults:
          - ../default        # path relative to this file, no .yaml suffix
        data:
          input_path: ...

    Each parent is loaded recursively and merged left-to-right; the
    final overrides come from the child file itself. The ``defaults:``
    key is stripped before the structured-schema merge so OmegaConf
    doesn't see it as an unknown field.
    """
    cfg = OmegaConf.load(path)
    defaults = cfg.pop("defaults", None) if hasattr(cfg, "pop") else None
    if defaults is None:
        return cfg
    merged = OmegaConf.create({})
    here = Path(path).resolve().parent
    for entry in defaults:
        entry_str = str(entry)
        parent_path = (here / f"{entry_str}.yaml").resolve()
        if not parent_path.exists():
            raise FileNotFoundError(
                f"defaults: entry {entry_str!r} not found at {parent_path}"
            )
        parent_cfg = load_with_defaults(str(parent_path))
        merged = OmegaConf.merge(merged, parent_cfg)
    return OmegaConf.merge(merged, cfg)


def load_config(
    config_path: Optional[str] = None,
    config_class: Type[T] = ToolMergeConfig,
) -> T:
    """Load YAML + CLI overrides into a dataclass instance.

    Supports an optional ``defaults: [paths]`` block in the YAML for
    Hydra-style inheritance (paths relative to the loaded file).

    CLI parsing matches the research tree's ``load_config``:
        python -m toolmerge.run run config=path.yaml key=value nested.field=val
        python -m toolmerge.run run --config path.yaml key=value
        python -m toolmerge.run run path.yaml key=value           (positional)
    """
    schema = OmegaConf.structured(config_class)
    cli_path, cli_args = extract_config_path_and_args(sys.argv[1:])
    selected = cli_path or config_path

    if selected and Path(selected).exists():
        file_cfg = load_with_defaults(selected)
        cfg = OmegaConf.merge(schema, file_cfg)
    else:
        cfg = schema

    cli_cfg = OmegaConf.from_cli(cli_args)
    if "config" in cli_cfg:
        del cli_cfg["config"]
    if cli_cfg:
        cfg = OmegaConf.merge(cfg, cli_cfg)

    return OmegaConf.to_object(cfg)


def save_config(config: Any, path: str) -> None:
    cfg = OmegaConf.structured(config)
    with open(path, "w") as f:
        OmegaConf.save(cfg, f)


def get_config_path_from_cli() -> Optional[str]:
    path, _ = extract_config_path_and_args(sys.argv[1:])
    return path


def looks_like_config_file(arg: str) -> bool:
    if not arg or arg.startswith("-") or "=" in arg:
        return False
    a = arg.lower()
    return a.endswith(".yaml") or a.endswith(".yml")


def extract_config_path_and_args(args: List[str]) -> Tuple[Optional[str], List[str]]:
    path: Optional[str] = None
    out: List[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            path = arg
            skip_next = False
            continue
        if arg.startswith("config="):
            path = arg.split("=", 1)[1]
            continue
        if arg.startswith("--config="):
            path = arg.split("=", 1)[1]
            continue
        if arg in {"--config", "-c"}:
            skip_next = True
            continue
        if looks_like_config_file(arg):
            path = arg
            continue
        out.append(arg)
    return path, out
