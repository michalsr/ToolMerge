"""Frame-selection backend for GRPO reward computation.

Runs the pipeline (load_video_caches → OCR judge →
gather_evidence) to convert a planner output into a list of selected frame
timestamps. No answerer VLM is loaded — the reward depends only on which
frames the planner selected, not on whether they yield a correct answer.
"""

import logging
import os
import sys
import hashlib
import json as _json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PKG_PATH = str(Path(__file__).resolve().parent.parent.parent)
if PKG_PATH not in sys.path:
    sys.path.insert(0, PKG_PATH)

from toolmerge.pipeline import gather_evidence
from toolmerge.caches import caches_for_video as load_video_caches

logger = logging.getLogger(__name__)

_COLORS = {
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "cyan": "\033[96m",
    "reset": "\033[0m",
}


def color(text: str, color: str) -> str:
    return f"{_COLORS.get(color, '')}{text}{_COLORS['reset']}"


class FrameSelectionBackend:
    """Runs the EP v2 evidence-gathering pipeline to produce selected frame
    timestamps for a planner's output. No VLM is loaded.

    Tool clients (SigLIP, T-REN) are loaded lazily based on inference_cfg.
    """

    def __init__(self, inference_cfg: Any):
        self._inference_cfg = inference_cfg
        self._siglip_client = None
        self._tren_client = None
        self._tool_clients_initialized = False
        # Per-batch eval cache so multiple reward fns share one pipeline pass.
        # Keyed by (uid, hash(queries+combine_expr)).
        self._eval_cache: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}

    def clear_eval_cache(self):
        self._eval_cache.clear()

    def tren_swap_enabled(self) -> bool:
        return os.environ.get("TREN_SWAP", "0").lower() in ("1", "true", "yes")

    def begin_tren_gpu_session(self):
        if not self.tren_swap_enabled() or self._tren_client is None:
            return
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._tren_client.move_to(f"cuda:{local_rank}")

    def end_tren_gpu_session(self):
        if not self.tren_swap_enabled() or self._tren_client is None:
            return
        self._tren_client.move_to("cpu")

    def ensure_tool_clients(self):
        if self._tool_clients_initialized:
            return
        self._tool_clients_initialized = True

        cfg = self._inference_cfg
        if cfg is None:
            return

        enabled_tools = cfg.enabled_tools
        if isinstance(enabled_tools, str):
            enabled_tools = [enabled_tools]

        if "siglip" in enabled_tools:
            from toolmerge.tools.siglip import SiglipClient
            self._siglip_client = SiglipClient()
            logger.info(color("SigLIP client initialized", "cyan"))

        if "tren" in enabled_tools:
            from toolmerge.tools.tren import TrenClient
            self._tren_client = TrenClient(lazy_init=True)
            logger.info(color("T-REN client initialized", "cyan"))

    def evaluate_plan(
        self,
        queries: List[dict],
        combine_expr: str,
        video_id: str,
        uid: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Run gather_evidence and return the selected timestamps.

        Returns {"timestamps": List[float]} or None on pipeline failure.
        Memoised per (uid, plan-hash) so multiple reward fns reuse one pass.
        """
        plan_repr = _json.dumps(
            {"q": queries, "c": combine_expr},
            sort_keys=True, default=str,
        )
        plan_hash = hashlib.md5(plan_repr.encode()).hexdigest()
        cache_key = (uid, plan_hash)
        if cache_key in self._eval_cache:
            return self._eval_cache[cache_key]

        self.ensure_tool_clients()
        cfg = self._inference_cfg

        try:
            video_caches = load_video_caches(video_id, cfg)
        except (FileNotFoundError, ValueError) as e:
            logger.warning(color(f"  Cannot load caches for {video_id}: {e}", "red"))
            self._eval_cache[cache_key] = None
            return None

        video_caches["siglip_client"] = self._siglip_client
        video_caches["tren_client"] = self._tren_client

        # OCR judge — disk cache only (no Azure during training)
        ocr_frames = []
        ocr_cache = video_caches.get("ocr_cache")
        if ocr_cache is not None and uid:
            ocr_judge_cache_dir = getattr(cfg, "ocr_judge_cache_dir", "")
            if ocr_judge_cache_dir:
                from toolmerge.tools.ocr_judge import load_judge_cache
                cached = load_judge_cache(ocr_judge_cache_dir, uid)
                if cached is not None:
                    ocr_frames = cached
                    logger.debug(color(
                        f"  OCR judge cache hit: {uid} ({len(cached)} frames)", "green"
                    ))

        try:
            _frames, timestamps, _gather_debug = gather_evidence(
                queries_with_ids=queries,
                combine_expr=combine_expr,
                video_caches=video_caches,
                cfg=cfg,
                ocr_frames=ocr_frames if ocr_frames else None,
            )
        except Exception as e:
            logger.warning(color(f"  gather_evidence failed for {video_id}: {e}", "red"))
            self._eval_cache[cache_key] = None
            return None

        result = {"timestamps": [float(t) for t in (timestamps or [])]}
        self._eval_cache[cache_key] = result
        return result
