"""LLM-judged OCR relevance.

Instead of doing OCR retrieval against the query, we send every extracted
text snippet to a small LLM judge and ask whether it helps answer the
question. Frames whose text gets a YES verdict are inserted into the merged
ranking at rank 1.

Per-question results are cached to disk (depend only on (video, question,
options) — independent of the planner / answerer / tool selection).

The actual per-frame OCR text extraction is done upstream during cache
build by ``cache_build/ocr.py`` (EasyOCR).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from toolmerge.prompts.ocr_judge import OCR_JUDGE_PROMPT, OCR_JUDGE_BATCH_PROMPT

logger = logging.getLogger(__name__)


# --- Disk cache ----------------------------------------------------------

def load_judge_cache(
    cache_dir: str, uid: str, current_fps: Optional[float] = None,
) -> Optional[List[int]]:
    """Return cached frame indices or ``None`` on miss.

    If the cache was written at a different fps, indices are scaled by
    ``current_fps / cached_fps``.
    """
    if not cache_dir or not uid:
        return None
    path = os.path.join(cache_dir, f"{uid}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        frames = list(data.get("frames") or [])
        cached_fps = float(data.get("fps", 2.0))
        if current_fps and cached_fps and abs(cached_fps - float(current_fps)) > 1e-6:
            ratio = float(current_fps) / cached_fps
            frames = sorted({int(idx * ratio) for idx in frames})
        return frames
    except Exception as e:  # noqa: BLE001
        logger.warning("OCR judge cache read failed (%s): %s", path, e)
        return None


def save_judge_cache(
    cache_dir: str, uid: str, frames: List[int], fps: Optional[float] = None,
) -> None:
    if not cache_dir or not uid:
        return
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{uid}.json")
    payload: Dict[str, Any] = {"uid": uid, "frames": frames}
    if fps is not None:
        payload["fps"] = float(fps)
    try:
        with open(path, "w") as f:
            json.dump(payload, f)
    except Exception as e:  # noqa: BLE001
        logger.warning("OCR judge cache write failed (%s): %s", path, e)


# --- Config / helpers ---------------------------------------------------

@dataclass
class OcrJudgeCfg:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.9
    top_k: int = 20
    do_sample: bool = False


def format_options(options: dict) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in options.items())


def format_text_list(texts: List[Tuple[int, str]]) -> str:
    return "\n".join(f"{i}. \"{text}\"" for i, text in texts)


# --- OCR text -> segments ------------------------------------------------

def deduplicate_ocr_segments(
    ocr_results: list, fps: float, pool_seconds: float = 5.0,
) -> List[dict]:
    """Group frames into segments by (case-insensitive text, temporal pool).

    Consecutive frames with the same normalized text within ``pool_seconds``
    are merged into one segment. Collapses a multi-second persistent subtitle
    into a single segment so the LLM only judges each unique text once.
    """
    pool_frames = int(pool_seconds * fps)

    frame_texts = []
    for frame_idx, detections in enumerate(ocr_results):
        if not detections:
            continue
        raw = " | ".join(d["text"].strip() for d in detections if d["text"].strip())
        if raw:
            frame_texts.append((frame_idx, raw.lower(), raw))

    if not frame_texts:
        return []

    segments = []
    cur_norm, cur_raw, cur_start, cur_end = (
        frame_texts[0][1], frame_texts[0][2], frame_texts[0][0], frame_texts[0][0]
    )
    for frame_idx, norm, raw in frame_texts[1:]:
        if norm == cur_norm and (frame_idx - cur_end) <= pool_frames:
            cur_end = frame_idx
        else:
            segments.append(segment(len(segments) + 1, cur_raw, cur_start, cur_end, fps))
            cur_norm, cur_raw, cur_start, cur_end = norm, raw, frame_idx, frame_idx
    segments.append(segment(len(segments) + 1, cur_raw, cur_start, cur_end, fps))
    return segments


def segment(seg_id, raw, start, end, fps):
    return {
        "id": seg_id,
        "text": raw,
        "start_frame": start,
        "end_frame": end,
        "start_time": start / fps,
        "end_time": end / fps,
    }


def prepare_segments(ocr_cache, start_idx, end_idx, fps=None):
    ocr_results = ocr_cache["ocr_results"]
    if fps is None:
        fps = ocr_cache.get("fps", 2.0)

    end_idx = min(end_idx, len(ocr_results))
    segments = deduplicate_ocr_segments(ocr_results[start_idx:end_idx], fps)
    if not segments:
        return [], {}, {}, fps

    # Shift to global frame indices.
    for seg in segments:
        seg["start_frame"] += start_idx
        seg["end_frame"] += start_idx

    # Group by unique normalized text so the LLM judges each only once.
    text_to_seg_ids: Dict[str, List[int]] = {}
    text_to_raw: Dict[str, str] = {}
    for seg in segments:
        key = seg["text"].lower()
        text_to_seg_ids.setdefault(key, []).append(seg["id"])
        if key not in text_to_raw:
            text_to_raw[key] = seg["text"]
    return segments, text_to_seg_ids, text_to_raw, fps


def segments_to_frames(segments, relevant_segment_ids):
    out: List[int] = []
    seen = set()
    for seg in segments:
        if seg["id"] in relevant_segment_ids:
            for frame_idx in range(seg["start_frame"], seg["end_frame"] + 1):
                if frame_idx not in seen:
                    seen.add(frame_idx)
                    out.append(frame_idx)
    out.sort()
    return out


def parse_batch_response(response: str, n_texts: int) -> List[bool]:
    """Parse N lines of ``<number>. YES/NO`` from the batch judge reply.

    Unparseable lines default to ``True`` (conservative — don't drop frames).
    """
    results = [True] * n_texts
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"(\d+)\s*[.):\-]\s*(YES|NO)", line, re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            if 1 <= idx <= n_texts:
                results[idx - 1] = m.group(2).upper() == "YES"
    return results


# --- Public API ----------------------------------------------------------

def judge_ocr_relevance(
    question: str,
    options: dict,
    ocr_cache: dict,
    start_idx: int,
    end_idx: int,
    backend: Any,
    cfg: Any,
    uid: str = "",
    cache_dir: str = "",
) -> List[int]:
    """One YES/NO LLM call per unique segment. Returns relevant frame indices."""
    current_fps = float(ocr_cache.get("fps", 2.0)) if isinstance(ocr_cache, dict) else None
    cached = load_judge_cache(cache_dir, uid, current_fps=current_fps)
    if cached is not None:
        logger.info("OCR judge: cache hit for %s (%d frames)", uid, len(cached))
        return cached

    segments, text_to_seg_ids, text_to_raw, fps = prepare_segments(ocr_cache, start_idx, end_idx)
    if not segments:
        logger.info("No OCR text found in video")
        save_judge_cache(cache_dir, uid, [], fps=current_fps)
        return []

    logger.info("OCR judge: %d segments, %d unique texts", len(segments), len(text_to_seg_ids))

    judge_cfg = OcrJudgeCfg(max_new_tokens=getattr(cfg, "ocr_llm_max_tokens", 16))
    options_text = format_options(options)
    relevant_segment_ids = set()

    t0 = time.time()
    for key, seg_ids in text_to_seg_ids.items():
        text = text_to_raw[key]
        prompt = OCR_JUDGE_PROMPT.format(question=question, options=options_text, ocr_text=text)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        try:
            response = backend.generate_text(messages, judge_cfg).strip().upper()
        except Exception as e:  # noqa: BLE001
            logger.warning("OCR judge call failed (skipping): %r", e)
            continue
        if response.startswith("YES"):
            relevant_segment_ids.update(seg_ids)

    logger.info(
        "OCR judge: %d unique texts judged in %.1fs",
        len(text_to_seg_ids), time.time() - t0,
    )

    relevant_frames = segments_to_frames(segments, relevant_segment_ids)
    logger.info(
        "OCR judge: %d/%d segments relevant, %d frames",
        len(relevant_segment_ids), len(segments), len(relevant_frames),
    )
    save_judge_cache(cache_dir, uid, relevant_frames, fps=current_fps)
    return relevant_frames


def judge_ocr_relevance_batched(
    question: str,
    options: dict,
    ocr_cache: dict,
    start_idx: int,
    end_idx: int,
    backend: Any,
    cfg: Any,
    batch_size: int = 20,
    uid: str = "",
    cache_dir: str = "",
) -> List[int]:
    """Pack up to ``batch_size`` unique snippets per LLM call. Returns frame indices."""
    current_fps = float(ocr_cache.get("fps", 2.0)) if isinstance(ocr_cache, dict) else None
    cached = load_judge_cache(cache_dir, uid, current_fps=current_fps)
    if cached is not None:
        logger.info("OCR judge: cache hit for %s (%d frames)", uid, len(cached))
        return cached

    segments, text_to_seg_ids, text_to_raw, fps = prepare_segments(ocr_cache, start_idx, end_idx)
    if not segments:
        logger.info("No OCR text found in video")
        save_judge_cache(cache_dir, uid, [], fps=current_fps)
        return []

    logger.info(
        "OCR judge (batched, bs=%d): %d segments, %d unique texts",
        batch_size, len(segments), len(text_to_seg_ids),
    )

    judge_cfg = OcrJudgeCfg(
        max_new_tokens=max(64, batch_size * 5),
        temperature=0.1,
        do_sample=False,
    )
    options_text = format_options(options)
    relevant_segment_ids = set()
    items = [(key, text_to_raw[key]) for key in text_to_seg_ids]

    t0 = time.time()
    n_batches = 0
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        numbered = [(i + 1, text) for i, (_k, text) in enumerate(batch)]
        prompt = OCR_JUDGE_BATCH_PROMPT.format(
            question=question,
            options=options_text,
            n_texts=len(batch),
            text_list=format_text_list(numbered),
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        try:
            response = backend.generate_text(messages, judge_cfg)
            verdicts = parse_batch_response(response, len(batch))
        except Exception as e:  # noqa: BLE001
            logger.warning("OCR batch call failed at offset %d (skipping): %r", start, e)
            verdicts = [False] * len(batch)
        n_batches += 1
        for (key, _text), is_relevant in zip(batch, verdicts):
            if is_relevant:
                relevant_segment_ids.update(text_to_seg_ids[key])

    logger.info(
        "OCR judge (batched): %d calls for %d texts in %.1fs",
        n_batches, len(items), time.time() - t0,
    )

    relevant_frames = segments_to_frames(segments, relevant_segment_ids)
    logger.info(
        "OCR judge: %d/%d segments relevant, %d frames",
        len(relevant_segment_ids), len(segments), len(relevant_frames),
    )
    save_judge_cache(cache_dir, uid, relevant_frames, fps=current_fps)
    return relevant_frames
