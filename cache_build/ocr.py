"""Build EasyOCR per-video caches (text per frame, no bbox).

Output: ``{"ocr_results": [[{"text": str}, ...], ...], "fps", "num_frames", "video_id"}``
at ``{video_id}.ocr_cache``. The inference-time OCR judge (``toolmerge/tools/ocr_judge.py``)
reads this cache and runs an LLM over the per-frame text strings.

`embed_ocr_caches` is an optional post-pass that embeds the unique OCR strings
with EmbeddingGemma so downstream baselines (e.g., SigLIP-Q) can rank by
OCR-string similarity. Writes ``{video_id}.ocr_embed_cache``.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import torch

from cache_build.utils import (
    CV2Reader,
    LOAD_CHUNK_SIZE,
    TARGET_FPS,
    get_frame_indices,
    load_chunk,
)


OCR_DETECT_BATCH = 64    # frames per readtext_batched call (CRAFT batch); 128 OOMs on A40 48GB
OCR_RECOG_BATCH = 512    # boxes per recognizer forward pass


def ocr_process_batch(reader, frames_rgb):
    """Batched CRAFT detection + per-frame batched recognition via EasyOCR.

    Returns ``list[list[{"text": str}]]`` aligned with ``frames_rgb``.
    """
    batch_np = np.stack(frames_rgb, axis=0)  # (N, H, W, 3) uint8 RGB
    results_per_frame = reader.readtext_batched(
        batch_np, batch_size=OCR_RECOG_BATCH, detail=0,
    )
    out = []
    for frame_results in results_per_frame:
        out.append(
            [{"text": t.strip()} for t in frame_results
             if isinstance(t, str) and t.strip()]
        )
    return out


def build_ocr(video_path, video_id, output_dir, reader,
              max_nframes=None, backend="decord"):
    """Run EasyOCR over every sampled frame; save per-frame text strings."""
    out_path = os.path.join(output_dir, f"{video_id}.ocr_cache")
    frame_idx, nframes, vr = get_frame_indices(
        video_path, max_nframes=max_nframes, backend=backend,
    )

    all_results = []
    buf = []

    def flush():
        if not buf:
            return
        all_results.extend(ocr_process_batch(reader, buf))
        buf.clear()

    if isinstance(vr, CV2Reader):
        for frame_batch in vr.iter_chunks(frame_idx, chunk_size=LOAD_CHUNK_SIZE):
            for frame_np in frame_batch:
                buf.append(frame_np)
                if len(buf) >= OCR_DETECT_BATCH:
                    flush()
            del frame_batch
            gc.collect()
        flush()
    else:
        for start in range(0, nframes, LOAD_CHUNK_SIZE):
            end = min(start + LOAD_CHUNK_SIZE, nframes)
            chunk = load_chunk(vr, frame_idx[start:end])
            for i in range(chunk.shape[0]):
                frame_np = chunk[i].permute(1, 2, 0).numpy().astype(np.uint8)
                buf.append(frame_np)
                if len(buf) >= OCR_DETECT_BATCH:
                    flush()
            del chunk
            gc.collect()
        flush()

    cache_obj = {
        "ocr_results": all_results,
        "fps": TARGET_FPS,
        "num_frames": len(all_results),
        "video_id": video_id,
    }
    torch.save(cache_obj, out_path)
    n_det = sum(len(fr) for fr in all_results)
    del vr
    gc.collect()
    return n_det, nframes


def embed_ocr_caches(ocr_output_dir, embed_dim=768):
    """Post-pass: embed unique OCR strings per video with EmbeddingGemma."""
    from sentence_transformers import SentenceTransformer
    MODEL_ID = "google/embeddinggemma-300m"

    cache_files = sorted(Path(ocr_output_dir).glob("*.ocr_cache"))
    if not cache_files:
        print("No OCR caches to embed")
        return

    print(f"Embedding {len(cache_files)} OCR caches with {MODEL_ID}...")
    model = SentenceTransformer(MODEL_ID, model_kwargs={"torch_dtype": torch.bfloat16})

    for cache_path in cache_files:
        video_id = cache_path.stem
        embed_path = cache_path.parent / f"{video_id}.ocr_embed_cache"
        if embed_path.exists():
            continue

        ocr_data = torch.load(str(cache_path), map_location="cpu", weights_only=False)
        ocr_results = ocr_data["ocr_results"]

        string_to_idx = {}
        strings = []
        frame_string_indices = []
        for frame_dets in ocr_results:
            frame_indices = []
            for det in frame_dets:
                text = det["text"].strip()
                if not text:
                    continue
                if text not in string_to_idx:
                    string_to_idx[text] = len(strings)
                    strings.append(text)
                frame_indices.append(string_to_idx[text])
            frame_string_indices.append(frame_indices)

        if not strings:
            embeddings = torch.zeros(0, embed_dim, dtype=torch.bfloat16)
        else:
            embeddings = model.encode_document(
                strings, batch_size=512, show_progress_bar=False,
                convert_to_tensor=True,
                truncate_dim=embed_dim if embed_dim < 768 else None,
            )
            embeddings = torch.nn.functional.normalize(embeddings.float(), p=2, dim=1)
            embeddings = embeddings.to(torch.bfloat16).cpu()

        result = {
            "strings": strings,
            "embeddings": embeddings,
            "string_to_idx": string_to_idx,
            "frame_string_indices": frame_string_indices,
            "model_id": MODEL_ID,
            "embed_dim": embed_dim,
            "video_id": video_id,
        }
        torch.save(result, str(embed_path))
        print(f"  {video_id}: {len(strings)} strings -> {list(embeddings.shape)}")

    print("OCR embedding done")
