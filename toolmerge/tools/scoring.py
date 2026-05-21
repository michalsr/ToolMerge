"""Frame scoring for SigLIP-2 and T-REN tools.

Both tools return a per-frame percentile rank in [0, 1] (best frame = 1.0).
This makes scores cross-tool comparable so the AND/OR merge in
``toolmerge.merging`` can combine them by min / max.

"""

from __future__ import annotations

import logging
from typing import List, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def normalize_to_percentiles(results: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """Convert raw scores to percentile ranks in [0, 1]. Higher = better.

    Each frame's value becomes the fraction of frames that score <= it, so
    the best frame is 1.0 and the worst is 0.0. Ties share the same
    percentile. Cross-tool comparable by construction.
    """
    if not results:
        return results
    n = len(results)
    if n == 1:
        return [(results[0][0], 1.0)]
    by_score = sorted(results, key=lambda x: x[1])
    percentiles = {idx: rank / (n - 1) for rank, (idx, _) in enumerate(by_score)}
    out = [(idx, percentiles[idx]) for idx in percentiles]
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def score_siglip(
    query: str,
    feature_cache: dict,
    start_idx: int,
    end_idx: int,
) -> List[Tuple[int, float]]:
    """SigLIP-2 text-frame cosine similarity, returned as percentiles.

    Args:
        query: query string.
        feature_cache: ``{"client": SiglipClient, "embeddings": (T, D) tensor}``.
        start_idx: first frame to score (inclusive).
        end_idx: last frame to score (exclusive).
    """
    client = feature_cache["client"]
    embeddings = feature_cache["embeddings"]

    text_feat = client.encode_texts(query)
    if isinstance(text_feat, torch.Tensor):
        text_feat = text_feat.float()
        emb_slice = embeddings[start_idx:end_idx].float()
    else:
        import numpy as np  # local import
        text_feat = torch.from_numpy(text_feat).float()
        emb_slice = (
            embeddings[start_idx:end_idx].float()
            if isinstance(embeddings, torch.Tensor)
            else torch.from_numpy(embeddings[start_idx:end_idx]).float()
        )

    if text_feat.ndim > 1:
        text_feat = text_feat.squeeze(0)

    text_feat = F.normalize(text_feat.unsqueeze(0), p=2, dim=1).squeeze(0)
    emb_slice = F.normalize(emb_slice, p=2, dim=1)

    scores = torch.matmul(emb_slice, text_feat)  # (N,)

    results = [(start_idx + i, float(s)) for i, s in enumerate(scores.tolist())]
    results.sort(key=lambda x: x[1], reverse=True)
    return normalize_to_percentiles(results)


def score_tren(
    query: str,
    tren_cache: dict,
    start_idx: int,
    end_idx: int,
    tren_client,
) -> List[Tuple[int, float]]:
    """T-REN region-text alignment per-frame scores, returned as percentiles.

    ``tren_client.get_frame_scores(cache, query) -> (T,)`` is computed once
    per (query, video) and percentile-normalized here for the AND/OR merge.
    """
    all_scores = tren_client.get_frame_scores(tren_cache, query)
    if isinstance(all_scores, torch.Tensor):
        all_scores = all_scores.cpu()
    end_idx = min(end_idx, len(all_scores))
    results = [(i, float(all_scores[i])) for i in range(start_idx, end_idx)]
    results.sort(key=lambda x: x[1], reverse=True)
    return normalize_to_percentiles(results)
