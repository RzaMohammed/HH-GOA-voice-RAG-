"""
Hybrid retrieval with Reciprocal Rank Fusion (RRF) and Weighted Score Fusion.

Combines dense (FAISS) and lexical (BM25) retrieval results into a single
ranked list using configurable fusion strategies.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from loguru import logger

from voice_rag.config import get_settings
from voice_rag.pipeline.schemas import RetrievedChunk


class FusionMethod(str, Enum):
    RRF = "rrf"
    WEIGHTED = "weighted"


# ═══════════════════════════════════════════════════════════════════════════
# Reciprocal Rank Fusion
# ═══════════════════════════════════════════════════════════════════════════

def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    lexical_results: list[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """
    Merge results using Reciprocal Rank Fusion (RRF).

    RRF score = Σ 1 / (k + rank) for each list the document appears in.
    Default k=60 following the original RRF paper.
    """
    # Build a map: chunk_id → merged RetrievedChunk
    merged: dict[str, RetrievedChunk] = {}

    for rank, rc in enumerate(dense_results, start=1):
        cid = rc.chunk.chunk_id
        if cid not in merged:
            merged[cid] = RetrievedChunk(
                chunk=rc.chunk,
                dense_score=rc.dense_score,
            )
        else:
            merged[cid].dense_score = rc.dense_score
        merged[cid].hybrid_score += 1.0 / (k + rank)

    for rank, rc in enumerate(lexical_results, start=1):
        cid = rc.chunk.chunk_id
        if cid not in merged:
            merged[cid] = RetrievedChunk(
                chunk=rc.chunk,
                bm25_score=rc.bm25_score,
            )
        else:
            merged[cid].bm25_score = rc.bm25_score
        merged[cid].hybrid_score += 1.0 / (k + rank)

    # Sort by hybrid score descending
    results = sorted(merged.values(), key=lambda r: r.hybrid_score, reverse=True)

    # Assign final ranks
    for i, rc in enumerate(results):
        rc.final_rank = i + 1

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Weighted / Convex Score Fusion
# ═══════════════════════════════════════════════════════════════════════════

def _min_max_normalize(values: list[float]) -> list[float]:
    """Min-max normalise a list of floats to [0, 1]."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    span = hi - lo
    if span == 0:
        return [1.0] * len(values)
    return [(v - lo) / span for v in values]


def weighted_score_fusion(
    dense_results: list[RetrievedChunk],
    lexical_results: list[RetrievedChunk],
    alpha: Optional[float] = None,
) -> list[RetrievedChunk]:
    """
    Merge results using Convex Score Fusion.

    hybrid_score = α · norm(dense_score) + (1-α) · norm(bm25_score)
    """
    cfg = get_settings()
    alpha = alpha if alpha is not None else cfg.hybrid_alpha

    # Collect all unique chunks
    merged: dict[str, RetrievedChunk] = {}

    for rc in dense_results:
        cid = rc.chunk.chunk_id
        merged[cid] = RetrievedChunk(
            chunk=rc.chunk,
            dense_score=rc.dense_score,
        )

    for rc in lexical_results:
        cid = rc.chunk.chunk_id
        if cid not in merged:
            merged[cid] = RetrievedChunk(chunk=rc.chunk)
        merged[cid].bm25_score = rc.bm25_score

    # Normalise scores within each source
    chunk_ids = list(merged.keys())
    dense_scores = [merged[cid].dense_score for cid in chunk_ids]
    bm25_scores = [merged[cid].bm25_score for cid in chunk_ids]

    norm_dense = _min_max_normalize(dense_scores)
    norm_bm25 = _min_max_normalize(bm25_scores)

    for i, cid in enumerate(chunk_ids):
        merged[cid].hybrid_score = alpha * norm_dense[i] + (1 - alpha) * norm_bm25[i]

    # Sort by hybrid score descending
    results = sorted(merged.values(), key=lambda r: r.hybrid_score, reverse=True)

    for i, rc in enumerate(results):
        rc.final_rank = i + 1

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Unified hybrid retrieval
# ═══════════════════════════════════════════════════════════════════════════

def hybrid_retrieve(
    dense_results: list[RetrievedChunk],
    lexical_results: list[RetrievedChunk],
    method: FusionMethod = FusionMethod.RRF,
    alpha: Optional[float] = None,
    top_k: Optional[int] = None,
) -> list[RetrievedChunk]:
    """
    Unified hybrid retrieval entry point.

    Args:
        dense_results:   Results from dense FAISS search.
        lexical_results: Results from BM25 search.
        method:          Fusion strategy (RRF or weighted).
        alpha:           Dense weight for weighted fusion.
        top_k:           Max results to return.

    Returns:
        Fused and ranked list of RetrievedChunk.
    """
    if method == FusionMethod.RRF:
        fused = reciprocal_rank_fusion(dense_results, lexical_results)
    else:
        fused = weighted_score_fusion(dense_results, lexical_results, alpha=alpha)

    if top_k:
        fused = fused[:top_k]
        for i, rc in enumerate(fused):
            rc.final_rank = i + 1

    return fused
