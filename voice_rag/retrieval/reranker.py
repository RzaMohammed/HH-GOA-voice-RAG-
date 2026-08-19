"""
Cross-encoder reranker.

Takes a candidate set (top 20-50 from hybrid retrieval) and reranks them
using a cross-encoder model to produce a high-precision top 3-5 set.
"""

from __future__ import annotations

import threading
from typing import Optional, Sequence

from loguru import logger

from voice_rag.config import get_settings
from voice_rag.pipeline.schemas import RetrievedChunk


# ═══════════════════════════════════════════════════════════════════════════
# Cross-Encoder wrapper
# ═══════════════════════════════════════════════════════════════════════════

class CrossEncoderReranker:
    """Reranks candidate chunks using a cross-encoder model."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        from sentence_transformers import CrossEncoder
        import torch

        cfg = get_settings()
        self.model_name = model_name or cfg.reranker_model

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info(f"Loading cross-encoder reranker: {self.model_name} on {self.device}")
        self._model = CrossEncoder(self.model_name, device=self.device)
        logger.info("Reranker model ready")

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """
        Rerank candidates using cross-encoder relevance scoring.

        Args:
            query:      The user query.
            candidates: Pre-filtered candidate chunks.
            top_k:      Number of top results to return.

        Returns:
            Reranked list of RetrievedChunk with rerank_score populated.
        """
        cfg = get_settings()
        top_k = top_k or cfg.rerank_top_k

        if not candidates:
            return []

        # Build query-passage pairs for the cross-encoder
        pairs = [(query, c.chunk.text) for c in candidates]

        # Score all pairs
        scores = self._model.predict(pairs, show_progress_bar=False)

        # Attach scores and sort
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)

        results: list[RetrievedChunk] = []
        for rank, (rc, score) in enumerate(scored[:top_k]):
            reranked = RetrievedChunk(
                chunk=rc.chunk,
                dense_score=rc.dense_score,
                bm25_score=rc.bm25_score,
                hybrid_score=rc.hybrid_score,
                rerank_score=float(score),
                final_rank=rank + 1,
            )
            results.append(reranked)

        return results


# ═══════════════════════════════════════════════════════════════════════════
# Singleton management
# ═══════════════════════════════════════════════════════════════════════════

_lock = threading.Lock()
_reranker: Optional[CrossEncoderReranker] = None


def get_reranker(model_name: Optional[str] = None) -> CrossEncoderReranker:
    """Return (and cache) the global CrossEncoderReranker instance."""
    global _reranker
    with _lock:
        if _reranker is None or (model_name and model_name != _reranker.model_name):
            _reranker = CrossEncoderReranker(model_name=model_name)
    return _reranker


def reset_reranker() -> None:
    """Clear the cached reranker (useful for tests)."""
    global _reranker
    with _lock:
        _reranker = None
