"""
Retrieval confidence gating.

Checks if the top retrieval candidates meet a minimum score threshold
before forwarding to the LLM. If evidence is too weak, triggers a safe
refusal rather than risking hallucinated output.
"""

from __future__ import annotations

from typing import Optional, Sequence

from loguru import logger

from voice_rag.config import get_settings
from voice_rag.pipeline.schemas import GuardrailResult, GuardrailVerdict, RetrievedChunk


def check_confidence(
    chunks: Sequence[RetrievedChunk],
    min_score: Optional[float] = None,
    min_chunks: int = 1,
) -> GuardrailResult:
    """
    Gate retrieval results by score confidence.

    Args:
        chunks:     Retrieved and ranked chunks.
        min_score:  Minimum acceptable score for the top result.
        min_chunks: Minimum number of chunks above threshold.

    Returns:
        GuardrailResult indicating whether evidence is sufficient.
    """
    cfg = get_settings()
    min_score = min_score if min_score is not None else cfg.min_retrieval_score

    if not chunks:
        return GuardrailResult(
            name="confidence",
            verdict=GuardrailVerdict.FAIL,
            score=0.0,
            reason="No retrieval results available",
        )

    # Use the best available score (prefer rerank > hybrid > dense > bm25)
    def _best_score(rc: RetrievedChunk) -> float:
        if rc.rerank_score > 0:
            return rc.rerank_score
        if rc.hybrid_score > 0:
            return rc.hybrid_score
        if rc.dense_score > 0:
            return rc.dense_score
        return rc.bm25_score

    top_score = _best_score(chunks[0])
    above_threshold = sum(1 for c in chunks if _best_score(c) >= min_score)

    if top_score < min_score:
        return GuardrailResult(
            name="confidence",
            verdict=GuardrailVerdict.FAIL,
            score=top_score,
            reason=f"Top retrieval score ({top_score:.3f}) below threshold ({min_score})",
        )

    if above_threshold < min_chunks:
        return GuardrailResult(
            name="confidence",
            verdict=GuardrailVerdict.WARN,
            score=top_score,
            reason=f"Only {above_threshold} chunks above threshold (need {min_chunks})",
        )

    return GuardrailResult(
        name="confidence",
        verdict=GuardrailVerdict.PASS,
        score=top_score,
        reason=f"Top score {top_score:.3f}, {above_threshold} chunks above threshold",
    )
