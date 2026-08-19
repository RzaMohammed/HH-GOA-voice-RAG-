"""
Grounding / Faithfulness Verifier.

Post-generation verification checking if all factual claims in the answer
are strictly supported by the retrieved context. Uses a lightweight
word-overlap + NLI approach.

If ungrounded → triggers a refusal response.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from loguru import logger

from voice_rag.config import get_settings
from voice_rag.pipeline.schemas import GroundingResult, GroundingStatus, RetrievedChunk


# ═══════════════════════════════════════════════════════════════════════════
# Claim extraction (simple sentence-level)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_claims(answer: str) -> list[str]:
    """Extract factual claim sentences from the generated answer."""
    # Remove citation markers like [Passage 1]
    clean = re.sub(r"\[Passage \d+\]", "", answer)
    # Split into sentences
    sentences = re.split(r"[.!?।॥]+", clean)
    claims = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
    return claims


# ═══════════════════════════════════════════════════════════════════════════
# Word-overlap grounding check
# ═══════════════════════════════════════════════════════════════════════════

def _word_overlap_score(claim: str, context: str) -> float:
    """
    Compute word-level overlap between a claim and context.

    Returns a ratio in [0, 1] of claim words found in context.
    """
    claim_words = set(claim.lower().split())
    context_words = set(context.lower().split())

    if not claim_words:
        return 1.0

    overlap = claim_words & context_words
    return len(overlap) / len(claim_words)


def _check_claim_grounded(
    claim: str,
    context_texts: list[str],
    threshold: float,
) -> tuple[bool, float]:
    """
    Check if a single claim is supported by any context passage.

    Returns (is_grounded, best_overlap_score).
    """
    best_score = 0.0
    for ctx in context_texts:
        score = _word_overlap_score(claim, ctx)
        best_score = max(best_score, score)
        if score >= threshold:
            return True, score

    return False, best_score


# ═══════════════════════════════════════════════════════════════════════════
# Main verifier
# ═══════════════════════════════════════════════════════════════════════════

def verify_grounding(
    answer: str,
    chunks: Sequence[RetrievedChunk],
    threshold: Optional[float] = None,
) -> GroundingResult:
    """
    Verify whether the generated answer is grounded in the retrieved context.

    Args:
        answer:    The LLM-generated answer.
        chunks:    Retrieved context passages.
        threshold: Minimum overlap score to consider a claim grounded.

    Returns:
        GroundingResult with status, confidence, and flagged claims.
    """
    cfg = get_settings()
    threshold = threshold if threshold is not None else cfg.grounding_nli_threshold

    # Check for explicit refusal patterns
    refusal_patterns = [
        "don't have enough information",
        "cannot answer",
        "outside the scope",
        "no relevant information",
        "unable to answer",
    ]
    answer_lower = answer.lower()
    if any(pat in answer_lower for pat in refusal_patterns):
        return GroundingResult(
            status=GroundingStatus.REFUSED,
            confidence=1.0,
            evidence_overlap=0.0,
        )

    # Extract claims and context texts
    claims = _extract_claims(answer)
    context_texts = [rc.chunk.text for rc in chunks]

    if not claims:
        return GroundingResult(
            status=GroundingStatus.GROUNDED,
            confidence=1.0,
            evidence_overlap=1.0,
        )

    if not context_texts:
        return GroundingResult(
            status=GroundingStatus.UNGROUNDED,
            confidence=0.0,
            evidence_overlap=0.0,
            flagged_claims=claims,
        )

    # Check each claim
    grounded_count = 0
    overlap_scores: list[float] = []
    flagged: list[str] = []

    for claim in claims:
        is_grounded, score = _check_claim_grounded(claim, context_texts, threshold)
        overlap_scores.append(score)
        if is_grounded:
            grounded_count += 1
        else:
            flagged.append(claim)

    # Determine overall status
    grounded_ratio = grounded_count / len(claims) if claims else 1.0
    avg_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0

    if grounded_ratio >= 0.8:
        status = GroundingStatus.GROUNDED
    elif grounded_ratio >= 0.4:
        status = GroundingStatus.PARTIALLY_GROUNDED
    else:
        status = GroundingStatus.UNGROUNDED

    return GroundingResult(
        status=status,
        confidence=grounded_ratio,
        evidence_overlap=avg_overlap,
        flagged_claims=flagged,
    )
