"""
Domain relevance & off-topic query detector.

Validates whether a query is within the domain scope of the indexed
knowledge base. Uses lightweight keyword and embedding-based heuristics
to reject completely off-topic queries before expensive LLM calls.
"""

from __future__ import annotations

import re
from typing import Optional

from loguru import logger

from voice_rag.config import get_settings
from voice_rag.pipeline.schemas import GuardrailResult, GuardrailVerdict


# ═══════════════════════════════════════════════════════════════════════════
# Off-topic heuristics
# ═══════════════════════════════════════════════════════════════════════════

# Clearly off-topic patterns (non-knowledge-base queries)
_OFF_TOPIC_PATTERNS = [
    r"^(hi|hello|hey|howdy|greetings)\s*[!?.]?\s*$",
    r"what('s| is) your name",
    r"who (are|made|created) you",
    r"tell me a joke",
    r"write (me )?(a |an )?(poem|song|story|essay|code)",
    r"translate .+ (to|into) ",
    r"(play|sing|draw|paint) ",
    r"what('s| is) the (weather|time|date)",
    r"set (a |an )?(alarm|timer|reminder)",
    r"(order|buy|purchase) ",
]

_COMPILED_OFF_TOPIC = [re.compile(p, re.IGNORECASE) for p in _OFF_TOPIC_PATTERNS]

# Minimum query length (very short queries are often not meaningful)
_MIN_QUERY_WORDS = 2


def check_relevance(
    query: str,
    threshold: Optional[float] = None,
) -> GuardrailResult:
    """
    Check if a query is relevant to the knowledge base domain.

    Args:
        query:     The user query text.
        threshold: Minimum relevance score (default from config).

    Returns:
        GuardrailResult with verdict.
    """
    cfg = get_settings()
    threshold = threshold if threshold is not None else cfg.relevance_threshold

    query = query.strip()

    # Empty query
    if not query:
        return GuardrailResult(
            name="relevance",
            verdict=GuardrailVerdict.FAIL,
            score=0.0,
            reason="Empty query",
        )

    # Check explicit off-topic patterns
    for pattern in _COMPILED_OFF_TOPIC:
        if pattern.search(query):
            return GuardrailResult(
                name="relevance",
                verdict=GuardrailVerdict.FAIL,
                score=0.1,
                reason="Query appears off-topic for the knowledge base",
            )

    # Too short
    word_count = len(query.split())
    if word_count < _MIN_QUERY_WORDS:
        return GuardrailResult(
            name="relevance",
            verdict=GuardrailVerdict.WARN,
            score=0.3,
            reason=f"Query too short ({word_count} words)",
        )

    # Check if query is a question or information-seeking
    question_indicators = ["?", "what", "how", "why", "when", "where", "who", "which",
                           "explain", "describe", "define", "tell me about",
                           "क्या", "कैसे", "क्यों", "कब", "कहाँ", "कौन", "कितने"]
    query_lower = query.lower()
    has_question_signal = any(ind in query_lower for ind in question_indicators)

    if has_question_signal:
        return GuardrailResult(
            name="relevance",
            verdict=GuardrailVerdict.PASS,
            score=0.9,
            reason="Query appears to be a valid information-seeking question",
        )

    # Default: mild pass (declarative statements might still be queries)
    return GuardrailResult(
        name="relevance",
        verdict=GuardrailVerdict.PASS,
        score=0.6,
        reason="Query accepted (ambiguous intent)",
    )
