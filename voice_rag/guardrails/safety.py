"""
Content safety & prompt injection guardrail.

Detects harmful, malicious, or prompt-injection inputs using
keyword pattern matching and heuristic scoring.
"""

from __future__ import annotations

import re

from loguru import logger

from voice_rag.config import get_settings
from voice_rag.pipeline.schemas import GuardrailResult, GuardrailVerdict


# ═══════════════════════════════════════════════════════════════════════════
# Dangerous patterns
# ═══════════════════════════════════════════════════════════════════════════

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|rules|prompts)",
    r"you\s+are\s+now\s+",
    r"pretend\s+(you\s+are|to\s+be)",
    r"act\s+as\s+if",
    r"disregard\s+(your|all|the)\s+",
    r"override\s+(your|the|system)",
    r"forget\s+(everything|all|your)",
    r"new\s+instructions?:",
    r"system\s+prompt",
    r"jailbreak",
    r"DAN\s+mode",
]

_HARMFUL_PATTERNS = [
    r"how\s+to\s+(make|build|create)\s+(a\s+)?(bomb|weapon|explosive|poison)",
    r"how\s+to\s+(hack|crack|break\s+into)",
    r"how\s+to\s+steal",
    r"how\s+to\s+kill",
    r"self.?harm",
    r"suicide\s+method",
]

_COMPILED_INJECTION = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]
_COMPILED_HARMFUL = [re.compile(p, re.IGNORECASE) for p in _HARMFUL_PATTERNS]


# ═══════════════════════════════════════════════════════════════════════════
# Safety checker
# ═══════════════════════════════════════════════════════════════════════════

def check_safety(query: str) -> GuardrailResult:
    """
    Check a query for safety violations and prompt injection attempts.

    Returns:
        GuardrailResult with verdict PASS, WARN, or FAIL.
    """
    cfg = get_settings()

    if not cfg.safety_enabled:
        return GuardrailResult(
            name="safety",
            verdict=GuardrailVerdict.PASS,
            score=1.0,
            reason="Safety checks disabled",
        )

    # Check for prompt injection
    for pattern in _COMPILED_INJECTION:
        if pattern.search(query):
            logger.warning(f"Prompt injection detected: {pattern.pattern}")
            return GuardrailResult(
                name="safety",
                verdict=GuardrailVerdict.FAIL,
                score=0.0,
                reason=f"Potential prompt injection detected",
            )

    # Check for harmful content
    for pattern in _COMPILED_HARMFUL:
        if pattern.search(query):
            logger.warning(f"Harmful content detected: {pattern.pattern}")
            return GuardrailResult(
                name="safety",
                verdict=GuardrailVerdict.FAIL,
                score=0.0,
                reason="Potentially harmful content detected",
            )

    # Heuristic: excessive special chars or very long inputs
    special_ratio = sum(1 for c in query if not c.isalnum() and not c.isspace()) / max(len(query), 1)
    if special_ratio > 0.4:
        return GuardrailResult(
            name="safety",
            verdict=GuardrailVerdict.WARN,
            score=0.5,
            reason="Unusually high proportion of special characters",
        )

    return GuardrailResult(
        name="safety",
        verdict=GuardrailVerdict.PASS,
        score=1.0,
        reason="Query passed safety checks",
    )
