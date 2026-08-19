"""
Text preprocessing utilities for Indic and English passage text.

Handles Unicode normalisation, whitespace cleanup, deduplication via
content hashing, and basic Indic script-aware tokenisation.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable

from loguru import logger


# ═══════════════════════════════════════════════════════════════════════════
# Unicode & Whitespace Normalisation
# ═══════════════════════════════════════════════════════════════════════════

def normalize_unicode(text: str) -> str:
    """Apply NFC normalisation and strip invisible/control chars."""
    text = unicodedata.normalize("NFC", text)
    # Remove zero-width chars, BOM, and other invisible codepoints
    text = re.sub(r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad]", "", text)
    return text


def clean_whitespace(text: str) -> str:
    """Collapse runs of whitespace, strip leading/trailing."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
# Indic-aware sentence splitting
# ═══════════════════════════════════════════════════════════════════════════

# Matches Indic dandas (।, ॥) and standard Western punctuation (.!?)
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[।॥\.!\?])\s+"
)


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences, respecting both English (.!?) and
    Indic (। ॥) sentence boundaries.
    """
    parts = _SENTENCE_BOUNDARY.split(text)
    return [s.strip() for s in parts if s.strip()]


# ═══════════════════════════════════════════════════════════════════════════
# Full preprocessing pipeline
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_text(text: str) -> str:
    """
    Full cleaning pipeline for a single piece of text:
    1. Unicode NFC normalisation
    2. Strip invisible chars
    3. Collapse whitespace
    """
    text = normalize_unicode(text)
    text = clean_whitespace(text)
    return text


def content_hash(text: str) -> str:
    """Return SHA-256 hex digest of cleaned text for deduplication."""
    return hashlib.sha256(preprocess_text(text).encode("utf-8")).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════
# Batch deduplication
# ═══════════════════════════════════════════════════════════════════════════

def deduplicate_texts(texts: Iterable[str]) -> list[str]:
    """Remove exact-duplicate texts (after normalisation)."""
    seen: set[str] = set()
    unique: list[str] = []
    for t in texts:
        h = content_hash(t)
        if h not in seen:
            seen.add(h)
            unique.append(t)
    return unique


# ═══════════════════════════════════════════════════════════════════════════
# Indic-aware tokeniser (word-level)
# ═══════════════════════════════════════════════════════════════════════════

# Splits on whitespace and common Indic punctuation
_WORD_SPLIT = re.compile(r"[\s।॥,;:!?\"\'\(\)\[\]\{\}]+")


def tokenize_words(text: str) -> list[str]:
    """
    Simple word-level tokeniser that handles both English and Indic scripts.
    Splits on whitespace and punctuation boundaries.
    """
    tokens = _WORD_SPLIT.split(text)
    return [t for t in tokens if t]
