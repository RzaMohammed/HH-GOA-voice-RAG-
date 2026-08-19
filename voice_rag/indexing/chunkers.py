"""
Multi-strategy text chunking engine.

Implements four chunking strategies as required:
  1. Fixed / Word-window  — deterministic word-count based windows
  2. Sentence-aware       — packs sentences up to a target token budget
  3. Semantic             — splits at topic boundaries via embedding cosine dips
  4. Adaptive             — sentence-aware default with semantic escalation

All chunkers produce standardised ``ChunkMetadata`` objects.
"""

from __future__ import annotations

import uuid
from typing import Optional

import numpy as np
from loguru import logger

from voice_rag.config import get_settings
from voice_rag.data.preprocessing import preprocess_text, split_sentences, tokenize_words
from voice_rag.pipeline.schemas import ChunkMetadata, ChunkStrategy


# ═══════════════════════════════════════════════════════════════════════════
# 1. Fixed / Word-Window Chunker
# ═══════════════════════════════════════════════════════════════════════════

def chunk_fixed(
    text: str,
    window_size: Optional[int] = None,
    overlap: Optional[int] = None,
    **meta_kwargs,
) -> list[ChunkMetadata]:
    """
    Split text into fixed-size word windows with configurable overlap.

    Args:
        text:        Source text.
        window_size: Number of words per chunk (default from config).
        overlap:     Number of overlapping words between chunks.
        **meta_kwargs: Extra fields forwarded to ``ChunkMetadata``.
    """
    cfg = get_settings()
    window_size = window_size if window_size is not None else cfg.chunk_size_words
    overlap = overlap if overlap is not None else cfg.chunk_overlap_words
    step = max(1, window_size - overlap)

    text = preprocess_text(text)
    words = tokenize_words(text)

    if not words:
        return []

    chunks: list[ChunkMetadata] = []
    for i in range(0, len(words), step):
        window = words[i : i + window_size]
        if not window:
            break
        chunks.append(
            ChunkMetadata(
                text=" ".join(window),
                chunk_strategy=ChunkStrategy.FIXED,
                **meta_kwargs,
            )
        )
        # Stop if we've consumed all words
        if i + window_size >= len(words):
            break

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 2. Sentence-Aware Chunker
# ═══════════════════════════════════════════════════════════════════════════

def chunk_sentence(
    text: str,
    max_words: Optional[int] = None,
    **meta_kwargs,
) -> list[ChunkMetadata]:
    """
    Pack complete sentences into chunks up to a word budget.

    Respects both Indic (। ॥) and English (.!?) sentence boundaries.
    """
    cfg = get_settings()
    max_words = max_words if max_words is not None else cfg.chunk_size_words

    text = preprocess_text(text)
    sentences = split_sentences(text)

    if not sentences:
        return []

    chunks: list[ChunkMetadata] = []
    current_sentences: list[str] = []
    current_word_count = 0

    for sent in sentences:
        sent_words = len(tokenize_words(sent))
        if current_word_count + sent_words > max_words and current_sentences:
            # Flush current buffer
            chunks.append(
                ChunkMetadata(
                    text=" ".join(current_sentences),
                    chunk_strategy=ChunkStrategy.SENTENCE,
                    **meta_kwargs,
                )
            )
            current_sentences = []
            current_word_count = 0
        current_sentences.append(sent)
        current_word_count += sent_words

    # Flush remaining
    if current_sentences:
        chunks.append(
            ChunkMetadata(
                text=" ".join(current_sentences),
                chunk_strategy=ChunkStrategy.SENTENCE,
                **meta_kwargs,
            )
        )

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 3. Semantic Chunker
# ═══════════════════════════════════════════════════════════════════════════

def chunk_semantic(
    text: str,
    similarity_threshold: Optional[float] = None,
    embedder=None,
    **meta_kwargs,
) -> list[ChunkMetadata]:
    """
    Split at topic boundaries where consecutive sentence embedding
    cosine similarity drops below ``similarity_threshold``.

    If no embedder is available, falls back to sentence-aware chunking.

    Args:
        text:                 Source text.
        similarity_threshold: Cosine similarity breakpoint.
        embedder:             Object with ``.encode(texts) -> np.ndarray``.
    """
    cfg = get_settings()
    similarity_threshold = similarity_threshold or cfg.semantic_similarity_threshold

    text = preprocess_text(text)
    sentences = split_sentences(text)

    if len(sentences) <= 1:
        return chunk_sentence(text, **meta_kwargs)

    # Lazy-load embedder if not provided
    if embedder is None:
        try:
            from voice_rag.indexing.embeddings import get_embedder
            embedder = get_embedder()
        except Exception:
            logger.warning("Embedder unavailable for semantic chunking, falling back to sentence")
            return chunk_sentence(text, **meta_kwargs)

    try:
        embeddings = embedder.encode(sentences, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Normalise for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms

        # Compute consecutive cosine similarities
        similarities = np.array([
            float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)
        ])

        # Find split points where similarity drops below threshold
        chunks: list[ChunkMetadata] = []
        current_sentences: list[str] = [sentences[0]]

        for i, sim in enumerate(similarities):
            if sim < similarity_threshold:
                # Topic boundary — flush
                chunks.append(
                    ChunkMetadata(
                        text=" ".join(current_sentences),
                        chunk_strategy=ChunkStrategy.SEMANTIC,
                        **meta_kwargs,
                    )
                )
                current_sentences = []
            current_sentences.append(sentences[i + 1])

        # Flush remaining
        if current_sentences:
            chunks.append(
                ChunkMetadata(
                    text=" ".join(current_sentences),
                    chunk_strategy=ChunkStrategy.SEMANTIC,
                    **meta_kwargs,
                )
            )

        return chunks

    except Exception as exc:
        logger.warning(f"Semantic chunking failed ({exc}), falling back to sentence")
        return chunk_sentence(text, **meta_kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Adaptive Chunker
# ═══════════════════════════════════════════════════════════════════════════

def chunk_adaptive(
    text: str,
    max_words: Optional[int] = None,
    dense_threshold: int = 300,
    embedder=None,
    **meta_kwargs,
) -> list[ChunkMetadata]:
    """
    Sentence-aware default that escalates to semantic splitting for
    long or topically diverse passages.

    Uses word count as the complexity heuristic:
      - Short passages (≤ dense_threshold words): sentence chunking.
      - Long passages (> dense_threshold words): semantic chunking.
    """
    cfg = get_settings()
    max_words = max_words or cfg.chunk_size_words

    text = preprocess_text(text)
    word_count = len(tokenize_words(text))

    if word_count <= dense_threshold:
        return chunk_sentence(text, max_words=max_words, **meta_kwargs)
    else:
        return chunk_semantic(text, embedder=embedder, **meta_kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Unified entry point
# ═══════════════════════════════════════════════════════════════════════════

_CHUNKERS = {
    ChunkStrategy.FIXED: chunk_fixed,
    ChunkStrategy.SENTENCE: chunk_sentence,
    ChunkStrategy.SEMANTIC: chunk_semantic,
    ChunkStrategy.ADAPTIVE: chunk_adaptive,
}


def chunk_text(
    text: str,
    strategy: Optional[ChunkStrategy] = None,
    embedder=None,
    **meta_kwargs,
) -> list[ChunkMetadata]:
    """
    Chunk text using the specified (or default) strategy.

    Args:
        text:       Source text to chunk.
        strategy:   Chunking strategy (defaults to config).
        embedder:   Optional embedding model for semantic/adaptive.
        **meta_kwargs: Forwarded to ChunkMetadata (document_id, language, etc.).

    Returns:
        List of ChunkMetadata objects.
    """
    cfg = get_settings()
    if strategy is None:
        strategy = ChunkStrategy(cfg.default_chunk_strategy)

    chunker_fn = _CHUNKERS[strategy]

    kwargs = dict(**meta_kwargs)
    if strategy in (ChunkStrategy.SEMANTIC, ChunkStrategy.ADAPTIVE) and embedder is not None:
        kwargs["embedder"] = embedder

    return chunker_fn(text, **kwargs)
