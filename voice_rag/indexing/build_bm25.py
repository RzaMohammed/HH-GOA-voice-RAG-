"""
Offline BM25 index builder.

Tokenises chunks with the Indic-aware word tokeniser, builds a
``BM25Okapi`` index from ``rank_bm25``, and persists the index
alongside chunk metadata for fast online lexical search.

Persists:
  - ``bm25_index.pkl``   — pickled BM25Okapi instance
  - ``bm25_chunks.json`` — ordered chunk metadata
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional, Sequence

from loguru import logger
from rank_bm25 import BM25Okapi

from voice_rag.config import get_settings
from voice_rag.data.preprocessing import preprocess_text, tokenize_words
from voice_rag.pipeline.schemas import ChunkMetadata


# ═══════════════════════════════════════════════════════════════════════════
# Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_bm25_index(
    chunks: Sequence[ChunkMetadata],
    index_dir: Optional[Path] = None,
) -> tuple[BM25Okapi, list[ChunkMetadata]]:
    """
    Build a BM25 index from chunks and persist to disk.

    Args:
        chunks:    Pre-chunked text segments with metadata.
        index_dir: Directory to save the index files.

    Returns:
        Tuple of (BM25Okapi, ordered_chunks).
    """
    cfg = get_settings()
    index_dir = Path(index_dir or cfg.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # --- Tokenise ---
    logger.info(f"Tokenising {len(chunks)} chunks for BM25...")
    tokenised_corpus = []
    for c in chunks:
        tokens = tokenize_words(preprocess_text(c.text).lower())
        tokenised_corpus.append(tokens)

    # --- Build BM25 ---
    logger.info("Building BM25Okapi index...")
    bm25 = BM25Okapi(tokenised_corpus)

    # --- Persist ---
    bm25_path = index_dir / "bm25_index.pkl"
    chunk_map_path = index_dir / "bm25_chunks.json"

    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    logger.info(f"BM25 index saved to {bm25_path}")

    ordered_chunks = list(chunks)
    chunk_dicts = [c.model_dump() for c in ordered_chunks]
    with open(chunk_map_path, "w", encoding="utf-8") as f:
        json.dump(chunk_dicts, f, ensure_ascii=False, indent=1)
    logger.info(f"BM25 chunk map saved to {chunk_map_path}")

    return bm25, ordered_chunks


# ═══════════════════════════════════════════════════════════════════════════
# Loader
# ═══════════════════════════════════════════════════════════════════════════

def load_bm25_index(
    index_dir: Optional[Path] = None,
) -> tuple[BM25Okapi, list[ChunkMetadata]]:
    """
    Load a previously-built BM25 index and chunk map from disk.

    Returns:
        Tuple of (BM25Okapi, ordered_chunks).
    """
    cfg = get_settings()
    index_dir = Path(index_dir or cfg.index_dir)

    bm25_path = index_dir / "bm25_index.pkl"
    chunk_map_path = index_dir / "bm25_chunks.json"

    if not bm25_path.exists():
        raise FileNotFoundError(f"BM25 index not found at {bm25_path}")
    if not chunk_map_path.exists():
        raise FileNotFoundError(f"BM25 chunk map not found at {chunk_map_path}")

    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    logger.info(f"Loaded BM25 index from {bm25_path}")

    with open(chunk_map_path, "r", encoding="utf-8") as f:
        chunk_dicts = json.load(f)
    chunks = [ChunkMetadata.model_validate(d) for d in chunk_dicts]
    logger.info(f"Loaded {len(chunks)} chunks from {chunk_map_path}")

    return bm25, chunks
