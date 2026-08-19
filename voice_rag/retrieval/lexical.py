"""
BM25 lexical retrieval.

Loads a pre-built BM25Okapi index and performs top-K keyword search
with the Indic-aware word tokeniser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from voice_rag.config import get_settings
from voice_rag.data.preprocessing import preprocess_text, tokenize_words
from voice_rag.indexing.build_bm25 import load_bm25_index
from voice_rag.pipeline.schemas import ChunkMetadata, RetrievedChunk


class LexicalRetriever:
    """Top-K lexical retrieval using BM25Okapi."""

    def __init__(
        self,
        bm25: Optional[BM25Okapi] = None,
        chunks: Optional[list[ChunkMetadata]] = None,
        index_dir: Optional[Path] = None,
    ):
        if bm25 is not None and chunks is not None:
            self._bm25 = bm25
            self._chunks = chunks
        else:
            self._bm25, self._chunks = load_bm25_index(index_dir)

    @property
    def num_documents(self) -> int:
        return len(self._chunks)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve top-K chunks via BM25 keyword scoring.

        Args:
            query: Natural language query string.
            top_k: Number of results to return (default from config).

        Returns:
            Sorted list of RetrievedChunk with bm25_score populated.
        """
        cfg = get_settings()
        top_k = top_k or cfg.bm25_top_k
        top_k = min(top_k, len(self._chunks))

        # Tokenise query the same way as the corpus
        query_tokens = tokenize_words(preprocess_text(query).lower())

        if not query_tokens:
            return []

        # Get BM25 scores for all documents
        scores = self._bm25.get_scores(query_tokens)

        # Get top-K indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[RetrievedChunk] = []
        for rank, idx in enumerate(top_indices):
            score = float(scores[idx])
            if score <= 0:
                continue  # Skip zero-score documents
            results.append(
                RetrievedChunk(
                    chunk=self._chunks[idx],
                    bm25_score=score,
                    final_rank=rank + 1,
                )
            )

        return results
