"""
FAISS dense vector retrieval.

Loads a pre-built FAISS index and performs top-K approximate nearest
neighbour search using the multilingual embedding model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import faiss
import numpy as np
from loguru import logger

from voice_rag.config import get_settings
from voice_rag.indexing.build_faiss import load_faiss_index
from voice_rag.indexing.embeddings import Embedder, get_embedder
from voice_rag.pipeline.schemas import ChunkMetadata, RetrievedChunk


class DenseRetriever:
    """Top-K dense retrieval using FAISS inner-product search."""

    def __init__(
        self,
        index: Optional[faiss.Index] = None,
        chunks: Optional[list[ChunkMetadata]] = None,
        embedder: Optional[Embedder] = None,
        index_dir: Optional[Path] = None,
    ):
        if index is not None and chunks is not None:
            self._index = index
            self._chunks = chunks
        else:
            self._index, self._chunks = load_faiss_index(index_dir)
        self._embedder = embedder or get_embedder()

    @property
    def num_vectors(self) -> int:
        return self._index.ntotal

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve top-K chunks most similar to the query.

        Args:
            query: Natural language query string.
            top_k: Number of results to return (default from config).

        Returns:
            Sorted list of RetrievedChunk with dense_score populated.
        """
        cfg = get_settings()
        top_k = top_k or cfg.dense_top_k
        top_k = min(top_k, self._index.ntotal)

        # Encode query
        query_vec = self._embedder.encode_single(query, normalize=True)
        query_vec = query_vec.reshape(1, -1).astype(np.float32)

        # FAISS search (inner product = cosine on normalised vectors)
        scores, indices = self._index.search(query_vec, top_k)

        results: list[RetrievedChunk] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:  # FAISS returns -1 for missing entries
                continue
            results.append(
                RetrievedChunk(
                    chunk=self._chunks[idx],
                    dense_score=float(score),
                    final_rank=rank + 1,
                )
            )

        return results

    def batch_search(
        self,
        queries: Sequence[str],
        top_k: Optional[int] = None,
    ) -> list[list[RetrievedChunk]]:
        """Batch-search multiple queries."""
        cfg = get_settings()
        top_k = top_k or cfg.dense_top_k
        top_k = min(top_k, self._index.ntotal)

        query_vecs = self._embedder.encode(queries, normalize=True)
        query_vecs = query_vecs.astype(np.float32)
        scores, indices = self._index.search(query_vecs, top_k)

        all_results: list[list[RetrievedChunk]] = []
        for q_scores, q_indices in zip(scores, indices):
            results: list[RetrievedChunk] = []
            for rank, (score, idx) in enumerate(zip(q_scores, q_indices)):
                if idx < 0:
                    continue
                results.append(
                    RetrievedChunk(
                        chunk=self._chunks[idx],
                        dense_score=float(score),
                        final_rank=rank + 1,
                    )
                )
            all_results.append(results)

        return all_results
