"""
Retrieval evaluation harness.

Evaluates chunking strategies on MSMARCO-XI query-passage pairs using
ground-truth ``is_selected`` labels to compute:
  - Recall@1, Recall@5, Recall@10
  - MRR (Mean Reciprocal Rank)
  - Average retrieval latency
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np
from loguru import logger

from voice_rag.pipeline.schemas import ChunkMetadata, RetrievalMetrics, RetrievedChunk


def compute_recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Compute Recall@K: fraction of relevant docs in top-K results."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = top_k & relevant_ids
    return len(hits) / len(relevant_ids)


def compute_mrr(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    Returns 1/rank of the first relevant result, or 0 if none found.
    """
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    queries: Sequence[str],
    ground_truth: Sequence[set[str]],
    retriever_fn,
    strategy_name: str = "",
) -> RetrievalMetrics:
    """
    Evaluate a retrieval function on a set of queries with ground truth.

    Args:
        queries:       List of query strings.
        ground_truth:  List of sets of relevant chunk/document IDs per query.
        retriever_fn:  Callable(query) -> list[RetrievedChunk].
        strategy_name: Name label for reporting.

    Returns:
        RetrievalMetrics with recall and MRR scores.
    """
    recall_1_scores: list[float] = []
    recall_5_scores: list[float] = []
    recall_10_scores: list[float] = []
    mrr_scores: list[float] = []
    latencies: list[float] = []

    for query, relevant in zip(queries, ground_truth):
        t0 = time.perf_counter()
        results: list[RetrievedChunk] = retriever_fn(query)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        retrieved_ids = [rc.chunk.chunk_id for rc in results]

        recall_1_scores.append(compute_recall_at_k(retrieved_ids, relevant, k=1))
        recall_5_scores.append(compute_recall_at_k(retrieved_ids, relevant, k=5))
        recall_10_scores.append(compute_recall_at_k(retrieved_ids, relevant, k=10))
        mrr_scores.append(compute_mrr(retrieved_ids, relevant))

    num_queries = len(queries)
    return RetrievalMetrics(
        strategy=strategy_name,
        num_queries=num_queries,
        recall_at_1=float(np.mean(recall_1_scores)) if recall_1_scores else 0.0,
        recall_at_5=float(np.mean(recall_5_scores)) if recall_5_scores else 0.0,
        recall_at_10=float(np.mean(recall_10_scores)) if recall_10_scores else 0.0,
        mrr=float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        avg_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
    )


def format_metrics_table(metrics_list: Sequence[RetrievalMetrics]) -> str:
    """Format multiple RetrievalMetrics as a comparison table."""
    header = f"{'Strategy':<15} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'Lat(ms)':>8}"
    sep = "-" * len(header)
    rows = [header, sep]
    for m in metrics_list:
        rows.append(
            f"{m.strategy:<15} {m.recall_at_1:>6.3f} {m.recall_at_5:>6.3f} "
            f"{m.recall_at_10:>6.3f} {m.mrr:>6.3f} {m.avg_latency_ms:>8.1f}"
        )
    return "\n".join(rows)


if __name__ == "__main__":
    logger.info("Retrieval evaluation — use as library or run from evaluation scripts")
