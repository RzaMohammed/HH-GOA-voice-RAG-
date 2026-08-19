"""Tests for retrieval evaluation metrics and hybrid fusion."""

import pytest

from voice_rag.evaluation.retrieval_eval import compute_mrr, compute_recall_at_k
from voice_rag.pipeline.schemas import ChunkMetadata, RetrievedChunk
from voice_rag.retrieval.hybrid import reciprocal_rank_fusion, weighted_score_fusion


class TestRecallAtK:
    def test_perfect_recall(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b"}
        assert compute_recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_zero_recall(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert compute_recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_partial_recall(self):
        retrieved = ["a", "x", "b"]
        relevant = {"a", "b", "c"}
        assert compute_recall_at_k(retrieved, relevant, k=2) == pytest.approx(1 / 3)

    def test_empty_relevant(self):
        assert compute_recall_at_k(["a"], set(), k=1) == 0.0

    def test_k_1(self):
        retrieved = ["a", "b"]
        relevant = {"a"}
        assert compute_recall_at_k(retrieved, relevant, k=1) == 1.0


class TestMRR:
    def test_first_is_relevant(self):
        assert compute_mrr(["a", "b", "c"], {"a"}) == 1.0

    def test_second_is_relevant(self):
        assert compute_mrr(["x", "a", "c"], {"a"}) == 0.5

    def test_no_relevant(self):
        assert compute_mrr(["x", "y", "z"], {"a"}) == 0.0


def _make_rc(chunk_id: str, dense_score: float = 0.0, bm25_score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=ChunkMetadata(chunk_id=chunk_id, text=f"Text for {chunk_id}"),
        dense_score=dense_score,
        bm25_score=bm25_score,
    )


class TestHybridFusion:
    def test_rrf_merges_results(self):
        dense = [_make_rc("a", dense_score=0.9), _make_rc("b", dense_score=0.7)]
        lexical = [_make_rc("b", bm25_score=5.0), _make_rc("c", bm25_score=3.0)]

        fused = reciprocal_rank_fusion(dense, lexical)

        ids = [r.chunk.chunk_id for r in fused]
        assert "a" in ids
        assert "b" in ids
        assert "c" in ids
        # b appears in both, should rank highest
        assert fused[0].chunk.chunk_id == "b"

    def test_rrf_ranks_assigned(self):
        dense = [_make_rc("a", dense_score=0.9)]
        lexical = [_make_rc("b", bm25_score=5.0)]

        fused = reciprocal_rank_fusion(dense, lexical)
        ranks = [r.final_rank for r in fused]
        assert ranks == [1, 2]

    def test_weighted_fusion(self):
        dense = [_make_rc("a", dense_score=0.9), _make_rc("b", dense_score=0.5)]
        lexical = [_make_rc("b", bm25_score=5.0), _make_rc("c", bm25_score=1.0)]

        fused = weighted_score_fusion(dense, lexical, alpha=0.5)
        assert len(fused) == 3
        assert all(r.hybrid_score >= 0 for r in fused)

    def test_empty_inputs(self):
        fused = reciprocal_rank_fusion([], [])
        assert fused == []
