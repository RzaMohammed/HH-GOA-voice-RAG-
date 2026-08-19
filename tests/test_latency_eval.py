"""Tests for latency evaluation metric calculations."""

import pytest

from voice_rag.evaluation.latency_eval import compute_percentiles, compute_stage_breakdown
from voice_rag.pipeline.schemas import LatencyTrace, StageLatency


class TestPercentileComputation:
    def test_basic_percentiles(self):
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        stats = compute_percentiles(latencies)

        assert stats["p50"] == pytest.approx(55.0, abs=1)
        assert stats["p100"] == pytest.approx(100.0)
        assert stats["mean"] == pytest.approx(55.0)
        assert stats["std"] > 0

    def test_single_value(self):
        stats = compute_percentiles([42.0])
        assert stats["p50"] == pytest.approx(42.0)
        assert stats["p70"] == pytest.approx(42.0)
        assert stats["p100"] == pytest.approx(42.0)
        assert stats["std"] == pytest.approx(0.0)

    def test_empty_list(self):
        stats = compute_percentiles([])
        assert stats["p50"] == 0
        assert stats["mean"] == 0

    def test_uniform_values(self):
        stats = compute_percentiles([5.0] * 100)
        assert stats["p50"] == pytest.approx(5.0)
        assert stats["p70"] == pytest.approx(5.0)
        assert stats["p100"] == pytest.approx(5.0)
        assert stats["std"] == pytest.approx(0.0)


class TestStageBreakdown:
    def test_average_computation(self):
        traces = [
            LatencyTrace(stages=[
                StageLatency(stage="retrieval", duration_ms=10.0),
                StageLatency(stage="generation", duration_ms=20.0),
            ]),
            LatencyTrace(stages=[
                StageLatency(stage="retrieval", duration_ms=30.0),
                StageLatency(stage="generation", duration_ms=40.0),
            ]),
        ]
        breakdown = compute_stage_breakdown(traces)

        assert breakdown["retrieval"] == pytest.approx(20.0)
        assert breakdown["generation"] == pytest.approx(30.0)

    def test_missing_stages(self):
        traces = [
            LatencyTrace(stages=[
                StageLatency(stage="retrieval", duration_ms=10.0),
            ]),
            LatencyTrace(stages=[
                StageLatency(stage="generation", duration_ms=20.0),
            ]),
        ]
        breakdown = compute_stage_breakdown(traces)

        assert breakdown["retrieval"] == pytest.approx(10.0)
        assert breakdown["generation"] == pytest.approx(20.0)

    def test_empty_traces(self):
        breakdown = compute_stage_breakdown([])
        assert breakdown == {}
