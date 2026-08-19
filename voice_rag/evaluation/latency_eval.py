"""
Latency evaluation benchmark harness.

Runs N warmup + M test queries through the pipeline, computing honest
P50, P70, P100 percentiles, standard deviation, and per-stage breakdowns.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional, Sequence

import numpy as np
from loguru import logger

from voice_rag.pipeline.orchestrator import RAGOrchestrator
from voice_rag.pipeline.schemas import LatencyReport, LatencyTrace, PipelineResponse


# ═══════════════════════════════════════════════════════════════════════════
# Percentile computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_percentiles(latencies: Sequence[float]) -> dict[str, float]:
    """
    Compute P50, P70, P100, mean, and std from a list of latencies (ms).
    """
    if not latencies:
        return {"p50": 0, "p70": 0, "p100": 0, "mean": 0, "std": 0}

    arr = np.array(latencies, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p70": float(np.percentile(arr, 70)),
        "p100": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def compute_stage_breakdown(traces: Sequence[LatencyTrace]) -> dict[str, float]:
    """
    Compute average duration per stage across all traces.
    """
    stage_totals: dict[str, list[float]] = {}
    for trace in traces:
        for stage in trace.stages:
            if stage.stage not in stage_totals:
                stage_totals[stage.stage] = []
            stage_totals[stage.stage].append(stage.duration_ms)

    return {
        stage: float(np.mean(durations))
        for stage, durations in stage_totals.items()
    }


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════════════════════

async def run_latency_benchmark(
    queries: Sequence[str],
    orchestrator: RAGOrchestrator,
    num_warmup: int = 3,
    num_test: Optional[int] = None,
) -> LatencyReport:
    """
    Run a latency benchmark with warmup and test phases.

    Args:
        queries:       List of test queries.
        orchestrator:  Configured RAG orchestrator.
        num_warmup:    Number of warmup queries (excluded from stats).
        num_test:      Number of test queries (defaults to all remaining).

    Returns:
        LatencyReport with P50, P70, P100 and stage breakdowns.
    """
    all_queries = list(queries)

    # Warmup phase
    warmup_queries = all_queries[:num_warmup]
    test_queries = all_queries[num_warmup:]
    if num_test:
        test_queries = test_queries[:num_test]

    logger.info(f"Warmup: {len(warmup_queries)} queries")
    for q in warmup_queries:
        await orchestrator.process(query=q)

    # Test phase
    logger.info(f"Benchmark: {len(test_queries)} queries")
    latencies: list[float] = []
    traces: list[LatencyTrace] = []

    for q in test_queries:
        t0 = time.perf_counter()
        response = await orchestrator.process(query=q)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)
        traces.append(response.latency)

    # Compute stats
    stats = compute_percentiles(latencies)
    breakdown = compute_stage_breakdown(traces)

    report = LatencyReport(
        num_queries=len(test_queries),
        num_warmup=len(warmup_queries),
        p50_ms=stats["p50"],
        p70_ms=stats["p70"],
        p100_ms=stats["p100"],
        mean_ms=stats["mean"],
        std_ms=stats["std"],
        stage_breakdown=breakdown,
    )

    return report


def format_latency_report(report: LatencyReport) -> str:
    """Format a LatencyReport as a human-readable string."""
    lines = [
        "╔══════════════════════════════════════════╗",
        "║     Latency Benchmark Report             ║",
        "╠══════════════════════════════════════════╣",
        f"║  Queries:  {report.num_queries:>5}  (warmup: {report.num_warmup})      ║",
        f"║  P50:     {report.p50_ms:>8.1f} ms                 ║",
        f"║  P70:     {report.p70_ms:>8.1f} ms                 ║",
        f"║  P100:    {report.p100_ms:>8.1f} ms                 ║",
        f"║  Mean:    {report.mean_ms:>8.1f} ms                 ║",
        f"║  Std:     {report.std_ms:>8.1f} ms                 ║",
        "╠══════════════════════════════════════════╣",
        "║  Stage Breakdown (avg ms)                ║",
        "╠──────────────────────────────────────────╣",
    ]
    for stage, avg_ms in report.stage_breakdown.items():
        lines.append(f"║  {stage:<25} {avg_ms:>8.1f}     ║")
    lines.append("╚══════════════════════════════════════════╝")
    return "\n".join(lines)


if __name__ == "__main__":
    from voice_rag.pipeline.orchestrator import RAGOrchestrator

    # Sample queries for benchmarking
    sample_queries = [
        "What is the capital of India?",
        "How does photosynthesis work?",
        "What are the benefits of machine learning?",
        "What is the Taj Mahal?",
        "How many states are in India?",
    ] * 4  # 20 queries total

    orch = RAGOrchestrator()  # Uses mock LLM and no retriever
    report = asyncio.run(run_latency_benchmark(sample_queries, orch, num_warmup=3))
    print(format_latency_report(report))
