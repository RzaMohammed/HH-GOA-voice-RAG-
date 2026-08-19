"""
Pydantic data models used across the entire Voice-RAG pipeline.

Every stage — from STT output through retrieval, generation, grounding
verification, and telemetry — speaks through these typed schemas.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"
    ADAPTIVE = "adaptive"


class GuardrailVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


class GroundingStatus(str, Enum):
    GROUNDED = "grounded"
    PARTIALLY_GROUNDED = "partially_grounded"
    UNGROUNDED = "ungrounded"
    REFUSED = "refused"


class LLMProvider(str, Enum):
    GEMINI = "gemini"
    GROQ = "groq"
    OPENAI = "openai"
    MOCK = "mock"


# ═══════════════════════════════════════════════════════════════════════════
# Chunk & Document Models
# ═══════════════════════════════════════════════════════════════════════════

class ChunkMetadata(BaseModel):
    """A single chunk of text with full provenance metadata."""
    chunk_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    document_id: str = ""
    query_id: str = ""
    text: str
    language: str = "en"
    chunk_strategy: ChunkStrategy = ChunkStrategy.FIXED
    is_selected: bool = False  # ground-truth relevance label from MSMARCO-XI
    word_count: int = 0
    passage_index: int = -1  # original passage index within document

    def model_post_init(self, __context: Any) -> None:
        if self.word_count == 0:
            self.word_count = len(self.text.split())


class DocumentRecord(BaseModel):
    """A single record from the MSMARCO-XI dataset."""
    query_id: str = ""
    query: str = ""
    query_type: str = ""
    answer: str = ""
    passages: list[dict[str, Any]] = Field(default_factory=list)
    translated_passages: list[dict[str, Any]] = Field(default_factory=list)
    language: str = "en"


# ═══════════════════════════════════════════════════════════════════════════
# Retrieval Models
# ═══════════════════════════════════════════════════════════════════════════

class RetrievedChunk(BaseModel):
    """A chunk returned by the retrieval pipeline with scores."""
    chunk: ChunkMetadata
    dense_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0
    final_rank: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# Guardrail Models
# ═══════════════════════════════════════════════════════════════════════════

class GuardrailResult(BaseModel):
    """Output from any guardrail check."""
    name: str
    verdict: GuardrailVerdict
    score: float = 1.0
    reason: str = ""


class GroundingResult(BaseModel):
    """Grounding verification output."""
    status: GroundingStatus
    confidence: float = 0.0
    evidence_overlap: float = 0.0
    flagged_claims: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# STT Models
# ═══════════════════════════════════════════════════════════════════════════

class STTResult(BaseModel):
    """Speech-to-text transcription result."""
    text: str
    language: str = "en"
    confidence: float = 1.0
    provider: str = "mock"
    duration_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Generation Models
# ═══════════════════════════════════════════════════════════════════════════

class GenerationResult(BaseModel):
    """LLM generation output."""
    answer: str
    provider: str = "mock"
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    generation_time_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Telemetry & Latency Trace
# ═══════════════════════════════════════════════════════════════════════════

class StageLatency(BaseModel):
    """Timing for a single pipeline stage."""
    stage: str
    start_ms: float = 0.0
    end_ms: float = 0.0
    duration_ms: float = 0.0


class LatencyTrace(BaseModel):
    """Full pipeline latency breakdown."""
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    total_ms: float = 0.0
    stages: list[StageLatency] = Field(default_factory=list)

    def start_stage(self, stage_name: str) -> StageLatency:
        """Begin timing a new stage."""
        s = StageLatency(stage=stage_name, start_ms=time.perf_counter() * 1000)
        self.stages.append(s)
        return s

    def end_stage(self, stage: StageLatency) -> None:
        """End timing for a stage."""
        stage.end_ms = time.perf_counter() * 1000
        stage.duration_ms = stage.end_ms - stage.start_ms

    def finalize(self) -> None:
        """Compute total latency from stages."""
        if self.stages:
            self.total_ms = sum(s.duration_ms for s in self.stages)


# ═══════════════════════════════════════════════════════════════════════════
# Full Pipeline Response
# ═══════════════════════════════════════════════════════════════════════════

class PipelineResponse(BaseModel):
    """Complete response from the RAG pipeline."""
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    query: str = ""
    stt_result: Optional[STTResult] = None
    guardrail_results: list[GuardrailResult] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    generation: Optional[GenerationResult] = None
    grounding: Optional[GroundingResult] = None
    final_answer: str = ""
    is_refused: bool = False
    refusal_reason: str = ""
    latency: LatencyTrace = Field(default_factory=LatencyTrace)


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation Models
# ═══════════════════════════════════════════════════════════════════════════

class RetrievalMetrics(BaseModel):
    """Retrieval quality metrics for a single evaluation run."""
    strategy: str = ""
    num_queries: int = 0
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    avg_latency_ms: float = 0.0


class LatencyReport(BaseModel):
    """Latency benchmark report."""
    num_queries: int = 0
    num_warmup: int = 0
    p50_ms: float = 0.0
    p70_ms: float = 0.0
    p100_ms: float = 0.0
    mean_ms: float = 0.0
    std_ms: float = 0.0
    stage_breakdown: dict[str, float] = Field(default_factory=dict)
