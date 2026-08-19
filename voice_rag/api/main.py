"""
FastAPI application with REST + WebSocket endpoints.

Provides:
  - POST /api/query          — text query endpoint
  - POST /api/voice          — audio file upload + transcription + RAG
  - WS   /ws/voice           — streaming WebSocket voice interface
  - GET  /api/health         — health check
  - GET  /api/benchmark      — run latency benchmark
  - Static file serving for the frontend
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from voice_rag.config import PROJECT_ROOT, get_settings
from voice_rag.pipeline.orchestrator import RAGOrchestrator
from voice_rag.pipeline.schemas import PipelineResponse


# ═══════════════════════════════════════════════════════════════════════════
# App setup
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Voice-Enabled RAG System",
    description="HH Goa 2026 — Voice-First Retrieval-Augmented Generation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator (lazy-initialised)
_orchestrator: Optional[RAGOrchestrator] = None


def _get_orchestrator() -> RAGOrchestrator:
    """Lazy-load the orchestrator with available indices."""
    global _orchestrator
    if _orchestrator is not None:
        return _orchestrator

    cfg = get_settings()

    dense = None
    lexical = None
    reranker = None

    # Try loading FAISS index
    try:
        from voice_rag.retrieval.dense import DenseRetriever
        dense = DenseRetriever()
        logger.info(f"Dense retriever loaded: {dense.num_vectors} vectors")
    except Exception as exc:
        logger.warning(f"Dense retriever unavailable: {exc}")

    # Try loading BM25 index
    try:
        from voice_rag.retrieval.lexical import LexicalRetriever
        lexical = LexicalRetriever()
        logger.info(f"Lexical retriever loaded: {lexical.num_documents} documents")
    except Exception as exc:
        logger.warning(f"Lexical retriever unavailable: {exc}")

    # Try loading reranker
    try:
        from voice_rag.retrieval.reranker import get_reranker
        reranker = get_reranker()
        logger.info("Reranker loaded")
    except Exception as exc:
        logger.warning(f"Reranker unavailable: {exc}")

    _orchestrator = RAGOrchestrator(
        dense_retriever=dense,
        lexical_retriever=lexical,
        reranker=reranker,
    )
    return _orchestrator


# ═══════════════════════════════════════════════════════════════════════════
# Request / Response models
# ═══════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    query: str
    language: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    indices_loaded: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    orch = _get_orchestrator()
    has_indices = orch._dense is not None or orch._lexical is not None
    return HealthResponse(indices_loaded=has_indices)


@app.post("/api/query")
async def text_query(request: QueryRequest) -> dict:
    """
    Process a text query through the RAG pipeline.

    Returns full pipeline response with answer, sources, and telemetry.
    """
    orch = _get_orchestrator()
    response = await orch.process(query=request.query)
    return response.model_dump()


@app.post("/api/voice")
async def voice_query(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
) -> dict:
    """
    Process a voice query: upload audio → STT → RAG pipeline.
    """
    audio_data = await file.read()
    orch = _get_orchestrator()
    response = await orch.process(audio_data=audio_data)
    return response.model_dump()


@app.get("/api/benchmark")
async def run_benchmark(
    num_queries: int = 10,
    num_warmup: int = 2,
) -> dict:
    """
    Run a latency benchmark and return the report.
    """
    from voice_rag.evaluation.latency_eval import run_latency_benchmark

    sample_queries = [
        "What is the capital of India?",
        "How does photosynthesis work?",
        "What are the benefits of machine learning?",
        "What is the Taj Mahal?",
        "How many states are in India?",
    ]
    # Repeat to get enough queries
    queries = (sample_queries * ((num_queries // len(sample_queries)) + 1))[:num_queries + num_warmup]

    orch = _get_orchestrator()
    report = await run_latency_benchmark(queries, orch, num_warmup=num_warmup)
    return report.model_dump()


@app.websocket("/ws/voice")
async def websocket_voice(websocket: WebSocket):
    """
    WebSocket endpoint for streaming voice interaction.

    Protocol:
      Client sends: JSON with {"type": "audio", "data": "<base64>"} or {"type": "text", "query": "..."}
      Server sends: JSON PipelineResponse
    """
    await websocket.accept()
    orch = _get_orchestrator()

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "text":
                response = await orch.process(query=msg.get("query", ""))
            elif msg.get("type") == "audio":
                import base64
                audio_bytes = base64.b64decode(msg.get("data", ""))
                response = await orch.process(audio_data=audio_bytes)
            else:
                response = PipelineResponse(
                    final_answer="Unknown message type",
                    is_refused=True,
                    refusal_reason="Invalid WebSocket message type",
                )

            await websocket.send_text(response.model_dump_json())

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.error(f"WebSocket error: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
# Static files (frontend)
# ═══════════════════════════════════════════════════════════════════════════

_frontend_dir = PROJECT_ROOT / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    cfg = get_settings()
    uvicorn.run(
        "voice_rag.api.main:app",
        host=cfg.api_host,
        port=cfg.api_port,
        reload=True,
    )
