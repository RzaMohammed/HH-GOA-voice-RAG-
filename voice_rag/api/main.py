"""
FastAPI application with REST + WebSocket endpoints.

Provides:
  - POST /api/query          — text query endpoint with provider selection
  - POST /api/voice          — audio file upload + STT + vector DB retrieval + RAG
  - POST /api/transcribe     — audio transcription only (Sarvam / ElevenLabs)
  - POST /api/tts            — text-to-speech audio synthesis (Sarvam / ElevenLabs)
  - GET  /api/config         — active provider status and model choices
  - GET  /api/health         — health check
  - GET  /api/benchmark      — run latency benchmark
  - WS   /ws/voice           — real-time streaming WebSocket voice interface
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
from pydantic import BaseModel, Field

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
    llm_provider: Optional[str] = None
    auto_tts: bool = False
    tts_provider: Optional[str] = None


class TTSRequest(BaseModel):
    text: str
    language: Optional[str] = None
    provider: Optional[str] = None
    voice: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    indices_loaded: bool = False
    num_vectors: int = 0
    num_documents: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    orch = _get_orchestrator()
    num_vec = orch._dense.num_vectors if orch._dense else 0
    num_doc = orch._lexical.num_documents if orch._lexical else 0
    has_indices = num_vec > 0 or num_doc > 0
    return HealthResponse(
        indices_loaded=has_indices,
        num_vectors=num_vec,
        num_documents=num_doc,
    )


@app.get("/api/config")
async def get_config_status() -> dict:
    """Return configured providers and system status without revealing API keys."""
    cfg = get_settings()
    orch = _get_orchestrator()
    return {
        "providers": {
            "sarvam": bool(cfg.sarvam_api_key),
            "elevenlabs": bool(cfg.elevenlabs_api_key),
        },
        "defaults": {
            "llm_provider": cfg.llm_provider,
            "sarvam_llm_model": cfg.sarvam_llm_model,
            "chunk_strategy": cfg.default_chunk_strategy,
            "dataset_language": cfg.dataset_language,
        },
        "indices": {
            "faiss_vectors": orch._dense.num_vectors if orch._dense else 0,
            "bm25_documents": orch._lexical.num_documents if orch._lexical else 0,
        },
    }


@app.post("/api/query")
async def text_query(request: QueryRequest) -> dict:
    """
    Process a text query through the RAG pipeline.

    Returns full pipeline response with answer, sources, and telemetry.
    """
    orch = _get_orchestrator()
    response = await orch.process(
        query=request.query,
        language=request.language,
        llm_provider=request.llm_provider,
        auto_tts=request.auto_tts,
        tts_provider=request.tts_provider,
    )
    return response.model_dump()


@app.post("/api/voice")
async def voice_query(
    file: UploadFile = File(...),
    transcript: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    stt_provider: Optional[str] = Form(None),
    llm_provider: Optional[str] = Form(None),
    auto_tts: bool = Form(False),
    tts_provider: Optional[str] = Form(None),
) -> dict:
    """
    Process a voice query: upload audio → STT (Sarvam/ElevenLabs) → RAG pipeline → TTS.
    """
    audio_data = await file.read()
    orch = _get_orchestrator()
    response = await orch.process(
        audio_data=audio_data,
        query=transcript if transcript else None,
        language=language,
        stt_provider=stt_provider,
        llm_provider=llm_provider,
        auto_tts=auto_tts,
        tts_provider=tts_provider,
    )
    return response.model_dump()


@app.post("/api/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    provider: Optional[str] = Form(None),
) -> dict:
    """
    Process audio for transcription only using Sarvam AI or ElevenLabs.
    """
    audio_data = await file.read()
    from voice_rag.voice.sarvam_stt import get_stt
    stt = get_stt(provider=provider or "auto")
    try:
        stt_result = await stt.transcribe(audio_data, language=language)
    except Exception as exc:
        logger.warning(f"STT transcription failed ({exc}), falling back to client transcript")
        stt_result = STTResult(
            text="",
            language=language or "en",
            confidence=0.0,
            provider="browser",
        )
    return stt_result.model_dump()


@app.post("/api/tts")
async def text_to_speech(request: TTSRequest) -> dict:
    """
    Synthesize speech audio from text using Sarvam Bulbul or ElevenLabs.
    """
    from voice_rag.voice.tts import get_tts
    tts = get_tts(provider=request.provider)
    result = await tts.synthesize(
        text=request.text,
        language=request.language,
        voice=request.voice,
    )
    return result.model_dump()


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
    queries = (sample_queries * ((num_queries // len(sample_queries)) + 1))[:num_queries + num_warmup]

    orch = _get_orchestrator()
    report = await run_latency_benchmark(queries, orch, num_warmup=num_warmup)
    return report.model_dump()


@app.websocket("/ws/voice")
async def websocket_voice(websocket: WebSocket):
    """
    Real-time streaming WebSocket endpoint for live voice interaction.

    Protocol:
      Client sends:
        JSON with:
          - "type": "audio", "data": "<base64_audio>", "language": "hi", "stt_provider": "sarvam", "llm_provider": "sarvam", "auto_tts": true
          - "type": "text", "query": "...", "language": "hi", "llm_provider": "gemini"
      Server sends:
        JSON PipelineResponse with answer, retrieved chunks, grounding, audio_base64, and latency trace.
    """
    await websocket.accept()
    orch = _get_orchestrator()

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type", "text")
            language = msg.get("language")
            stt_provider = msg.get("stt_provider")
            llm_provider = msg.get("llm_provider")
            auto_tts = bool(msg.get("auto_tts", False))
            tts_provider = msg.get("tts_provider")

            if msg_type == "text":
                response = await orch.process(
                    query=msg.get("query", ""),
                    language=language,
                    llm_provider=llm_provider,
                    auto_tts=auto_tts,
                    tts_provider=tts_provider,
                )
            elif msg_type == "audio":
                import base64
                audio_bytes = base64.b64decode(msg.get("data", ""))
                response = await orch.process(
                    audio_data=audio_bytes,
                    query=msg.get("query"),
                    language=language,
                    stt_provider=stt_provider,
                    llm_provider=llm_provider,
                    auto_tts=auto_tts,
                    tts_provider=tts_provider,
                )
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
