"""
Async typed pipeline orchestrator.

Connects all stages of the Voice-RAG pipeline with:
  - Per-stage timing telemetry
  - Multilingual voice STT (Sarvam AI / ElevenLabs)
  - Parallel hybrid retrieval (FAISS dense + BM25 lexical)
  - Score fusion & Cross-Encoder reranking
  - Multi-provider grounded LLM generation (Sarvam, Gemini, Groq, OpenAI)
  - Grounding verification & hallucination guardrails
  - Optional real-time TTS voice synthesis
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional, Union

from loguru import logger

from voice_rag.config import get_settings
from voice_rag.generation.grounding import verify_grounding
from voice_rag.generation.llm import BaseLLM, get_llm
from voice_rag.generation.prompts import REFUSAL_NO_EVIDENCE, REFUSAL_OFF_TOPIC, REFUSAL_SAFETY
from voice_rag.guardrails.confidence import check_confidence
from voice_rag.guardrails.relevance import check_relevance
from voice_rag.guardrails.safety import check_safety
from voice_rag.pipeline.schemas import (
    GenerationResult,
    GroundingStatus,
    GuardrailResult,
    GuardrailVerdict,
    LatencyTrace,
    PipelineResponse,
    RetrievedChunk,
    STTResult,
)
from voice_rag.retrieval.dense import DenseRetriever
from voice_rag.retrieval.hybrid import FusionMethod, hybrid_retrieve
from voice_rag.retrieval.lexical import LexicalRetriever
from voice_rag.retrieval.reranker import CrossEncoderReranker


class RAGOrchestrator:
    """
    End-to-end async RAG pipeline orchestrator.

    Stages:
      1. STT (optional, if audio provided via Sarvam or ElevenLabs)
      2. Query guardrails (safety + relevance)
      3. Parallel Retrieval (FAISS Dense + BM25 Lexical)
      4. Hybrid fusion (RRF or weighted)
      5. Cross-encoder reranking
      6. Confidence gating
      7. LLM generation (Sarvam, Gemini, Groq, OpenAI)
      8. Grounding verification
      9. Text-to-Speech (optional, if auto_tts enabled)
    """

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        lexical_retriever: Optional[LexicalRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        llm: Optional[BaseLLM] = None,
        fusion_method: FusionMethod = FusionMethod.RRF,
    ):
        self._dense = dense_retriever
        self._lexical = lexical_retriever
        self._reranker = reranker
        self._llm = llm or get_llm()
        self._fusion_method = fusion_method

    def _make_refusal(
        self,
        query: str,
        reason: str,
        refusal_text: str,
        trace: LatencyTrace,
        guardrail_results: Optional[list[GuardrailResult]] = None,
        stt_result: Optional[STTResult] = None,
        retrieved_chunks: Optional[list[RetrievedChunk]] = None,
    ) -> PipelineResponse:
        """Build a refusal response."""
        trace.finalize()
        return PipelineResponse(
            query=query,
            stt_result=stt_result,
            guardrail_results=guardrail_results or [],
            retrieved_chunks=retrieved_chunks or [],
            final_answer=refusal_text,
            is_refused=True,
            refusal_reason=reason,
            latency=trace,
        )

    async def process(
        self,
        query: Optional[str] = None,
        stt_result: Optional[STTResult] = None,
        audio_data: Optional[Union[bytes, str]] = None,
        language: Optional[str] = None,
        stt_provider: Optional[str] = None,
        llm_provider: Optional[str] = None,
        auto_tts: bool = False,
        tts_provider: Optional[str] = None,
    ) -> PipelineResponse:
        """
        Run the full RAG pipeline.

        Args:
            query: Direct text query
            stt_result: Pre-transcribed STT result
            audio_data: Raw audio bytes to transcribe first
            language: Spoken / query language code hint
            stt_provider: STT engine ('sarvam', 'elevenlabs', 'auto', 'mock')
            llm_provider: LLM engine ('sarvam', 'gemini', 'groq', 'openai', 'mock')
            auto_tts: Whether to synthesize speech audio for the final answer
            tts_provider: TTS engine ('sarvam', 'elevenlabs', 'auto')
        """
        trace = LatencyTrace()
        response = PipelineResponse(latency=trace)

        # ── Stage 1: STT ──────────────────────────────────────────────
        if audio_data is not None:
            stage = trace.start_stage("stt")
            try:
                from voice_rag.voice.sarvam_stt import get_stt
                stt = get_stt(provider=stt_provider or "auto")
                stt_res = await stt.transcribe(audio_data, language=language)
                if query and stt_res.provider == "mock":
                    # Use client's real-time spoken transcript if backend STT is mock
                    stt_res.text = query
                else:
                    query = stt_res.text or query
                response.stt_result = stt_res
            except Exception as exc:
                logger.warning(f"STT failed with {stt_provider or 'auto'}: {exc}")
                if query:
                    # Fall back to client transcript
                    logger.info("Using client speech transcript fallback")
                    response.stt_result = STTResult(text=query, language=language or "en", provider="browser", confidence=0.95)
                else:
                    trace.end_stage(stage)
                    return self._make_refusal(
                        "", "Speech transcription unavailable", "Speech transcription was unavailable. Please speak clearly into your microphone or type your question in the box.", trace
                    )
            trace.end_stage(stage)

        if stt_result and not query:
            query = stt_result.text
            response.stt_result = stt_result

        if not query:
            return self._make_refusal("", "No query provided", "Please provide a question.", trace)

        response.query = query

        # ── Stage 2: Safety guardrail ─────────────────────────────────
        stage = trace.start_stage("guardrail_safety")
        safety = check_safety(query)
        response.guardrail_results.append(safety)
        trace.end_stage(stage)

        if safety.verdict == GuardrailVerdict.FAIL:
            return self._make_refusal(
                query, safety.reason, REFUSAL_SAFETY, trace,
                guardrail_results=response.guardrail_results,
                stt_result=response.stt_result,
            )

        # ── Stage 3: Relevance guardrail ──────────────────────────────
        stage = trace.start_stage("guardrail_relevance")
        relevance = check_relevance(query)
        response.guardrail_results.append(relevance)
        trace.end_stage(stage)

        if relevance.verdict == GuardrailVerdict.FAIL:
            return self._make_refusal(
                query, relevance.reason, REFUSAL_OFF_TOPIC, trace,
                guardrail_results=response.guardrail_results,
                stt_result=response.stt_result,
            )

        # ── Stage 4 & 5: Parallel hybrid retrieval ─────────────────────
        dense_results: list[RetrievedChunk] = []
        lexical_results: list[RetrievedChunk] = []

        if self._dense and self._lexical:
            stage = trace.start_stage("retrieval")
            try:
                loop = asyncio.get_running_loop()
                dense_future = loop.run_in_executor(None, self._dense.search, query)
                lexical_future = loop.run_in_executor(None, self._lexical.search, query)
                dense_results, lexical_results = await asyncio.gather(
                    dense_future, lexical_future
                )
            except Exception as exc:
                logger.error(f"Parallel retrieval failed: {exc}")
            trace.end_stage(stage)
        elif self._dense:
            stage = trace.start_stage("retrieval")
            dense_results = self._dense.search(query)
            trace.end_stage(stage)
        elif self._lexical:
            stage = trace.start_stage("retrieval")
            lexical_results = self._lexical.search(query)
            trace.end_stage(stage)

        # ── Stage 6: Hybrid fusion ────────────────────────────────────
        if dense_results or lexical_results:
            stage = trace.start_stage("fusion")
            fused = hybrid_retrieve(
                dense_results, lexical_results, method=self._fusion_method
            )
            trace.end_stage(stage)
        else:
            fused = []

        # ── Stage 7: Cross-Encoder Reranking ──────────────────────────
        if self._reranker and fused:
            stage = trace.start_stage("reranking")
            try:
                reranked = self._reranker.rerank(query, fused)
            except Exception as exc:
                logger.error(f"Reranking failed: {exc}, using fusion results")
                reranked = fused[:get_settings().rerank_top_k]
            trace.end_stage(stage)
        else:
            reranked = fused[:get_settings().rerank_top_k]

        response.retrieved_chunks = reranked

        # ── Stage 8: Confidence gating ────────────────────────────────
        stage = trace.start_stage("guardrail_confidence")
        confidence = check_confidence(reranked)
        response.guardrail_results.append(confidence)
        trace.end_stage(stage)

        if confidence.verdict == GuardrailVerdict.FAIL:
            return self._make_refusal(
                query, confidence.reason, REFUSAL_NO_EVIDENCE, trace,
                guardrail_results=response.guardrail_results,
                stt_result=response.stt_result,
                retrieved_chunks=response.retrieved_chunks,
            )

        # ── Stage 9: LLM generation ──────────────────────────────────
        stage = trace.start_stage("generation")
        llm_engine = get_llm(llm_provider) if llm_provider else self._llm
        try:
            gen_result = llm_engine.generate(query, reranked)
            response.generation = gen_result
        except Exception as exc:
            logger.warning(f"LLM generation failed with {llm_provider or 'default'} ({exc}), falling back to local extractor")
            try:
                from voice_rag.generation.llm import MockLLM
                gen_result = MockLLM().generate(query, reranked)
                response.generation = gen_result
            except Exception as exc2:
                trace.end_stage(stage)
                return self._make_refusal(
                    query, f"Generation failed: {exc}", REFUSAL_NO_EVIDENCE, trace,
                    guardrail_results=response.guardrail_results,
                    stt_result=response.stt_result,
                )
        trace.end_stage(stage)

        # ── Stage 10: Grounding verification ──────────────────────────
        stage = trace.start_stage("grounding")
        grounding = verify_grounding(gen_result.answer, reranked)
        response.grounding = grounding
        trace.end_stage(stage)

        if grounding.status == GroundingStatus.UNGROUNDED:
            response.final_answer = REFUSAL_NO_EVIDENCE
            response.is_refused = True
            response.refusal_reason = (
                f"Answer failed grounding check — "
                f"{len(grounding.flagged_claims)} ungrounded claims"
            )
        else:
            response.final_answer = gen_result.answer

        # ── Stage 11: Text-to-Speech (Optional) ───────────────────────
        if auto_tts and response.final_answer and not response.is_refused:
            stage = trace.start_stage("tts")
            try:
                from voice_rag.voice.tts import get_tts
                tts = get_tts(tts_provider)
                # Determine language hint for TTS
                tts_lang = language or (response.stt_result.language if response.stt_result else "hi-IN")
                tts_res = await tts.synthesize(response.final_answer, language=tts_lang)
                if tts_res.audio_base64:
                    response.audio_base64 = tts_res.audio_base64
                    response.audio_mime_type = tts_res.mime_type
            except Exception as exc:
                logger.warning(f"Auto-TTS synthesis skipped/failed: {exc}")
            trace.end_stage(stage)

        trace.finalize()
        return response


# ═══════════════════════════════════════════════════════════════════════════
# Convenience
# ═══════════════════════════════════════════════════════════════════════════

async def run_query(query: str, orchestrator: Optional[RAGOrchestrator] = None) -> PipelineResponse:
    """Quick helper to run a query through the pipeline."""
    if orchestrator is None:
        orchestrator = RAGOrchestrator()
    return await orchestrator.process(query=query)
