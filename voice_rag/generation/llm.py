"""
LLM Generation Module.

Exclusively powered by Sarvam AI (sarvam-105b, sarvam-2b) and an intelligent
local extractive QA fallback with citation generation.
"""

from __future__ import annotations

import re
import time
from typing import Optional, Sequence

from loguru import logger

from voice_rag.config import get_settings
from voice_rag.pipeline.schemas import GenerationResult, RetrievedChunk


SYSTEM_PROMPT = """You are an accurate, helpful multilingual AI assistant specialized in Indian knowledge and general facts.
Answer the user's question using ONLY the provided context passages.
If the context does not contain enough information to answer the question, state:
"I don't have enough information in the provided knowledge base to answer this question."

Rules:
1. Answer directly and concisely in the same language as the question (Hindi, English, etc.).
2. Do not hallucinate or use outside knowledge not present in the passages.
3. Cite sources using [Passage 1], [Passage 2], etc.
"""


def build_user_prompt(query: str, chunks: Sequence[RetrievedChunk]) -> str:
    """Format retrieved chunks and query into an LLM prompt with numbered citations."""
    context_parts: list[str] = []
    for i, item in enumerate(chunks, 1):
        context_parts.append(f"[Passage {i}]:\n{item.chunk.text.strip()}")

    context_str = "\n\n".join(context_parts)
    return f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"


class BaseLLM:
    """Abstract base class for LLM providers."""

    provider_name: str = "base"

    def generate(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
# Mock & Local Extractive LLM
# ═══════════════════════════════════════════════════════════════════════════

class MockLLM(BaseLLM):
    """
    Intelligent local extractive QA engine and deterministic fallback.
    Dynamically extracts relevant answers from retrieved database chunks.
    """

    provider_name = "mock"

    def generate(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        t0 = time.perf_counter()

        if not chunks:
            answer = "I don't have enough information in the provided knowledge base to answer this question."
            return GenerationResult(
                answer=answer,
                provider="mock",
                model="local-extractive-v1",
                prompt_tokens=len(query.split()),
                completion_tokens=len(answer.split()),
                generation_time_ms=0.1,
            )

        # Extract words from query (ignoring common stop words)
        stop_words = {"what", "is", "the", "of", "and", "in", "to", "are", "how", "does", "who", "where", "which", "a", "an", "for", "tell", "me", "about", "कहाँ", "कौन", "क्या", "है", "का", "की", "के", "में", "से", "पर", "और", "बारे", "बताओ"}
        query_words = set(w.lower() for w in re.split(r'[\s,.?!]+', query) if w.lower() and w.lower() not in stop_words)

        selected_sentences: list[tuple[str, int, float]] = []

        for idx, item in enumerate(chunks[:4]):
            passage_text = item.chunk.text.strip()
            sentences = re.split(r'(?<=[.?!।])\s+', passage_text)

            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) < 12:
                    continue
                s_words = set(re.split(r'[\s,.?!]+', s_clean.lower()))
                overlap = len(query_words.intersection(s_words))
                if query_words and overlap > 0:
                    score = overlap + (item.rerank_score * 0.5 if item.rerank_score else 0.5)
                    selected_sentences.append((s_clean, idx + 1, score))

        if selected_sentences:
            selected_sentences.sort(key=lambda x: x[2], reverse=True)
            top_sentences = []
            seen_texts = set()
            for s_text, p_num, _ in selected_sentences[:3]:
                if s_text not in seen_texts:
                    top_sentences.append(f"{s_text} [Passage {p_num}]")
                    seen_texts.add(s_text)
            answer = " ".join(top_sentences)
        else:
            top_text = chunks[0].chunk.text.strip()
            answer = f"{top_text} [Passage 1]"

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return GenerationResult(
            answer=answer,
            provider="mock",
            model="local-extractive-v1",
            prompt_tokens=len(query.split()),
            completion_tokens=len(answer.split()),
            generation_time_ms=max(0.5, elapsed_ms),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Sarvam AI LLM
# ═══════════════════════════════════════════════════════════════════════════

SARVAM_CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"


class SarvamLLM(BaseLLM):
    """Sarvam AI LLM provider (supporting sarvam-105b, sarvam-2b)."""

    provider_name = "sarvam"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        cfg = get_settings()
        self.model_name = model or cfg.sarvam_llm_model
        self.api_key = api_key or cfg.sarvam_api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not self.api_key:
            raise ValueError("SARVAM_API_KEY is required for Sarvam LLM provider")

        # Try to initialize official SarvamAI SDK
        self._sdk_client = None
        try:
            from sarvamai import SarvamAI
            self._sdk_client = SarvamAI(api_subscription_key=self.api_key)
        except Exception as e:
            logger.debug(f"Sarvam SDK client initialization: {e}")

        logger.info(f"Sarvam LLM initialised: {self.model_name}")

    def generate(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        import httpx
        user_prompt = build_user_prompt(query, chunks)
        sys_prompt = system_prompt or SYSTEM_PROMPT

        t0 = time.perf_counter()

        # Try SDK first
        if self._sdk_client:
            try:
                res = self._sdk_client.chat.completions(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000
                answer = res.choices[0].message.content if res.choices else ""
                return GenerationResult(
                    answer=answer,
                    provider="sarvam",
                    model=self.model_name,
                    prompt_tokens=len(user_prompt.split()),
                    completion_tokens=len(answer.split()),
                    generation_time_ms=elapsed_ms,
                )
            except Exception as e:
                logger.warning(f"Sarvam SDK chat call failed ({e}), trying REST fallback")

        # REST fallback
        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        with httpx.Client(timeout=45.0) as client:
            response = client.post(SARVAM_CHAT_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        choices = data.get("choices", [])
        answer = choices[0].get("message", {}).get("content", "") if choices else ""
        usage = data.get("usage", {})

        return GenerationResult(
            answer=answer,
            provider="sarvam",
            model=self.model_name,
            prompt_tokens=usage.get("prompt_tokens", len(user_prompt.split())),
            completion_tokens=usage.get("completion_tokens", len(answer.split())),
            generation_time_ms=elapsed_ms,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════

def get_llm(provider: Optional[str] = None) -> BaseLLM:
    """
    Create an LLM instance based on the configured provider.
    Strictly supports Sarvam AI and MockLLM.
    """
    cfg = get_settings()
    provider = provider or cfg.llm_provider

    if provider == "mock":
        return MockLLM()

    if provider == "sarvam":
        try:
            return SarvamLLM()
        except ValueError as exc:
            logger.warning(f"Sarvam LLM unavailable ({exc}), falling back to smart local extractor")
            return MockLLM()

    return MockLLM()
