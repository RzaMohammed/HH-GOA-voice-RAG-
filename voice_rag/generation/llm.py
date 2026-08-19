"""
Multi-provider LLM interface.

Supports Gemini, Groq, OpenAI, and a deterministic zero-latency mock
for offline testing and latency benchmarking. All providers expose a
unified ``generate()`` method returning a ``GenerationResult``.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional, Sequence

from loguru import logger

from voice_rag.config import get_settings
from voice_rag.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from voice_rag.pipeline.schemas import GenerationResult, LLMProvider, RetrievedChunk


# ═══════════════════════════════════════════════════════════════════════════
# Base class
# ═══════════════════════════════════════════════════════════════════════════

class BaseLLM(ABC):
    """Abstract base for LLM providers."""

    provider_name: str = "base"

    @abstractmethod
    def generate(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        """Generate a grounded answer from the query and retrieved chunks."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Mock LLM (zero latency, deterministic)
# ═══════════════════════════════════════════════════════════════════════════

class MockLLM(BaseLLM):
    """Deterministic mock LLM for testing and benchmarking."""

    provider_name = "mock"

    def generate(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        if not chunks:
            answer = "I don't have enough information in the provided knowledge base to answer this question."
        else:
            top_text = chunks[0].chunk.text[:200]
            answer = f"Based on the retrieved passages: {top_text}... [Passage 1]"

        return GenerationResult(
            answer=answer,
            provider="mock",
            model="mock-v1",
            prompt_tokens=len(query.split()),
            completion_tokens=len(answer.split()),
            generation_time_ms=0.1,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Gemini LLM
# ═══════════════════════════════════════════════════════════════════════════

class GeminiLLM(BaseLLM):
    """Google Gemini API provider."""

    provider_name = "gemini"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        cfg = get_settings()
        self.model_name = model or cfg.llm_model
        self.api_key = api_key or cfg.gemini_api_key

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini provider")

        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self._model = genai.GenerativeModel(self.model_name)
        logger.info(f"Gemini LLM initialised: {self.model_name}")

    def generate(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        user_prompt = build_user_prompt(query, chunks)
        sys = system_prompt or SYSTEM_PROMPT

        t0 = time.perf_counter()
        response = self._model.generate_content(
            f"{sys}\n\n{user_prompt}",
            generation_config={"temperature": 0.1, "max_output_tokens": 1024},
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        answer = response.text or ""
        usage = getattr(response, "usage_metadata", None)

        return GenerationResult(
            answer=answer,
            provider="gemini",
            model=self.model_name,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
            generation_time_ms=elapsed_ms,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Groq LLM
# ═══════════════════════════════════════════════════════════════════════════

class GroqLLM(BaseLLM):
    """Groq API provider (fast inference)."""

    provider_name = "groq"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        cfg = get_settings()
        self.model_name = model or "llama-3.1-8b-instant"
        self.api_key = api_key or cfg.groq_api_key

        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required for Groq provider")

        from groq import Groq
        self._client = Groq(api_key=self.api_key)
        logger.info(f"Groq LLM initialised: {self.model_name}")

    def generate(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        user_prompt = build_user_prompt(query, chunks)

        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        answer = response.choices[0].message.content or ""
        usage = response.usage

        return GenerationResult(
            answer=answer,
            provider="groq",
            model=self.model_name,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            generation_time_ms=elapsed_ms,
        )


# ═══════════════════════════════════════════════════════════════════════════
# OpenAI LLM
# ═══════════════════════════════════════════════════════════════════════════

class OpenAILLM(BaseLLM):
    """OpenAI API provider."""

    provider_name = "openai"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        cfg = get_settings()
        self.model_name = model or "gpt-4o-mini"
        self.api_key = api_key or cfg.openai_api_key

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI provider")

        from openai import OpenAI
        self._client = OpenAI(api_key=self.api_key)
        logger.info(f"OpenAI LLM initialised: {self.model_name}")

    def generate(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        user_prompt = build_user_prompt(query, chunks)

        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        answer = response.choices[0].message.content or ""
        usage = response.usage

        return GenerationResult(
            answer=answer,
            provider="openai",
            model=self.model_name,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            generation_time_ms=elapsed_ms,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════

def get_llm(provider: Optional[str] = None) -> BaseLLM:
    """
    Create an LLM instance based on the configured provider.

    Falls back to MockLLM if the requested provider's API key is missing.
    """
    cfg = get_settings()
    provider = provider or cfg.llm_provider

    _PROVIDERS = {
        "gemini": GeminiLLM,
        "groq": GroqLLM,
        "openai": OpenAILLM,
        "mock": MockLLM,
    }

    if provider == "mock":
        return MockLLM()

    cls = _PROVIDERS.get(provider)
    if cls is None:
        logger.warning(f"Unknown LLM provider '{provider}', falling back to mock")
        return MockLLM()

    try:
        return cls()
    except ValueError as exc:
        logger.warning(f"LLM provider '{provider}' unavailable ({exc}), falling back to mock")
        return MockLLM()
