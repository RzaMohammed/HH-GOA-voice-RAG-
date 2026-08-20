"""
Centralized configuration for the Voice-Enabled RAG System.

All tunables live here and are loaded from environment variables / .env file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings
from pydantic import Field

# ---------------------------------------------------------------------------
# Resolve project root (two levels up from this file)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data_cache"
INDEX_DIR = PROJECT_ROOT / "index_cache"


class Settings(BaseSettings):
    """Single source of truth for every tunable knob."""

    # ── LLM Providers ───────────────────────────────────────────────────
    gemini_api_key: str = ""
    groq_api_key: str = ""
    openai_api_key: str = ""

    # ── Voice / STT ─────────────────────────────────────────────────────
    sarvam_api_key: str = ""
    elevenlabs_api_key: str = ""

    # ── Model Choices ───────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_provider: Literal["gemini", "groq", "openai", "mock"] = "mock"
    llm_model: str = "gemini-2.0-flash"

    # ── Dataset ─────────────────────────────────────────────────────────
    dataset_name: str = "ai4bharat/MSMARCO-XI"
    dataset_split: str = "train"
    dataset_language: str = "hi"  # default Indic language: "hi" (Hindi), "bn", "te", "ta", etc.
    dataset_max_samples: int = 5000  # cap for rapid prototyping; 0 = unlimited
    dataset_streaming: bool = True

    # ── Chunking ────────────────────────────────────────────────────────
    default_chunk_strategy: Literal["fixed", "sentence", "semantic", "adaptive"] = "adaptive"
    chunk_size_words: int = 120
    chunk_overlap_words: int = 25
    semantic_similarity_threshold: float = 0.65

    # ── Retrieval ───────────────────────────────────────────────────────
    dense_top_k: int = 50
    bm25_top_k: int = 50
    hybrid_alpha: float = 0.6  # weight for dense in fusion
    rerank_top_k: int = 5
    min_retrieval_score: float = 0.25

    # ── Guardrails ──────────────────────────────────────────────────────
    safety_enabled: bool = True
    relevance_threshold: float = 0.30
    grounding_nli_threshold: float = 0.60

    # ── Server ──────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Paths ───────────────────────────────────────────────────────────
    data_dir: Path = DATA_DIR
    index_dir: Path = INDEX_DIR

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# ---------------------------------------------------------------------------
# Singleton instance
# ---------------------------------------------------------------------------
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return (and cache) the global Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        # Ensure cache directories exist
        _settings.data_dir.mkdir(parents=True, exist_ok=True)
        _settings.index_dir.mkdir(parents=True, exist_ok=True)
    return _settings
