"""
Multilingual embedding model abstraction.

Wraps ``sentence-transformers`` with:
  - Automatic GPU/CPU device detection
  - Configurable model selection (bge-m3, multilingual-e5, MiniLM)
  - Batch encoding with progress tracking
  - Thread-safe singleton caching
"""

from __future__ import annotations

import threading
from typing import Optional, Sequence

import numpy as np
from loguru import logger

from voice_rag.config import get_settings


# ═══════════════════════════════════════════════════════════════════════════
# Embedder wrapper
# ═══════════════════════════════════════════════════════════════════════════

class Embedder:
    """Thin wrapper around SentenceTransformer with normalisation."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        import torch

        cfg = get_settings()
        self.model_name = model_name or cfg.embedding_model

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
        self._model = SentenceTransformer(self.model_name, device=self.device)
        try:
            self._dim = self._model.get_embedding_dimension()
        except AttributeError:
            self._dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model ready — dim={self._dim}")

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimensionality."""
        return self._dim

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        normalize: bool = True,
        show_progress_bar: bool = False,
        is_query: bool = False,
    ) -> np.ndarray:
        """
        Encode texts into dense vectors with optional prefix routing and GPU acceleration.
        """
        import torch

        formatted_texts = list(texts)
        # Prefix routing for e5 models
        if "e5" in self.model_name.lower():
            prefix = "query: " if is_query else "passage: "
            formatted_texts = [f"{prefix}{t}" if not t.startswith(prefix) else t for t in formatted_texts]

        with torch.inference_mode():
            if self.device == "cuda":
                with torch.cuda.amp.autocast(enabled=True):
                    embeddings = self._model.encode(
                        formatted_texts,
                        batch_size=batch_size,
                        show_progress_bar=show_progress_bar,
                        normalize_embeddings=normalize,
                        convert_to_numpy=True,
                    )
            else:
                embeddings = self._model.encode(
                    formatted_texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress_bar,
                    normalize_embeddings=normalize,
                    convert_to_numpy=True,
                )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_single(self, text: str, normalize: bool = True, is_query: bool = True) -> np.ndarray:
        """Encode a single text string."""
        return self.encode([text], normalize=normalize, is_query=is_query)[0]


# ═══════════════════════════════════════════════════════════════════════════
# Singleton management
# ═══════════════════════════════════════════════════════════════════════════

_lock = threading.Lock()
_embedder: Optional[Embedder] = None


def get_embedder(model_name: Optional[str] = None) -> Embedder:
    """Return (and cache) the global Embedder instance."""
    global _embedder
    with _lock:
        if _embedder is None or (model_name and model_name != _embedder.model_name):
            _embedder = Embedder(model_name=model_name)
    return _embedder


def reset_embedder() -> None:
    """Clear the cached embedder (useful for tests)."""
    global _embedder
    with _lock:
        _embedder = None
