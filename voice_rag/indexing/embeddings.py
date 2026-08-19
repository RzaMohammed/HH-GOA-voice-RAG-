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
    ) -> np.ndarray:
        """
        Encode texts into dense vectors.

        Args:
            texts:             Input strings.
            batch_size:        Encoding batch size.
            normalize:         L2-normalise vectors (for cosine via dot product).
            show_progress_bar: Show tqdm progress.

        Returns:
            Array of shape ``(len(texts), dimension)``.
        """
        embeddings = self._model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode a single text string."""
        return self.encode([text], normalize=normalize)[0]


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
