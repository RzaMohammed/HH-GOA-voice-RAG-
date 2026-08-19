"""
Offline FAISS index builder.

Embeds a list of ``ChunkMetadata`` objects and creates a FAISS inner-product
index (``IndexFlatIP``).  Because embeddings are L2-normalised, inner product
equals cosine similarity.

Persists:
  - ``faiss_index.bin``   — the FAISS binary index
  - ``chunk_map.json``    — ordered list of serialised ChunkMetadata (id → index)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import faiss
import numpy as np
from loguru import logger
from tqdm import tqdm

from voice_rag.config import get_settings
from voice_rag.indexing.embeddings import Embedder, get_embedder
from voice_rag.pipeline.schemas import ChunkMetadata


# ═══════════════════════════════════════════════════════════════════════════
# Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_faiss_index(
    chunks: Sequence[ChunkMetadata],
    embedder: Optional[Embedder] = None,
    index_dir: Optional[Path] = None,
    batch_size: int = 128,
    show_progress: bool = True,
) -> tuple[faiss.Index, list[ChunkMetadata]]:
    """
    Build a FAISS index from chunks and persist to disk.

    Args:
        chunks:        Pre-chunked text segments with metadata.
        embedder:      Embedding model (uses default if None).
        index_dir:     Directory to save the index files.
        batch_size:    Encoding batch size.
        show_progress: Show progress bar during embedding.

    Returns:
        Tuple of (faiss.Index, ordered_chunks).
    """
    cfg = get_settings()
    index_dir = index_dir or cfg.index_dir
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    embedder = embedder or get_embedder()

    # --- Embed all chunk texts ---
    texts = [c.text for c in chunks]
    logger.info(f"Embedding {len(texts)} chunks (batch_size={batch_size})...")

    all_embeddings: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", disable=not show_progress):
        batch = texts[i : i + batch_size]
        embs = embedder.encode(batch, batch_size=batch_size, normalize=True)
        all_embeddings.append(embs)

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    dim = embeddings.shape[1]

    logger.info(f"Building FAISS IndexFlatIP — {len(embeddings)} vectors × {dim}d")

    # --- Build index ---
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # --- Persist ---
    index_path = index_dir / "faiss_index.bin"
    chunk_map_path = index_dir / "chunk_map.json"

    faiss.write_index(index, str(index_path))
    logger.info(f"FAISS index saved to {index_path} ({index.ntotal} vectors)")

    # Serialise chunk metadata in order
    ordered_chunks = list(chunks)
    chunk_dicts = [c.model_dump() for c in ordered_chunks]
    with open(chunk_map_path, "w", encoding="utf-8") as f:
        json.dump(chunk_dicts, f, ensure_ascii=False, indent=1)
    logger.info(f"Chunk map saved to {chunk_map_path}")

    return index, ordered_chunks


# ═══════════════════════════════════════════════════════════════════════════
# Loader
# ═══════════════════════════════════════════════════════════════════════════

def load_faiss_index(
    index_dir: Optional[Path] = None,
) -> tuple[faiss.Index, list[ChunkMetadata]]:
    """
    Load a previously-built FAISS index and chunk map from disk.

    Returns:
        Tuple of (faiss.Index, ordered_chunks).
    """
    cfg = get_settings()
    index_dir = Path(index_dir or cfg.index_dir)

    index_path = index_dir / "faiss_index.bin"
    chunk_map_path = index_dir / "chunk_map.json"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    if not chunk_map_path.exists():
        raise FileNotFoundError(f"Chunk map not found at {chunk_map_path}")

    index = faiss.read_index(str(index_path))
    logger.info(f"Loaded FAISS index from {index_path} ({index.ntotal} vectors)")

    with open(chunk_map_path, "r", encoding="utf-8") as f:
        chunk_dicts = json.load(f)
    chunks = [ChunkMetadata.model_validate(d) for d in chunk_dicts]
    logger.info(f"Loaded {len(chunks)} chunks from {chunk_map_path}")

    return index, chunks
