"""Embedding generation module using BAAI/bge-small-en-v1.5."""

from __future__ import annotations

import logging
from typing import Sequence
import numpy as np
from sentence_transformers import SentenceTransformer

from analyst.config import settings

logger = logging.getLogger("analyst.embeddings")

_EMBEDDER_INSTANCE: SentenceTransformer | None = None
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def get_embedder() -> SentenceTransformer:
    """Lazy-load and return the singleton SentenceTransformer model instance."""
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        logger.info("Loading embedding model: %s", settings.embed_model_name)
        _EMBEDDER_INSTANCE = SentenceTransformer(settings.embed_model_name)
    return _EMBEDDER_INSTANCE


def embed_passages(texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
    """Embed a sequence of passage texts with L2 normalization.

    Args:
        texts: Passage strings to embed.
        batch_size: Batch size for encoding.

    Returns:
        A 2D float32 numpy array of shape (N, dim).
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    embedder = get_embedder()
    embeddings = embedder.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def embed_queries(texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
    """Embed a sequence of search queries with the required BGE prompt prefix.

    Args:
        texts: Raw query strings.
        batch_size: Batch size for encoding.

    Returns:
        A 2D float32 numpy array of shape (N, dim).
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    prefixed = [f"{BGE_QUERY_PREFIX}{q.strip()}" for q in texts]
    return embed_passages(prefixed, batch_size=batch_size)