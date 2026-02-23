"""Nomic-embed-text-v1.5 embedding wrapper.

Uses sentence-transformers with task-specific instruction prefixes.
Models are lazy-loaded on first use to allow fast startup.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np
import structlog

log = structlog.get_logger()

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIM = 768

# Task prefixes required by nomic-embed-text
PREFIX_DOCUMENT = "search_document: "
PREFIX_QUERY = "search_query: "


class NomicEmbedder:
    """Wraps nomic-embed-text-v1.5 for document and query embedding.

    Thread-safe: uses an asyncio lock to prevent concurrent model loading.
    """

    def __init__(self):
        self._model = None
        self._lock = asyncio.Lock()

    async def _ensure_loaded(self):
        """Lazy-load the model on first call."""
        if self._model is not None:
            return
        async with self._lock:
            if self._model is not None:
                return
            log.info("embedder_loading_model", model=MODEL_NAME)
            self._model = await asyncio.to_thread(self._load_model)
            log.info("embedder_model_ready", model=MODEL_NAME)

    @staticmethod
    def _load_model():
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    async def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a list of document texts.

        Applies 'search_document:' prefix and L2-normalises output
        so inner product = cosine similarity.

        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for encoding.

        Returns:
            Float32 numpy array of shape (len(texts), 768).
        """
        await self._ensure_loaded()
        prefixed = [PREFIX_DOCUMENT + t for t in texts]
        vectors = await asyncio.to_thread(
            self._model.encode,
            prefixed,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalise for cosine via IP
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    async def embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query.

        Applies 'search_query:' prefix and L2-normalises output.

        Args:
            query: Natural language query string.

        Returns:
            Float32 numpy array of shape (1, 768).
        """
        await self._ensure_loaded()
        prefixed = [PREFIX_QUERY + query]
        vector = await asyncio.to_thread(
            self._model.encode,
            prefixed,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.astype(np.float32)
