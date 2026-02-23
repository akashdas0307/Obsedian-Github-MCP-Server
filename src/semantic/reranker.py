"""Cross-encoder reranker using ms-marco-MiniLM-L-6-v2.

Lazy-loaded on first use. Takes top-N FAISS candidates and scores them
with a cross-encoder (query, passage) → relevance score.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.semantic.chunker import Chunk

log = structlog.get_logger()

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Reranks candidate chunks using a cross-encoder model.

    Scores each (query, chunk_text) pair and returns them sorted
    by relevance descending.
    """

    def __init__(self):
        self._model = None
        self._lock = asyncio.Lock()

    async def _ensure_loaded(self):
        if self._model is not None:
            return
        async with self._lock:
            if self._model is not None:
                return
            log.info("reranker_loading_model", model=RERANKER_MODEL)
            self._model = await asyncio.to_thread(self._load_model)
            log.info("reranker_model_ready", model=RERANKER_MODEL)

    @staticmethod
    def _load_model():
        from sentence_transformers import CrossEncoder
        return CrossEncoder(RERANKER_MODEL)

    async def rerank(
        self,
        query: str,
        candidates: list["Chunk"],
        top_k: int = 10,
    ) -> list[tuple["Chunk", float]]:
        """Score and rerank a list of candidate chunks.

        Args:
            query: The search query.
            candidates: List of Chunk objects retrieved by FAISS.
            top_k: Number of top results to return.

        Returns:
            List of (Chunk, score) tuples sorted by score descending,
            capped at top_k.
        """
        if not candidates:
            return []

        await self._ensure_loaded()

        # Build (query, passage) pairs
        pairs = [(query, chunk.text) for chunk in candidates]

        scores = await asyncio.to_thread(self._model.predict, pairs)

        # Pair chunks with their scores and sort
        ranked = sorted(
            zip(candidates, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked[:top_k]
