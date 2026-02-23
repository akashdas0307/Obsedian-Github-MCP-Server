"""Semantic Search Engine — orchestrates chunking, embedding, and retrieval.

Lifecycle:
  1. init(): Load saved index or build from scratch.
  2. search(query): Embed → FAISS top-50 → rerank → return top-k.
  3. on_file_changed(path, type): Incremental index update.
  4. save(): Persist index to disk (called on shutdown).
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Literal

import structlog

from src.config import Settings
from src.semantic.chunker import FileChunker
from src.semantic.embedder import NomicEmbedder
from src.semantic.reranker import CrossEncoderReranker
from src.semantic.faiss_index import FAISSIndex

log = structlog.get_logger()

ChangeType = Literal["A", "M", "D"]  # Added, Modified, Deleted


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 of a file's content."""
    h = hashlib.sha256()
    try:
        h.update(path.read_bytes())
    except Exception:
        pass
    return h.hexdigest()


class SemanticSearchEngine:
    """Full semantic search pipeline: chunking → embedding → FAISS → reranking."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._repo_path = Path(settings.repo_dir)
        self._index_path = Path(settings.index_dir)

        self._chunker = FileChunker(chunk_size=1000, overlap=200)
        self._embedder = NomicEmbedder()
        self._reranker = CrossEncoderReranker()
        self._faiss = FAISSIndex(self._index_path)

        # Lock prevents concurrent index modification
        self._write_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def init(self):
        """Load saved index, or build from scratch if none exists."""
        loaded = self._faiss.load()
        if loaded:
            log.info(
                "search_engine_loaded_existing",
                vectors=self._faiss.total_vectors,
            )
        else:
            log.info("search_engine_building_index")
            await self._build_full_index()

    async def _build_full_index(self):
        """Scan entire repo and build index from scratch."""
        async with self._write_lock:
            all_chunks = await asyncio.to_thread(
                self._chunker.chunk_directory, self._repo_path
            )

            if not all_chunks:
                log.warning("search_engine_no_chunks")
                return

            log.info("search_engine_embedding", chunks=len(all_chunks))
            texts = [c.text for c in all_chunks]
            vectors = await self._embedder.embed_documents(texts)

            self._faiss.add(all_chunks, vectors)

            # Record file hashes
            for chunk in all_chunks:
                file_abs = self._repo_path / chunk.file_path
                if file_abs.exists():
                    self._faiss.set_file_hash(
                        chunk.file_path, _file_sha256(file_abs)
                    )

            self._faiss.save()
            log.info(
                "search_engine_index_built",
                total_vectors=self._faiss.total_vectors,
            )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search the repo with semantic similarity + reranking.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.

        Returns:
            List of result dicts with file_name, position_range, score, preview.
        """
        # Embed query
        query_vector = await self._embedder.embed_query(query)

        # FAISS retrieval (get top 50 candidates for reranking)
        candidates_meta = self._faiss.search(query_vector, top_k=50)
        if not candidates_meta:
            return []

        # Reconstruct chunk objects for reranking (we need the text)
        from src.semantic.chunker import Chunk

        chunks = []
        for meta in candidates_meta:
            try:
                file_abs = self._repo_path / meta["file_path"]
                if not file_abs.is_file():
                    continue
                content = file_abs.read_text(encoding="utf-8", errors="ignore")
                text = content[meta["char_start"]:meta["char_end"]]
                chunk = Chunk(
                    id=meta["chunk_id"],
                    file_path=meta["file_path"],
                    char_start=meta["char_start"],
                    char_end=meta["char_end"],
                    text=text,
                )
                chunks.append(chunk)
            except Exception:
                log.exception(
                    "search_chunk_reconstruct_error",
                    file=meta.get("file_path"),
                )

        if not chunks:
            return []

        # Rerank
        ranked = await self._reranker.rerank(query, chunks, top_k=top_k)

        # Format results
        results = []
        for chunk, score in ranked:
            results.append({
                "file_name": chunk.file_path,
                "position_range": chunk.position_range,
                "score": round(score, 4),
                "preview": chunk.preview,
            })

        return results

    # ------------------------------------------------------------------
    # Incremental updates
    # ------------------------------------------------------------------

    async def on_file_changed(self, file_path: str, change_type: ChangeType):
        """Update index when a file is added, modified, or deleted.

        Args:
            file_path: Relative path from repo root.
            change_type: "A" (added), "M" (modified), "D" (deleted).
        """
        file_abs = self._repo_path / file_path

        # Check if this is a file we should index
        if change_type in ("A", "M"):
            if not self._chunker.is_indexable(file_abs):
                return
            if not file_abs.is_file():
                return

        async with self._write_lock:
            if change_type == "D":
                self._faiss.remove_file(file_path)
                log.debug("search_index_removed", file=file_path)

            elif change_type in ("A", "M"):
                # Check if content actually changed (skip if identical hash)
                new_hash = _file_sha256(file_abs)
                old_hash = self._faiss.get_file_hash(file_path)

                if new_hash == old_hash:
                    log.debug("search_index_skip_unchanged", file=file_path)
                    return

                # Remove old chunks (if any)
                self._faiss.remove_file(file_path)

                # Chunk + embed the new version
                try:
                    content = file_abs.read_text(encoding="utf-8", errors="ignore")
                    new_chunks = self._chunker.chunk_file(file_path, content)

                    if new_chunks:
                        texts = [c.text for c in new_chunks]
                        vectors = await self._embedder.embed_documents(texts)
                        self._faiss.add(new_chunks, vectors)
                        self._faiss.set_file_hash(file_path, new_hash)

                    log.debug(
                        "search_index_updated",
                        file=file_path,
                        chunks=len(new_chunks),
                    )
                except Exception:
                    log.exception("search_index_update_error", file=file_path)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        """Save index to disk (called on shutdown)."""
        self._faiss.save()
