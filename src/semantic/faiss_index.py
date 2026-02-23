"""FAISS index wrapper with incremental add/remove support.

Uses IndexIDMap(IndexFlatIP) so:
  - Each chunk gets a custom int64 ID
  - Vectors can be removed by ID without full rebuild
  - Inner product on L2-normalised vectors = cosine similarity

Two sidecar JSON files are persisted alongside the FAISS binary:
  - chunk_meta.json  : {chunk_id_str: {file_path, char_start, char_end}}
  - file_hashes.json : {file_path: sha256_hash}  (for incremental detection)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    from src.semantic.chunker import Chunk

log = structlog.get_logger()

EMBEDDING_DIM = 768
INDEX_FILENAME = "index.faiss"
CHUNK_META_FILENAME = "chunk_meta.json"
FILE_HASHES_FILENAME = "file_hashes.json"


class FAISSIndex:
    """FAISS IndexIDMap wrapping IndexFlatIP for cosine-similarity search."""

    def __init__(self, index_dir: Path):
        self._index_dir = index_dir
        self._index_dir.mkdir(parents=True, exist_ok=True)

        self._index = None          # faiss.IndexIDMap
        self._chunk_meta: dict[str, dict] = {}   # id_str -> {file_path, char_start, char_end}
        self._file_hashes: dict[str, str] = {}   # file_path -> sha256

        self._init_index()

    def _init_index(self):
        """Create a fresh in-memory FAISS index."""
        import faiss
        flat = faiss.IndexFlatIP(EMBEDDING_DIM)
        self._index = faiss.IndexIDMap(flat)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        """Save FAISS index and sidecar metadata to disk."""
        import faiss
        faiss.write_index(
            self._index,
            str(self._index_dir / INDEX_FILENAME),
        )
        (self._index_dir / CHUNK_META_FILENAME).write_text(
            json.dumps(self._chunk_meta), encoding="utf-8"
        )
        (self._index_dir / FILE_HASHES_FILENAME).write_text(
            json.dumps(self._file_hashes), encoding="utf-8"
        )
        log.info("faiss_saved", vectors=self._index.ntotal)

    def load(self) -> bool:
        """Load FAISS index and metadata from disk.

        Returns True if loaded successfully, False if no saved index exists.
        """
        import faiss

        index_path = self._index_dir / INDEX_FILENAME
        meta_path = self._index_dir / CHUNK_META_FILENAME
        hashes_path = self._index_dir / FILE_HASHES_FILENAME

        if not index_path.exists():
            return False

        try:
            self._index = faiss.read_index(str(index_path))
            if meta_path.exists():
                self._chunk_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if hashes_path.exists():
                self._file_hashes = json.loads(hashes_path.read_text(encoding="utf-8"))

            log.info("faiss_loaded", vectors=self._index.ntotal)
            return True
        except Exception:
            log.exception("faiss_load_failed")
            self._init_index()
            self._chunk_meta = {}
            self._file_hashes = {}
            return False

    # ------------------------------------------------------------------
    # Incremental operations
    # ------------------------------------------------------------------

    def add(self, chunks: list["Chunk"], vectors: np.ndarray):
        """Add chunks and their vectors to the index.

        Args:
            chunks: List of Chunk objects (for metadata).
            vectors: Float32 array of shape (len(chunks), 768).
        """
        if not chunks:
            return

        ids = np.array([c.id for c in chunks], dtype=np.int64)
        self._index.add_with_ids(vectors.astype(np.float32), ids)

        for chunk in chunks:
            self._chunk_meta[str(chunk.id)] = {
                "file_path": chunk.file_path,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
            }

        log.debug("faiss_add", count=len(chunks), total=self._index.ntotal)

    def remove_file(self, file_path: str):
        """Remove all chunks belonging to a file from the index.

        Args:
            file_path: Relative file path to remove.
        """
        import faiss

        # Find chunk IDs for this file
        ids_to_remove = [
            int(id_str)
            for id_str, meta in self._chunk_meta.items()
            if meta["file_path"] == file_path
        ]

        if not ids_to_remove:
            return

        id_array = np.array(ids_to_remove, dtype=np.int64)
        selector = faiss.IDSelectorBatch(
            len(id_array), faiss.swig_ptr(id_array)
        )
        self._index.remove_ids(selector)

        # Remove from metadata
        for id_str in [str(i) for i in ids_to_remove]:
            self._chunk_meta.pop(id_str, None)

        # Remove file hash
        self._file_hashes.pop(file_path, None)

        log.debug(
            "faiss_remove_file",
            file=file_path,
            removed=len(ids_to_remove),
            total=self._index.ntotal,
        )

    def search(self, query_vector: np.ndarray, top_k: int = 50) -> list[dict]:
        """Search for nearest neighbours.

        Args:
            query_vector: Float32 array of shape (1, 768).
            top_k: Number of candidates to retrieve.

        Returns:
            List of dicts: {chunk_id, score, file_path, char_start, char_end}
        """
        if self._index.ntotal == 0:
            return []

        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(query_vector, k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = self._chunk_meta.get(str(idx))
            if meta is None:
                continue
            results.append({
                "chunk_id": idx,
                "score": float(score),
                "file_path": meta["file_path"],
                "char_start": meta["char_start"],
                "char_end": meta["char_end"],
            })

        return results

    # ------------------------------------------------------------------
    # Hash helpers for incremental indexing
    # ------------------------------------------------------------------

    def get_file_hash(self, file_path: str) -> str | None:
        return self._file_hashes.get(file_path)

    def set_file_hash(self, file_path: str, hash_val: str):
        self._file_hashes[file_path] = hash_val

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0
