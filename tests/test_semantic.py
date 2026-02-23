"""Tests for the semantic search components (chunker, FAISS index)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.semantic.chunker import FileChunker, Chunk, make_chunk_id


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------

def test_chunk_id_stable():
    id1 = make_chunk_id("src/main.py", 0)
    id2 = make_chunk_id("src/main.py", 0)
    assert id1 == id2


def test_chunk_id_different_for_different_inputs():
    id1 = make_chunk_id("src/main.py", 0)
    id2 = make_chunk_id("src/main.py", 1000)
    assert id1 != id2


def test_chunk_id_positive_int64():
    cid = make_chunk_id("any/path.txt", 0)
    assert isinstance(cid, int)
    assert cid >= 0
    assert cid <= 0x7FFFFFFFFFFFFFFF


def test_chunk_file_short():
    """File shorter than chunk_size produces single chunk."""
    chunker = FileChunker(chunk_size=1000, overlap=200)
    content = "Hello world"
    chunks = chunker.chunk_file("test.txt", content)
    assert len(chunks) == 1
    assert chunks[0].char_start == 0
    assert chunks[0].text == content


def test_chunk_file_empty():
    chunker = FileChunker()
    chunks = chunker.chunk_file("empty.txt", "")
    assert chunks == []


def test_chunk_file_whitespace_only():
    chunker = FileChunker()
    chunks = chunker.chunk_file("ws.txt", "   \n\n  ")
    assert chunks == []


def test_chunk_file_positions_contiguous():
    """Chunk positions should cover the entire file (with overlaps)."""
    chunker = FileChunker(chunk_size=100, overlap=20)
    content = "a" * 350
    chunks = chunker.chunk_file("long.txt", content)

    assert len(chunks) > 1
    # First chunk starts at 0
    assert chunks[0].char_start == 0
    # Last chunk ends at or near file end
    assert chunks[-1].char_end <= len(content)
    assert chunks[-1].char_end >= len(content) - 20  # Close to end


def test_chunk_preview():
    chunker = FileChunker()
    content = "The quick brown fox jumps over the lazy dog\n" * 30
    chunks = chunker.chunk_file("fox.txt", content)
    for chunk in chunks:
        assert len(chunk.preview) <= 200


def test_is_indexable_python():
    chunker = FileChunker()
    assert chunker.is_indexable(Path("src/main.py"))


def test_is_indexable_markdown():
    chunker = FileChunker()
    assert chunker.is_indexable(Path("README.md"))


def test_is_not_indexable_binary():
    chunker = FileChunker()
    assert not chunker.is_indexable(Path("image.png"))
    assert not chunker.is_indexable(Path("archive.zip"))
    assert not chunker.is_indexable(Path("binary.exe"))


def test_chunk_directory(tmp_path):
    """chunk_directory should find and chunk text files, skip .git."""
    # Create test files
    (tmp_path / "main.py").write_text("print('hello')\n" * 5)
    (tmp_path / "README.md").write_text("# Title\n\nSome content\n")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[core]\n  repositoryformatversion = 0\n")

    chunker = FileChunker(chunk_size=100, overlap=20)
    all_chunks = chunker.chunk_directory(tmp_path)

    # Should have chunks from .py and .md files
    files_indexed = {c.file_path for c in all_chunks}
    assert "main.py" in files_indexed
    assert "README.md" in files_indexed

    # Should NOT index .git internals
    assert not any(".git" in c.file_path for c in all_chunks)
    # Should NOT index binary files
    assert not any("image.png" in c.file_path for c in all_chunks)


# ---------------------------------------------------------------------------
# FAISS index tests (no model — uses random vectors)
# Skipped automatically if faiss-cpu is not installed.
# ---------------------------------------------------------------------------

faiss = pytest.importorskip("faiss", reason="faiss-cpu not installed")


@pytest.fixture
def index_dir(tmp_path):
    return tmp_path / "index"


def test_faiss_add_and_search(index_dir):
    from src.semantic.faiss_index import FAISSIndex, EMBEDDING_DIM

    idx = FAISSIndex(index_dir)
    assert idx.total_vectors == 0

    # Create dummy chunks and vectors
    from src.semantic.chunker import Chunk

    chunks = [
        Chunk(id=1000, file_path="a.py", char_start=0, char_end=100, text="foo"),
        Chunk(id=1001, file_path="b.py", char_start=0, char_end=100, text="bar"),
    ]
    vectors = np.random.randn(2, EMBEDDING_DIM).astype(np.float32)
    # L2 normalise
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    idx.add(chunks, vectors)
    assert idx.total_vectors == 2

    # Search
    query = np.random.randn(1, EMBEDDING_DIM).astype(np.float32)
    query = query / np.linalg.norm(query)
    results = idx.search(query, top_k=2)
    assert len(results) == 2
    assert all("file_path" in r for r in results)


def test_faiss_remove_file(index_dir):
    from src.semantic.faiss_index import FAISSIndex, EMBEDDING_DIM
    from src.semantic.chunker import Chunk

    idx = FAISSIndex(index_dir)

    chunks = [
        Chunk(id=2000, file_path="keep.py", char_start=0, char_end=100, text="keep"),
        Chunk(id=2001, file_path="remove.py", char_start=0, char_end=100, text="remove"),
        Chunk(id=2002, file_path="remove.py", char_start=100, char_end=200, text="remove2"),
    ]
    vectors = np.random.randn(3, EMBEDDING_DIM).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    idx.add(chunks, vectors)
    assert idx.total_vectors == 3

    idx.remove_file("remove.py")
    assert idx.total_vectors == 1


def test_faiss_save_and_load(index_dir):
    from src.semantic.faiss_index import FAISSIndex, EMBEDDING_DIM
    from src.semantic.chunker import Chunk

    idx = FAISSIndex(index_dir)
    chunks = [Chunk(id=3000, file_path="saved.py", char_start=0, char_end=50, text="x")]
    vectors = np.random.randn(1, EMBEDDING_DIM).astype(np.float32)
    idx.add(chunks, vectors)
    idx.set_file_hash("saved.py", "abc123")
    idx.save()

    # Load into a new instance
    idx2 = FAISSIndex(index_dir)
    loaded = idx2.load()
    assert loaded
    assert idx2.total_vectors == 1
    assert idx2.get_file_hash("saved.py") == "abc123"


def test_faiss_empty_search(index_dir):
    from src.semantic.faiss_index import FAISSIndex, EMBEDDING_DIM

    idx = FAISSIndex(index_dir)
    query = np.random.randn(1, EMBEDDING_DIM).astype(np.float32)
    results = idx.search(query, top_k=10)
    assert results == []
