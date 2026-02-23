"""File chunker — splits text into overlapping character-based chunks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger()

# Text file extensions we will index
INDEXABLE_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".py", ".js", ".ts", ".jsx", ".tsx",
    ".html", ".htm", ".css", ".scss", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".conf", ".sh", ".bash", ".zsh",
    ".env", ".example", ".go", ".rs", ".java", ".kt", ".swift",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".r",
    ".sql", ".graphql", ".proto", ".xml", ".rst", ".tex",
    ".dockerfile", ".makefile", ".gitignore", ".env.example",
}

# Max file size to index (5 MB)
MAX_FILE_SIZE = 5 * 1024 * 1024


@dataclass
class Chunk:
    """A chunk of text from a file, with position tracking."""
    id: int                # Stable int64 ID for FAISS (derived from path + offset)
    file_path: str         # Relative path from repo root
    char_start: int        # Start position (inclusive) in the file
    char_end: int          # End position (exclusive) in the file
    text: str              # Chunk text content

    @property
    def position_range(self) -> str:
        """Human-readable position string e.g. '0-1000'."""
        return f"{self.char_start}-{self.char_end}"

    @property
    def preview(self) -> str:
        """First 200 chars of the chunk as a preview."""
        return self.text[:200].replace("\n", " ").strip()


def make_chunk_id(file_path: str, char_start: int) -> int:
    """Generate a stable int64 ID from file path + char offset.

    Uses SHA-256 truncated to 63 bits (positive int64 for FAISS).
    """
    key = f"{file_path}:{char_start}"
    digest = hashlib.sha256(key.encode()).digest()
    # Take first 8 bytes, mask to positive int64
    raw = int.from_bytes(digest[:8], "big")
    return raw & 0x7FFFFFFFFFFFFFFF  # Ensure positive


class FileChunker:
    """Splits files into overlapping character-based chunks."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def is_indexable(self, path: Path | str) -> bool:
        """Return True if the file should be semantically indexed."""
        p = Path(path)
        # Check extension
        if p.suffix.lower() in INDEXABLE_EXTENSIONS:
            return True
        # Extensionless files with known names
        if p.name.lower() in {"makefile", "dockerfile", "rakefile", "gemfile",
                               "procfile", ".gitignore", ".env.example"}:
            return True
        return False

    def chunk_file(self, file_path: str, content: str) -> list[Chunk]:
        """Split file content into overlapping chunks.

        Args:
            file_path: Relative file path (used for ID and metadata).
            content: Full text content of the file.

        Returns:
            List of Chunk objects with IDs and position metadata.
        """
        chunks: list[Chunk] = []

        if not content.strip():
            return chunks

        step = self.chunk_size - self.overlap
        if step <= 0:
            step = self.chunk_size

        pos = 0
        total = len(content)

        while pos < total:
            end = min(pos + self.chunk_size, total)
            chunk_text = content[pos:end]

            # Try to break at a newline or space for cleaner chunks
            if end < total:
                # Look back up to 50 chars for a newline
                nl = chunk_text.rfind("\n")
                if nl > self.chunk_size // 2:
                    end = pos + nl + 1
                    chunk_text = content[pos:end]

            chunk = Chunk(
                id=make_chunk_id(file_path, pos),
                file_path=file_path,
                char_start=pos,
                char_end=end,
                text=chunk_text,
            )
            chunks.append(chunk)
            pos = max(pos + step, end - self.overlap)

            # Avoid infinite loop on very short content
            if pos >= total:
                break

        return chunks

    def chunk_directory(
        self, repo_path: Path, skip_extensions: Optional[set[str]] = None
    ) -> list[Chunk]:
        """Walk a directory and chunk all indexable files.

        Args:
            repo_path: Path to the repository root.
            skip_extensions: Optional set of extensions to skip.

        Returns:
            All chunks from all indexable files.
        """
        all_chunks: list[Chunk] = []
        skip = skip_extensions or set()

        for file_path in sorted(repo_path.rglob("*")):
            # Skip .git and hidden dirs
            if any(part.startswith(".git") for part in file_path.parts):
                continue

            if not file_path.is_file():
                continue

            if file_path.suffix.lower() in skip:
                continue

            if not self.is_indexable(file_path):
                continue

            if file_path.stat().st_size > MAX_FILE_SIZE:
                log.debug("chunker_skip_large", path=str(file_path))
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                rel_path = str(file_path.relative_to(repo_path))
                chunks = self.chunk_file(rel_path, content)
                all_chunks.extend(chunks)
            except Exception:
                log.exception("chunker_file_error", path=str(file_path))

        log.info("chunker_directory_done", files_chunked=len(all_chunks))
        return all_chunks
