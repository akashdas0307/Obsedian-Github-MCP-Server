"""Path validation utilities to prevent directory traversal attacks."""

from pathlib import Path


class PathValidationError(Exception):
    """Raised when a path fails security validation."""
    pass


def safe_path(path: str, base_dir: Path) -> Path:
    """Resolve a user-supplied path and ensure it stays within base_dir.

    Args:
        path: Relative path supplied by the MCP client (e.g. "src/main.py").
        base_dir: Absolute path to the repository root.

    Returns:
        Resolved absolute Path guaranteed to be inside base_dir.

    Raises:
        PathValidationError: If the path escapes the base directory.
    """
    # Strip leading slashes — all paths are relative to repo root
    clean = path.lstrip("/").lstrip("\\")

    # Resolve against base
    resolved = (base_dir / clean).resolve()
    base_resolved = base_dir.resolve()

    # Ensure resolved path is within base
    if not str(resolved).startswith(str(base_resolved)):
        raise PathValidationError(
            f"Path '{path}' resolves outside the repository root"
        )

    # Reject symlinks that point outside base
    if resolved.is_symlink():
        real = resolved.resolve()
        if not str(real).startswith(str(base_resolved)):
            raise PathValidationError(
                f"Symlink '{path}' points outside the repository root"
            )

    return resolved


def relative_to_repo(absolute_path: Path, base_dir: Path) -> str:
    """Convert an absolute path back to a repo-relative string.

    Always uses forward slashes for cross-platform compatibility.
    """
    try:
        return str(absolute_path.relative_to(base_dir.resolve())).replace("\\", "/")
    except ValueError:
        return str(absolute_path).replace("\\", "/")
