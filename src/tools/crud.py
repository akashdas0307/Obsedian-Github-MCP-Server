"""14 CRUD MCP tools — all operate on the local Git clone."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import structlog

from mcp.server.fastmcp import FastMCP

from src.tools.validators import safe_path, relative_to_repo, PathValidationError

log = structlog.get_logger()

# These will be set by main.py at startup via app.state
_repo_dir: Path | None = None
_debouncer = None
_search_engine = None


def _get_repo_dir():
    """Get the repo directory — injected from main at startup."""
    from src.main import settings
    return Path(settings.repo_dir)


def _get_debouncer():
    from src.main import debouncer
    return debouncer


def _get_search_engine():
    from src.main import search_engine
    return search_engine


async def _notify_write(file_path: str, change_type: str = "M"):
    """Notify debouncer and search engine of a file change."""
    deb = _get_debouncer()
    se = _get_search_engine()
    if deb:
        await deb.notify_write(file_path)
    if se:
        await se.on_file_changed(file_path, change_type)


def register_crud_tools(mcp: FastMCP):
    """Register all 14 CRUD tools on the given FastMCP server."""

    # ------------------------------------------------------------------
    # READ operations
    # ------------------------------------------------------------------

    @mcp.tool()
    async def list_folder(path: str = "") -> str:
        """List contents of a folder in the repository.

        Args:
            path: Relative path from repo root. Empty string for root.

        Returns:
            JSON with 'dirs' and 'files' arrays.
        """
        repo_dir = _get_repo_dir()
        try:
            target = safe_path(path, repo_dir)
        except PathValidationError as e:
            return json.dumps({"error": str(e)})

        if not target.exists():
            return json.dumps({"error": f"Path '{path}' does not exist"})
        if not target.is_dir():
            return json.dumps({"error": f"Path '{path}' is not a directory"})

        dirs = []
        files = []
        for entry in sorted(target.iterdir()):
            # Skip .git directory
            if entry.name == ".git":
                continue
            rel = relative_to_repo(entry, repo_dir)
            if entry.is_dir():
                dirs.append({"name": entry.name, "path": rel})
            else:
                files.append({
                    "name": entry.name,
                    "path": rel,
                    "size": entry.stat().st_size,
                })

        return json.dumps({"dirs": dirs, "files": files}, indent=2)

    @mcp.tool()
    async def list_files(path: str = "", recursive: bool = False) -> str:
        """List files in a folder.

        Args:
            path: Relative path from repo root. Empty for root.
            recursive: If True, list files recursively in all subdirectories.

        Returns:
            JSON array of file objects with name, path, and size.
        """
        repo_dir = _get_repo_dir()
        try:
            target = safe_path(path, repo_dir)
        except PathValidationError as e:
            return json.dumps({"error": str(e)})

        if not target.exists() or not target.is_dir():
            return json.dumps({"error": f"Directory '{path}' not found"})

        files = []
        iterator = target.rglob("*") if recursive else target.iterdir()
        for entry in sorted(iterator):
            if entry.is_file() and ".git" not in entry.parts:
                rel = relative_to_repo(entry, repo_dir)
                files.append({
                    "name": entry.name,
                    "path": rel,
                    "size": entry.stat().st_size,
                })

        return json.dumps(files, indent=2)

    @mcp.tool()
    async def read_file(path: str) -> str:
        """Read the full content of a file.

        Args:
            path: Relative path to the file from repo root.

        Returns:
            The file content as text.
        """
        repo_dir = _get_repo_dir()
        try:
            target = safe_path(path, repo_dir)
        except PathValidationError as e:
            return json.dumps({"error": str(e)})

        if not target.exists():
            return json.dumps({"error": f"File '{path}' not found"})
        if not target.is_file():
            return json.dumps({"error": f"'{path}' is not a file"})

        try:
            content = target.read_text(encoding="utf-8")
            return content
        except UnicodeDecodeError:
            return json.dumps({"error": f"File '{path}' is binary, cannot read as text"})

    @mcp.tool()
    async def read_range(path: str, start: int, end: int) -> str:
        """Read a specific character range from a file.

        Args:
            path: Relative path to the file.
            start: Start character position (0-indexed, inclusive).
            end: End character position (exclusive).

        Returns:
            The substring from start to end, plus metadata.
        """
        repo_dir = _get_repo_dir()
        try:
            target = safe_path(path, repo_dir)
        except PathValidationError as e:
            return json.dumps({"error": str(e)})

        if not target.exists() or not target.is_file():
            return json.dumps({"error": f"File '{path}' not found"})

        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return json.dumps({"error": "Binary file"})

        total = len(content)
        start = max(0, start)
        end = min(end, total)

        return json.dumps({
            "content": content[start:end],
            "range": f"{start}-{end}",
            "total_length": total,
        })

    @mcp.tool()
    async def search_files(query: str, path: str = "", case_sensitive: bool = False) -> str:
        """Search for text across files using grep.

        Args:
            query: Text pattern to search for.
            path: Subfolder to limit the search. Empty for entire repo.
            case_sensitive: Whether the search is case-sensitive.

        Returns:
            JSON array of matches with file, line number, and content.
        """
        repo_dir = _get_repo_dir()
        try:
            target = safe_path(path, repo_dir) if path else repo_dir
        except PathValidationError as e:
            return json.dumps({"error": str(e)})

        cmd = ["grep", "-rn", "--include=*"]
        if not case_sensitive:
            cmd.append("-i")
        cmd.extend(["--", query, str(target)])

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
                cwd=str(repo_dir),
            )
            matches = []
            for line in result.stdout.strip().split("\n")[:100]:  # Cap at 100
                if not line:
                    continue
                # Format: file:line_num:content
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    file_abs = parts[0]
                    try:
                        rel = relative_to_repo(Path(file_abs), repo_dir)
                    except Exception:
                        rel = file_abs
                    matches.append({
                        "file": rel,
                        "line": int(parts[1]) if parts[1].isdigit() else 0,
                        "content": parts[2].strip(),
                    })

            return json.dumps(matches, indent=2)
        except subprocess.TimeoutExpired:
            return json.dumps({"error": "Search timed out after 30 seconds"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ------------------------------------------------------------------
    # WRITE operations
    # ------------------------------------------------------------------

    @mcp.tool()
    async def add_folder(path: str) -> str:
        """Create a new folder in the repository.

        Args:
            path: Relative path for the new folder.

        Returns:
            Success or error message.
        """
        repo_dir = _get_repo_dir()
        try:
            target = safe_path(path, repo_dir)
        except PathValidationError as e:
            return json.dumps({"error": str(e)})

        if target.exists():
            return json.dumps({"error": f"Path '{path}' already exists"})

        target.mkdir(parents=True, exist_ok=True)

        # Create .gitkeep so git tracks the empty folder
        gitkeep = target / ".gitkeep"
        gitkeep.touch()

        await _notify_write(path, "A")
        return json.dumps({"success": True, "path": path})

    @mcp.tool()
    async def add_file(path: str, content: str) -> str:
        """Create a new file in the repository.

        Args:
            path: Relative path for the new file.
            content: The file content to write.

        Returns:
            Success with path and size, or error if file already exists.
        """
        repo_dir = _get_repo_dir()
        try:
            target = safe_path(path, repo_dir)
        except PathValidationError as e:
            return json.dumps({"error": str(e)})

        if target.exists():
            return json.dumps({"error": f"File '{path}' already exists. Use edit_file to modify."})

        # Ensure parent directories exist
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

        await _notify_write(path, "A")
        return json.dumps({
            "success": True,
            "path": path,
            "size": len(content),
        })

    @mcp.tool()
    async def edit_file(path: str, content: str) -> str:
        """Replace the entire content of an existing file.

        Args:
            path: Relative path to the file.
            content: New file content (replaces existing content entirely).

        Returns:
            Success with path and new size, or error.
        """
        repo_dir = _get_repo_dir()
        try:
            target = safe_path(path, repo_dir)
        except PathValidationError as e:
            return json.dumps({"error": str(e)})

        if not target.exists():
            return json.dumps({"error": f"File '{path}' not found. Use add_file to create."})
        if not target.is_file():
            return json.dumps({"error": f"'{path}' is not a file"})

        target.write_text(content, encoding="utf-8")

        await _notify_write(path, "M")
        return json.dumps({
            "success": True,
            "path": path,
            "size": len(content),
        })

    @mcp.tool()
    async def rename_file(old_path: str, new_path: str) -> str:
        """Rename or move a file within the repository.

        Args:
            old_path: Current relative path of the file.
            new_path: New relative path for the file.

        Returns:
            Success or error message.
        """
        repo_dir = _get_repo_dir()
        try:
            source = safe_path(old_path, repo_dir)
            dest = safe_path(new_path, repo_dir)
        except PathValidationError as e:
            return json.dumps({"error": str(e)})

        if not source.exists() or not source.is_file():
            return json.dumps({"error": f"File '{old_path}' not found"})
        if dest.exists():
            return json.dumps({"error": f"Destination '{new_path}' already exists"})

        dest.parent.mkdir(parents=True, exist_ok=True)
        source.rename(dest)

        await _notify_write(old_path, "D")
        await _notify_write(new_path, "A")
        return json.dumps({
            "success": True,
            "old_path": old_path,
            "new_path": new_path,
        })

    @mcp.tool()
    async def rename_folder(old_path: str, new_path: str) -> str:
        """Rename or move a folder within the repository.

        Args:
            old_path: Current relative path of the folder.
            new_path: New relative path for the folder.

        Returns:
            Success or error message.
        """
        repo_dir = _get_repo_dir()
        try:
            source = safe_path(old_path, repo_dir)
            dest = safe_path(new_path, repo_dir)
        except PathValidationError as e:
            return json.dumps({"error": str(e)})

        if not source.exists() or not source.is_dir():
            return json.dumps({"error": f"Folder '{old_path}' not found"})
        if dest.exists():
            return json.dumps({"error": f"Destination '{new_path}' already exists"})

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(dest))

        await _notify_write(old_path, "D")
        await _notify_write(new_path, "A")
        return json.dumps({
            "success": True,
            "old_path": old_path,
            "new_path": new_path,
        })

    @mcp.tool()
    async def delete_file(path: str) -> str:
        """Delete a file from the repository.

        Args:
            path: Relative path to the file to delete.

        Returns:
            Success or error message.
        """
        repo_dir = _get_repo_dir()
        try:
            target = safe_path(path, repo_dir)
        except PathValidationError as e:
            return json.dumps({"error": str(e)})

        if not target.exists():
            return json.dumps({"error": f"File '{path}' not found"})
        if not target.is_file():
            return json.dumps({"error": f"'{path}' is not a file"})

        target.unlink()

        await _notify_write(path, "D")
        return json.dumps({"success": True, "deleted": path})

    @mcp.tool()
    async def delete_folder(path: str) -> str:
        """Delete a folder and all its contents from the repository.

        Args:
            path: Relative path to the folder to delete.

        Returns:
            Success or error message.
        """
        repo_dir = _get_repo_dir()
        try:
            target = safe_path(path, repo_dir)
        except PathValidationError as e:
            return json.dumps({"error": str(e)})

        if not target.exists():
            return json.dumps({"error": f"Folder '{path}' not found"})
        if not target.is_dir():
            return json.dumps({"error": f"'{path}' is not a folder"})

        # Safety: don't delete the repo root
        if target.resolve() == repo_dir.resolve():
            return json.dumps({"error": "Cannot delete the repository root"})

        shutil.rmtree(str(target))

        await _notify_write(path, "D")
        return json.dumps({"success": True, "deleted": path})

    @mcp.tool()
    async def append_to_file(path: str, content: str) -> str:
        """Append content to the end of an existing file.

        Args:
            path: Relative path to the file.
            content: Content to append at the end.

        Returns:
            Success with new file size, or error.
        """
        repo_dir = _get_repo_dir()
        try:
            target = safe_path(path, repo_dir)
        except PathValidationError as e:
            return json.dumps({"error": str(e)})

        if not target.exists() or not target.is_file():
            return json.dumps({"error": f"File '{path}' not found"})

        with open(target, "a", encoding="utf-8") as f:
            f.write(content)

        new_size = target.stat().st_size
        await _notify_write(path, "M")
        return json.dumps({
            "success": True,
            "path": path,
            "new_size": new_size,
        })

    @mcp.tool()
    async def insert_at_position(path: str, position: int, content: str) -> str:
        """Insert content at a specific character position in a file.

        Args:
            path: Relative path to the file.
            position: Character position where content should be inserted (0-indexed).
            content: Content to insert.

        Returns:
            Success with new file size, or error.
        """
        repo_dir = _get_repo_dir()
        try:
            target = safe_path(path, repo_dir)
        except PathValidationError as e:
            return json.dumps({"error": str(e)})

        if not target.exists() or not target.is_file():
            return json.dumps({"error": f"File '{path}' not found"})

        try:
            existing = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return json.dumps({"error": "Binary file"})

        pos = max(0, min(position, len(existing)))
        new_content = existing[:pos] + content + existing[pos:]
        target.write_text(new_content, encoding="utf-8")

        await _notify_write(path, "M")
        return json.dumps({
            "success": True,
            "path": path,
            "position": pos,
            "inserted_length": len(content),
            "new_size": len(new_content),
        })
