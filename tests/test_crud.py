"""Tests for CRUD filesystem operations and path validation."""

import json
import tempfile
from pathlib import Path

import pytest

from src.tools.validators import PathValidationError, safe_path, relative_to_repo


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------

def test_safe_path_normal():
    with tempfile.TemporaryDirectory() as base:
        base_path = Path(base)
        result = safe_path("src/main.py", base_path)
        assert result == (base_path / "src/main.py").resolve()


def test_safe_path_rejects_traversal():
    with tempfile.TemporaryDirectory() as base:
        base_path = Path(base)
        with pytest.raises(PathValidationError):
            safe_path("../../etc/passwd", base_path)


def test_safe_path_absolute_stripped_safely():
    # An "absolute" path like /etc/passwd has its leading slash stripped,
    # so it becomes {base}/etc/passwd (safely inside base). This is the
    # intended behaviour — not an error.
    with tempfile.TemporaryDirectory() as base:
        base_path = Path(base)
        result = safe_path("/etc/passwd", base_path)
        assert str(result).startswith(str(base_path.resolve()))


def test_safe_path_rejects_double_dot_escape():
    # ../.. style traversal must always be rejected.
    with tempfile.TemporaryDirectory() as base:
        base_path = Path(base)
        with pytest.raises(PathValidationError):
            safe_path("../../etc/shadow", base_path)


def test_safe_path_strips_leading_slash():
    with tempfile.TemporaryDirectory() as base:
        base_path = Path(base)
        # Leading slash should be stripped, not treated as absolute
        result = safe_path("/src/main.py", base_path)
        assert str(result).startswith(str(base_path.resolve()))


def test_relative_to_repo():
    with tempfile.TemporaryDirectory() as base:
        base_path = Path(base)
        abs_path = base_path / "src" / "main.py"
        rel = relative_to_repo(abs_path, base_path)
        assert rel == "src/main.py"


# ---------------------------------------------------------------------------
# Filesystem CRUD tests (using temp directories to mock the repo)
# ---------------------------------------------------------------------------

@pytest.fixture
def repo_dir(tmp_path):
    """Create a temporary repo directory."""
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


def test_add_and_read_file(repo_dir):
    """Writing and reading a file should return the same content."""
    path = safe_path("hello.txt", repo_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Hello, World!", encoding="utf-8")
    assert path.read_text(encoding="utf-8") == "Hello, World!"


def test_add_folder(repo_dir):
    target = repo_dir / "new_folder"
    target.mkdir(parents=True, exist_ok=True)
    assert target.is_dir()


def test_edit_file(repo_dir):
    path = repo_dir / "edit_me.txt"
    path.write_text("original", encoding="utf-8")
    path.write_text("updated", encoding="utf-8")
    assert path.read_text(encoding="utf-8") == "updated"


def test_append_to_file(repo_dir):
    path = repo_dir / "append_me.txt"
    path.write_text("line1\n", encoding="utf-8")
    with open(path, "a", encoding="utf-8") as f:
        f.write("line2\n")
    assert path.read_text(encoding="utf-8") == "line1\nline2\n"


def test_insert_at_position(repo_dir):
    path = repo_dir / "insert_me.txt"
    path.write_text("Hello World", encoding="utf-8")
    content = path.read_text(encoding="utf-8")
    pos = 5
    new_content = content[:pos] + ", Beautiful" + content[pos:]
    path.write_text(new_content, encoding="utf-8")
    assert path.read_text(encoding="utf-8") == "Hello, Beautiful World"


def test_read_range(repo_dir):
    path = repo_dir / "range_me.txt"
    content = "0123456789abcdefghij"
    path.write_text(content, encoding="utf-8")
    result = content[5:10]
    assert result == "56789"


def test_rename_file(repo_dir):
    src = repo_dir / "old.txt"
    dst = repo_dir / "new.txt"
    src.write_text("content", encoding="utf-8")
    src.rename(dst)
    assert not src.exists()
    assert dst.exists()
    assert dst.read_text(encoding="utf-8") == "content"


def test_delete_file(repo_dir):
    path = repo_dir / "delete_me.txt"
    path.write_text("bye", encoding="utf-8")
    path.unlink()
    assert not path.exists()


def test_delete_folder(repo_dir):
    import shutil
    folder = repo_dir / "delete_folder"
    folder.mkdir()
    (folder / "file.txt").write_text("x", encoding="utf-8")
    shutil.rmtree(str(folder))
    assert not folder.exists()


def test_list_folder(repo_dir):
    (repo_dir / "subdir").mkdir()
    (repo_dir / "file.txt").write_text("x", encoding="utf-8")

    dirs = [e.name for e in repo_dir.iterdir() if e.is_dir()]
    files = [e.name for e in repo_dir.iterdir() if e.is_file()]

    assert "subdir" in dirs
    assert "file.txt" in files
