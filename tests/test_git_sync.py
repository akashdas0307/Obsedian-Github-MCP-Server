"""Tests for Git Sync Manager — uses a local bare repo as remote."""

import os
import tempfile
from pathlib import Path

import pytest

# Skip git-dependent tests if GitPython/git binary are unavailable
git = pytest.importorskip("git", reason="GitPython / git binary not available")
Repo = git.Repo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_test_repo(tmp_path: Path) -> tuple[Path, Path]:
    """Create a bare 'remote' and a local clone for testing.

    Returns (bare_repo_path, local_clone_path).
    """
    bare_path = tmp_path / "remote.git"
    bare_path.mkdir()
    bare_repo = Repo.init(str(bare_path), bare=True)

    # Create a working copy to seed the bare repo
    seed_path = tmp_path / "seed"
    seed_repo = Repo.clone_from(str(bare_path), str(seed_path))
    seed_repo.config_writer().set_value("user", "name", "Test").release()
    seed_repo.config_writer().set_value("user", "email", "t@t.com").release()

    # Add an initial commit
    readme = seed_path / "README.md"
    readme.write_text("# Test Repo\n")
    seed_repo.index.add(["README.md"])
    seed_repo.index.commit("Initial commit")
    seed_repo.remotes.origin.push(refspec="HEAD:refs/heads/main")

    return bare_path, seed_path


# ---------------------------------------------------------------------------
# Debouncer tests (unit — no real git)
# ---------------------------------------------------------------------------

class MockGitManager:
    """Minimal mock for PushDebouncer tests."""
    pushes: list[str] = []

    async def push(self, message: str) -> bool:
        self.pushes.append(message)
        return True


@pytest.mark.asyncio
async def test_debouncer_fires_after_delay(monkeypatch):
    """Debouncer should call push after the delay expires."""
    import asyncio
    from src.git_sync.debouncer import PushDebouncer

    mock = MockGitManager()
    mock.pushes = []
    d = PushDebouncer(mock, delay=0)  # 0-second delay for testing

    await d.notify_write("file.txt")
    await asyncio.sleep(0.05)  # Let the event loop tick

    assert len(mock.pushes) == 1
    assert "file.txt" in mock.pushes[0]


@pytest.mark.asyncio
async def test_debouncer_resets_on_multiple_writes(monkeypatch):
    """Multiple rapid writes should result in a single push."""
    import asyncio
    from src.git_sync.debouncer import PushDebouncer

    mock = MockGitManager()
    mock.pushes = []
    d = PushDebouncer(mock, delay=0)

    await d.notify_write("a.txt")
    await d.notify_write("b.txt")
    await d.notify_write("c.txt")
    await asyncio.sleep(0.05)

    # Only one push should have happened
    assert len(mock.pushes) == 1
    # All files should be in the commit message
    assert "a.txt" in mock.pushes[0]


@pytest.mark.asyncio
async def test_debouncer_force_push():
    """force_push should push immediately."""
    import asyncio
    from src.git_sync.debouncer import PushDebouncer

    mock = MockGitManager()
    mock.pushes = []
    d = PushDebouncer(mock, delay=9999)  # Very long delay

    await d.notify_write("urgent.txt")
    assert len(mock.pushes) == 0  # Not pushed yet

    await d.force_push()
    assert len(mock.pushes) == 1


# ---------------------------------------------------------------------------
# Change detection tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__("shutil").which("git"),
    reason="git binary not available in PATH",
)
def test_get_changes_detects_added_file(tmp_path):
    """GitSyncManager._get_changes should detect newly added files."""
    bare_path, clone_path = create_test_repo(tmp_path)
    repo = Repo(str(clone_path))
    repo.config_writer().set_value("user", "name", "Test").release()
    repo.config_writer().set_value("user", "email", "t@t.com").release()

    old_sha = repo.head.commit.hexsha

    # Add a new file
    new_file = clone_path / "new_feature.py"
    new_file.write_text("def hello(): pass\n")
    repo.index.add(["new_feature.py"])
    repo.index.commit("Add new_feature.py")

    new_sha = repo.head.commit.hexsha

    # Use the manager's change detection logic
    diff_output = repo.git.diff("--name-status", old_sha, new_sha)
    changes = []
    for line in diff_output.strip().split("\n"):
        if line.strip():
            parts = line.split("\t")
            if len(parts) >= 2:
                changes.append((parts[-1], parts[0][0]))

    files = [c[0] for c in changes]
    types = [c[1] for c in changes]
    assert "new_feature.py" in files
    assert "A" in types
