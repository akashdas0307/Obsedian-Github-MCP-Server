"""Git Sync Manager — handles clone, pull, push, and change detection."""

import asyncio
from pathlib import Path
from typing import Optional

import structlog
from git import Repo, GitCommandError

from src.config import Settings

log = structlog.get_logger()

# Change types matching git diff --name-status output
CHANGE_ADDED = "A"
CHANGE_MODIFIED = "M"
CHANGE_DELETED = "D"
CHANGE_RENAMED = "R"


class GitSyncManager:
    """Manages a local Git clone with periodic pull and debounced push."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._repo: Optional[Repo] = None
        self._repo_path = Path(settings.repo_dir)
        self._lock = asyncio.Lock()

    @property
    def repo(self) -> Repo:
        assert self._repo is not None, "GitSyncManager not initialised"
        return self._repo

    @property
    def repo_path(self) -> Path:
        return self._repo_path

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def init(self):
        """Clone the repository if it doesn't exist, otherwise open and pull."""
        await asyncio.to_thread(self._init_sync)

    def _init_sync(self):
        if (self._repo_path / ".git").exists():
            log.info("git_open_existing", path=str(self._repo_path))
            self._repo = Repo(self._repo_path)
            self._configure_remote()
            self._pull_sync()
        else:
            log.info("git_cloning", url=self._settings.github_repo_url)
            self._repo_path.mkdir(parents=True, exist_ok=True)
            self._repo = Repo.clone_from(
                self._settings.authenticated_repo_url,
                str(self._repo_path),
                branch=self._settings.github_branch,
            )
            log.info("git_cloned", branch=self._settings.github_branch)

    def _configure_remote(self):
        """Ensure the origin remote uses the authenticated URL."""
        origin = self._repo.remotes.origin
        with origin.config_writer as cw:
            cw.set("url", self._settings.authenticated_repo_url)

    # ------------------------------------------------------------------
    # Pull
    # ------------------------------------------------------------------

    async def pull(self) -> list[tuple[str, str]]:
        """Pull from remote with rebase.

        Returns list of (file_path, change_type) tuples for changed files.
        """
        async with self._lock:
            return await asyncio.to_thread(self._pull_sync)

    def _pull_sync(self) -> list[tuple[str, str]]:
        """Synchronous pull implementation."""
        try:
            old_head = self._repo.head.commit.hexsha

            # Fetch first to see if there are changes
            self._repo.remotes.origin.fetch()

            # Check if remote has new commits
            branch = self._settings.github_branch
            remote_ref = f"origin/{branch}"

            try:
                remote_commit = self._repo.commit(remote_ref)
            except Exception:
                log.warning("git_remote_ref_not_found", ref=remote_ref)
                return []

            if remote_commit.hexsha == old_head:
                return []  # Nothing new

            # Pull with rebase
            try:
                self._repo.git.pull("--rebase", "origin", branch)
            except GitCommandError as e:
                if "conflict" in str(e).lower() or "merge" in str(e).lower():
                    log.error("git_pull_conflict", error=str(e))
                    # Abort rebase to keep repo in clean state
                    try:
                        self._repo.git.rebase("--abort")
                    except GitCommandError:
                        pass
                    return []
                raise

            new_head = self._repo.head.commit.hexsha
            if new_head == old_head:
                return []

            # Detect changed files
            changes = self._get_changes(old_head, new_head)
            log.info("git_pulled", changes=len(changes))
            return changes

        except Exception:
            log.exception("git_pull_failed")
            return []

    # ------------------------------------------------------------------
    # Push
    # ------------------------------------------------------------------

    async def push(self, message: str = "MCP server auto-commit") -> bool:
        """Stage all changes, commit, pull --rebase, and push.

        Returns True on success, False on failure.
        """
        async with self._lock:
            return await asyncio.to_thread(self._push_sync, message)

    def _push_sync(self, message: str) -> bool:
        """Synchronous push implementation."""
        try:
            # Check for changes
            if not self._repo.is_dirty(untracked_files=True):
                log.info("git_push_skip_clean")
                return True

            # Stage everything
            self._repo.git.add("-A")

            # Commit
            self._repo.git.commit("-m", message)
            log.info("git_committed", message=message)

            # Pull rebase before push to avoid conflicts
            branch = self._settings.github_branch
            try:
                self._repo.git.pull("--rebase", "origin", branch)
            except GitCommandError as e:
                if "conflict" in str(e).lower():
                    log.error("git_push_conflict", error=str(e))
                    try:
                        self._repo.git.rebase("--abort")
                    except GitCommandError:
                        pass
                    return False
                raise

            # Push
            self._repo.git.push("origin", branch)
            log.info("git_pushed", branch=branch)
            return True

        except Exception:
            log.exception("git_push_failed")
            return False

    # ------------------------------------------------------------------
    # Change detection
    # ------------------------------------------------------------------

    def _get_changes(self, old_sha: str, new_sha: str) -> list[tuple[str, str]]:
        """Get list of (file_path, change_type) between two commits."""
        changes = []
        try:
            diff_output = self._repo.git.diff(
                "--name-status", old_sha, new_sha
            )
            for line in diff_output.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    change_type = parts[0][0]  # First char: A, M, D, R, etc.
                    file_path = parts[-1]  # Last element is the file path
                    changes.append((file_path, change_type))
        except Exception:
            log.exception("git_diff_failed", old=old_sha, new=new_sha)
        return changes

    async def get_current_status(self) -> dict:
        """Return current repo status for diagnostics."""
        return await asyncio.to_thread(self._status_sync)

    def _status_sync(self) -> dict:
        return {
            "branch": str(self._repo.active_branch),
            "head": self._repo.head.commit.hexsha[:8],
            "dirty": self._repo.is_dirty(untracked_files=True),
            "untracked": len(self._repo.untracked_files),
        }
