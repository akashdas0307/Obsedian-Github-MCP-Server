"""Push Debouncer — batches write operations into single git pushes."""

import asyncio
from datetime import datetime, timezone
from typing import Optional

import structlog

from src.git_sync.manager import GitSyncManager

log = structlog.get_logger()


class PushDebouncer:
    """Debounces write notifications into a single git push.

    After the first write notification, waits `delay` seconds. If another
    write arrives during that window, the timer resets. When the timer
    finally fires, all accumulated changes are pushed in one commit.
    """

    def __init__(self, git_manager: GitSyncManager, delay: int = 120):
        self._git_manager = git_manager
        self._delay = delay
        self._timer_handle: Optional[asyncio.TimerHandle] = None
        self._pending_files: list[str] = []
        self._first_write_time: Optional[datetime] = None
        self._lock = asyncio.Lock()

    @property
    def has_pending(self) -> bool:
        return len(self._pending_files) > 0

    async def notify_write(self, file_path: str = ""):
        """Called after any write operation. Resets the debounce timer."""
        async with self._lock:
            if file_path and file_path not in self._pending_files:
                self._pending_files.append(file_path)

            if self._first_write_time is None:
                self._first_write_time = datetime.now(timezone.utc)

            # Cancel existing timer
            if self._timer_handle is not None:
                self._timer_handle.cancel()

            # Start new timer
            loop = asyncio.get_event_loop()
            self._timer_handle = loop.call_later(
                self._delay, lambda: asyncio.ensure_future(self._do_push())
            )
            log.debug(
                "debounce_reset",
                pending=len(self._pending_files),
                delay=self._delay,
            )

    async def _do_push(self):
        """Execute the debounced push."""
        async with self._lock:
            if not self._pending_files:
                return

            file_count = len(self._pending_files)
            files_summary = ", ".join(self._pending_files[:5])
            if file_count > 5:
                files_summary += f" (+{file_count - 5} more)"

            message = f"MCP: Update {file_count} file(s) — {files_summary}"

            log.info("debounce_push_start", files=file_count)
            success = await self._git_manager.push(message)

            if success:
                self._pending_files.clear()
                self._first_write_time = None
                self._timer_handle = None
                log.info("debounce_push_done", files=file_count)
            else:
                log.error("debounce_push_failed", files=file_count)

    async def force_push(self):
        """Force an immediate push (used during shutdown)."""
        if self._timer_handle is not None:
            self._timer_handle.cancel()
        await self._do_push()
