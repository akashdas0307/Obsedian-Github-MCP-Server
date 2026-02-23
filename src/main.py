"""
MCP GitHub Server — Main entry point.

A self-hosted MCP HTTP server that provides CRUD operations and semantic
search on a single GitHub repository. Connects to Claude.ai as a custom
connector via OAuth 2.1 + Streamable HTTP MCP transport.

Architecture:
  - Starlette app (root):  OAuth endpoints + health + MCP mount
  - FastMCP sub-app:       15 MCP tools (14 CRUD + 1 semantic search)
  - GitSyncManager:        Local clone, periodic pull, debounced push
  - SemanticSearchEngine:  nomic-embed + FAISS + cross-encoder reranker
  - OAuthMiddleware:       Bearer token validation on /mcp routes
"""

import asyncio
import contextlib

import structlog
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from mcp.server.fastmcp import FastMCP

from src.config import get_settings
from src.git_sync.debouncer import PushDebouncer
from src.git_sync.manager import GitSyncManager
from src.oauth.middleware import OAuthMiddleware
from src.oauth.provider import oauth_routes
from src.semantic.engine import SemanticSearchEngine

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Load & validate configuration at import time
# ---------------------------------------------------------------------------
settings = get_settings()

# ---------------------------------------------------------------------------
# Shared component singletons (populated during lifespan startup)
# ---------------------------------------------------------------------------
git_manager: GitSyncManager | None = None
debouncer: PushDebouncer | None = None
search_engine: SemanticSearchEngine | None = None
_pull_task: asyncio.Task | None = None

# ---------------------------------------------------------------------------
# FastMCP server — all tools registered here
# ---------------------------------------------------------------------------
mcp_server = FastMCP(
    "GitHub MCP Server",
    stateless_http=True,
    json_response=True,
)

# Register tools (imported after mcp_server is created)
from src.tools.crud import register_crud_tools  # noqa: E402
from src.tools.search import register_search_tool  # noqa: E402

register_crud_tools(mcp_server)
register_search_tool(mcp_server)

# ---------------------------------------------------------------------------
# Background pull loop
# ---------------------------------------------------------------------------
async def _periodic_pull():
    """Pull from remote every settings.pull_interval seconds."""
    while True:
        await asyncio.sleep(settings.pull_interval)
        try:
            changed = await git_manager.pull()
            if changed and search_engine is not None:
                for file_path, change_type in changed:
                    await search_engine.on_file_changed(file_path, change_type)
                if changed:
                    log.info("periodic_pull_indexed", files=len(changed))
        except Exception:
            log.exception("periodic_pull_error")

# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------
@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    """Initialise all components on startup; clean up on shutdown."""
    global git_manager, debouncer, search_engine, _pull_task

    log.info("startup_begin", repo=settings.github_repo_url)

    # 1 — Git clone / pull
    git_manager = GitSyncManager(settings)
    await git_manager.init()
    log.info("git_sync_ready")

    # 2 — Push debouncer (waits push_debounce seconds after last write)
    debouncer = PushDebouncer(git_manager, settings.push_debounce)

    # 3 — Semantic search engine (loads saved index or builds from scratch)
    search_engine = SemanticSearchEngine(settings)
    await search_engine.init()
    log.info("search_engine_ready", vectors=search_engine._faiss.total_vectors)

    # 4 — Background periodic pull task
    _pull_task = asyncio.create_task(_periodic_pull())

    # Expose components via app.state so they're accessible to tools
    app.state.settings = settings
    app.state.git_manager = git_manager
    app.state.debouncer = debouncer
    app.state.search_engine = search_engine

    log.info("startup_complete", port=settings.port)

    # ---- yield (server runs here) ----
    yield

    # Shutdown sequence
    log.info("shutdown_begin")

    if _pull_task:
        _pull_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _pull_task

    # Final push of any pending changes
    if debouncer and debouncer.has_pending:
        log.info("shutdown_final_push")
        await debouncer.force_push()

    # Persist semantic index
    if search_engine:
        search_engine.save()
        log.info("shutdown_index_saved")

    log.info("shutdown_complete")

# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------
async def health(request: Request) -> JSONResponse:
    """Health check endpoint."""
    status: dict = {"status": "ok", "service": "GitHub MCP Server"}
    if git_manager:
        try:
            git_status = await git_manager.get_current_status()
            status["git"] = git_status
        except Exception:
            status["git"] = {"error": "unavailable"}

    if search_engine:
        status["index"] = {"vectors": search_engine._faiss.total_vectors}

    return JSONResponse(status)

# ---------------------------------------------------------------------------
# Starlette application
# ---------------------------------------------------------------------------
app = Starlette(
    routes=[
        # Health check (public)
        Route("/health", health, methods=["GET"]),
        # OAuth 2.1 endpoints (public)
        *oauth_routes(settings),
        # MCP Streamable HTTP transport (protected by OAuthMiddleware)
        Mount("/mcp", app=mcp_server.streamable_http_app()),
    ],
    lifespan=lifespan,
)

# Add OAuth middleware (validates Bearer tokens on /mcp/* paths)
app.add_middleware(OAuthMiddleware, settings=settings)

# ---------------------------------------------------------------------------
# CLI / Docker entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False,
    )
