"""OAuth Bearer token middleware for Starlette.

Protects the /mcp endpoint. All other paths pass through without auth.
Returns a standards-compliant 401 with WWW-Authenticate header when
the token is missing or invalid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.oauth.tokens import TokenError

if TYPE_CHECKING:
    from src.config import Settings

log = structlog.get_logger()

# Paths that do NOT require authentication
PUBLIC_PATHS = {
    "/health",
    "/.well-known/oauth-authorization-server",
    "/.well-known/oauth-protected-resource",
    "/authorize",
    "/token",
}


class OAuthMiddleware(BaseHTTPMiddleware):
    """Validates Bearer tokens on protected routes."""

    def __init__(self, app, settings: "Settings"):
        super().__init__(app)
        self._settings = settings
        self._token_manager = None

    def _get_token_manager(self):
        if self._token_manager is None:
            from src.oauth.tokens import TokenManager
            self._token_manager = TokenManager(
                secret_key=self._settings.jwt_secret_key,
                issuer=self._settings.oauth_issuer_url,
            )
        return self._token_manager

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path

        # Allow public paths and anything under /.well-known/
        if path in PUBLIC_PATHS or path.startswith("/.well-known/"):
            return await call_next(request)

        # Only enforce auth on /mcp routes
        if not path.startswith("/mcp"):
            return await call_next(request)

        # Extract Bearer token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            log.warning("oauth_missing_token", path=path)
            return self._unauthorized("Bearer token required")

        token = auth_header[len("Bearer "):]

        # Verify token
        tm = self._get_token_manager()
        try:
            claims = tm.verify_access_token(token)
            # Attach claims to request state for downstream use
            request.state.oauth_claims = claims
            return await call_next(request)
        except TokenError as e:
            log.warning("oauth_invalid_token", error=str(e), path=path)
            return self._unauthorized(str(e))

    def _unauthorized(self, detail: str) -> JSONResponse:
        base = self._settings.oauth_issuer_url.rstrip("/")
        return JSONResponse(
            {
                "error": "unauthorized",
                "error_description": detail,
            },
            status_code=401,
            headers={
                "WWW-Authenticate": (
                    f'Bearer realm="{base}",'
                    f' error="invalid_token",'
                    f' error_description="{detail}"'
                )
            },
        )
