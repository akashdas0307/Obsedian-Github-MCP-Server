"""OAuth 2.1 Authorization Server implementation.

Exposes the endpoints required by Claude custom connectors:
  GET  /.well-known/oauth-authorization-server   — Discovery metadata
  GET  /.well-known/oauth-protected-resource     — Resource metadata
  GET  /authorize                                — Auth code + PKCE flow
  POST /token                                    — Token exchange
  POST /token  (refresh_token grant)             — Token refresh

The authorization endpoint renders a minimal HTML consent page.
For a single-user home-server deployment, it auto-approves after
the user clicks "Authorize".

PKCE (S256) is mandatory — plain code_challenge_method is rejected.
"""

from __future__ import annotations

import base64
import hashlib
import html
import secrets
import time
from typing import TYPE_CHECKING

import structlog
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route

if TYPE_CHECKING:
    from src.config import Settings

log = structlog.get_logger()

# In-memory store for pending authorization codes
# Structure: {code: {client_id, redirect_uri, code_challenge, expires_at}}
_auth_codes: dict[str, dict] = {}
AUTH_CODE_TTL = 600  # 10 minutes

_token_manager = None  # Initialised in build_oauth_routes()


def _get_token_manager(settings: "Settings"):
    """Lazily create the TokenManager."""
    global _token_manager
    if _token_manager is None:
        from src.oauth.tokens import TokenManager
        _token_manager = TokenManager(
            secret_key=settings.jwt_secret_key,
            issuer=settings.oauth_issuer_url,
        )
    return _token_manager


def _verify_pkce(code_verifier: str, code_challenge: str) -> bool:
    """Verify PKCE S256: SHA256(verifier) == base64url(challenge)."""
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    computed = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return computed == code_challenge


def _clean_expired_codes():
    """Remove expired auth codes from the in-memory store."""
    now = time.time()
    expired = [k for k, v in _auth_codes.items() if v["expires_at"] < now]
    for k in expired:
        del _auth_codes[k]


# ---------------------------------------------------------------------------
# Endpoint handlers
# ---------------------------------------------------------------------------

async def well_known_oauth_server(request: Request) -> JSONResponse:
    """OAuth 2.1 Authorization Server Metadata (RFC 8414)."""
    settings: Settings = request.app.state.settings
    base = settings.oauth_issuer_url.rstrip("/")
    return JSONResponse({
        "issuer": base,
        "authorization_endpoint": f"{base}/authorize",
        "token_endpoint": f"{base}/token",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["client_secret_post", "none"],
        "scopes_supported": ["mcp"],
    })


async def well_known_protected_resource(request: Request) -> JSONResponse:
    """OAuth 2.0 Protected Resource Metadata (RFC 9728)."""
    settings: Settings = request.app.state.settings
    base = settings.oauth_issuer_url.rstrip("/")
    return JSONResponse({
        "resource": base,
        "authorization_servers": [base],
        "scopes_supported": ["mcp"],
        "bearer_methods_supported": ["header"],
    })


async def authorize(request: Request) -> Response:
    """Authorization endpoint — handles both GET (show form) and POST (confirm)."""
    settings: Settings = request.app.state.settings

    if request.method == "GET":
        return await _authorize_get(request, settings)
    else:
        return await _authorize_post(request, settings)


async def _authorize_get(request: Request, settings: "Settings") -> Response:
    """Show the consent page."""
    params = dict(request.query_params)
    client_id = params.get("client_id", "")
    redirect_uri = params.get("redirect_uri", "")
    code_challenge = params.get("code_challenge", "")
    code_challenge_method = params.get("code_challenge_method", "")
    state = params.get("state", "")
    scope = params.get("scope", "mcp")

    # Validate required fields
    if not all([client_id, redirect_uri, code_challenge]):
        return JSONResponse(
            {"error": "invalid_request", "error_description": "Missing required parameters"},
            status_code=400,
        )

    if client_id != settings.oauth_client_id:
        return JSONResponse(
            {"error": "unauthorized_client", "error_description": "Unknown client_id"},
            status_code=401,
        )

    if code_challenge_method != "S256":
        return JSONResponse(
            {"error": "invalid_request", "error_description": "Only S256 code_challenge_method is supported"},
            status_code=400,
        )

    # Build a minimal consent HTML page
    safe_client = html.escape(client_id)
    safe_scope = html.escape(scope)

    consent_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Authorize — GitHub MCP Server</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: system-ui, sans-serif; background: #0f0f0f; color: #e8e8e8;
           display: flex; align-items: center; justify-content: center; min-height: 100vh; }}
    .card {{ background: #1a1a1a; border: 1px solid #2d2d2d; border-radius: 12px;
             padding: 2rem; max-width: 420px; width: 100%; }}
    h1 {{ font-size: 1.25rem; font-weight: 600; margin-bottom: 0.5rem; }}
    p {{ color: #aaa; font-size: 0.9rem; margin-bottom: 1.5rem; }}
    .scope-box {{ background: #111; border: 1px solid #2d2d2d; border-radius: 8px;
                 padding: 0.75rem 1rem; margin-bottom: 1.5rem; font-size: 0.85rem;
                 color: #ccc; }}
    .scope-box strong {{ color: #fff; }}
    .actions {{ display: flex; gap: 0.75rem; }}
    button {{ flex: 1; padding: 0.65rem 1rem; border-radius: 8px; border: none;
              font-size: 0.9rem; cursor: pointer; font-weight: 500; }}
    .btn-allow {{ background: #2563eb; color: #fff; }}
    .btn-allow:hover {{ background: #1d4ed8; }}
    .btn-deny {{ background: #2d2d2d; color: #ccc; }}
    .btn-deny:hover {{ background: #3d3d3d; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Authorization Request</h1>
    <p><strong style="color:#fff">{safe_client}</strong> is requesting access to your GitHub MCP Server.</p>
    <div class="scope-box">
      <strong>Requested scope:</strong> {safe_scope}
    </div>
    <form method="POST" action="/authorize">
      <input type="hidden" name="client_id" value="{html.escape(client_id)}">
      <input type="hidden" name="redirect_uri" value="{html.escape(redirect_uri)}">
      <input type="hidden" name="code_challenge" value="{html.escape(code_challenge)}">
      <input type="hidden" name="state" value="{html.escape(state)}">
      <div class="actions">
        <button type="submit" name="decision" value="allow" class="btn-allow">Authorize</button>
        <button type="submit" name="decision" value="deny" class="btn-deny">Deny</button>
      </div>
    </form>
  </div>
</body>
</html>"""
    return HTMLResponse(consent_html)


async def _authorize_post(request: Request, settings: "Settings") -> Response:
    """Process consent form submission."""
    form = await request.form()
    decision = form.get("decision", "deny")
    client_id = str(form.get("client_id", ""))
    redirect_uri = str(form.get("redirect_uri", ""))
    code_challenge = str(form.get("code_challenge", ""))
    state = str(form.get("state", ""))

    if decision != "allow":
        sep = "&" if "?" in redirect_uri else "?"
        return RedirectResponse(
            f"{redirect_uri}{sep}error=access_denied&state={state}",
            status_code=303,
        )

    # Generate authorization code
    _clean_expired_codes()
    code = secrets.token_urlsafe(32)
    _auth_codes[code] = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_challenge": code_challenge,
        "expires_at": time.time() + AUTH_CODE_TTL,
    }

    sep = "&" if "?" in redirect_uri else "?"
    redirect_url = f"{redirect_uri}{sep}code={code}"
    if state:
        redirect_url += f"&state={state}"

    log.info("oauth_code_issued", client_id=client_id)
    return RedirectResponse(redirect_url, status_code=303)


async def token(request: Request) -> JSONResponse:
    """Token endpoint — handles authorization_code and refresh_token grants."""
    settings: Settings = request.app.state.settings
    tm = _get_token_manager(settings)

    try:
        form = await request.form()
    except Exception:
        return JSONResponse(
            {"error": "invalid_request", "error_description": "Could not parse form body"},
            status_code=400,
        )

    grant_type = str(form.get("grant_type", ""))

    if grant_type == "authorization_code":
        return await _token_auth_code(form, settings, tm)
    elif grant_type == "refresh_token":
        return await _token_refresh(form, settings, tm)
    else:
        return JSONResponse(
            {"error": "unsupported_grant_type"},
            status_code=400,
        )


async def _token_auth_code(form, settings: "Settings", tm) -> JSONResponse:
    """Exchange authorization code for access + refresh tokens."""
    code = str(form.get("code", ""))
    code_verifier = str(form.get("code_verifier", ""))
    client_id = str(form.get("client_id", ""))
    redirect_uri = str(form.get("redirect_uri", ""))

    if not all([code, code_verifier, client_id]):
        return JSONResponse(
            {"error": "invalid_request", "error_description": "Missing code, code_verifier, or client_id"},
            status_code=400,
        )

    # Look up the auth code
    _clean_expired_codes()
    code_data = _auth_codes.pop(code, None)
    if code_data is None:
        return JSONResponse(
            {"error": "invalid_grant", "error_description": "Invalid or expired authorization code"},
            status_code=400,
        )

    # Verify client
    if code_data["client_id"] != client_id or client_id != settings.oauth_client_id:
        return JSONResponse(
            {"error": "invalid_client"},
            status_code=401,
        )

    # Verify redirect_uri matches
    if redirect_uri and code_data["redirect_uri"] != redirect_uri:
        return JSONResponse(
            {"error": "invalid_grant", "error_description": "redirect_uri mismatch"},
            status_code=400,
        )

    # Verify PKCE
    if not _verify_pkce(code_verifier, code_data["code_challenge"]):
        return JSONResponse(
            {"error": "invalid_grant", "error_description": "PKCE verification failed"},
            status_code=400,
        )

    # Issue tokens
    access_token = tm.create_access_token(client_id)
    refresh_token = tm.create_refresh_token(client_id)

    log.info("oauth_token_issued", client_id=client_id)
    return JSONResponse({
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": 3600,
        "refresh_token": refresh_token,
        "scope": "mcp",
    })


async def _token_refresh(form, settings: "Settings", tm) -> JSONResponse:
    """Exchange a refresh token for a new access token."""
    refresh_token_val = str(form.get("refresh_token", ""))
    client_id = str(form.get("client_id", ""))

    client_id_from_token = tm.use_refresh_token(refresh_token_val)
    if client_id_from_token is None:
        return JSONResponse(
            {"error": "invalid_grant", "error_description": "Invalid or expired refresh token"},
            status_code=400,
        )

    # Issue new tokens
    new_access_token = tm.create_access_token(client_id_from_token)
    new_refresh_token = tm.create_refresh_token(client_id_from_token)

    log.info("oauth_token_refreshed", client_id=client_id_from_token)
    return JSONResponse({
        "access_token": new_access_token,
        "token_type": "Bearer",
        "expires_in": 3600,
        "refresh_token": new_refresh_token,
        "scope": "mcp",
    })


# ---------------------------------------------------------------------------
# Route builder
# ---------------------------------------------------------------------------

def oauth_routes(settings: "Settings") -> list[Route]:
    """Return list of Starlette routes for OAuth endpoints."""
    return [
        Route(
            "/.well-known/oauth-authorization-server",
            well_known_oauth_server,
            methods=["GET"],
        ),
        Route(
            "/.well-known/oauth-protected-resource",
            well_known_protected_resource,
            methods=["GET"],
        ),
        Route("/authorize", authorize, methods=["GET", "POST"]),
        Route("/token", token, methods=["POST"]),
    ]
