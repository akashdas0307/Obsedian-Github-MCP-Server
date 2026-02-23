"""JWT access token and refresh token management for OAuth 2.1."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional

import jwt
import structlog

log = structlog.get_logger()

ACCESS_TOKEN_TTL = 3600        # 1 hour
REFRESH_TOKEN_TTL = 86400 * 30  # 30 days
ALGORITHM = "HS256"


class TokenError(Exception):
    """Raised when a token cannot be validated."""
    pass


class TokenManager:
    """Creates and validates JWT access tokens, and manages refresh tokens."""

    def __init__(self, secret_key: str, issuer: str):
        self._secret = secret_key
        self._issuer = issuer
        # In-memory refresh token store: {token: {client_id, expires_at}}
        self._refresh_tokens: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Access tokens
    # ------------------------------------------------------------------

    def create_access_token(
        self,
        client_id: str,
        scopes: list[str] | None = None,
    ) -> str:
        """Create a signed JWT access token.

        Args:
            client_id: The OAuth client that authenticated.
            scopes: Granted scopes (defaults to ["mcp"]).

        Returns:
            Signed JWT string.
        """
        now = datetime.now(timezone.utc)
        payload = {
            "iss": self._issuer,
            "sub": client_id,
            "aud": self._issuer,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(seconds=ACCESS_TOKEN_TTL)).timestamp()),
            "scopes": scopes or ["mcp"],
        }
        return jwt.encode(payload, self._secret, algorithm=ALGORITHM)

    def verify_access_token(self, token: str) -> dict:
        """Verify a JWT access token.

        Args:
            token: JWT string from Authorization header.

        Returns:
            Decoded payload dict on success.

        Raises:
            TokenError: If the token is invalid, expired, or malformed.
        """
        try:
            payload = jwt.decode(
                token,
                self._secret,
                algorithms=[ALGORITHM],
                audience=self._issuer,
                issuer=self._issuer,
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise TokenError("Token has expired")
        except jwt.InvalidAudienceError:
            raise TokenError("Invalid token audience")
        except jwt.InvalidIssuerError:
            raise TokenError("Invalid token issuer")
        except jwt.DecodeError as e:
            raise TokenError(f"Token decode error: {e}")

    # ------------------------------------------------------------------
    # Refresh tokens
    # ------------------------------------------------------------------

    def create_refresh_token(self, client_id: str) -> str:
        """Create an opaque refresh token.

        Args:
            client_id: The client this token belongs to.

        Returns:
            Random 64-char hex token string.
        """
        token = secrets.token_hex(32)
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=REFRESH_TOKEN_TTL)
        self._refresh_tokens[token] = {
            "client_id": client_id,
            "expires_at": expires_at.isoformat(),
        }
        return token

    def use_refresh_token(self, token: str) -> Optional[str]:
        """Validate and consume a refresh token.

        Args:
            token: The refresh token to validate.

        Returns:
            client_id if valid, None if invalid or expired.
        """
        entry = self._refresh_tokens.get(token)
        if not entry:
            return None

        expires_at = datetime.fromisoformat(entry["expires_at"])
        if datetime.now(timezone.utc) > expires_at:
            del self._refresh_tokens[token]
            return None

        # Rotate: delete old, caller will create new
        del self._refresh_tokens[token]
        return entry["client_id"]
