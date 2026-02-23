"""Tests for OAuth 2.1 token management and PKCE verification."""

import base64
import hashlib
import secrets
import time

import pytest

from src.oauth.tokens import TokenManager, TokenError


# ---------------------------------------------------------------------------
# Token Manager tests
# ---------------------------------------------------------------------------

@pytest.fixture
def tm():
    return TokenManager(secret_key="test-secret-key-32-bytes-long!!", issuer="https://example.com")


def test_create_and_verify_access_token(tm):
    token = tm.create_access_token(client_id="test-client")
    claims = tm.verify_access_token(token)
    assert claims["sub"] == "test-client"
    assert claims["iss"] == "https://example.com"
    assert "exp" in claims


def test_verify_invalid_token(tm):
    with pytest.raises(TokenError):
        tm.verify_access_token("not.a.valid.token")


def test_verify_wrong_secret():
    tm1 = TokenManager("secret-1", "https://example.com")
    tm2 = TokenManager("secret-2", "https://example.com")
    token = tm1.create_access_token("client")
    with pytest.raises(TokenError):
        tm2.verify_access_token(token)


def test_refresh_token_roundtrip(tm):
    refresh = tm.create_refresh_token("client-abc")
    assert len(refresh) == 64  # 32 bytes hex = 64 chars
    client_id = tm.use_refresh_token(refresh)
    assert client_id == "client-abc"


def test_refresh_token_single_use(tm):
    refresh = tm.create_refresh_token("client-abc")
    tm.use_refresh_token(refresh)  # First use — OK
    result = tm.use_refresh_token(refresh)  # Second use — should return None
    assert result is None


def test_refresh_token_unknown(tm):
    result = tm.use_refresh_token("unknown-token")
    assert result is None


def test_access_token_scopes(tm):
    token = tm.create_access_token("client", scopes=["mcp", "read"])
    claims = tm.verify_access_token(token)
    assert claims["scopes"] == ["mcp", "read"]


# ---------------------------------------------------------------------------
# PKCE tests (testing the helper function directly)
# ---------------------------------------------------------------------------

def _make_pkce_pair():
    """Generate a valid PKCE code_verifier and code_challenge (S256)."""
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def test_pkce_verify_valid():
    from src.oauth.provider import _verify_pkce
    verifier, challenge = _make_pkce_pair()
    assert _verify_pkce(verifier, challenge) is True


def test_pkce_verify_wrong_verifier():
    from src.oauth.provider import _verify_pkce
    _, challenge = _make_pkce_pair()
    assert _verify_pkce("wrong-verifier", challenge) is False


def test_pkce_verify_tampered_challenge():
    from src.oauth.provider import _verify_pkce
    verifier, challenge = _make_pkce_pair()
    tampered = challenge[:-4] + "AAAA"
    assert _verify_pkce(verifier, tampered) is False
