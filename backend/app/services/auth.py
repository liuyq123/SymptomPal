import json
import os
import secrets
import time
import threading
from typing import Optional
from urllib.parse import urlparse

from fastapi import Header, HTTPException, Request


# ── API key → user_id map (cached at startup) ────────────────────────────────

_api_key_user_map: Optional[dict[str, str]] = None
_api_key_map_lock = threading.Lock()


def _load_api_key_user_map() -> dict[str, str]:
    """Load and cache optional API key -> user_id bindings from JSON."""
    global _api_key_user_map
    if _api_key_user_map is not None:
        return _api_key_user_map

    with _api_key_map_lock:
        if _api_key_user_map is not None:
            return _api_key_user_map

        raw = os.environ.get("API_KEYS_JSON", "").strip()
        if not raw:
            _api_key_user_map = {}
            return _api_key_user_map

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=500, detail="Invalid API_KEYS_JSON") from exc

        if not isinstance(parsed, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()):
            raise HTTPException(status_code=500, detail="API_KEYS_JSON must map API key strings to user_id strings")
        _api_key_user_map = parsed
        return _api_key_user_map


# ── Session token store (in-memory, for cookie-based auth) ───────────────────

_sessions: dict[str, dict] = {}  # token -> {"user_id": str | None, "bound_user": str | None, "created_at": float}
_sessions_lock = threading.Lock()
SESSION_TTL = 24 * 3600  # 24 hours


def create_session(bound_user: str | None) -> str:
    """Create a session token after successful API key authentication."""
    token = secrets.token_urlsafe(32)
    with _sessions_lock:
        # Prune expired sessions (lazy cleanup)
        now = time.time()
        expired = [k for k, v in _sessions.items() if now - v["created_at"] > SESSION_TTL]
        for k in expired:
            del _sessions[k]
        _sessions[token] = {"bound_user": bound_user, "created_at": now}
    return token


def _validate_session(token: str) -> dict | None:
    """Validate a session token and return session data, or None if invalid."""
    with _sessions_lock:
        session = _sessions.get(token)
        if session is None:
            return None
        if time.time() - session["created_at"] > SESSION_TTL:
            del _sessions[token]
            return None
        return session


# ── Core authentication ──────────────────────────────────────────────────────

def _authenticate_and_get_bound_user(x_api_key: str) -> str | None:
    """
    Authenticate API key and return optional bound user_id.

    Priority:
    1) `API_KEYS_JSON` (multi-key mapping)
    2) `API_KEY` with optional `API_KEY_USER_ID` binding
    """
    key_user_map = _load_api_key_user_map()
    if key_user_map:
        for key, user_id in key_user_map.items():
            if secrets.compare_digest(x_api_key, key):
                return user_id
        raise HTTPException(status_code=401, detail="Invalid API key")

    expected = os.environ.get("API_KEY")
    if not expected:
        raise HTTPException(status_code=500, detail="API key not configured")
    if not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")

    return os.environ.get("API_KEY_USER_ID")


def _resolve_request_user_id(bound_user_id: str | None, x_user_id: str | None) -> str:
    """Resolve effective request user_id from binding and optional header."""
    if bound_user_id:
        if x_user_id and x_user_id != bound_user_id:
            raise HTTPException(status_code=403, detail="Authenticated API key cannot act on this user")
        return bound_user_id

    if not x_user_id:
        raise HTTPException(status_code=400, detail="X-User-Id header is required")
    return x_user_id


# ── FastAPI dependencies ─────────────────────────────────────────────────────

def _authenticate_request(request: Request) -> str | None:
    """
    Authenticate via session cookie OR API key header.
    Returns bound user_id (or None if no binding).
    """
    # Try session cookie first
    session_token = request.cookies.get("session_token")
    if session_token:
        session = _validate_session(session_token)
        if session is not None:
            _enforce_csrf(request)
            return session["bound_user"]

    # Fall back to API key header
    api_key = request.headers.get("x-api-key")
    if api_key:
        return _authenticate_and_get_bound_user(api_key)

    raise HTTPException(status_code=401, detail="Authentication required (session cookie or X-API-Key header)")


def _enforce_csrf(request: Request) -> None:
    """Basic CSRF protection for cookie-based auth."""
    if os.environ.get("CSRF_CHECK_ENABLED", "true").lower() == "false":
        return
    if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
        return

    origin = request.headers.get("origin")
    if not origin:
        referer = request.headers.get("referer")
        if referer:
            parsed = urlparse(referer)
            if parsed.scheme and parsed.netloc:
                origin = f"{parsed.scheme}://{parsed.netloc}"

    if not origin:
        raise HTTPException(status_code=403, detail="Missing Origin/Referer for CSRF check")

    default_origins = "http://localhost:5173,http://127.0.0.1:5173"
    raw_origins = os.environ.get("CSRF_TRUSTED_ORIGINS") or os.environ.get("CORS_ORIGINS", default_origins)
    allowed = {o.strip() for o in raw_origins.split(",") if o.strip()}
    if origin not in allowed:
        raise HTTPException(status_code=403, detail="Origin not allowed for CSRF check")


async def require_api_key(request: Request) -> None:
    """Enforce authentication for all /api routes (session cookie or API key)."""
    _authenticate_request(request)


async def get_request_user_id(
    request: Request,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> str:
    """Return effective request user_id, enforcing key-user binding when configured."""
    bound_user = _authenticate_request(request)
    return _resolve_request_user_id(bound_user, x_user_id)


def enforce_user_match(request_user_id: str, expected_user_id: str) -> None:
    """Ensure the request user id matches the resource user id."""
    if request_user_id != expected_user_id:
        raise HTTPException(status_code=403, detail="User does not own this resource")
