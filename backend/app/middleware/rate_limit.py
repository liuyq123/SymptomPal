"""Simple in-memory rate limiting middleware using a sliding window."""

import hashlib
import os
import time
from collections import defaultdict
from threading import Lock

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Per-IP sliding window rate limiter.

    Configure via environment variables:
    - RATE_LIMIT_RPM: requests per minute (default 60)
    - RATE_LIMIT_ENABLED: set to "false" to disable (default "true")
    """

    def __init__(self, app, rpm: int | None = None):
        super().__init__(app)
        self.rpm = rpm or int(os.environ.get("RATE_LIMIT_RPM", "60"))
        self.enabled = os.environ.get("RATE_LIMIT_ENABLED", "true").lower() != "false"
        self.trust_proxy = os.environ.get("TRUST_PROXY", "false").lower() == "true"
        self.window = 60.0  # 1 minute
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def _client_key(self, request: Request) -> str:
        """Extract client identifier (IP + API key if present)."""
        forwarded = request.headers.get("x-forwarded-for") if self.trust_proxy else None
        ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")
        api_key = request.headers.get("x-api-key", "")
        session_token = request.cookies.get("session_token", "")
        # Use API key as additional discriminator so different users behind
        # the same IP each get their own bucket.
        key = f"{ip}"
        if api_key:
            key = f"{key}:k:{hashlib.sha256(api_key.encode()).hexdigest()[:8]}"
        if session_token:
            key = f"{key}:s:{hashlib.sha256(session_token.encode()).hexdigest()[:8]}"
        return key

    def _is_rate_limited(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            timestamps = self._requests[key]
            # Prune old entries
            cutoff = now - self.window
            self._requests[key] = [t for t in timestamps if t > cutoff]
            if len(self._requests[key]) >= self.rpm:
                return True
            self._requests[key].append(now)
            return False

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self.enabled:
            return await call_next(request)

        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/"):
            return await call_next(request)

        key = self._client_key(request)
        if self._is_rate_limited(key):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.rpm)
        return response
