"""Rate limiting middleware for Nvidia Model Bridge."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from threading import Lock
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse


DEFAULT_RATE_LIMIT = 60  # requests per minute
DEFAULT_RATE_WINDOW = 60  # seconds


class TokenBucket:
    """Token bucket rate limiter."""

    def __init__(self, rate: int, window: int) -> None:
        self.rate = rate
        self.window = window
        self.tokens = float(rate)
        self.last_refill = time.monotonic()

    def consume(self) -> bool:
        """Try to consume a token. Returns True if allowed."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.rate, self.tokens + elapsed * (self.rate / self.window))
        self.last_refill = now

        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

    def time_until_available(self) -> float:
        """Seconds until the next token is available."""
        if self.tokens >= 1:
            return 0.0
        deficit = 1 - self.tokens
        return deficit * (self.window / self.rate)


class RateLimiter:
    """Per-client rate limiter using token buckets."""

    def __init__(
        self,
        default_rate: int | None = None,
        default_window: int | None = None,
    ) -> None:
        env_rate = os.getenv("NVIDIA_BRIDGE_RATE_LIMIT")
        env_window = os.getenv("NVIDIA_BRIDGE_RATE_WINDOW")
        self.default_rate = int(env_rate) if env_rate else (default_rate or DEFAULT_RATE_LIMIT)
        self.default_window = int(env_window) if env_window else (default_window or DEFAULT_RATE_WINDOW)
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = Lock()
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.monotonic()

    def is_allowed(self, client_id: str) -> tuple[bool, float]:
        """Check if a request from client_id is allowed.
        
        Returns (allowed, retry_after_seconds).
        """
        self._maybe_cleanup()
        with self._lock:
            if client_id not in self._buckets:
                self._buckets[client_id] = TokenBucket(
                    self.default_rate, self.default_window
                )
            bucket = self._buckets[client_id]
            allowed = bucket.consume()
            retry_after = 0.0 if allowed else bucket.time_until_available()
            return allowed, retry_after

    def get_client_id(self, request: Request) -> str:
        """Extract a client identifier from the request."""
        # Use X-Forwarded-For if behind a proxy, otherwise use client host
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"

    def _maybe_cleanup(self) -> None:
        now = time.monotonic()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        with self._lock:
            stale = [
                client_id
                for client_id, bucket in self._buckets.items()
                if now - bucket.last_refill > self.default_window * 2
            ]
            for client_id in stale:
                del self._buckets[client_id]
            self._last_cleanup = now

    def get_stats(self) -> dict[str, Any]:
        """Return rate limiter statistics."""
        with self._lock:
            return {
                "active_clients": len(self._buckets),
                "default_rate": self.default_rate,
                "default_window": self.default_window,
            }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    # Endpoints exempt from rate limiting
    EXEMPT_PATHS = {"/health", "/metrics", "/docs", "/openapi.json", "/redoc"}

    def __init__(self, app, rate_limiter: RateLimiter | None = None) -> None:
        super().__init__(app)
        self.limiter = rate_limiter or RateLimiter()

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        client_id = self.limiter.get_client_id(request)
        allowed, retry_after = self.limiter.is_allowed(client_id)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": "Rate limit exceeded. Please retry later.",
                        "type": "rate_limit_error",
                        "retry_after_seconds": round(retry_after, 2),
                    }
                },
                headers={
                    "Retry-After": str(int(retry_after) + 1),
                    "X-RateLimit-Limit": str(self.limiter.default_rate),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.limiter.default_rate)
        return response
