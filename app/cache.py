"""Caching layer for Nvidia Model Bridge with Redis and in-memory backends."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from cachetools import TTLCache

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Abstract interface for cache backends."""

    @abstractmethod
    def get(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Any | None: ...

    @abstractmethod
    def put(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        response: Any,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None: ...

    @abstractmethod
    def invalidate(self, model_id: str | None = None) -> int: ...

    @abstractmethod
    def get_stats(self) -> dict[str, Any]: ...

    @abstractmethod
    def reset_stats(self) -> None: ...


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    hit_count: int = 0


class ResponseCache(CacheBackend):
    """Thread-safe in-memory response cache with TTL."""

    def __init__(self, maxsize: int = 1000, ttl: int = 300) -> None:
        self._cache: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._lock = Lock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def _make_key(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a deterministic cache key from request parameters."""
        key_data = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        raw = json.dumps(key_data, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Any | None:
        """Retrieve a cached response, or None if not found."""
        key = self._make_key(model_id, messages, temperature, max_tokens)
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                self._stats["hits"] += 1
                entry.hit_count += 1
                return entry.value
            self._stats["misses"] += 1
            return None

    def put(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        response: Any,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Store a response in the cache."""
        key = self._make_key(model_id, messages, temperature, max_tokens)
        entry = CacheEntry(key=key, value=response, created_at=time.time())
        with self._lock:
            self._cache[key] = entry

    def invalidate(self, model_id: str | None = None) -> int:
        """Invalidate cache entries. If model_id given, only that model's entries."""
        with self._lock:
            if model_id is None:
                count = len(self._cache)
                self._cache.clear()
                return count
            keys_to_remove = []
            for key, entry in self._cache.items():
                if hasattr(entry, "value") and isinstance(entry.value, dict):
                    if entry.value.get("model") == model_id:
                        keys_to_remove.append(key)
            for key in keys_to_remove:
                del self._cache[key]
            return len(keys_to_remove)

    def get_stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0.0
            return {
                "backend": "in-memory",
                "size": len(self._cache),
                "maxsize": self._cache.maxsize,
                "ttl": self._cache.ttl,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": round(hit_rate, 4),
                "total_requests": total,
            }

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        with self._lock:
            self._stats = {"hits": 0, "misses": 0, "evictions": 0}


class RedisCacheBackend(CacheBackend):
    """Redis-backed response cache with TTL."""

    def __init__(
        self, redis_url: str, ttl: int = 300, prefix: str = "nvbridge:cache:"
    ) -> None:
        import redis as _redis

        self._client = _redis.from_url(redis_url, decode_responses=True)
        self._ttl = ttl
        self._prefix = prefix
        self._stats = {"hits": 0, "misses": 0}
        # Verify connectivity
        self._client.ping()
        # Log without leaking credentials
        safe_url = redis_url.split("@")[-1] if "@" in redis_url else redis_url
        logger.info(
            "Redis cache backend initialized", extra={"url": safe_url}
        )

    def _make_key(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a deterministic cache key from request parameters."""
        key_data = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        raw = json.dumps(key_data, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Any | None:
        """Retrieve a cached response from Redis, or None if not found."""
        key = self._prefix + self._make_key(
            model_id, messages, temperature, max_tokens
        )
        data = self._client.get(key)
        if data is not None:
            self._stats["hits"] += 1
            return json.loads(data)
        self._stats["misses"] += 1
        return None

    def put(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        response: Any,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Store a response in Redis with TTL."""
        key = self._prefix + self._make_key(
            model_id, messages, temperature, max_tokens
        )
        self._client.setex(key, self._ttl, json.dumps(response))

    def invalidate(self, model_id: str | None = None) -> int:
        """Invalidate cache entries stored in Redis."""
        if model_id is None:
            keys = self._client.keys(f"{self._prefix}*")
            if keys:
                self._client.delete(*keys)
            return len(keys)
        # Model-specific invalidation not supported for Redis
        return 0

    def get_stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0.0
        return {
            "backend": "redis",
            "ttl": self._ttl,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": round(hit_rate, 4),
            "total_requests": total,
        }

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._stats = {"hits": 0, "misses": 0}


# Module-level singleton
_response_cache: CacheBackend | None = None


def get_response_cache(maxsize: int = 1000, ttl: int = 300) -> CacheBackend:
    """Get or create the module-level response cache singleton.

    If the ``NVIDIA_BRIDGE_REDIS_URL`` environment variable is set, a Redis
    backend is used.  If Redis is unavailable or the variable is not set, the
    in-memory backend is used as a fallback.
    """
    global _response_cache
    if _response_cache is None:
        redis_url = os.environ.get("NVIDIA_BRIDGE_REDIS_URL")
        if redis_url:
            try:
                _response_cache = RedisCacheBackend(
                    redis_url=redis_url, ttl=ttl
                )
            except Exception as exc:
                logger.warning(
                    "Redis unavailable, falling back to in-memory cache",
                    extra={"error": str(exc)},
                )
                _response_cache = ResponseCache(maxsize=maxsize, ttl=ttl)
        else:
            _response_cache = ResponseCache(maxsize=maxsize, ttl=ttl)
    return _response_cache
