"""Request deduplication to prevent redundant API calls."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


class RequestDeduplicator:
    """Prevents duplicate concurrent requests from hitting the API multiple times.
    
    Uses a promise/future pattern: the first request for a given key starts
    the actual work, subsequent identical requests wait for the same result.
    """
    
    def __init__(self, ttl: float = 30.0) -> None:
        self._pending: dict[str, asyncio.Future] = {}
        self._lock = Lock()
        self._ttl = ttl
        self._stats = {"deduplicated": 0, "total": 0}
    
    def _make_key(self, model_id: str, messages: list[dict], **kwargs) -> str:
        key_data = {"model": model_id, "messages": messages}
        key_data.update({k: v for k, v in kwargs.items() if v is not None})
        raw = json.dumps(key_data, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(raw.encode()).hexdigest()
    
    async def deduplicate(self, key: str) -> asyncio.Future | None:
        """Check if a request with this key is already in flight.
        
        Returns the existing Future if deduplicated, None if this is a new request.
        """
        with self._lock:
            self._stats["total"] += 1
            if key in self._pending:
                self._stats["deduplicated"] += 1
                logger.debug("request_deduplicated", extra={"key": key[:12]})
                return self._pending[key]
            # Create a new future for this request
            loop = asyncio.get_event_loop()
            future = loop.create_future()
            self._pending[key] = future
            return None
    
    def resolve(self, key: str, result: Any) -> None:
        """Resolve a pending request, notifying all waiters."""
        with self._lock:
            future = self._pending.pop(key, None)
        if future and not future.done():
            future.set_result(result)
    
    def reject(self, key: str, error: Exception) -> None:
        """Reject a pending request with an error."""
        with self._lock:
            future = self._pending.pop(key, None)
        if future and not future.done():
            future.set_exception(error)
    
    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "pending_requests": len(self._pending),
                "total_requests": self._stats["total"],
                "deduplicated": self._stats["deduplicated"],
                "dedup_rate": round(
                    self._stats["deduplicated"] / max(self._stats["total"], 1), 4
                ),
            }


_deduplicator: RequestDeduplicator | None = None

def get_deduplicator() -> RequestDeduplicator:
    global _deduplicator
    if _deduplicator is None:
        _deduplicator = RequestDeduplicator()
    return _deduplicator
