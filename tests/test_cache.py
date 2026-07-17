"""Tests for caching module."""

from __future__ import annotations

from app.cache import ResponseCache


class TestResponseCache:
    def test_put_and_get(self):
        cache = ResponseCache(maxsize=10, ttl=60)
        cache.put("model/a", [{"role": "user", "content": "hello"}], {"content": "world"})
        result = cache.get("model/a", [{"role": "user", "content": "hello"}])
        assert result is not None
        assert result["content"] == "world"

    def test_cache_miss(self):
        cache = ResponseCache(maxsize=10, ttl=60)
        result = cache.get("model/a", [{"role": "user", "content": "hello"}])
        assert result is None

    def test_different_messages_different_keys(self):
        cache = ResponseCache(maxsize=10, ttl=60)
        cache.put("model/a", [{"role": "user", "content": "hello"}], {"content": "a"})
        result = cache.get("model/a", [{"role": "user", "content": "world"}])
        assert result is None

    def test_invalidate_all(self):
        cache = ResponseCache(maxsize=10, ttl=60)
        cache.put("model/a", [{"role": "user", "content": "1"}], {"content": "a"})
        cache.put("model/b", [{"role": "user", "content": "2"}], {"content": "b"})
        count = cache.invalidate()
        assert count == 2

    def test_stats(self):
        cache = ResponseCache(maxsize=10, ttl=60)
        cache.put("model/a", [{"role": "user", "content": "1"}], {"content": "a"})
        cache.get("model/a", [{"role": "user", "content": "1"}])  # hit
        cache.get("model/a", [{"role": "user", "content": "2"}])  # miss
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
