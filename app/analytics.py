"""Usage analytics and dashboard data for Nvidia Model Bridge."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class RequestRecord:
    """A single request record for analytics."""
    timestamp: float
    model_id: str
    task_type: str
    success: bool
    latency_seconds: float
    fallback_used: bool = False
    cache_hit: bool = False
    endpoint: str = ""


class AnalyticsCollector:
    """In-memory analytics collector for real-time dashboard data."""

    def __init__(self, max_records: int = 10000) -> None:
        self._records: list[RequestRecord] = []
        self._max_records = max_records
        self._lock = Lock()
        self._started_at = time.time()

    def record(
        self,
        model_id: str,
        task_type: str,
        success: bool,
        latency_seconds: float,
        fallback_used: bool = False,
        cache_hit: bool = False,
        endpoint: str = "",
    ) -> None:
        """Record a request for analytics."""
        record = RequestRecord(
            timestamp=time.time(),
            model_id=model_id,
            task_type=task_type,
            success=success,
            latency_seconds=latency_seconds,
            fallback_used=fallback_used,
            cache_hit=cache_hit,
            endpoint=endpoint,
        )
        with self._lock:
            self._records.append(record)
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records:]

    def get_dashboard_data(self, hours: int = 24) -> dict[str, Any]:
        """Get comprehensive dashboard data for the specified time window."""
        cutoff = time.time() - (hours * 3600)
        with self._lock:
            recent = [r for r in self._records if r.timestamp > cutoff]

        if not recent:
            return self._empty_dashboard(hours)

        total = len(recent)
        successes = sum(1 for r in recent if r.success)
        failures = total - successes
        cache_hits = sum(1 for r in recent if r.cache_hit)
        fallbacks = sum(1 for r in recent if r.fallback_used)
        latencies = [r.latency_seconds for r in recent if r.success]

        model_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"requests": 0, "successes": 0, "total_latency": 0.0}
        )
        for r in recent:
            stats = model_stats[r.model_id]
            stats["requests"] += 1
            if r.success:
                stats["successes"] += 1
                stats["total_latency"] += r.latency_seconds

        task_type_stats: dict[str, int] = defaultdict(int)
        for r in recent:
            task_type_stats[r.task_type] += 1

        endpoint_stats: dict[str, int] = defaultdict(int)
        for r in recent:
            endpoint_stats[r.endpoint] += 1

        return {
            "period_hours": hours,
            "uptime_seconds": round(time.time() - self._started_at, 2),
            "total_requests": total,
            "successful_requests": successes,
            "failed_requests": failures,
            "success_rate": round(successes / total, 4) if total > 0 else 0.0,
            "cache_hits": cache_hits,
            "cache_hit_rate": round(cache_hits / total, 4) if total > 0 else 0.0,
            "fallback_activations": fallbacks,
            "fallback_rate": round(fallbacks / total, 4) if total > 0 else 0.0,
            "latency": {
                "avg": round(sum(latencies) / len(latencies), 4) if latencies else 0,
                "min": round(min(latencies), 4) if latencies else 0,
                "max": round(max(latencies), 4) if latencies else 0,
                "p50": round(sorted(latencies)[len(latencies) // 2], 4) if latencies else 0,
                "p95": round(
                    sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 4
                ),
                "p99": round(
                    sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0, 4
                ),
            },
            "models": {
                model_id: {
                    "requests": stats["requests"],
                    "successes": stats["successes"],
                    "success_rate": round(stats["successes"] / stats["requests"], 4)
                    if stats["requests"] > 0 else 0.0,
                    "avg_latency": round(
                        stats["total_latency"] / stats["successes"], 4
                    ) if stats["successes"] > 0 else 0.0,
                }
                for model_id, stats in sorted(
                    model_stats.items(), key=lambda x: x[1]["requests"], reverse=True
                )
            },
            "task_types": dict(
                sorted(task_type_stats.items(), key=lambda x: x[1], reverse=True)
            ),
            "endpoints": dict(
                sorted(endpoint_stats.items(), key=lambda x: x[1], reverse=True)
            ),
        }

    def _empty_dashboard(self, hours: int) -> dict[str, Any]:
        return {
            "period_hours": hours,
            "uptime_seconds": round(time.time() - self._started_at, 2),
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "success_rate": 0.0,
            "cache_hits": 0,
            "cache_hit_rate": 0.0,
            "fallback_activations": 0,
            "fallback_rate": 0.0,
            "latency": {"avg": 0, "min": 0, "max": 0, "p50": 0, "p95": 0, "p99": 0},
            "models": {},
            "task_types": {},
            "endpoints": {},
        }

    def get_record_count(self) -> int:
        with self._lock:
            return len(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


# Module-level singleton
_collector: AnalyticsCollector | None = None


def get_analytics_collector(max_records: int = 10000) -> AnalyticsCollector:
    """Get or create the analytics collector singleton."""
    global _collector
    if _collector is None:
        _collector = AnalyticsCollector(max_records=max_records)
    return _collector
