"""Prometheus metrics for Nvidia Model Bridge."""

from __future__ import annotations

from typing import Any

from prometheus_client import Counter, Gauge, Histogram, Info


# Request metrics
REQUEST_COUNT = Counter(
    "nvidia_bridge_requests_total",
    "Total number of requests",
    ["endpoint", "method", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "nvidia_bridge_request_duration_seconds",
    "Request duration in seconds",
    ["endpoint", "method"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

# Model metrics
MODEL_REQUEST_COUNT = Counter(
    "nvidia_bridge_model_requests_total",
    "Total requests per model",
    ["model_id", "task_type", "success"],
)

MODEL_LATENCY = Histogram(
    "nvidia_bridge_model_latency_seconds",
    "Model response latency in seconds",
    ["model_id"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

FALLBACK_COUNT = Counter(
    "nvidia_bridge_fallback_total",
    "Total number of fallback activations",
    ["primary_model", "fallback_model"],
)

# Circuit breaker metrics
CIRCUIT_STATE = Gauge(
    "nvidia_bridge_circuit_state",
    "Circuit breaker state (0=closed, 1=open, 2=half_open)",
    ["model_id"],
)

# Cache metrics
CACHE_HITS = Counter("nvidia_bridge_cache_hits_total", "Total cache hits")
CACHE_MISSES = Counter("nvidia_bridge_cache_misses_total", "Total cache misses")
CACHE_SIZE = Gauge("nvidia_bridge_cache_size", "Current cache size")

# Auth metrics
AUTH_SUCCESS = Counter("nvidia_bridge_auth_success_total", "Successful authentications")
AUTH_FAILURE = Counter("nvidia_bridge_auth_failure_total", "Failed authentications")

# Rate limit metrics
RATE_LIMIT_HITS = Counter(
    "nvidia_bridge_rate_limit_hits_total",
    "Total rate limit hits",
    ["client_ip"],
)

# Service info
SERVICE_INFO = Info("nvidia_bridge", "Nvidia Model Bridge service information")


def record_request(endpoint: str, method: str, status_code: int) -> None:
    """Record an API request."""
    REQUEST_COUNT.labels(
        endpoint=endpoint, method=method, status_code=str(status_code)
    ).inc()


def record_model_request(
    model_id: str, task_type: str, success: bool, latency: float | None = None
) -> None:
    """Record a model inference request."""
    MODEL_REQUEST_COUNT.labels(
        model_id=model_id, task_type=task_type, success=str(success)
    ).inc()
    if latency is not None:
        MODEL_LATENCY.labels(model_id=model_id).observe(latency)


def record_fallback(primary_model: str, fallback_model: str) -> None:
    """Record a fallback activation."""
    FALLBACK_COUNT.labels(
        primary_model=primary_model, fallback_model=fallback_model
    ).inc()


def record_cache_hit() -> None:
    CACHE_HITS.inc()


def record_cache_miss() -> None:
    CACHE_MISSES.inc()


def update_cache_size(size: int) -> None:
    CACHE_SIZE.set(size)


def record_auth_success() -> None:
    AUTH_SUCCESS.inc()


def record_auth_failure() -> None:
    AUTH_FAILURE.inc()


def record_rate_limit(client_ip: str) -> None:
    RATE_LIMIT_HITS.labels(client_ip=client_ip).inc()


def update_circuit_state(model_id: str, state: str) -> None:
    """Update circuit breaker state gauge."""
    state_map = {"closed": 0, "open": 1, "half_open": 2}
    CIRCUIT_STATE.labels(model_id=model_id).set(state_map.get(state, -1))


def set_service_info(version: str, base_url: str) -> None:
    """Set service metadata."""
    SERVICE_INFO.info({"version": version, "nvidia_base_url": base_url})


def get_metrics_summary() -> dict[str, Any]:
    """Return a summary of current metrics for the dashboard endpoint."""
    return {
        "cache_hits": CACHE_HITS._value.get(),
        "cache_misses": CACHE_MISSES._value.get(),
        "cache_size": CACHE_SIZE._value.get(),
        "auth_successes": AUTH_SUCCESS._value.get(),
        "auth_failures": AUTH_FAILURE._value.get(),
    }
