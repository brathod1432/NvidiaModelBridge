"""Circuit breaker pattern for model routing resilience."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any

try:
    from app.metrics import update_circuit_state
    _HAS_METRICS = True
except ImportError:
    _HAS_METRICS = False


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitStats:
    """Track failure/success stats for a single model."""
    failures: int = 0
    successes: int = 0
    consecutive_failures: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    total_requests: int = 0
    state_changed_at: float = field(default_factory=time.time)


class ModelCircuitBreaker:
    """Circuit breaker for a single model endpoint."""

    def __init__(
        self,
        model_id: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ) -> None:
        self.model_id = model_id
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._half_open_calls = 0
        self._lock = Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._stats.last_failure_time and (
                    time.time() - self._stats.last_failure_time > self.recovery_timeout
                ):
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    self._stats.state_changed_at = time.time()
                    if _HAS_METRICS:
                        update_circuit_state(self.model_id, "half_open")
            return self._state

    def is_available(self) -> bool:
        """Check if requests should be allowed through."""
        current_state = self.state
        if current_state == CircuitState.CLOSED:
            return True
        if current_state == CircuitState.HALF_OPEN:
            with self._lock:
                return self._half_open_calls < self.half_open_max_calls
        return False

    def record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            self._stats.successes += 1
            self._stats.consecutive_failures = 0
            self._stats.last_success_time = time.time()
            self._stats.total_requests += 1
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._stats.state_changed_at = time.time()
                if _HAS_METRICS:
                    update_circuit_state(self.model_id, self._state.value)

    def record_failure(self) -> None:
        """Record a failed request."""
        with self._lock:
            self._stats.failures += 1
            self._stats.consecutive_failures += 1
            self._stats.last_failure_time = time.time()
            self._stats.total_requests += 1
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._stats.state_changed_at = time.time()
                if _HAS_METRICS:
                    update_circuit_state(self.model_id, self._state.value)
            elif self._stats.consecutive_failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._stats.state_changed_at = time.time()
                if _HAS_METRICS:
                    update_circuit_state(self.model_id, self._state.value)

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._stats = CircuitStats()
            self._half_open_calls = 0
            if _HAS_METRICS:
                update_circuit_state(self.model_id, "closed")

    def get_status(self) -> dict[str, Any]:
        """Return current circuit breaker status."""
        current_state = self.state
        with self._lock:
            return {
                "model_id": self.model_id,
                "state": current_state.value,
                "failures": self._stats.failures,
                "successes": self._stats.successes,
                "consecutive_failures": self._stats.consecutive_failures,
                "total_requests": self._stats.total_requests,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
                "last_failure_time": self._stats.last_failure_time,
                "last_success_time": self._stats.last_success_time,
            }


class CircuitBreakerRegistry:
    """Registry of circuit breakers for all models."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._breakers: dict[str, ModelCircuitBreaker] = {}
        self._lock = Lock()

    def get_breaker(self, model_id: str) -> ModelCircuitBreaker:
        """Get or create a circuit breaker for a model."""
        with self._lock:
            if model_id not in self._breakers:
                self._breakers[model_id] = ModelCircuitBreaker(
                    model_id=model_id,
                    failure_threshold=self.failure_threshold,
                    recovery_timeout=self.recovery_timeout,
                )
            return self._breakers[model_id]

    def is_model_available(self, model_id: str) -> bool:
        """Check if a model is available (circuit not open)."""
        return self.get_breaker(model_id).is_available()

    def record_success(self, model_id: str) -> None:
        self.get_breaker(model_id).record_success()

    def record_failure(self, model_id: str) -> None:
        self.get_breaker(model_id).record_failure()

    def get_all_statuses(self) -> list[dict[str, Any]]:
        """Return status of all tracked circuit breakers."""
        with self._lock:
            return [breaker.get_status() for breaker in self._breakers.values()]

    def get_open_circuits(self) -> list[str]:
        """Return model IDs with open circuits."""
        with self._lock:
            return [
                model_id
                for model_id, breaker in self._breakers.items()
                if breaker.state == CircuitState.OPEN
            ]

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()

    def reset_model(self, model_id: str) -> bool:
        """Reset a specific model's circuit breaker."""
        with self._lock:
            if model_id in self._breakers:
                self._breakers[model_id].reset()
                return True
            return False


# Module-level singleton
_registry: CircuitBreakerRegistry | None = None


def get_circuit_breaker_registry(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
) -> CircuitBreakerRegistry:
    """Get or create the module-level circuit breaker registry."""
    global _registry
    if _registry is None:
        _registry = CircuitBreakerRegistry(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )
    return _registry
