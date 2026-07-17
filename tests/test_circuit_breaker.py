"""Tests for circuit breaker module."""

from __future__ import annotations

import time
from app.circuit_breaker import (
    CircuitBreakerRegistry,
    CircuitState,
    ModelCircuitBreaker,
)


class TestModelCircuitBreaker:
    def test_starts_closed(self):
        cb = ModelCircuitBreaker("test/model", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED
        assert cb.is_available() is True

    def test_opens_after_threshold(self):
        cb = ModelCircuitBreaker("test/model", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.is_available() is False

    def test_success_resets_consecutive_failures(self):
        cb = ModelCircuitBreaker("test/model", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_after_recovery(self):
        cb = ModelCircuitBreaker("test/model", failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.is_available() is True

    def test_half_open_success_closes(self):
        cb = ModelCircuitBreaker("test/model", failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = ModelCircuitBreaker("test/model", failure_threshold=2, recovery_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_reset(self):
        cb = ModelCircuitBreaker("test/model", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_get_status(self):
        cb = ModelCircuitBreaker("test/model")
        status = cb.get_status()
        assert status["model_id"] == "test/model"
        assert status["state"] == "closed"
        assert "failures" in status
        assert "successes" in status


class TestCircuitBreakerRegistry:
    def test_get_or_create(self):
        registry = CircuitBreakerRegistry(failure_threshold=3)
        breaker = registry.get_breaker("model/a")
        assert breaker.model_id == "model/a"

    def test_is_model_available(self):
        registry = CircuitBreakerRegistry(failure_threshold=2)
        assert registry.is_model_available("model/a") is True
        registry.record_failure("model/a")
        registry.record_failure("model/a")
        assert registry.is_model_available("model/a") is False

    def test_get_open_circuits(self):
        registry = CircuitBreakerRegistry(failure_threshold=1)
        registry.record_failure("model/a")
        assert "model/a" in registry.get_open_circuits()

    def test_reset_all(self):
        registry = CircuitBreakerRegistry(failure_threshold=1)
        registry.record_failure("model/a")
        registry.record_failure("model/b")
        registry.reset_all()
        assert registry.get_open_circuits() == []

    def test_reset_model(self):
        registry = CircuitBreakerRegistry(failure_threshold=1)
        registry.record_failure("model/a")
        registry.reset_model("model/a")
        assert registry.is_model_available("model/a") is True
