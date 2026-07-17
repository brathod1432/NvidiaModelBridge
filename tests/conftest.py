"""Shared test fixtures for Nvidia Model Bridge."""

from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch

# Ensure test environment
os.environ.setdefault("NVIDIA_API_KEY", "nvapi-test-key-for-unit-tests")
os.environ.setdefault("NVIDIA_BRIDGE_API_KEY", "")
os.environ.setdefault("NVIDIA_BRIDGE_AUTH_ENABLED", "false")


@pytest.fixture
def mock_nvidia_settings():
    """Create mock NvidiaSettings for testing."""
    from app.config import NvidiaSettings
    return NvidiaSettings(
        api_key="nvapi-test-key-for-unit-tests",
        api_key_source="test",
        base_url="https://integrate.api.nvidia.com/v1",
        timeout_seconds=30,
    )


@pytest.fixture
def mock_bridge_settings():
    """Create mock BridgeSettings for testing."""
    from app.config import BridgeSettings
    return BridgeSettings(
        api_key="nvapi-test-key-for-unit-tests",
        api_key_source="test",
        base_url="https://integrate.api.nvidia.com/v1",
        timeout_seconds=30,
        enable_cache=True,
        cache_maxsize=100,
        cache_ttl=60,
        circuit_breaker_threshold=3,
        circuit_breaker_recovery=10.0,
        enable_metrics=False,
        enable_request_logging=False,
        enable_streaming=True,
        enable_system_prompts=True,
        enable_analytics=True,
        enable_job_queue=True,
    )


@pytest.fixture
def test_client():
    """Create a FastAPI test client."""
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)
