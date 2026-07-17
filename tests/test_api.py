"""Tests for API endpoints."""

from __future__ import annotations

import os

# Ensure auth is disabled for tests
os.environ["NVIDIA_BRIDGE_API_KEY"] = ""
os.environ["NVIDIA_BRIDGE_AUTH_ENABLED"] = "false"
os.environ["NVIDIA_BRIDGE_ENABLE_METRICS"] = "false"

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Nvidia Model Bridge"
        assert data["version"] == "2.0.0"
        assert "api_key_found" in data
        assert "auth_enabled" in data
        assert "cache_enabled" in data
        assert "uptime_seconds" in data

    def test_health_has_security_headers(self, client):
        response = client.get("/health")
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"
        assert "X-Request-ID" in response.headers

    def test_health_has_request_id(self, client):
        response = client.get("/health")
        assert response.headers.get("X-Request-ID") is not None
        assert len(response.headers["X-Request-ID"]) > 0


class TestModelEndpoints:
    def test_models_priority(self, client):
        response = client.get("/models/priority")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data

    def test_models_recommended(self, client):
        response = client.get("/models/recommended")
        assert response.status_code == 200
        data = response.json()
        assert "general" in data
        assert "coding" in data

    def test_models_avoid(self, client):
        response = client.get("/models/avoid")
        assert response.status_code == 200
        data = response.json()
        assert "avoid" in data


class TestAskEndpoint:
    def test_ask_validation_empty_prompt(self, client):
        response = client.post("/ask", json={"prompt": ""})
        assert response.status_code == 422

    def test_ask_validation_bad_temperature(self, client):
        response = client.post(
            "/ask", json={"prompt": "hello", "temperature": 5.0}
        )
        assert response.status_code == 422


class TestChatCompletionsEndpoint:
    def test_missing_model(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 400
        assert "model is required" in response.json()["error"]["message"]


class TestObservabilityEndpoints:
    def test_analytics_dashboard(self, client):
        response = client.get("/analytics/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data

    def test_cache_stats(self, client):
        response = client.get("/cache/stats")
        assert response.status_code == 200
        data = response.json()
        assert "hits" in data
        assert "misses" in data

    def test_circuit_breakers(self, client):
        response = client.get("/circuit-breakers")
        assert response.status_code == 200
        data = response.json()
        assert "breakers" in data
        assert "open_circuits" in data

    def test_templates(self, client):
        response = client.get("/templates")
        assert response.status_code == 200
        data = response.json()
        assert "templates" in data
        assert "general" in data["templates"]


class TestJobEndpoints:
    def test_list_jobs(self, client):
        response = client.get("/jobs")
        assert response.status_code == 200
        assert "jobs" in response.json()

    def test_job_not_found(self, client):
        response = client.get("/jobs/nonexistent-id")
        assert response.status_code == 404
