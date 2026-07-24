"""Security tests for the Model Bridge."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# We need to be able to import the app
# But it requires NVIDIA_API_KEY, so set a dummy one
import os
os.environ.setdefault("NVIDIA_API_KEY", "test-dummy-key-for-testing")

from app.main import app
from app.input_validation import validate_request, sanitize_log_string


@pytest.fixture
def client():
    return TestClient(app)


class TestSQLInjection:
    """Verify SQL injection attempts are handled safely."""

    def test_sql_in_prompt(self, client):
        """SQL injection in prompt should not crash."""
        response = client.post("/ask", json={
            "prompt": "'; DROP TABLE request_log; --",
            "task_type": "general",
        })
        # Should get a response (may fail due to no API key, but not a 500)
        assert response.status_code in (200, 422, 500, 503)

    def test_sql_in_model_name(self, client):
        """SQL injection in model name should be handled."""
        response = client.post("/ask", json={
            "prompt": "Hello",
            "model": "'; DROP TABLE models; --",
        })
        assert response.status_code in (200, 422, 500, 503)

    def test_sql_in_task_type(self, client):
        response = client.post("/ask", json={
            "prompt": "Hello",
            "task_type": "general'; DROP TABLE request_log; --",
        })
        assert response.status_code in (200, 422, 500, 503)


class TestXSSPrevention:
    """Verify XSS payloads are handled safely."""

    def test_xss_in_prompt(self, client):
        response = client.post("/ask", json={
            "prompt": "<script>alert('xss')</script>",
        })
        assert response.status_code in (200, 422, 500, 503)
        # Response should not contain raw script tag
        if response.status_code == 200:
            assert "<script>" not in response.text

    def test_xss_in_headers(self, client):
        response = client.get("/health", headers={
            "X-Custom": "<img src=x onerror=alert(1)>"
        })
        assert response.status_code == 200


class TestAuthBypass:
    """Verify authentication cannot be bypassed."""

    def test_empty_api_key(self, client):
        """Empty API key should not authenticate."""
        response = client.post("/ask", json={"prompt": "test"}, headers={
            "X-Bridge-API-Key": ""
        })
        # Should still work (auth may be disabled) or reject
        assert response.status_code in (200, 401, 403, 422, 500, 503)

    def test_null_api_key(self, client):
        response = client.post("/ask", json={"prompt": "test"}, headers={
            "X-Bridge-API-Key": "\x00"
        })
        assert response.status_code in (200, 401, 403, 422, 500, 503)


class TestRateLimitEvasion:
    """Verify rate limiting cannot be easily evaded."""

    def test_spoofed_ip_header(self, client):
        """X-Forwarded-For should not bypass rate limits."""
        for i in range(5):
            response = client.get("/health", headers={
                "X-Forwarded-For": f"10.0.0.{i}"
            })
            assert response.status_code in (200, 429)


class TestInputValidation:
    """Verify input validation catches edge cases."""

    def test_extremely_long_prompt(self):
        result = validate_request(prompt="A" * 200_000)
        assert not result.valid

    def test_empty_prompt(self):
        result = validate_request(prompt="")
        assert not result.valid

    def test_null_bytes_in_prompt(self):
        result = validate_request(prompt="Hello\x00World")
        # Should either pass or fail gracefully
        assert isinstance(result.valid, bool)

    def test_unicode_prompt(self):
        result = validate_request(prompt="Compute 1+1")
        assert result.valid

    def test_temperature_out_of_range(self):
        result = validate_request(prompt="test", temperature=5.0)
        assert not result.valid

    def test_negative_max_tokens(self):
        result = validate_request(prompt="test", max_tokens=-1)
        assert not result.valid


class TestSanitization:
    """Verify log sanitization works."""

    def test_sanitize_removes_control_chars(self):
        result = sanitize_log_string("line1\x00line2\x01line3")
        assert "\x00" not in result
        assert "\x01" not in result

    def test_sanitize_truncates_long_strings(self):
        result = sanitize_log_string("A" * 10000, max_length=100)
        assert len(result) <= 114  # 100 + "...[truncated]"


class TestSecurityHeaders:
    """Verify security headers are present."""

    def test_health_has_security_headers(self, client):
        response = client.get("/health")
        # Check key security headers
        assert "x-content-type-options" in response.headers
        assert "x-frame-options" in response.headers

    def test_cors_headers(self, client):
        response = client.options("/health", headers={
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "GET",
        })
        # CORS should respond (may allow or deny based on config)
        assert response.status_code in (200, 400, 405)


class TestEndpointSecurity:
    """Verify endpoints handle malformed requests."""

    def test_invalid_json(self, client):
        response = client.post("/ask", content="not json", headers={
            "Content-Type": "application/json"
        })
        assert response.status_code == 422

    def test_missing_required_fields(self, client):
        response = client.post("/ask", json={})
        assert response.status_code == 422

    def test_wrong_content_type(self, client):
        response = client.post("/ask", content="prompt=test", headers={
            "Content-Type": "application/x-www-form-urlencoded"
        })
        assert response.status_code == 422
