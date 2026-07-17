"""Tests for input validation module."""

from __future__ import annotations

from app.input_validation import (
    validate_max_tokens,
    validate_messages,
    validate_model_id,
    validate_prompt,
    validate_request,
    validate_temperature,
    sanitize_log_string,
)


class TestValidatePrompt:
    def test_valid_prompt(self):
        result = validate_prompt("Hello, world!")
        assert result.valid is True

    def test_empty_prompt(self):
        result = validate_prompt("")
        assert result.valid is False

    def test_whitespace_only(self):
        result = validate_prompt("   ")
        assert result.valid is False

    def test_too_long(self):
        result = validate_prompt("x" * 200_000)
        assert result.valid is False


class TestValidateMessages:
    def test_valid_messages(self):
        result = validate_messages([
            {"role": "user", "content": "hello"},
        ])
        assert result.valid is True

    def test_empty_messages(self):
        result = validate_messages([])
        assert result.valid is False

    def test_invalid_role(self):
        result = validate_messages([
            {"role": "invalid_role", "content": "hello"},
        ])
        assert result.valid is False

    def test_too_many_messages(self):
        msgs = [{"role": "user", "content": "hi"}] * 101
        result = validate_messages(msgs)
        assert result.valid is False

    def test_valid_roles(self):
        result = validate_messages([
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Thanks"},
        ])
        assert result.valid is True


class TestValidateModelId:
    def test_valid(self):
        result = validate_model_id("nvidia/nemotron-mini-4b")
        assert result.valid is True

    def test_empty(self):
        result = validate_model_id("")
        assert result.valid is False

    def test_invalid_chars(self):
        result = validate_model_id("model with spaces")
        assert result.valid is False


class TestValidateTemperature:
    def test_none_valid(self):
        assert validate_temperature(None).valid is True

    def test_valid_range(self):
        assert validate_temperature(0.5).valid is True

    def test_too_high(self):
        assert validate_temperature(3.0).valid is False

    def test_negative(self):
        assert validate_temperature(-1.0).valid is False


class TestValidateMaxTokens:
    def test_none_valid(self):
        assert validate_max_tokens(None).valid is True

    def test_valid(self):
        assert validate_max_tokens(1024).valid is True

    def test_zero(self):
        assert validate_max_tokens(0).valid is False


class TestSanitizeLogString:
    def test_normal_string(self):
        assert sanitize_log_string("hello") == "hello"

    def test_truncation(self):
        result = sanitize_log_string("x" * 600, max_length=500)
        assert len(result) < 600
        assert "truncated" in result

    def test_control_chars_removed(self):
        result = sanitize_log_string("hello\x00world")
        assert "\x00" not in result
