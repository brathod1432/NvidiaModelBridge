"""Tests for configuration module."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from app.config import (
    BridgeSettings,
    NvidiaSettings,
    _clean_api_key,
    _parse_bool,
    _parse_int,
    mask_api_key,
    get_key_shape_warning,
)


class TestParseBool:
    def test_none_returns_default(self):
        assert _parse_bool(None, True) is True
        assert _parse_bool(None, False) is False

    def test_empty_returns_default(self):
        assert _parse_bool("", True) is True

    def test_truthy_values(self):
        for val in ["1", "true", "yes", "y", "on", "TRUE", "Yes"]:
            assert _parse_bool(val, False) is True

    def test_falsy_values(self):
        for val in ["0", "false", "no", "n", "off"]:
            assert _parse_bool(val, True) is False


class TestParseInt:
    def test_none_returns_default(self):
        assert _parse_int(None, 42) == 42

    def test_valid_int(self):
        assert _parse_int("10", 0) == 10

    def test_invalid_returns_default(self):
        assert _parse_int("abc", 5) == 5


class TestMaskApiKey:
    def test_none(self):
        assert mask_api_key(None) == ""

    def test_empty(self):
        assert mask_api_key("") == ""

    def test_short_key(self):
        assert mask_api_key("abc") == "****"

    def test_nvapi_prefix(self):
        result = mask_api_key("nvapi-abcdefghijklmnop")
        assert result.startswith("nvapi-****")
        assert result.endswith("mnop")

    def test_regular_key(self):
        result = mask_api_key("abcdefghijklmnop")
        assert result.startswith("****")
        assert result.endswith("mnop")


class TestCleanApiKey:
    def test_none(self):
        assert _clean_api_key(None) is None

    def test_placeholder(self):
        assert _clean_api_key("your_api_key_here") is None
        assert _clean_api_key("changeme") is None
        assert _clean_api_key("none") is None

    def test_valid_key(self):
        assert _clean_api_key("nvapi-realkey123") == "nvapi-realkey123"

    def test_strips_whitespace(self):
        assert _clean_api_key("  nvapi-key  ") == "nvapi-key"


class TestKeyShapeWarning:
    def test_no_warning_for_nvapi(self):
        assert get_key_shape_warning("nvapi-test") == ""

    def test_warning_for_non_nvapi(self):
        assert "nvapi-" in get_key_shape_warning("sk-test")

    def test_no_warning_for_none(self):
        assert get_key_shape_warning(None) == ""


class TestBridgeSettings:
    def test_has_security_fields(self):
        settings = BridgeSettings(
            api_key="test", api_key_source="test"
        )
        assert hasattr(settings, "cors_origins")
        assert hasattr(settings, "auth_enabled")
        assert hasattr(settings, "rate_limit_per_minute")
        assert hasattr(settings, "enable_cache")
        assert hasattr(settings, "circuit_breaker_threshold")
        assert hasattr(settings, "enable_streaming")
        assert hasattr(settings, "enable_system_prompts")
        assert hasattr(settings, "log_level")

    def test_masked_api_key_property(self):
        settings = BridgeSettings(
            api_key="nvapi-test1234567890", api_key_source="test"
        )
        assert "****" in settings.masked_api_key

    def test_api_key_found_property(self):
        settings = BridgeSettings(api_key="key", api_key_source="test")
        assert settings.api_key_found is True

        settings_empty = BridgeSettings(api_key="", api_key_source="test")
        assert settings_empty.api_key_found is False
