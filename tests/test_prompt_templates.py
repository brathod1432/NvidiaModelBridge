"""Tests for prompt templates module."""

from __future__ import annotations

from app.prompt_templates import (
    SYSTEM_PROMPTS,
    build_messages_with_system_prompt,
    get_system_prompt,
    get_template_info,
    list_available_templates,
)


class TestGetSystemPrompt:
    def test_general(self):
        prompt = get_system_prompt("general")
        assert prompt is not None
        assert "helpful" in prompt.lower()

    def test_coding(self):
        prompt = get_system_prompt("coding")
        assert prompt is not None
        assert "code" in prompt.lower()

    def test_unknown(self):
        assert get_system_prompt("nonexistent") is None


class TestBuildMessages:
    def test_basic(self):
        messages = build_messages_with_system_prompt("hello", "general")
        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "hello"

    def test_with_history(self):
        history = [
            {"role": "user", "content": "prev question"},
            {"role": "assistant", "content": "prev answer"},
        ]
        messages = build_messages_with_system_prompt(
            "follow up", "general", conversation_history=history
        )
        assert len(messages) == 4  # system + 2 history + user
        assert messages[-1]["content"] == "follow up"

    def test_system_prompt_override(self):
        messages = build_messages_with_system_prompt(
            "hello", "general", system_prompt_override="Custom system"
        )
        assert messages[0]["content"] == "Custom system"

    def test_no_system_prompt_for_unknown_type(self):
        messages = build_messages_with_system_prompt("hello", "nonexistent")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"


class TestListTemplates:
    def test_returns_dict(self):
        templates = list_available_templates()
        assert isinstance(templates, dict)
        assert "general" in templates
        assert "coding" in templates

    def test_template_info(self):
        info = get_template_info()
        assert len(info) > 0
        assert "task_type" in info[0]
        assert "system_prompt" in info[0]
