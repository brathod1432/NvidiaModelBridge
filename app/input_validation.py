"""Input validation and content safety for Nvidia Model Bridge."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


MAX_PROMPT_LENGTH = 100_000
MAX_MESSAGE_COUNT = 100
MAX_SINGLE_MESSAGE_LENGTH = 50_000
MAX_MODEL_ID_LENGTH = 200
BLOCKED_PATTERNS: list[str] = []


@dataclass(frozen=True)
class ValidationResult:
    """Result of input validation."""
    valid: bool
    error: str | None = None
    warnings: list[str] | None = None


def validate_prompt(prompt: str) -> ValidationResult:
    """Validate a single prompt string."""
    if not prompt or not prompt.strip():
        return ValidationResult(valid=False, error="Prompt cannot be empty.")

    if len(prompt) > MAX_PROMPT_LENGTH:
        return ValidationResult(
            valid=False,
            error=f"Prompt exceeds maximum length of {MAX_PROMPT_LENGTH:,} characters.",
        )

    warnings = []
    if len(prompt) > MAX_PROMPT_LENGTH * 0.8:
        warnings.append("Prompt is approaching the maximum length limit.")

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, prompt, re.IGNORECASE):
            return ValidationResult(
                valid=False, error="Prompt contains blocked content patterns."
            )

    return ValidationResult(valid=True, warnings=warnings if warnings else None)


def validate_messages(messages: list[dict[str, str]]) -> ValidationResult:
    """Validate a list of chat messages."""
    if not messages:
        return ValidationResult(valid=False, error="Messages list cannot be empty.")

    if len(messages) > MAX_MESSAGE_COUNT:
        return ValidationResult(
            valid=False,
            error=f"Too many messages. Maximum is {MAX_MESSAGE_COUNT}.",
        )

    valid_roles = {"system", "user", "assistant", "tool"}
    warnings = []

    for i, message in enumerate(messages):
        role = message.get("role", "")
        content = message.get("content", "")

        if role not in valid_roles:
            return ValidationResult(
                valid=False,
                error=f"Message {i}: invalid role '{role}'. Valid roles: {', '.join(sorted(valid_roles))}.",
            )

        if not content and role != "assistant":
            return ValidationResult(
                valid=False,
                error=f"Message {i}: content cannot be empty for role '{role}'.",
            )

        if len(content) > MAX_SINGLE_MESSAGE_LENGTH:
            return ValidationResult(
                valid=False,
                error=f"Message {i}: content exceeds maximum length of {MAX_SINGLE_MESSAGE_LENGTH:,} characters.",
            )

    total_length = sum(len(m.get("content", "")) for m in messages)
    if total_length > MAX_PROMPT_LENGTH:
        return ValidationResult(
            valid=False,
            error=f"Total message content exceeds maximum of {MAX_PROMPT_LENGTH:,} characters.",
        )

    if total_length > MAX_PROMPT_LENGTH * 0.8:
        warnings.append("Total message content is approaching the maximum length limit.")

    return ValidationResult(valid=True, warnings=warnings if warnings else None)


def validate_model_id(model_id: str) -> ValidationResult:
    """Validate a model ID string."""
    if not model_id or not model_id.strip():
        return ValidationResult(valid=False, error="Model ID cannot be empty.")

    if len(model_id) > MAX_MODEL_ID_LENGTH:
        return ValidationResult(
            valid=False,
            error=f"Model ID exceeds maximum length of {MAX_MODEL_ID_LENGTH} characters.",
        )

    if not re.match(r'^[\w\-\.\/]+$', model_id):
        return ValidationResult(
            valid=False,
            error="Model ID contains invalid characters. Use alphanumeric, hyphens, dots, and slashes only.",
        )

    return ValidationResult(valid=True)


def validate_temperature(temperature: float | None) -> ValidationResult:
    """Validate temperature parameter."""
    if temperature is None:
        return ValidationResult(valid=True)
    if not isinstance(temperature, (int, float)):
        return ValidationResult(valid=False, error="Temperature must be a number.")
    if temperature < 0 or temperature > 2.0:
        return ValidationResult(
            valid=False, error="Temperature must be between 0 and 2.0."
        )
    return ValidationResult(valid=True)


def validate_max_tokens(max_tokens: int | None) -> ValidationResult:
    """Validate max_tokens parameter."""
    if max_tokens is None:
        return ValidationResult(valid=True)
    if not isinstance(max_tokens, int):
        return ValidationResult(valid=False, error="max_tokens must be an integer.")
    if max_tokens < 1 or max_tokens > 131072:
        return ValidationResult(
            valid=False, error="max_tokens must be between 1 and 131072."
        )
    return ValidationResult(valid=True)


def validate_request(
    prompt: str | None = None,
    messages: list[dict[str, str]] | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ValidationResult:
    """Validate a complete request with all parameters."""
    if prompt is not None:
        result = validate_prompt(prompt)
        if not result.valid:
            return result

    if messages is not None:
        result = validate_messages(messages)
        if not result.valid:
            return result

    if model is not None:
        result = validate_model_id(model)
        if not result.valid:
            return result

    result = validate_temperature(temperature)
    if not result.valid:
        return result

    result = validate_max_tokens(max_tokens)
    if not result.valid:
        return result

    return ValidationResult(valid=True)


def sanitize_log_string(text: str, max_length: int = 500) -> str:
    """Sanitize a string for safe logging (remove control chars, truncate)."""
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    if len(cleaned) > max_length:
        return cleaned[:max_length] + "...[truncated]"
    return cleaned
