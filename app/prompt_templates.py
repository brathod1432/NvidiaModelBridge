"""System prompt templates per task type for Nvidia Model Bridge."""

from __future__ import annotations

from typing import Any


SYSTEM_PROMPTS: dict[str, str] = {
    "general": (
        "You are a helpful, accurate, and concise AI assistant. "
        "Provide clear and well-structured responses."
    ),
    "coding": (
        "You are an expert software engineer. Write clean, efficient, "
        "well-documented code. Include error handling and follow best practices. "
        "When providing code, use appropriate language-specific conventions."
    ),
    "reasoning": (
        "You are a logical reasoning expert. Break down complex problems "
        "step by step. Show your work clearly and arrive at well-justified "
        "conclusions. Consider edge cases and alternative approaches."
    ),
    "nvidia_reasoning": (
        "You are an advanced reasoning AI. Use deep analytical thinking "
        "to solve complex problems. Provide detailed step-by-step reasoning "
        "and clearly state your final answer."
    ),
    "json": (
        "You are a structured data assistant. Always respond with valid JSON. "
        "Follow the requested schema exactly. Do not include any text outside "
        "the JSON structure."
    ),
    "fast": (
        "You are a concise assistant. Provide brief, direct answers. "
        "Avoid unnecessary elaboration."
    ),
}


def get_system_prompt(task_type: str) -> str | None:
    """Get the system prompt for a task type, or None if not configured."""
    return SYSTEM_PROMPTS.get(task_type.lower())


def build_messages_with_system_prompt(
    prompt: str,
    task_type: str,
    system_prompt_override: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build a complete messages list with system prompt and conversation history.
    
    Args:
        prompt: The user's current prompt.
        task_type: The task type for selecting system prompt.
        system_prompt_override: Optional override for the default system prompt.
        conversation_history: Optional list of previous messages.
    
    Returns:
        Complete messages list ready for the API.
    """
    messages: list[dict[str, str]] = []

    system_prompt = system_prompt_override or get_system_prompt(task_type)
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": prompt})
    return messages


def list_available_templates() -> dict[str, str]:
    """Return all available system prompt templates."""
    return dict(SYSTEM_PROMPTS)


def get_template_info() -> list[dict[str, Any]]:
    """Return detailed info about all templates."""
    return [
        {
            "task_type": task_type,
            "system_prompt": prompt,
            "prompt_length": len(prompt),
        }
        for task_type, prompt in SYSTEM_PROMPTS.items()
    ]
