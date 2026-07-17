"""FastAPI request and response schemas for Nvidia Model Bridge."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    prompt: str
    task_type: str | None = None
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False


class AskResponse(BaseModel):
    success: bool
    model: str | None = None
    task_type: str
    selected_by: str
    selection_reason: str
    fallback_used: bool
    latency_seconds: float | None = None
    content: str | None = None
    reasoning: str | None = None
    error: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    extra_body: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    service: str
    api_key_found: bool
    api_key_masked: str
    base_url: str


class ModelResponse(BaseModel):
    id: str
    display_name: str
    provider: str
    priority: bool
    status: str
    recommended_for: list[str]
    avg_latency: float | None = None
    benchmark_pass_count: int
    benchmark_total_count: int
    notes: str
    default_temperature: float
    default_top_p: float
    default_max_tokens: int
    extra_body: dict[str, Any] = Field(default_factory=dict)
    category: str
    recommended_use: str
    supports_streaming: bool
    uses_reasoning: bool | str
    source_reference: str
    rank: int | None = None


class ModelListResponse(BaseModel):
    models: list[ModelResponse]


class ErrorEnvelope(BaseModel):
    error: dict[str, Any]
