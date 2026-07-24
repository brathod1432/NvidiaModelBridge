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
    system_prompt: str | None = None
    conversation_history: list[ChatMessage] | None = None


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
    request_id: str | None = None
    cache_hit: bool = False


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
    version: str = "2.0.0"
    api_key_found: bool
    api_key_masked: str
    base_url: str
    auth_enabled: bool = False
    cache_enabled: bool = False
    streaming_enabled: bool = False
    metrics_enabled: bool = False
    uptime_seconds: float | None = None


class DeepHealthResponse(BaseModel):
    status: str
    service: str
    version: str = "2.0.0"
    api_key_found: bool
    base_url: str
    nvidia_reachable: bool = False
    nvidia_status_code: int | None = None
    nvidia_latency_seconds: float | None = None
    nvidia_models_found: int = 0
    cache_stats: dict[str, Any] = Field(default_factory=dict)
    circuit_breaker_summary: dict[str, Any] = Field(default_factory=dict)
    auth_enabled: bool = False
    rate_limit_stats: dict[str, Any] = Field(default_factory=dict)


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


class BatchRequest(BaseModel):
    """Batch inference request."""
    requests: list[AskRequest]
    max_concurrency: int = Field(default=5, ge=1, le=20, description="Max parallel requests")


class BatchResponseItem(BaseModel):
    index: int
    success: bool
    response: dict[str, Any] | None = None
    error: str | None = None


class BatchResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: list[BatchResponseItem]
    total_latency_seconds: float


class JobSubmitRequest(BaseModel):
    prompt: str
    task_type: str | None = None
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    system_prompt: str | None = None


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    id: str
    status: str
    created_at: float
    started_at: float | None = None
    completed_at: float | None = None
    elapsed_seconds: float | None = None
    progress: str = ""
    has_result: bool = False
    error: str | None = None
    result: dict[str, Any] | None = None
