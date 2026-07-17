"""FastAPI application for Nvidia Model Bridge."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api_schemas import (
    AskRequest,
    AskResponse,
    ChatCompletionRequest,
    ErrorEnvelope,
    HealthResponse,
    ModelListResponse,
    ModelResponse,
)
from app.config import BridgeSettings
from app.coordinator import Coordinator, UNSUPPORTED_STREAMING_ERROR
from app.model_registry import (
    LATEST_AUDIT_SOURCE,
    get_avoid_models,
    get_partial_models,
    get_priority_models,
    get_retest_models,
    model_to_dict,
)


SERVICE_NAME = "Nvidia Model Bridge"

settings = BridgeSettings.load()
coordinator = Coordinator(settings=settings)
app = FastAPI(title=SERVICE_NAME, version="1.0.0")


def _as_model_response(model) -> ModelResponse:
    return ModelResponse.model_validate(model_to_dict(model))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    status = "ok" if settings.api_key else "degraded"
    return HealthResponse(
        status=status,
        service=SERVICE_NAME,
        api_key_found=bool(settings.api_key),
        api_key_masked=settings.masked_api_key,
        base_url=settings.base_url,
    )


@app.get("/models/priority", response_model=ModelListResponse)
def models_priority() -> ModelListResponse:
    return ModelListResponse(models=[_as_model_response(model) for model in get_priority_models()])


@app.get("/models/recommended")
def models_recommended() -> dict[str, str]:
    return {
        "source": LATEST_AUDIT_SOURCE,
        "general": "qwen/qwen3-next-80b-a3b-instruct",
        "coding": "qwen/qwen3-next-80b-a3b-instruct",
        "reasoning": "qwen/qwen3-next-80b-a3b-instruct",
        "nvidia_reasoning": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "fast": "nvidia/nemotron-mini-4b-instruct",
        "fallback": "mistralai/mistral-nemotron",
        "lightweight": "openai/gpt-oss-20b",
        "deepseek": "deepseek-ai/deepseek-v4-pro",
    }


@app.get("/models/avoid")
def models_avoid() -> dict[str, Any]:
    return {
        "avoid": [_as_model_response(model) for model in get_avoid_models()],
        "retest": [model_to_dict(model) for model in get_retest_models()],
        "partial": [model_to_dict(model) for model in get_partial_models()],
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    outcome = coordinator.ask(
        prompt=request.prompt,
        task_type=request.task_type,
        model=request.model,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stream=request.stream,
    )
    return AskResponse(**outcome.to_api_response())


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": UNSUPPORTED_STREAMING_ERROR,
                    "type": "unsupported_feature",
                }
            },
        )

    if not request.model:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "model is required for /v1/chat/completions. Use /ask for coordinator routing.",
                    "type": "invalid_request_error",
                }
            },
        )

    messages = [message.model_dump() for message in request.messages]
    status_code, payload = coordinator.client.forward_chat_completion_raw(
        model_id=request.model,
        messages=messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        extra_body=request.extra_body or None,
    )
    return JSONResponse(status_code=status_code, content=payload)
