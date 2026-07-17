"""FastAPI application for Nvidia Model Bridge."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.analytics import get_analytics_collector
from app.api_schemas import (
    AskRequest,
    AskResponse,
    ChatCompletionRequest,
    DeepHealthResponse,
    ErrorEnvelope,
    HealthResponse,
    JobStatusResponse,
    JobSubmitRequest,
    JobSubmitResponse,
    ModelListResponse,
    ModelResponse,
)
from app.cache import get_response_cache
from app.circuit_breaker import get_circuit_breaker_registry
from app.config import BridgeSettings
from app.coordinator import Coordinator, UNSUPPORTED_STREAMING_ERROR
from app.database import init_database, log_request
from app.input_validation import validate_request, ValidationResult
from app.job_queue import JobStatus, get_job_queue
from app.logging_config import get_logger, setup_logging
from app.metrics import set_service_info
from app.middleware.auth import is_auth_enabled, verify_api_key
from app.middleware.rate_limit import RateLimitMiddleware, RateLimiter
from app.middleware.request_id import RequestIDMiddleware, get_request_id
from app.middleware.security_headers import SecurityHeadersMiddleware
from app.model_registry import (
    LATEST_AUDIT_SOURCE,
    get_avoid_models,
    get_partial_models,
    get_priority_models,
    get_retest_models,
    model_to_dict,
)
from app.prompt_templates import get_template_info, list_available_templates
from app.streaming import create_sse_response


SERVICE_NAME = "Nvidia Model Bridge"
SERVICE_VERSION = "2.0.0"
_start_time = time.time()

settings = BridgeSettings.load()

# Setup structured logging
setup_logging(log_level=settings.log_level, json_output=settings.log_json)
logger = get_logger("nvidia_bridge")

coordinator = Coordinator(settings=settings)
app = FastAPI(
    title=SERVICE_NAME,
    version=SERVICE_VERSION,
    description="A FastAPI gateway for routing prompts to NVIDIA-hosted OpenAI-compatible models with security, caching, circuit breakers, and observability.",
)

# --- Middleware stack (order matters: last added = first executed) ---

# Security headers on all responses
app.add_middleware(SecurityHeadersMiddleware)

# Request ID for tracing
app.add_middleware(RequestIDMiddleware)

# Rate limiting
if settings.rate_limit_per_minute > 0:
    rate_limiter = RateLimiter(
        default_rate=settings.rate_limit_per_minute,
        default_window=settings.rate_limit_window,
    )
    app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
)

# Prometheus metrics
if settings.enable_metrics:
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
        Instrumentator().instrument(app).expose(app, endpoint="/metrics")
        set_service_info(version=SERVICE_VERSION, base_url=settings.base_url)
    except ImportError:
        logger.warning("prometheus-fastapi-instrumentator not available, metrics disabled")


# --- Startup ---

@app.on_event("startup")
async def startup_event():
    logger.info(
        "service_starting",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        base_url=settings.base_url,
        auth_enabled=is_auth_enabled(),
        cache_enabled=settings.enable_cache,
        streaming_enabled=settings.enable_streaming,
    )
    try:
        init_database()
        logger.info("database_initialized")
    except Exception as exc:
        logger.warning("database_init_failed", error=str(exc))


# --- Auth dependency ---

def _auth_dependency():
    """Return the auth dependency if enabled, otherwise a no-op."""
    if is_auth_enabled():
        return Depends(verify_api_key)
    return None


def _as_model_response(model) -> ModelResponse:
    return ModelResponse.model_validate(model_to_dict(model))


# --- Health Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    status = "ok" if settings.api_key else "degraded"
    return HealthResponse(
        status=status,
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        api_key_found=bool(settings.api_key),
        api_key_masked=settings.masked_api_key,
        base_url=settings.base_url,
        auth_enabled=is_auth_enabled(),
        cache_enabled=settings.enable_cache,
        streaming_enabled=settings.enable_streaming,
        metrics_enabled=settings.enable_metrics,
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@app.get("/health/deep", response_model=DeepHealthResponse)
async def health_deep() -> DeepHealthResponse:
    """Deep health check that verifies NVIDIA API connectivity."""
    loop = asyncio.get_event_loop()
    nvidia_result = await loop.run_in_executor(None, coordinator.client.list_models)

    cache = get_response_cache()
    cb_registry = get_circuit_breaker_registry()
    open_circuits = cb_registry.get_open_circuits()

    overall = "ok"
    if not settings.api_key:
        overall = "degraded"
    elif not nvidia_result.get("success"):
        overall = "degraded"
    elif open_circuits:
        overall = "warning"

    return DeepHealthResponse(
        status=overall,
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        api_key_found=bool(settings.api_key),
        base_url=settings.base_url,
        nvidia_reachable=nvidia_result.get("success", False),
        nvidia_status_code=nvidia_result.get("status_code"),
        nvidia_latency_seconds=nvidia_result.get("latency_seconds"),
        nvidia_models_found=nvidia_result.get("number_of_models_discovered", 0),
        cache_stats=cache.get_stats(),
        circuit_breaker_summary={
            "open_circuits": open_circuits,
            "total_tracked": len(cb_registry.get_all_statuses()),
        },
        auth_enabled=is_auth_enabled(),
    )


# --- Model Endpoints ---

@app.get("/models/priority", response_model=ModelListResponse)
async def models_priority() -> ModelListResponse:
    return ModelListResponse(models=[_as_model_response(model) for model in get_priority_models()])


@app.get("/models/recommended")
async def models_recommended() -> dict[str, str]:
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
async def models_avoid() -> dict[str, Any]:
    return {
        "avoid": [_as_model_response(model) for model in get_avoid_models()],
        "retest": [model_to_dict(model) for model in get_retest_models()],
        "partial": [model_to_dict(model) for model in get_partial_models()],
    }


# --- Core Inference Endpoints ---

@app.post("/ask", response_model=AskResponse)
async def ask(request_body: AskRequest, request: Request):
    request_id = getattr(request.state, "request_id", "")

    # Input validation
    validation = validate_request(
        prompt=request_body.prompt,
        model=request_body.model,
        temperature=request_body.temperature,
        max_tokens=request_body.max_tokens,
    )
    if not validation.valid:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "message": validation.error,
                    "type": "validation_error",
                }
            },
        )

    # Convert conversation history if present
    conversation_history = None
    if request_body.conversation_history:
        conversation_history = [
            {"role": msg.role, "content": msg.content}
            for msg in request_body.conversation_history
        ]

    loop = asyncio.get_event_loop()
    outcome = await loop.run_in_executor(
        None,
        lambda: coordinator.ask(
            prompt=request_body.prompt,
            task_type=request_body.task_type,
            model=request_body.model,
            temperature=request_body.temperature,
            top_p=request_body.top_p,
            max_tokens=request_body.max_tokens,
            stream=request_body.stream,
            system_prompt=request_body.system_prompt,
            conversation_history=conversation_history,
        ),
    )

    # Log to database
    if settings.enable_request_logging:
        try:
            log_request(
                request_id=request_id,
                endpoint="/ask",
                model_id=outcome.model or "",
                task_type=outcome.task_type,
                prompt_length=len(request_body.prompt),
                response_length=len(outcome.content or ""),
                latency_seconds=outcome.latency_seconds or 0,
                status_code=200 if outcome.success else 500,
                success=outcome.success,
                fallback_used=outcome.fallback_used,
                fallback_model=outcome.fallback_model or "",
                error_message=outcome.error or "",
                cache_hit=outcome.cache_hit,
            )
        except Exception:
            pass  # Don't fail the request due to logging errors

    response_data = outcome.to_api_response()
    response_data["request_id"] = request_id
    return AskResponse(**response_data)


@app.post("/v1/chat/completions")
async def chat_completions(request_body: ChatCompletionRequest, request: Request):
    request_id = getattr(request.state, "request_id", "")

    # Streaming support
    if request_body.stream:
        if not settings.enable_streaming:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Streaming is disabled on this instance.",
                        "type": "unsupported_feature",
                    }
                },
            )
        if not request_body.model:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "model is required for streaming. Use /ask for coordinator routing.",
                        "type": "invalid_request_error",
                    }
                },
            )

        # Validate messages
        messages_dicts = [msg.model_dump() for msg in request_body.messages]
        validation = validate_request(messages=messages_dicts, model=request_body.model)
        if not validation.valid:
            return JSONResponse(
                status_code=422,
                content={"error": {"message": validation.error, "type": "validation_error"}},
            )

        return await create_sse_response(
            base_url=settings.base_url,
            api_key=settings.api_key,
            model_id=request_body.model,
            messages=messages_dicts,
            temperature=request_body.temperature,
            top_p=request_body.top_p,
            max_tokens=request_body.max_tokens,
            extra_body=request_body.extra_body or None,
            timeout=settings.timeout_seconds,
        )

    if not request_body.model:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "model is required for /v1/chat/completions. Use /ask for coordinator routing.",
                    "type": "invalid_request_error",
                }
            },
        )

    messages = [message.model_dump() for message in request_body.messages]

    # Input validation
    validation = validate_request(messages=messages, model=request_body.model)
    if not validation.valid:
        return JSONResponse(
            status_code=422,
            content={"error": {"message": validation.error, "type": "validation_error"}},
        )

    loop = asyncio.get_event_loop()
    status_code, payload = await loop.run_in_executor(
        None,
        lambda: coordinator.client.forward_chat_completion_raw(
            model_id=request_body.model,
            messages=messages,
            temperature=request_body.temperature,
            top_p=request_body.top_p,
            max_tokens=request_body.max_tokens,
            extra_body=request_body.extra_body or None,
        ),
    )

    # Log to database
    if settings.enable_request_logging:
        try:
            log_request(
                request_id=request_id,
                endpoint="/v1/chat/completions",
                model_id=request_body.model or "",
                prompt_length=sum(len(m.content) for m in request_body.messages),
                status_code=status_code,
                success=status_code < 400,
            )
        except Exception:
            pass

    return JSONResponse(status_code=status_code, content=payload)


# --- Job Queue Endpoints ---

@app.post("/jobs/submit", response_model=JobSubmitResponse)
async def submit_job(request_body: JobSubmitRequest, request: Request):
    """Submit a long-running request as an async job."""
    if not settings.enable_job_queue:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Job queue is disabled.", "type": "feature_disabled"}},
        )

    validation = validate_request(
        prompt=request_body.prompt,
        model=request_body.model,
        temperature=request_body.temperature,
        max_tokens=request_body.max_tokens,
    )
    if not validation.valid:
        return JSONResponse(
            status_code=422,
            content={"error": {"message": validation.error, "type": "validation_error"}},
        )

    queue = get_job_queue()
    job = queue.create_job(request_data=request_body.model_dump())

    # Run the job in the background
    asyncio.create_task(_process_job(job.id, request_body))

    return JobSubmitResponse(
        job_id=job.id,
        status=job.status.value,
        message="Job submitted successfully. Poll /jobs/{job_id} for status.",
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of an async job."""
    queue = get_job_queue()
    job = queue.get_job(job_id)
    if job is None:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": f"Job {job_id} not found.", "type": "not_found"}},
        )

    elapsed = None
    if job.started_at:
        end = job.completed_at or time.time()
        elapsed = round(end - job.started_at, 4)

    return JobStatusResponse(
        id=job.id,
        status=job.status.value,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        elapsed_seconds=elapsed,
        progress=job.progress,
        has_result=job.result is not None,
        error=job.error,
        result=job.result,
    )


@app.get("/jobs")
async def list_jobs(limit: int = 50):
    """List recent jobs."""
    queue = get_job_queue()
    return {"jobs": queue.list_jobs(limit=limit)}


async def _process_job(job_id: str, request_body: JobSubmitRequest):
    """Background task to process a job."""
    queue = get_job_queue()
    queue.update_job(job_id, status=JobStatus.RUNNING, progress="Processing...")

    try:
        loop = asyncio.get_event_loop()
        outcome = await loop.run_in_executor(
            None,
            lambda: coordinator.ask(
                prompt=request_body.prompt,
                task_type=request_body.task_type,
                model=request_body.model,
                temperature=request_body.temperature,
                top_p=request_body.top_p,
                max_tokens=request_body.max_tokens,
                stream=False,
                system_prompt=request_body.system_prompt,
            ),
        )

        if outcome.success:
            queue.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                result=outcome.to_api_response(),
                progress="Completed",
            )
        else:
            queue.update_job(
                job_id,
                status=JobStatus.FAILED,
                error=outcome.error,
                result=outcome.to_api_response(),
                progress="Failed",
            )
    except Exception as exc:
        queue.update_job(
            job_id,
            status=JobStatus.FAILED,
            error=str(exc),
            progress="Failed with exception",
        )


# --- Analytics & Observability Endpoints ---

@app.get("/analytics/dashboard")
async def analytics_dashboard(hours: int = 24):
    """Get usage analytics dashboard data."""
    collector = get_analytics_collector()
    return collector.get_dashboard_data(hours=hours)


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    cache = get_response_cache()
    return cache.get_stats()


@app.post("/cache/invalidate")
async def cache_invalidate(model_id: str | None = None):
    """Invalidate cache entries."""
    cache = get_response_cache()
    count = cache.invalidate(model_id=model_id)
    return {"invalidated": count, "model_id": model_id}


@app.get("/circuit-breakers")
async def circuit_breaker_status():
    """Get circuit breaker status for all models."""
    registry = get_circuit_breaker_registry()
    return {
        "breakers": registry.get_all_statuses(),
        "open_circuits": registry.get_open_circuits(),
    }


@app.post("/circuit-breakers/reset")
async def circuit_breaker_reset(model_id: str | None = None):
    """Reset circuit breaker(s)."""
    registry = get_circuit_breaker_registry()
    if model_id:
        success = registry.reset_model(model_id)
        return {"reset": success, "model_id": model_id}
    registry.reset_all()
    return {"reset": True, "model_id": "all"}


# --- System Prompt Endpoints ---

@app.get("/templates")
async def list_templates():
    """List available system prompt templates."""
    return {
        "templates": list_available_templates(),
        "details": get_template_info(),
    }
