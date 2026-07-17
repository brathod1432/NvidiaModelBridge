# Service API Reference

## Authentication

When authentication is enabled (`NVIDIA_BRIDGE_AUTH_ENABLED=true`), include the API key in every request:

```
X-Bridge-API-Key: nvbridge-your-key-here
```

Endpoints `/health` and `/metrics` are always accessible without authentication.

## `GET /health`

Returns service status and feature flags.

```json
{
  "status": "ok",
  "service": "Nvidia Model Bridge",
  "version": "2.0.0",
  "api_key_found": true,
  "api_key_masked": "nvapi-****abcd",
  "base_url": "https://integrate.api.nvidia.com/v1",
  "auth_enabled": false,
  "cache_enabled": true,
  "streaming_enabled": true,
  "metrics_enabled": true,
  "uptime_seconds": 3600.0
}
```

## `GET /health/deep`

Deep health check that verifies NVIDIA API connectivity, cache status, and circuit breakers.

```json
{
  "status": "ok",
  "service": "Nvidia Model Bridge",
  "version": "2.0.0",
  "api_key_found": true,
  "base_url": "https://integrate.api.nvidia.com/v1",
  "nvidia_reachable": true,
  "nvidia_status_code": 200,
  "nvidia_latency_seconds": 0.45,
  "nvidia_models_found": 121,
  "cache_stats": {"size": 42, "hits": 150, "misses": 30, "hit_rate": 0.8333},
  "circuit_breaker_summary": {"open_circuits": [], "total_tracked": 5},
  "auth_enabled": false
}
```

## `POST /ask`

Coordinator routing endpoint with intelligent model selection.

### Request

```json
{
  "prompt": "Write a Python function to reverse a string.",
  "task_type": "coding",
  "model": null,
  "temperature": null,
  "top_p": null,
  "max_tokens": null,
  "stream": false,
  "system_prompt": null,
  "conversation_history": [
    {"role": "user", "content": "Previous question"},
    {"role": "assistant", "content": "Previous answer"}
  ]
}
```

### Response

```json
{
  "success": true,
  "model": "qwen/qwen3-next-80b-a3b-instruct",
  "task_type": "coding",
  "selected_by": "coordinator",
  "selection_reason": "recommended coding/default model from latest benchmark",
  "fallback_used": false,
  "latency_seconds": 1.23,
  "content": "...",
  "reasoning": null,
  "error": null,
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "cache_hit": false
}
```

## `POST /v1/chat/completions`

OpenAI-compatible forwarding endpoint. Supports streaming.

### Non-Streaming

```json
{
  "model": "qwen/qwen3-next-80b-a3b-instruct",
  "messages": [{"role": "user", "content": "Hello"}],
  "temperature": 0.7,
  "max_tokens": 1024,
  "stream": false
}
```

### Streaming (SSE)

Set `"stream": true` to receive Server-Sent Events.

## `POST /jobs/submit`

Submit a long-running request as an async job.

### Request

```json
{
  "prompt": "Solve this complex problem...",
  "task_type": "reasoning",
  "model": null,
  "system_prompt": null
}
```

### Response

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Job submitted successfully. Poll /jobs/{job_id} for status."
}
```

## `GET /jobs/{job_id}`

Get job status. Returns result when completed.

## `GET /analytics/dashboard?hours=24`

Returns usage analytics for the specified time window.

## `GET /cache/stats`

Returns cache hit rate, size, and configuration.

## `POST /cache/invalidate?model_id=model/name`

Invalidate cache entries. Omit `model_id` to clear all.

## `GET /circuit-breakers`

Returns circuit breaker state for all tracked models.

## `POST /circuit-breakers/reset?model_id=model/name`

Reset a circuit breaker. Omit `model_id` to reset all.

## `GET /templates`

Returns available system prompt templates and their content.

## Error Responses

All errors follow this format:

```json
{
  "error": {
    "message": "Description of the error.",
    "type": "error_type"
  }
}
```

Error types: `authentication_error`, `rate_limit_error`, `validation_error`, `invalid_request_error`, `unsupported_feature`, `not_found`.
