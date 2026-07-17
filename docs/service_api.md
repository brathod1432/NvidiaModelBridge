# Service API

## `GET /health`

Returns service status, whether `NVIDIA_API_KEY` is present, the masked key, and the configured NVIDIA base URL.

Example:

```json
{
  "status": "ok",
  "service": "Nvidia Model Bridge",
  "api_key_found": true,
  "api_key_masked": "nvapi-****abcd",
  "base_url": "https://integrate.api.nvidia.com/v1"
}
```

If the key is missing, the service reports `degraded`.

## `GET /models/priority`

Returns the three priority models as editable model records.

## `GET /models/recommended`

Returns the current benchmark-backed routing map:

- default/general: `qwen/qwen3-next-80b-a3b-instruct`
- reasoning: `qwen/qwen3-next-80b-a3b-instruct`
- nvidia reasoning: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`
- coding: `qwen/qwen3-next-80b-a3b-instruct`
- fast: `nvidia/nemotron-mini-4b-instruct`
- fallback: `mistralai/mistral-nemotron`
- lightweight: `openai/gpt-oss-20b`
- deepseek: `deepseek-ai/deepseek-v4-pro`

## `GET /models/avoid`

Returns avoid, partial, and retest candidate lists.

## `POST /ask`

Coordinator routing endpoint.

Request example:

```json
{
  "prompt": "Write a Python function to reverse a string.",
  "task_type": "coding",
  "model": null,
  "temperature": null,
  "top_p": null,
  "max_tokens": null,
  "stream": false
}
```

Behavior:

- If `model` is supplied, the gateway uses that exact model.
- If `model` is missing, the coordinator chooses a model from the task type.
- If `task_type` is missing, the gateway uses `general`.
- If the coordinator-selected model fails, it tries one fallback model.
- If a user-specified model fails, it returns a clear error instead of silently switching models.

Example response:

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
  "error": null
}
```

## `POST /v1/chat/completions`

OpenAI-compatible forwarding endpoint.

Rules:

- `model` is required.
- If `model` is missing, the service returns HTTP `400` with an `invalid_request_error`.
- If `stream=true`, the service returns HTTP `400` with an `unsupported_feature` error.
- The request is forwarded to NVIDIA’s `POST https://integrate.api.nvidia.com/v1/chat/completions` endpoint using the exact model provided.

Use `/ask` when you want routing. Use `/v1/chat/completions` when you want exact OpenAI-compatible forwarding.
