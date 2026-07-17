# Nvidia Model Bridge

A production-grade FastAPI gateway for routing prompts to NVIDIA-hosted OpenAI-compatible models, featuring security hardening, intelligent routing, caching, circuit breakers, and full observability.

## Features

### Core
- Intelligent model routing by task type (general, coding, reasoning, fast, etc.)
- Multi-level fallback chain with automatic failover
- OpenAI-compatible `/v1/chat/completions` endpoint
- Multi-turn conversation support with conversation history
- Configurable system prompts per task type
- SSE streaming support

### Security
- API key authentication for gateway consumers (`X-Bridge-API-Key` header)
- Per-client rate limiting with token bucket algorithm
- Input validation and content safety (prompt length, message count, parameter bounds)
- Security headers (CORS, CSP, HSTS, X-Frame-Options, etc.)
- Request ID tracking for audit trails
- Sensitive data redaction in logs

### Infrastructure
- In-memory response caching with TTL
- Circuit breaker pattern for failing models (auto-recovery)
- Structured logging via structlog (JSON or console output)
- Prometheus metrics endpoint (`/metrics`)
- SQLite database for request logging and usage analytics
- Async job queue for long-running requests
- Deep health checks with NVIDIA API connectivity verification

### Observability
- Real-time analytics dashboard endpoint
- Cache statistics and invalidation API
- Circuit breaker status and reset API
- Request logging with latency, model, and error tracking
- Prometheus-compatible metrics for Grafana dashboards

## Quick Start

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### Configure

```bash
# Auto-detect NVIDIA_API_KEY from environment/registry
python scripts/prepare_env.py

# Or manually create .env
cp .env.example .env
# Edit .env and add your NVIDIA_API_KEY
```

### Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or:

```bash
python main.py
```

### Docker

```bash
docker compose up --build
```

## API Endpoints

### Health
| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Basic health check with feature flags |
| `/health/deep` | GET | Deep health check with NVIDIA API verification |

### Inference
| Endpoint | Method | Description |
|---|---|---|
| `/ask` | POST | Coordinator routing with auto model selection |
| `/v1/chat/completions` | POST | OpenAI-compatible forwarding (supports streaming) |

### Models
| Endpoint | Method | Description |
|---|---|---|
| `/models/priority` | GET | Priority model list |
| `/models/recommended` | GET | Recommended routing map |
| `/models/avoid` | GET | Avoid, partial, and retest lists |

### Jobs (Async)
| Endpoint | Method | Description |
|---|---|---|
| `/jobs/submit` | POST | Submit async job for long-running requests |
| `/jobs/{job_id}` | GET | Get job status and result |
| `/jobs` | GET | List recent jobs |

### Observability
| Endpoint | Method | Description |
|---|---|---|
| `/metrics` | GET | Prometheus metrics |
| `/analytics/dashboard` | GET | Usage analytics dashboard |
| `/cache/stats` | GET | Cache statistics |
| `/cache/invalidate` | POST | Invalidate cache entries |
| `/circuit-breakers` | GET | Circuit breaker status |
| `/circuit-breakers/reset` | POST | Reset circuit breakers |
| `/templates` | GET | System prompt templates |

## Usage Examples

### Basic Ask (Auto-Routing)

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python function to reverse a string.", "task_type": "coding"}'
```

### With Authentication

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-Bridge-API-Key: nvbridge-your-key-here" \
  -d '{"prompt": "Explain quantum computing.", "task_type": "reasoning"}'
```

### Multi-Turn Conversation

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Now explain it to a 5-year-old.",
    "task_type": "general",
    "conversation_history": [
      {"role": "user", "content": "What is gravity?"},
      {"role": "assistant", "content": "Gravity is a fundamental force..."}
    ]
  }'
```

### Streaming (SSE)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/qwen3-next-80b-a3b-instruct",
    "messages": [{"role": "user", "content": "Tell me a story."}],
    "stream": true
  }'
```

### Async Job

```bash
# Submit job
curl -X POST http://localhost:8000/jobs/submit \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Solve this complex math problem...", "task_type": "reasoning"}'

# Poll for result
curl http://localhost:8000/jobs/{job_id}
```

## Configuration

All settings are configurable via environment variables. See `.env.example` for the full list.

### Key Settings

| Variable | Default | Description |
|---|---|---|
| `NVIDIA_API_KEY` | (required) | NVIDIA API key |
| `NVIDIA_BRIDGE_API_KEY` | (empty) | Gateway consumer API key(s) |
| `NVIDIA_BRIDGE_AUTH_ENABLED` | `false` | Enable API key authentication |
| `NVIDIA_BRIDGE_RATE_LIMIT` | `60` | Requests per minute per client |
| `NVIDIA_BRIDGE_CORS_ORIGINS` | `*` | Allowed CORS origins |
| `NVIDIA_BRIDGE_ENABLE_CACHE` | `true` | Enable response caching |
| `NVIDIA_BRIDGE_CACHE_TTL` | `300` | Cache TTL in seconds |
| `NVIDIA_BRIDGE_CB_THRESHOLD` | `5` | Circuit breaker failure threshold |
| `NVIDIA_BRIDGE_LOG_LEVEL` | `INFO` | Log level |
| `NVIDIA_BRIDGE_LOG_JSON` | `false` | JSON log output |

## Architecture

See [docs/architecture.md](docs/architecture.md) for detailed architecture documentation.

## Security

See [docs/security.md](docs/security.md) for security documentation.

## Testing

```bash
pytest tests/ -v
```

## Validation

```bash
python scripts/check_env.py
python scripts/test_nvidia_models.py
curl http://localhost:8000/health
curl http://localhost:8000/health/deep
curl http://localhost:8000/models/recommended
```

## Documentation

- [Architecture](docs/architecture.md)
- [Security](docs/security.md)
- [Deployment](docs/deployment.md)
- [API Reference](docs/service_api.md)
- [Environment Setup](docs/env_setup.md)
- [Docker Setup](docs/docker_setup.md)
- [Model Selection Notes](docs/model_selection_notes.md)
- [API Changelog](docs/api_changelog.md)
