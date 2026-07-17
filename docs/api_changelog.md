# API Changelog

## v2.0.0 (2026-07-17)

### New Endpoints
- `GET /health/deep` - Deep health check with NVIDIA API connectivity verification
- `POST /jobs/submit` - Async job submission for long-running requests
- `GET /jobs/{job_id}` - Job status polling
- `GET /jobs` - List recent jobs
- `GET /analytics/dashboard` - Usage analytics dashboard
- `GET /cache/stats` - Cache statistics
- `POST /cache/invalidate` - Cache invalidation
- `GET /circuit-breakers` - Circuit breaker status
- `POST /circuit-breakers/reset` - Circuit breaker reset
- `GET /templates` - System prompt templates
- `GET /metrics` - Prometheus metrics

### Enhanced Endpoints
- `POST /ask` - Added `system_prompt`, `conversation_history`, `request_id`, `cache_hit` fields
- `POST /v1/chat/completions` - Added SSE streaming support (`stream: true`)
- `GET /health` - Added `version`, `auth_enabled`, `cache_enabled`, `streaming_enabled`, `metrics_enabled`, `uptime_seconds`

### Security
- API key authentication via `X-Bridge-API-Key` header
- Per-client rate limiting (token bucket algorithm)
- Input validation on all inference endpoints
- Security headers on all responses
- Request ID tracking (`X-Request-ID`)
- CORS configuration

### Infrastructure
- In-memory response caching with TTL
- Circuit breaker pattern for model resilience
- Structured logging (structlog)
- SQLite request logging and analytics
- Prometheus metrics
- Multi-turn conversation support
- System prompt templates per task type
- Async job queue for long-running requests
- Provider abstraction layer

### Docker
- Multi-stage build for smaller images
- Non-root user for security
- Built-in HEALTHCHECK

## v1.0.0

- Initial release
- Basic model routing
- Benchmark system
- Docker support
