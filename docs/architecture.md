# Architecture

## Overview

Nvidia Model Bridge is structured as a layered FastAPI application with clear separation of concerns:

```
                    Client Request
                         |
                    [Middleware Stack]
                    |  Security Headers
                    |  Request ID
                    |  Rate Limiting
                    |  CORS
                         |
                    [FastAPI Router]
                         |
              +----------+----------+
              |          |          |
           /ask    /v1/chat    /jobs/submit
              |          |          |
         [Input Validation]        |
              |                    |
         [Coordinator]        [Job Queue]
              |                    |
         [Cache Check]       [Background Task]
              |                    |
         [Circuit Breaker]        |
              |                    |
         [Model Router]           |
              |                    |
         [NvidiaClient]           |
              |                    |
         [NVIDIA API]             |
              |                    |
         [Analytics + Metrics + DB Logging]
```

## Module Responsibilities

### Core
- **`main.py`** - FastAPI app, middleware registration, endpoint definitions
- **`coordinator.py`** - Request orchestration with caching, circuit breakers, and fallback
- **`nvidia_client.py`** - HTTP/SDK communication with NVIDIA API
- **`config.py`** - Environment-based configuration with validation

### Routing
- **`model_registry.py`** - Curated model database with benchmark data
- **`model_router.py`** - Task-type-based model selection and fallback chains

### Security
- **`middleware/auth.py`** - API key authentication
- **`middleware/rate_limit.py`** - Token bucket rate limiting
- **`middleware/security_headers.py`** - Security response headers
- **`middleware/request_id.py`** - Request ID generation and propagation
- **`input_validation.py`** - Prompt, message, and parameter validation

### Infrastructure
- **`cache.py`** - In-memory response cache with TTL
- **`circuit_breaker.py`** - Circuit breaker pattern for model resilience
- **`logging_config.py`** - Structured logging configuration
- **`metrics.py`** - Prometheus metrics definitions
- **`database.py`** - SQLite for request logs and analytics
- **`streaming.py`** - SSE streaming support

### Features
- **`prompt_templates.py`** - System prompts per task type
- **`analytics.py`** - Real-time usage analytics
- **`job_queue.py`** - Async job processing
- **`provider_base.py`** - Provider abstraction for multi-provider future

### Benchmarking
- **`model_tester.py`** - Model benchmark runner
- **`benchmark_tasks.py`** - Benchmark task definitions and evaluators
- **`audit.py`** - Audit report generation

## Data Flow

1. **Request arrives** -> Middleware processes security headers, request ID, rate limit, CORS
2. **Input validation** -> Validates prompt length, message format, parameter ranges
3. **Model selection** -> Coordinator selects model by task type or uses explicit model
4. **Cache check** -> Returns cached response if available (cache hit)
5. **Circuit breaker check** -> Skips models with open circuits, tries fallbacks
6. **API call** -> NvidiaClient makes HTTP request to NVIDIA API
7. **Response processing** -> Extract content, record metrics, cache response
8. **Fallback** -> If primary fails, try fallback chain (up to 3 models)
9. **Logging** -> Log to SQLite, update analytics, emit Prometheus metrics
