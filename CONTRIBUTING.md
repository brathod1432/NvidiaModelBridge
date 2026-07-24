# Contributing to NvidiaModelBridge

**A contribution by Brijesh Rathod to the AI community.**

---

## Why This Exists

I built NvidiaModelBridge because I believe access to powerful AI models
should be simpler, more reliable, and more observable. The gap between
"I have an API key" and "I have a production-ready AI gateway" is too wide
for most developers and teams. This project bridges that gap.

This is my contribution to making AI infrastructure better -- not just for
myself, but for anyone who needs it. Whether you are a student exploring
AI for the first time, a startup shipping your first product, or an
engineer hardening a production system, this toolkit is for you.

**Use it freely. Learn from it. Build on it. Make it better.**

---

## What This Project Stands For

- **Open access** -- AI tooling should be available to everyone, not locked
  behind enterprise paywalls
- **Production quality** -- Open source does not mean half-finished; this
  project has 133 tests, security hardening, and observability built in
- **Practical engineering** -- Every feature exists because it solves a real
  problem: circuit breakers for resilience, caching for cost savings,
  rate limiting for safety, batch processing for efficiency
- **Transparency** -- The code is the documentation; read it, understand it,
  challenge it

---

## How to Contribute

### Reporting Issues

- Open a GitHub issue with a clear description
- Include steps to reproduce, expected vs actual behavior
- Mention your Python version, OS, and relevant configuration

### Submitting Code

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-improvement`
3. Write your code following the standards below
4. Add or update tests for your changes
5. Run the test suite: `python -m pytest tests/ -v`
6. Commit with a clear message describing *what* and *why*
7. Open a pull request against `master`

### Coding Standards

- **Python 3.10+** compatibility
- `from __future__ import annotations` in every module
- Type hints on all public functions
- Docstrings on classes and public methods
- Use `logging` module, never `print()` in application code
- Follow existing patterns (middleware, singletons, config loading)
- Keep dependencies minimal -- don't add heavy frameworks without discussion

### Test Requirements

- All existing tests must continue to pass
- New features must include tests
- Security-sensitive code must have negative/boundary tests
- Run: `python -m pytest tests/ -v --tb=short`

---

## Project Architecture

```
app/
  main.py              FastAPI application and endpoint definitions
  coordinator.py       Request orchestration (routing, fallback, caching)
  nvidia_client.py     NVIDIA API communication layer
  config.py            Environment-based configuration
  model_router.py      Task-type to model mapping with fallback chains
  model_registry.py    Curated model database from benchmarks
  cache.py             Response caching (in-memory + Redis backends)
  circuit_breaker.py   Circuit breaker pattern for resilience
  deduplication.py     Request deduplication (promise/future pattern)
  database.py          SQLite audit trail with migrations
  streaming.py         SSE streaming with retry and error sanitization
  job_queue.py         Async job processing with timeout/cancellation
  input_validation.py  Request validation and sanitization
  analytics.py         Usage analytics and dashboard
  metrics.py           Prometheus metrics
  middleware/          Auth, rate limiting, request ID, security headers
tests/                 133 tests covering functionality and security
docs/                  Architecture, deployment, security, API docs
```

---

## Areas Where Help Is Welcome

- **Additional model providers** -- Azure OpenAI, Anthropic, Google AI
- **Advanced caching** -- Semantic similarity-based cache lookup
- **WebSocket support** -- Real-time bidirectional streaming
- **Admin dashboard** -- Web UI for monitoring and configuration
- **Kubernetes manifests** -- Helm charts for cloud deployment
- **Load testing** -- Locust or k6 performance benchmarks
- **Documentation** -- Tutorials, deployment guides, architecture diagrams

---

## Code of Conduct

- Be respectful and constructive in all interactions
- Focus on the code and ideas, not the person
- Welcome newcomers and help them get started
- Give credit where it is due
- Keep discussions technical and productive

---

## License

This project is released under the [MIT License](LICENSE).

Free to use, modify, and distribute for any purpose. If it helps you
build something great, that is all the thanks I need.

---

*"The best way to predict the future of AI is to build the infrastructure
that makes it accessible to everyone."*

-- Brijesh Rathod
