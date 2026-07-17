# Security Documentation

## Authentication

### API Key Authentication
When `NVIDIA_BRIDGE_AUTH_ENABLED=true` and `NVIDIA_BRIDGE_API_KEY` is set, all endpoints (except `/health` and `/metrics`) require a valid API key.

**Header:** `X-Bridge-API-Key: <your-key>`

**Generate a key:**
```bash
python -c "import secrets; print(f'nvbridge-{secrets.token_urlsafe(32)}')"
```

**Multiple keys:** Comma-separated in `NVIDIA_BRIDGE_API_KEY`.

**Security properties:**
- Constant-time comparison (prevents timing attacks)
- Keys are never logged or exposed in responses
- Separate from the NVIDIA API key (defense in depth)

### Disabling Authentication
Set `NVIDIA_BRIDGE_AUTH_ENABLED=false` or leave `NVIDIA_BRIDGE_API_KEY` empty. Suitable for development or trusted internal networks only.

## Rate Limiting

Token bucket algorithm per client IP:
- Default: 60 requests per 60-second window
- Configurable via `NVIDIA_BRIDGE_RATE_LIMIT` and `NVIDIA_BRIDGE_RATE_WINDOW`
- Returns `429 Too Many Requests` with `Retry-After` header
- Exempt paths: `/health`, `/metrics`, `/docs`

## Input Validation

| Parameter | Limit | Error |
|---|---|---|
| Prompt length | 100,000 chars | 422 |
| Message count | 100 messages | 422 |
| Single message | 50,000 chars | 422 |
| Model ID | 200 chars, alphanumeric | 422 |
| Temperature | 0.0 - 2.0 | 422 |
| Max tokens | 1 - 131,072 | 422 |

## Security Headers

All responses include:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Content-Security-Policy: default-src 'self'`
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`
- `Permissions-Policy: camera=(), microphone=(), geolocation=(), payment=()`
- `Cache-Control: no-store, no-cache, must-revalidate`

## CORS

Configurable via `NVIDIA_BRIDGE_CORS_ORIGINS`. Default: `*` (all origins).

For production, restrict to specific origins:
```
NVIDIA_BRIDGE_CORS_ORIGINS=https://app.example.com,https://admin.example.com
```

## Request Tracing

Every request gets a unique `X-Request-ID` header (UUID v4). Clients can send their own `X-Request-ID` for correlation. All logs include the request ID.

## Sensitive Data Protection

- NVIDIA API keys are masked in health endpoints and logs
- API response content is not logged (only metadata)
- Error messages are redacted of account identifiers
- SQLite database stores no API keys or response content

## Docker Security

- Multi-stage build (minimal attack surface)
- Runs as non-root user (`bridgeuser`)
- Health check configured
- No unnecessary packages installed

## Recommendations for Production

1. **Enable authentication:** Set `NVIDIA_BRIDGE_AUTH_ENABLED=true`
2. **Use HTTPS:** Deploy behind a reverse proxy (nginx, Traefik) with TLS
3. **Restrict CORS:** Set specific origins instead of `*`
4. **Set rate limits:** Tune to your expected traffic
5. **Monitor metrics:** Connect Prometheus to `/metrics`
6. **Rotate API keys:** Regenerate consumer keys periodically
7. **Restrict network access:** Use firewall rules to limit who can reach port 8000
