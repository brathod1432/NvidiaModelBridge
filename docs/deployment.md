# Deployment Guide

## Local Development

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Configure
python scripts/prepare_env.py
# Or: cp .env.example .env && edit .env

# Run
python main.py
# Or: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Docker

### Build and Run

```bash
docker compose up --build -d
```

### Environment Variables

Pass via `docker-compose.yml` environment section or `.env` file.

### Persistent Storage

- `./audits:/app/audits` - Benchmark audit reports
- `./docs:/app/docs` - Generated documentation
- `bridge-data:/app/data` - SQLite database

## Production Deployment

### Behind Nginx (Recommended)

```nginx
upstream nvidia_bridge {
    server 127.0.0.1:8000;
}

server {
    listen 443 ssl;
    server_name bridge.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://nvidia_bridge;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE streaming support
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;
    }
}
```

### Monitoring

Connect Prometheus to `http://bridge:8000/metrics` and use the analytics dashboard at `/analytics/dashboard`.

## Validation

```bash
# Health check
curl http://localhost:8000/health

# Deep health check (verifies NVIDIA connectivity)
curl http://localhost:8000/health/deep

# Environment check
python scripts/check_env.py
```
