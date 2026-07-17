# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

# Security: run as non-root
RUN groupadd -r bridgeuser && useradd -r -g bridgeuser -d /app -s /sbin/nologin bridgeuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy application code
COPY app ./app
COPY scripts ./scripts
COPY main.py .

# Create data directories with proper ownership
RUN mkdir -p /app/audits /app/docs /app/data && \
    chown -R bridgeuser:bridgeuser /app

# Copy static files
COPY audits/.gitkeep ./audits/
COPY docs ./docs

# Switch to non-root user
USER bridgeuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
