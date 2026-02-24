# Multi-stage build for SymptomPal backend
# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Install runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY backend/ ./backend/

# Create data directory for SQLite
RUN mkdir -p /app/backend/data && chown -R appuser:appuser /app

USER appuser

# Environment defaults — connects to Ollama sidecar by default.
# Set USE_STUB_MEDGEMMA=true (etc.) to run without model access.
ENV PYTHONUNBUFFERED=1
ENV USE_OLLAMA_MEDGEMMA=true
ENV OLLAMA_HOST=http://ollama:11434
ENV RATE_LIMIT_RPM=60

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
