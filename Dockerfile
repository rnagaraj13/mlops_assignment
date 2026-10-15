# Stage 1: build / install dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

# Install OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: runtime image
FROM python:3.11-slim AS runtime

LABEL description="Heart disease prediction API"

# non-root user
RUN addgroup --gid 1001 appgroup && \
    adduser  --uid 1001 --gid 1001 --no-create-home --disabled-password appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/       ./src/
COPY api/       ./api/
COPY models/    ./models/

# Ensure model directory exists
RUN mkdir -p /app/models

ENV MODEL_PATH=/app/models/best_model.pkl \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

EXPOSE 8000

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
