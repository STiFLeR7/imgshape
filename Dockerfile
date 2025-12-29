# Dockerfile — imgshape v4.0.0 Atlas
# Backend-only image for Google Cloud Run
FROM python:3.12-slim

# Environment configuration for Cloud Run
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    PYTHONPATH=/app/src

# Avoid interactive prompts
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package files
COPY pyproject.toml setup.py requirements.txt README.md LICENSE /app/

# Copy Python source code
COPY src/ /app/src/
COPY service/ /app/service/

# Ensure container is API-only (no UI assets mounted)
RUN rm -rf /app/ui /app/service/templates /app/service/static || true

# Upgrade pip and install build tools
RUN python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN if [ -f requirements.txt ]; then \
    pip install --no-cache-dir -r requirements.txt || true; \
    fi && \
    pip install --no-cache-dir --upgrade .

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port for Cloud Run (uses $PORT)
EXPOSE 8080

# Run FastAPI with imgshape CLI (which handles service startup)
# Or directly use uvicorn for better Cloud Run compatibility
CMD ["sh", "-c", "uvicorn service.app:app --host 0.0.0.0 --port ${PORT:-8080} --timeout-keep-alive 120 --access-log"]
