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

# Install system dependencies (for OpenCV, WeasyPrint, and C-extension compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    ca-certificates \
    curl \
    # OpenCV dependencies
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0t64 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # WeasyPrint dependencies (v60+)
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libpangoft2-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    libharfbuzz-subset0 \
    libopenjp2-7 \
    libjpeg-dev \
    # Optional tools
    nano \
    && apt-get clean \
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

# Install Python dependencies with verification
RUN pip install --no-cache-dir uvicorn[standard] && \
    if [ -f requirements.txt ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    fi && \
    pip install --no-cache-dir -e . && \
    python -c "import service.app; print('✓ service.app imports successfully')" && \
    python -c "import uvicorn; print('✓ uvicorn available')"

# Expose port for Cloud Run (uses $PORT)
EXPOSE 8080

# Run FastAPI with diagnostic output
CMD set -x && \
    echo "PORT=${PORT:-8080}" && \
    echo "PYTHONPATH=${PYTHONPATH}" && \
    python -c "import sys; print('Python:', sys.version); import service.app" && \
    exec python -m uvicorn service.app:app \
    --host 0.0.0.0 \
    --port ${PORT:-8080} \
    --timeout-keep-alive 120 \
    --log-level info
