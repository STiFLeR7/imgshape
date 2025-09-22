# Dockerfile — run FastAPI service at service/app.py using uvicorn
FROM python:3.10-slim

# environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

# avoid prompts
ARG DEBIAN_FRONTEND=noninteractive

# system deps (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy package files (requirements + setup)
COPY pyproject.toml setup.py requirements.txt README.md /app/

# copy source and service folders
COPY src/ /app/src/
COPY service/ /app/service/

# make sure pip/setuptools are recent
RUN python -m pip install --upgrade pip setuptools wheel

# install runtime deps: prefer requirements.txt if provided; otherwise install editable package
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt || true; fi \
    && pip install --no-cache-dir --upgrade .

# expose port used by uvicorn
EXPOSE 8080

# Use uvicorn to serve the FastAPI app at service/app.py
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "120"]
