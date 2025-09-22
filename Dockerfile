# Start from slim Python base
FROM python:3.10-slim

# Env vars for Python + Streamlit
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_HEADLESS=true

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create app dir
WORKDIR /app

# Copy everything (source + requirements)
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt || true \
    && pip install --no-cache-dir .

# Expose port
EXPOSE 8080

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.headless=true"]
