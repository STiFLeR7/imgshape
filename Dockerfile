# Use official Python slim image
FROM python:3.10-slim

# Set envs for Streamlit
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_HEADLESS=true

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-dev libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy requirements
COPY requirements.txt ./

# Install deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy src
COPY . .

# Expose port
EXPOSE 8080

# Entrypoint (Streamlit app)
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
