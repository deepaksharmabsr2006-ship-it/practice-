# Project 119: Driver Fatigue & Distraction Safety Monitor
# Multi-Stage Production Container for Cloud & Edge Deployment

FROM python:3.10-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications and install
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-ml.txt

# Copy full application codebase
COPY . .

# Pre-train and serialize ML models during container build
RUN python ml/model_trainer.py

# Expose web gateway port
EXPOSE 8000

# Healthcheck probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/v1/health || exit 1

# Start FastAPI production server
CMD uvicorn backend.app:app --host 0.0.0.0 --port ${PORT}
