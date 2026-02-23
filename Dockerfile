# Multi-stage Dockerfile for Real Estate Prediction Pipeline

# Base stage with Python
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Development stage
FROM base as development

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models reports/figures

# Default command for development
CMD ["python", "run_all.py"]

# Production stage
FROM base as production

# Copy requirements first
COPY requirements.txt .

# Install only production dependencies (you can create requirements-prod.txt without dev tools)
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy only necessary application files
COPY src/ ./src/
COPY api/ ./api/
COPY config.yaml .
COPY run_all.py .

# Create necessary directories
RUN mkdir -p models data/raw data/processed reports/figures

# Add non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose port for API (if running API server)
EXPOSE 8000

# Default command for production
CMD ["python", "run_all.py"]

# Training stage - for running model training
FROM production as training

USER root
COPY data/ ./data/
RUN chown -R appuser:appuser /app/data
USER appuser

CMD ["python", "run_all.py"]

# API stage - for serving predictions
FROM production as api

# Copy pre-trained models (should be built separately)
COPY models/ ./models/

# Run API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
