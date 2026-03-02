# ---- Dockerfile for IISC-HACK CCPA Compliance Analyzer ----
# Single-stage build: fast, reliable, push-ready for Docker Hub

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -g 1000 appuser && \
    useradd -m -u 1000 -g 1000 -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy and install Python dependencies (done as root before switching user)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and data
COPY app/ ./app/
COPY data/ ./data/
COPY build_kb.py .

# Copy pre-built FAISS index if it exists (speeds up cold start)
COPY faiss_index/ ./faiss_index/

# Runtime environment variables
ENV MODEL_ID="Qwen/Qwen2.5-3B-Instruct"
ENV HF_HOME="/app/.cache/huggingface"
ENV TRANSFORMERS_CACHE="/app/.cache/huggingface"

# Expose FastAPI port
EXPOSE 8000

# Switch to non-root user
USER appuser

# Start the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
