# ---- Multi‑stage Dockerfile for IISC‑HACK ----
# Builder stage: install Python, dependencies, and pre‑download model weights
FROM python:3.10-slim AS builder

# Install system build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first for caching
COPY requirements.txt .
# Upgrade pip and install dependencies into /install
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    mkdir -p /install && \
    pip install --no-cache-dir -r requirements.txt -t /install

# Copy source code and data
COPY app/ ./app/
COPY data/ ./data/
COPY build_kb.py .

# Build‑time argument for HuggingFace token (optional)
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}
# Pre‑download model weights (uses the script which respects HF_TOKEN)
RUN python build_kb.py

# Runtime stage: lightweight CUDA runtime
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Create non‑root user
ARG USERNAME=appuser
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID $USERNAME && \
    useradd -m -u $UID -g $GID -s /bin/bash $USERNAME

# Set work directory
WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local/lib/python3.10/site-packages
# Copy application code and data
COPY --from=builder /app/app ./app
COPY --from=builder /app/data ./data
COPY --from=builder /app/build_kb.py .

# Default model ID (can be overridden at runtime)
ENV MODEL_ID="Qwen/Qwen2.5-3B-Instruct"
# Ensure HF token is cleared
ENV HF_TOKEN=""

# Expose FastAPI port
EXPOSE 8000

# Switch to non‑root user
USER $USERNAME

# Start the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
