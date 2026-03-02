# ---- Base image with CUDA support for GPU inference ----
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    build-essential \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Install Python dependencies ----
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Copy source code and data ----
COPY data/ccpa_statute.pdf data/
COPY app/ app/
COPY build_kb.py .

# ---- Set default model ----
ENV MODEL_ID="Qwen/Qwen2.5-3B-Instruct"

# ---- Accept HF token as build arg to download gated models ----
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# ---- Pre-download ALL weights at build time (critical hackathon requirement) ----
RUN python build_kb.py

# ---- Clear the build-time token so it is NOT baked into final image ----
ENV HF_TOKEN=""

# ---- Expose FastAPI port ----
EXPOSE 8000

# ---- Start the server ----
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
