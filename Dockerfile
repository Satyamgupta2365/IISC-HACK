FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p data app

# Copy application code and data
COPY data/ccpa_statute.pdf data/
COPY app/ app/
COPY build_kb.py .

# Set default model ID. Llama-3-8B is also an excellent option if you have access.
ENV MODEL_ID="mistralai/Mistral-7B-Instruct-v0.2"

# You can pass the HuggingFace token as a build argument
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Pre-build vector DB and download model weights (no download at runtime)
RUN python build_kb.py

# Expose port 8000 for FastAPI
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
