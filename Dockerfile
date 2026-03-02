# ---- Dockerfile for IISC-HACK CCPA Compliance Analyzer ----
# Build time: ~5-6 minutes
# - Installs all Python dependencies           (~3-4 min)
# - Pre-downloads embedding model (90 MB)      (~1 min)
# - Builds FAISS vector index from PDF         (~30 sec)
# - LLM weights are downloaded on first run    (keeps image size manageable)

FROM python:3.10-slim

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl && \
    rm -rf /var/lib/apt/lists/*

# ── Non-root user ────────────────────────────────────────────────────────────
RUN groupadd -g 1000 appuser && \
    useradd -m -u 1000 -g 1000 -s /bin/bash appuser

WORKDIR /app

# ── Python dependencies (~3-4 min) ───────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy source code and data ────────────────────────────────────────────────
COPY app/       ./app/
COPY data/      ./data/
COPY build_kb.py .

# ── HuggingFace cache directory (writable by appuser) ────────────────────────
RUN mkdir -p /app/.cache/huggingface && \
    chown -R appuser:appuser /app

ENV HF_HOME="/app/.cache/huggingface"
ENV TRANSFORMERS_CACHE="/app/.cache/huggingface"

# ── Pre-download embedding model + build FAISS index (~1-2 min) ──────────────
# Only downloads sentence-transformers/all-MiniLM-L6-v2 (~90 MB, very fast).
# Does NOT download the large LLM — that happens on first container startup.
USER appuser
RUN python - <<'EOF'
from huggingface_hub import snapshot_download
import sys, os

print("==> Pre-downloading embedding model (all-MiniLM-L6-v2)...")
try:
    snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2")
    print("    Embedding model cached successfully.")
except Exception as e:
    print(f"    WARNING: {e}")

print("==> Building FAISS index from CCPA PDF...")
try:
    os.chdir("/app")
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    pdf = "data/ccpa_statute.pdf"
    if os.path.exists(pdf):
        docs   = PyPDFLoader(pdf).load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        emb    = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db     = FAISS.from_documents(chunks, emb)
        db.save_local("faiss_index")
        print("    FAISS index built and saved.")
    else:
        print("    PDF not found — skipping index build (will build at runtime).")
except Exception as e:
    print(f"    WARNING: FAISS build failed: {e}")

print("==> Build-time pre-warming complete.")
EOF

# ── Runtime environment ───────────────────────────────────────────────────────
ENV MODEL_ID="Qwen/Qwen2.5-3B-Instruct"
ENV PORT=8000

EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
