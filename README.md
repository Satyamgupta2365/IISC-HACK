# 📄 CCPA Compliance Analyzer

> AI-powered API to analyze privacy policies for **California Consumer Privacy Act (CCPA)** compliance.

Accepts a privacy policy (PDF or plain text), runs semantic analysis, and returns structured JSON results with compliance status, evidence, and confidence scores.

---

## 🧠 How It Works

| Stage | Description |
|---|---|
| **Text Extraction** | Extracts text from uploaded PDF or text files |
| **Chunking & Embedding** | Splits text into segments and generates vector embeddings via sentence-transformers |
| **Semantic Search** | Uses a FAISS vector store to retrieve the most relevant policy segments per requirement |
| **Classification** | A transformer model evaluates each CCPA requirement against retrieved segments |
| **Structured Output** | Returns compliance status, evidence, confidence score, and explanation per requirement |

### Architecture
```
Client (File Upload)
       │
       ▼
 FastAPI Server
       │
       ▼
Text Extraction → Chunking → Embeddings
       │
       ▼
 FAISS Semantic Search
       │
       ▼
Model-Based Compliance Evaluation
       │
       ▼
    JSON Response
```

---

## 🐳 Docker Setup (Recommended)

### Pull the Image
```bash
docker pull satyampy/iischack:latest
```

### Run with GPU (Recommended)
```bash
docker run --gpus all \
  -p 8000:8000 \
  -e HF_TOKEN=<your_hf_token> \
  satyampy/iischack:latest
```

### Run on CPU (Fallback)
```bash
docker run \
  -p 8000:8000 \
  -e HF_TOKEN=<your_hf_token> \
  -e DEVICE=cpu \
  satyampy/iischack:latest
```

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | ✅ Yes | — | Hugging Face token to download gated models |
| `DEVICE` | No | `cuda` | Inference device: `cuda` or `cpu` |
| `MODEL_NAME` | No | built-in | Override the default transformer model |
| `EMBED_MODEL` | No | built-in | Override the default sentence-transformer model |

> **GPU Recommendation:** At least 8GB VRAM. CPU-only mode is supported but will be slower.

---

## 🛠 Local Setup (Without Docker)

> ⚠️ Only use this if Docker is unavailable.
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/iischack.git
cd iischack

# 2. Create and activate a virtual environment
python3.10 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Set environment variables
export HF_TOKEN=<your_hf_token>
export DEVICE=cuda   # or cpu

# 5. Start the server
uvicorn app:app --host 0.0.0.0 --port 8000
```

Visit the interactive API docs at `http://localhost:8000/docs`

---

## 📡 API Reference

### `GET /health`
```bash
curl -X GET http://localhost:8000/health
```
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda"
}
```

---

### `POST /analyze`

Upload a privacy policy file for CCPA compliance analysis.
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: multipart/form-data" \
  -F "file=@privacy_policy.pdf"
```
```json
{
  "document_name": "privacy_policy.pdf",
  "overall_compliance": "Partially Compliant",
  "results": [
    {
      "requirement": "Right to Know",
      "status": "Compliant",
      "confidence": 0.92,
      "evidence": "Consumers have the right to request disclosure...",
      "explanation": "The document clearly defines consumer disclosure rights."
    },
    {
      "requirement": "Right to Delete",
      "status": "Non-Compliant",
      "confidence": 0.85,
      "evidence": "",
      "explanation": "No deletion mechanism was found."
    }
  ]
}
```

---

## 📦 Dependencies
```
fastapi        uvicorn         pydantic
PyPDF2         sentence-transformers    faiss-cpu
torch          transformers    huggingface-hub
langchain      langchain-community
```

---

## 🔐 Hugging Face Token

- Generate a **read-only** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- Pass it via the `HF_TOKEN` environment variable
- ⚠️ **Never** hardcode your token in source code or committed files

---

## ✅ Design Notes

- **Modular** — swap or upgrade models via environment variables, no code changes needed
- **Scalable** — FAISS enables fast semantic search on large documents
- **Pipeline-ready** — designed to integrate into broader privacy automation workflows

