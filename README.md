# CCPA Compliance Detector

## Architecture
This project is an AI system built to detect California Consumer Privacy Act (CCPA) violations from business practice descriptions. 
The architecture utilizes a Retrieval-Augmented Generation (RAG) approach:
1. **Document Embedding**: The `ccpa_statute.pdf` is chunked and embedded using the sentence-transformers (`all-MiniLM-L6-v2`) via Langchain and saved to a local FAISS vector store. 
2. **Retrieval**: When a request comes in, the specific query is embedded and the top-k most similar excerpts from the CCPA text are retrieved.
3. **Generative Inference**: The user prompt and the retrieved context are passed to a Large Language Model (`Mistral-7B-Instruct-v0.2` by default) via HuggingFace's transformers pipeline to analyze and determine if the practice is harmful (violates the CCPA) and return the corresponding articles as completely valid JSON.
4. **FastAPI**: The system is packaged into a containerized FastAPI server listening on Port 8000. All weights are downloaded statically at Docker build time.

## Local Setup (Without Docker)
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Build the FAISS Vector Database and download model weights (will download to Hugging Face cache):
   ```bash
   python build_kb.py
   ```
3. Run the FastAPI Application locally:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## Docker Setup & Build
To build the Docker image (replace `<your-token>` and `<your-dockerhub-username>`):
```bash
docker build --build-arg HF_TOKEN=<your-token> -t <your-dockerhub-username>/ccpa-compliance:latest .
```
_Note: The inference model weights and vector store embeddings are packaged immediately during the Docker build process to ensure maximum runtime speed and no delayed initialization._

To push to Docker Hub:
```bash
docker push <your-dockerhub-username>/ccpa-compliance:latest
```

## Docker Run Command
The service requires an NVIDIA GPU to run the LLM inference within the 120-second timeout constraints. Example run command:
```bash
docker run --gpus all -p 8000:8000 -e HF_TOKEN=<your-token> <your-dockerhub-username>/ccpa-compliance:latest
```
_Wait until the uvicorn worker successfully completes loading the LLM instance._

## Environment Variables & GPU Requirements
- `HF_TOKEN`: Needed to authenticate gated model downloads. Must be passed at Build Time `build-arg` and at Runtime via `-e`.
- `MODEL_ID`: Defaults to `mistralai/Mistral-7B-Instruct-v0.2`. Change to a smaller Llama-3 parameter model if necessary.
- **GPU Requirement:** Minimum 16GB VRAM (e.g. Nvidia T4, A10G) configured with nvidia-container-toolkit (`--gpus all`) due to loading a 7B float16 LLM.

## API Examples

### 1. Health Check
```bash
curl -X GET "http://localhost:8000/health"
```
**Response:**
```json
{"status": "ok"}
```

### 2. Analyze Endpoint - Harmful Request
```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "We charge customers who opted out of data selling a higher price for the same service."}'
```
**Response:**
```json
{
  "harmful": true,
  "articles": ["Section 1798.125"]
}
```

### 3. Analyze Endpoint - Non-Harmful Request
```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "We securely delete user data upon proper verified request within maximum 45 days."}'
```
**Response:**
```json
{
  "harmful": false,
  "articles": []
}
```
