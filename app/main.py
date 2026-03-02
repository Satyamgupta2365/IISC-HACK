from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from .rag import get_analyzer

app = FastAPI()

class AnalysisRequest(BaseModel):
    prompt: str

class AnalysisResponse(BaseModel):
    harmful: bool
    articles: List[str]

@app.on_event("startup")
async def startup_event():
    # Pre-load the analyzer on startup
    _ = get_analyzer()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/analyze", response_model=AnalysisResponse)
def analyze_endpoint(request: AnalysisRequest):
    analyzer = get_analyzer()
    print(f"Request prompt: {request.prompt}")
    result = analyzer.analyze(request.prompt)
    print(f"Response: {result}")
    return result
