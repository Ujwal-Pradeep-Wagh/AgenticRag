"""
app.py
FastAPI Backend Entry Point

Provides REST API endpoints for the Agentic RAG system.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os

from pipeline import AgenticRAGPipeline, TraditionalRAGPipeline
from ingestion.pipeline import DocumentIngestionPipeline

app = FastAPI(
    title="Agentic RAG API",
    description="Multi-Agent Retrieval-Augmented Generation System",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    conversation_history: Optional[List[Dict[str, str]]] = []
    compare_baseline: bool = False


class QueryResponse(BaseModel):
    query: str
    final_answer: str
    agent_decisions: List[Dict[str, Any]]
    retrieved_documents: List[Dict[str, Any]]
    reflection: Dict[str, Any]
    iterations: int
    baseline_answer: Optional[str] = None


@app.get("/")
def root():
    return {"message": "Agentic RAG API", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ingest")
def ingest_document(file: UploadFile = File(...)):
    """Upload and ingest a PDF document."""
    os.makedirs("./data/uploads", exist_ok=True)
    file_path = f"./data/uploads/{file.filename}"

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    pipeline = DocumentIngestionPipeline()
    result = pipeline.ingest(file_path)

    return result


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Process a query through the Agentic RAG pipeline."""
    agentic = AgenticRAGPipeline()
    result = agentic.run(request.query, request.conversation_history)

    response = QueryResponse(
        query=result["query"],
        final_answer=result["final_answer"],
        agent_decisions=result["agent_decisions"],
        retrieved_documents=result["retrieved_documents"],
        reflection=result["reflection"],
        iterations=result["iterations"]
    )

    if request.compare_baseline:
        try:
            traditional = TraditionalRAGPipeline()
            trad_result = traditional.run(request.query)
            response.baseline_answer = trad_result["answer"]
        except Exception:
            response.baseline_answer = None

    return response


@app.get("/stats")
def stats():
    """Get vector database statistics."""
    pipeline = DocumentIngestionPipeline()
    return pipeline.get_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
