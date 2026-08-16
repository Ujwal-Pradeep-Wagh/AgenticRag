"""
Agentic RAG - Central Configuration
Manages all project settings, models, and environment variables.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central configuration for the Agentic RAG system."""

    # LLM Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    # Default to llama-3.1-8b-instant (fast and free tier compatible)
    LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    LLM_TEMPERATURE = 0.1
    MAX_TOKENS = 4096

    # Embedding Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    EMBEDDING_DEVICE = "cpu"

    # Vector DB
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./vector_db/chroma_store")
    COLLECTION_NAME = "enterprise_docs"

    # Retrieval Settings
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "6"))
    TOP_K_RERANK = 4

    # Chunking — larger chunks preserve context better for BGE models
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Agent Settings
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    # Lower threshold: validation was over-filtering with 0.7
    CONFIDENCE_THRESHOLD = 0.5
    # Only do one reflection iteration to keep latency down
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "1"))

    # Validation: documents scoring below this are filtered
    VALIDATION_THRESHOLD = 0.4

    # Paths
    DATA_DIR = os.getenv("DATA_DIR", "./data")
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads")

    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration."""
        if not cls.GROQ_API_KEY and not cls.OPENROUTER_API_KEY:
            print("ERROR: No LLM API key configured. Set GROQ_API_KEY or OPENROUTER_API_KEY in .env")
            return False
        return True
