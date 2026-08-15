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
    # Other options: llama-3.3-70b-versatile, qwen/qwen3-32b, openai/gpt-oss-120b
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
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    TOP_K_RERANK = 3

    # Chunking
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Agent Settings
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    CONFIDENCE_THRESHOLD = 0.7
    MAX_ITERATIONS = 2

    # Paths
    DATA_DIR = "./data"
    UPLOADS_DIR = "./data/uploads"

    @classmethod
    def validate(cls):
        if not cls.GROQ_API_KEY and not cls.OPENROUTER_API_KEY:
            raise ValueError("Either GROQ_API_KEY or OPENROUTER_API_KEY must be set")
        return True
