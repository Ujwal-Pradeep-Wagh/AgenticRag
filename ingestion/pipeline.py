"""
ingestion/pipeline.py
Document Ingestion Pipeline

Steps: Load PDF -> Chunk -> Embed -> Store in ChromaDB
Supports single file and bulk directory ingestion.
Includes deduplication to prevent re-ingesting the same file.
"""
import os
import hashlib
from typing import List, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class DocumentIngestionPipeline:
    """
    Ingests PDF documents into the ChromaDB vector store.

    Features:
    - MD5-based deduplication (skip already-ingested files)
    - Bulk directory ingestion
    - Larger chunks (1024 chars) with more overlap (200 chars) for better context
    - Rich metadata per chunk
    """

    def __init__(self):
        print("[Ingestion] Initializing embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={"device": Config.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        self.vector_store = None
        self._init_vector_store()

    def _init_vector_store(self):
        """Initialize or connect to existing ChromaDB."""
        os.makedirs(Config.CHROMA_PERSIST_DIR, exist_ok=True)
        self.vector_store = Chroma(
            collection_name=Config.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=Config.CHROMA_PERSIST_DIR
        )

    def _is_already_ingested(self, file_hash: str) -> bool:
        """
        Check if a file with this hash already exists in the vector store.
        Prevents duplicate ingestion on re-upload.
        """
        try:
            existing = self.vector_store._collection.get(
                where={"file_hash": file_hash},
                limit=1
            )
            return len(existing.get("ids", [])) > 0
        except Exception:
            return False

    def load_pdf(self, file_path: str) -> List[Document]:
        """Load a PDF and enrich each page with metadata."""
        print(f"[Ingestion] Loading: {file_path}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        file_name = Path(file_path).name
        file_hash = self._compute_file_hash(file_path)

        for i, doc in enumerate(documents):
            doc.metadata.update({
                "source": file_name,
                "file_path": file_path,
                "file_hash": file_hash,
                "page_number": doc.metadata.get("page", i + 1),
                "total_pages": len(documents),
                "document_type": "pdf"
            })

        print(f"[Ingestion] Loaded {len(documents)} pages")
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split pages into overlapping chunks with chunk-level metadata."""
        chunks = self.text_splitter.split_documents(documents)

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = (
                f"{chunk.metadata.get('file_hash', 'unknown')}_{i}"
            )
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_length"] = len(chunk.page_content)

        print(f"[Ingestion] Created {len(chunks)} chunks")
        return chunks

    def ingest(self, file_path: str) -> Dict[str, Any]:
        """
        Full ingestion pipeline for a single PDF file.

        Skips the file if it has already been ingested (deduplication).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_hash = self._compute_file_hash(file_path)

        # Deduplication check
        if self._is_already_ingested(file_hash):
            existing_count = self.vector_store._collection.count()
            print(f"[Ingestion] Skipping '{Path(file_path).name}' — already in DB.")
            return {
                "file": file_path,
                "status": "skipped",
                "reason": "already_ingested",
                "pages_loaded": 0,
                "chunks_created": 0,
                "collection_count": existing_count
            }

        documents = self.load_pdf(file_path)
        chunks = self.chunk_documents(documents)

        print(f"[Ingestion] Storing {len(chunks)} chunks...")
        self.vector_store.add_documents(chunks)

        total = self.vector_store._collection.count()
        print(f"[Ingestion] Done. Total docs in DB: {total}")

        return {
            "file": file_path,
            "status": "success",
            "pages_loaded": len(documents),
            "chunks_created": len(chunks),
            "collection_count": total
        }

    def ingest_multiple(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Ingest multiple PDF files in one call.
        Supports bulk upload from the frontend.
        """
        results = []
        for file_path in file_paths:
            try:
                result = self.ingest(file_path)
                results.append(result)
            except Exception as e:
                print(f"[Ingestion] Error ingesting '{file_path}': {e}")
                results.append({
                    "file": file_path,
                    "status": "error",
                    "error": str(e)
                })
        return results

    def ingest_directory(self, directory: str) -> List[Dict[str, Any]]:
        """Ingest all PDFs in a directory (bulk ingestion)."""
        pdf_files = list(Path(directory).glob("*.pdf"))
        print(f"[Ingestion] Found {len(pdf_files)} PDF(s) in '{directory}'")
        return self.ingest_multiple([str(p) for p in pdf_files])

    def get_stats(self) -> Dict[str, Any]:
        """Return current vector store statistics."""
        return {
            "total_documents": self.vector_store._collection.count(),
            "collection_name": Config.COLLECTION_NAME,
            "embedding_model": Config.EMBEDDING_MODEL,
            "chunk_size": Config.CHUNK_SIZE,
            "chunk_overlap": Config.CHUNK_OVERLAP,
            "persist_directory": Config.CHROMA_PERSIST_DIR
        }

    def clear_collection(self) -> Dict[str, Any]:
        """
        Delete all documents from the vector store.
        Use carefully — this is irreversible.
        """
        self.vector_store._collection.delete(
            where={"document_type": "pdf"}
        )
        return {"status": "cleared", "collection_count": self.vector_store._collection.count()}

    @staticmethod
    def _compute_file_hash(file_path: str) -> str:
        """MD5 hash for deduplication."""
        h = hashlib.md5()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                h.update(block)
        return h.hexdigest()[:16]


def ingest_document(file_path: str) -> Dict[str, Any]:
    """Convenience function for quick single-file ingestion."""
    return DocumentIngestionPipeline().ingest(file_path)


if __name__ == "__main__":
    pipeline = DocumentIngestionPipeline()
    print("Document Ingestion Pipeline Ready!")
    print(f"Stats: {pipeline.get_stats()}")
