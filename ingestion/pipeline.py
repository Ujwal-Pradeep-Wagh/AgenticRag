"""
ingestion/pipeline.py
Complete document ingestion pipeline.
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
    Pipeline for ingesting enterprise documents into the vector database.

    Steps:
    1. Load PDF documents
    2. Extract text and metadata
    3. Chunk documents intelligently
    4. Generate embeddings
    5. Store in ChromaDB
    """

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={"device": Config.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.vector_store = None
        self._init_vector_store()

    def _init_vector_store(self):
        """Initialize or load existing ChromaDB vector store."""
        os.makedirs(Config.CHROMA_PERSIST_DIR, exist_ok=True)
        self.vector_store = Chroma(
            collection_name=Config.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=Config.CHROMA_PERSIST_DIR
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a PDF file and extract documents with metadata.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of Document objects with page content and metadata
        """
        print(f"Loading PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Enrich metadata
        file_name = Path(file_path).name
        file_hash = self._compute_file_hash(file_path)

        for i, doc in enumerate(documents):
            doc.metadata.update({
                "source": file_name,
                "file_path": file_path,
                "file_hash": file_hash,
                "page_number": doc.metadata.get("page", i + 1),
                "chunk_index": i,
                "total_pages": len(documents),
                "document_type": "pdf"
            })

        print(f"   Loaded {len(documents)} pages")
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into semantically meaningful chunks.

        Why chunking matters:
        - LLMs have context limits
        - Smaller chunks improve retrieval precision
        - Overlap preserves context across chunk boundaries

        Args:
            documents: List of Document objects

        Returns:
            List of chunked Document objects
        """
        print(f"Chunking {len(documents)} documents...")
        chunks = self.text_splitter.split_documents(documents)

        # Add chunk-level metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"{chunk.metadata.get('file_hash', 'unknown')}_{i}"
            chunk.metadata["chunk_length"] = len(chunk.page_content)

        print(f"   Created {len(chunks)} chunks")
        return chunks

    def ingest(self, file_path: str) -> Dict[str, Any]:
        """
        Complete ingestion pipeline for a single file.

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary with ingestion results and statistics
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Step 1: Load
        documents = self.load_pdf(file_path)

        # Step 2: Chunk
        chunks = self.chunk_documents(documents)

        # Step 3: Store in vector DB
        print(f"Storing {len(chunks)} chunks in vector database...")
        self.vector_store.add_documents(chunks)

        result = {
            "file": file_path,
            "pages_loaded": len(documents),
            "chunks_created": len(chunks),
            "status": "success",
            "collection_count": self.vector_store._collection.count()
        }

        print(f"   Ingestion complete! Total docs in DB: {result['collection_count']}")
        return result

    def ingest_directory(self, directory: str) -> List[Dict[str, Any]]:
        """Ingest all PDFs in a directory."""
        results = []
        pdf_files = list(Path(directory).glob("*.pdf"))

        print(f"Found {len(pdf_files)} PDF files in {directory}")

        for pdf_file in pdf_files:
            try:
                result = self.ingest(str(pdf_file))
                results.append(result)
            except Exception as e:
                results.append({
                    "file": str(pdf_file),
                    "status": "error",
                    "error": str(e)
                })

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get current vector store statistics."""
        return {
            "total_documents": self.vector_store._collection.count(),
            "collection_name": Config.COLLECTION_NAME,
            "embedding_model": Config.EMBEDDING_MODEL,
            "persist_directory": Config.CHROMA_PERSIST_DIR
        }

    @staticmethod
    def _compute_file_hash(file_path: str) -> str:
        """Compute MD5 hash of file for deduplication."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()[:12]


# Convenience function for quick ingestion
def ingest_document(file_path: str) -> Dict[str, Any]:
    """Quick ingest a single document."""
    pipeline = DocumentIngestionPipeline()
    return pipeline.ingest(file_path)


if __name__ == "__main__":
    # Example usage
    pipeline = DocumentIngestionPipeline()
    print("Document Ingestion Pipeline Ready!")
    print(f"Stats: {pipeline.get_stats()}")
