"""
agents/retrieval.py
Retrieval Agent

Executes vector search and returns ranked documents.
"""
import os
from typing import Dict, Any, List, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from agents.base import BaseAgent

# Module-level singleton for embeddings — avoids re-loading the model on every agent init
_embeddings_instance = None
_vector_store_instance = None


def _get_vector_store():
    """Return a shared vector store instance (loaded once per process)."""
    global _embeddings_instance, _vector_store_instance
    if _embeddings_instance is None:
        print("[Retrieval] Loading embedding model (first time)...")
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={"device": Config.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True}
        )
    if _vector_store_instance is None:
        _vector_store_instance = Chroma(
            collection_name=Config.COLLECTION_NAME,
            embedding_function=_embeddings_instance,
            persist_directory=Config.CHROMA_PERSIST_DIR
        )
    return _vector_store_instance


class RetrievalAgent(BaseAgent):
    """
    Retrieves relevant documents from the vector store.

    Supports: vector_search, multi_query, hybrid_search strategies.
    Falls back to original query if rewritten query returns 0 results.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use the shared singleton — no per-instance model loading
        self.vector_store = _get_vector_store()

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query_rewrite = state.get("query_rewrite", {})
        routing = state.get("routing_decision", {})
        original_query = state.get("query", "")

        rewritten_query = query_rewrite.get("rewritten_query", original_query)
        strategy = routing.get("strategy", "vector_search")
        top_k = routing.get("top_k", Config.TOP_K_RETRIEVAL)
        filters = routing.get("filters") or {}

        print(f"[Retrieval] strategy={strategy}, query='{rewritten_query}', top_k={top_k}")

        try:
            if strategy == "direct_answer":
                return {**state, "retrieved_documents": [], "retrieval_metadata": {
                    "strategy": "direct_answer",
                    "document_count": 0
                }}

            elif strategy == "multi_query":
                variations = query_rewrite.get("query_variations", [])
                documents = self._multi_query_retrieval(rewritten_query, variations, top_k, filters)

            else:
                # vector_search or hybrid_search (both use semantic search)
                documents = self._vector_search(rewritten_query, top_k, filters)

            # Fallback: if rewritten query found nothing, try original
            if not documents and rewritten_query != original_query:
                print(f"[Retrieval] Rewritten query found nothing. Falling back to original.")
                documents = self._vector_search(original_query, top_k, filters)
                if documents:
                    rewritten_query = original_query

            # Second fallback: if original query also found nothing, try key entities
            if not documents:
                entities = state.get("query_understanding", {}).get("entities", [])
                if entities:
                    entity_query = " ".join(entities[:4])
                    print(f"[Retrieval] Trying entity-based query: '{entity_query}'")
                    documents = self._vector_search(entity_query, top_k, filters)

            print(f"[Retrieval] Found {len(documents)} documents")

            return {**state, "retrieved_documents": documents, "retrieval_metadata": {
                "strategy": strategy,
                "query_used": rewritten_query,
                "document_count": len(documents),
                "top_k_requested": top_k
            }}

        except Exception as e:
            print(f"[Retrieval] Error: {e}")
            return {**state, "retrieved_documents": [], "retrieval_metadata": {
                "strategy": strategy,
                "error": str(e),
                "document_count": 0
            }}

    def _vector_search(self, query: str, top_k: int, filters: Dict) -> List[Document]:
        """Semantic similarity search."""
        results: List[Tuple[Document, float]] = self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=filters if filters else None
        )
        documents = []
        for doc, score in results:
            # ChromaDB returns L2 distance — lower = more similar.
            # Convert to similarity score 0-1 for consistency.
            similarity = max(0.0, 1.0 - float(score))
            doc.metadata["retrieval_score"] = round(similarity, 4)
            documents.append(doc)
        return documents

    def _multi_query_retrieval(self, primary: str, variations: List[str],
                                top_k: int, filters: Dict) -> List[Document]:
        """Search with primary + variation queries, deduplicate by chunk_id."""
        seen: Dict[str, Tuple[Document, float]] = {}

        def _search_and_collect(q: str, k: int):
            for doc, score in self.vector_store.similarity_search_with_score(
                query=q, k=k, filter=filters if filters else None
            ):
                doc_id = doc.metadata.get("chunk_id", doc.page_content[:80])
                if doc_id not in seen or score < seen[doc_id][1]:
                    seen[doc_id] = (doc, score)

        _search_and_collect(primary, top_k)
        for variation in variations[:2]:
            if variation and variation != primary:
                _search_and_collect(variation, max(2, top_k // 2))

        # Sort by score ascending (lower L2 = better), return top_k
        sorted_docs = sorted(seen.values(), key=lambda x: x[1])
        documents = []
        for doc, score in sorted_docs[:top_k]:
            similarity = max(0.0, 1.0 - float(score))
            doc.metadata["retrieval_score"] = round(similarity, 4)
            doc.metadata["multi_query"] = True
            documents.append(doc)
        return documents
