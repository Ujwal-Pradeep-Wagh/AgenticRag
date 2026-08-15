"""
agents/retrieval.py
Retrieval Agent

Responsibilities:
- Execute retrieval based on routing decision
- Perform vector search, hybrid search, or multi-query retrieval
- Return ranked documents with relevance scores
- Handle retrieval failures gracefully

Why this is needed:
This is the core information retrieval component. It translates the
rewritten query into actual document chunks from the vector database.
"""
import os
from typing import Dict, Any, List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from agents.base import BaseAgent


class RetrievalAgent(BaseAgent):
    """
    Agent that retrieves relevant documents from the vector store.

    Inputs:
        - query_rewrite: Dict with rewritten_query and variations
        - routing_decision: Dict with strategy and top_k

    Outputs:
        - retrieved_documents: List[Document] - Retrieved chunks
        - retrieval_metadata: Dict - Scores, strategy used, etc.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL,
            model_kwargs={"device": Config.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vector_store = Chroma(
            collection_name=Config.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=Config.CHROMA_PERSIST_DIR
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve documents based on routing strategy.

        Args:
            state: Current graph state

        Returns:
            Updated state with retrieved documents
        """
        query_rewrite = state.get("query_rewrite", {})
        routing = state.get("routing_decision", {})

        rewritten_query = query_rewrite.get("rewritten_query", state.get("query", ""))
        strategy = routing.get("strategy", "vector_search")
        top_k = routing.get("top_k", Config.TOP_K_RETRIEVAL)
        filters = routing.get("filters", {})

        print(f"[Retrieval] Executing with strategy: {strategy}")

        try:
            if strategy == "direct_answer":
                # No retrieval needed
                return {**state, "retrieved_documents": [], "retrieval_metadata": {
                    "strategy": "direct_answer",
                    "document_count": 0,
                    "reason": "Routing decided no retrieval needed"
                }}

            elif strategy == "multi_query":
                # Retrieve with multiple query variations
                documents = self._multi_query_retrieval(
                    rewritten_query, 
                    query_rewrite.get("query_variations", []),
                    top_k, 
                    filters
                )

            elif strategy == "hybrid_search":
                # Semantic + keyword hybrid (simplified: just semantic for now)
                documents = self._vector_search(rewritten_query, top_k, filters)

            else:  # vector_search (default)
                documents = self._vector_search(rewritten_query, top_k, filters)
                
                # Fallback: if rewritten query returns 0 results, try original query
                if not documents and rewritten_query != state.get("query", ""):
                    print(f"   WARNING: Rewritten query returned 0 results. Trying original query...")
                    original_query = state.get("query", "")
                    documents = self._vector_search(original_query, top_k, filters)
                    if documents:
                        print(f"   SUCCESS: Original query found {len(documents)} documents!")
                        rewritten_query = original_query  # Update for metadata

            # Add retrieval metadata
            metadata = {
                "strategy": strategy,
                "query_used": rewritten_query,
                "document_count": len(documents),
                "top_k_requested": top_k,
                "filters_applied": filters
            }

            print(f"   Retrieved {len(documents)} documents")

            return {**state, "retrieved_documents": documents, "retrieval_metadata": metadata}

        except Exception as e:
            print(f"   Retrieval error: {str(e)}")
            return {**state, "retrieved_documents": [], "retrieval_metadata": {
                "strategy": strategy,
                "error": str(e),
                "document_count": 0
            }}

    def _vector_search(self, query: str, top_k: int, filters: Dict) -> List[Document]:
        """Perform vector similarity search."""
        print(f"   Searching for: '{query}' (top_k={top_k})")
        
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=filters if filters else None
        )

        print(f"   Raw results count: {len(results)}")
        
        # Attach scores to metadata
        documents = []
        for doc, score in results:
            doc.metadata["retrieval_score"] = float(score)
            documents.append(doc)
            if len(documents) <= 2:  # Only print first 2
                print(f"     Doc: score={score:.4f}, content={doc.page_content[:60]}...")

        return documents

    def _multi_query_retrieval(self, primary_query: str, variations: List[str], 
                                top_k: int, filters: Dict) -> List[Document]:
        """Retrieve using multiple query variations and deduplicate."""
        all_docs = {}

        # Search with primary query
        for doc, score in self.vector_store.similarity_search_with_score(
            query=primary_query, k=top_k, filter=filters if filters else None
        ):
            doc_id = doc.metadata.get("chunk_id", hash(doc.page_content))
            all_docs[doc_id] = (doc, score)

        # Search with variations
        for variation in variations[:2]:  # Limit to 2 variations
            for doc, score in self.vector_store.similarity_search_with_score(
                query=variation, k=top_k//2, filter=filters if filters else None
            ):
                doc_id = doc.metadata.get("chunk_id", hash(doc.page_content))
                if doc_id not in all_docs or score < all_docs[doc_id][1]:
                    all_docs[doc_id] = (doc, score)

        # Sort by score and return top results
        sorted_docs = sorted(all_docs.values(), key=lambda x: x[1])
        documents = []
        for doc, score in sorted_docs[:top_k]:
            doc.metadata["retrieval_score"] = float(score)
            doc.metadata["multi_query"] = True
            documents.append(doc)

        return documents


if __name__ == "__main__":
    agent = RetrievalAgent()
    test_state = {
        "query": "What is the remote work policy?",
        "query_rewrite": {
            "rewritten_query": "remote work policy guidelines requirements"
        },
        "routing_decision": {
            "strategy": "vector_search",
            "top_k": 5
        }
    }
    result = agent.run(test_state)
    print(f"Retrieved {len(result.get('retrieved_documents', []))} documents")
