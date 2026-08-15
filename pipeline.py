"""
pipeline.py
Complete Agentic RAG Pipeline

This is the main entry point. It runs the full graph and returns results.
Also includes the Traditional RAG baseline for comparison.
"""

import os
from typing import Dict, Any, Optional

from state import AgentState
from graph import graph
from config import Config


class AgenticRAGPipeline:
    """
    Main pipeline for the Agentic RAG system.

    Usage:
        pipeline = AgenticRAGPipeline()
        result = pipeline.run("What is the remote work policy?")
    """

    def __init__(self):
        self.graph = graph

    def run(self, query: str, conversation_history: Optional[list] = None) -> Dict[str, Any]:
        """
        Run the complete Agentic RAG pipeline.

        Args:
            query: User question
            conversation_history: Optional previous conversation turns

        Returns:
            Dictionary with final answer and full trace
        """
        # Initialize state
        initial_state: AgentState = {
            "query": query,
            "conversation_history": conversation_history or [],
            "iteration_count": 0,
            "agent_decisions": []
        }

        print(f"\n{'='*60}")
        print(f"Agentic RAG Pipeline Started")
        print(f"Query: {query}")
        print(f"{'='*60}\n")

        # Run the graph
        config = {"configurable": {"thread_id": "1"}}

        for event in self.graph.stream(initial_state, config):
            if "__end__" not in event:
                for node_name, node_state in event.items():
                    print(f"   [Node: {node_name}]")

        # Get final state
        final_state = self.graph.get_state(config)

        # Extract results
        result = {
            "query": query,
            "final_answer": final_state.values.get("generated_answer", "No answer generated"),
            "agent_decisions": final_state.values.get("agent_decisions", []),
            "retrieved_documents": [
                {
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page_number", "N/A"),
                    "score": doc.metadata.get("relevance_score", 0)
                }
                for doc in final_state.values.get("validated_documents", [])
            ],
            "reflection": final_state.values.get("reflection_result", {}),
            "iterations": final_state.values.get("iteration_count", 0)
        }

        print(f"\n{'='*60}")
        print(f"Pipeline Complete")
        print(f"Iterations: {result['iterations']}")
        print(f"{'='*60}\n")

        return result


# Baseline Traditional RAG (for comparison)
class TraditionalRAGPipeline:
    """
    Simple baseline RAG without agentic features.
    Used for experimental comparison to prove agentic improvements.
    """

    def __init__(self):
        from ingestion.pipeline import DocumentIngestionPipeline
        from langchain_groq import ChatGroq

        self.embeddings = DocumentIngestionPipeline().embeddings
        self.vector_store = DocumentIngestionPipeline().vector_store
        self.llm = ChatGroq(
            model=Config.LLM_MODEL,
            temperature=0.1,
            api_key=Config.GROQ_API_KEY
        )

    def run(self, query: str) -> Dict[str, Any]:
        """Simple retrieve-then-generate pipeline."""
        # Direct retrieval
        docs = self.vector_store.similarity_search(query, k=5)

        # Simple prompt
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        answer = self.llm.invoke(prompt).content

        return {
            "query": query,
            "answer": answer,
            "documents_retrieved": len(docs),
            "method": "traditional"
        }


if __name__ == "__main__":
    print("Agentic RAG Pipeline Ready!")
    print("Usage:")
    print("  from pipeline import AgenticRAGPipeline")
    print("  p = AgenticRAGPipeline()")
    print("  result = p.run('Your question here')")
