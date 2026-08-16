"""
pipeline.py
Complete Agentic RAG Pipeline and Traditional RAG Baseline
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

    def run(self, query: str, conversation_history: Optional[list] = None,
            thread_id: str = "default") -> Dict[str, Any]:
        """
        Run the complete Agentic RAG pipeline.

        Args:
            query: User question
            conversation_history: Optional previous conversation turns
            thread_id: Unique thread ID for stateful memory (use different IDs for different users)

        Returns:
            Dictionary with final_answer, agent_decisions, retrieved_documents, reflection, iterations
        """
        initial_state: AgentState = {
            "query": query,
            "conversation_history": conversation_history or [],
            "iteration_count": 0,
            "agent_decisions": [],
            "improvement_feedback": ""
        }

        print(f"\n{'='*60}")
        print(f"Agentic RAG Pipeline Started")
        print(f"Query: {query}")
        print(f"{'='*60}\n")

        config = {"configurable": {"thread_id": thread_id}}

        # Stream through graph nodes
        for event in self.graph.stream(initial_state, config):
            if "__end__" not in event:
                for node_name in event:
                    print(f"   [Completed: {node_name}]")

        # Get final state
        final_state = self.graph.get_state(config)
        values = final_state.values

        # final_answer is set by response_generation and confirmed by reflection
        # Fall back to generated_answer if final_answer is somehow empty
        final_answer = (
            values.get("final_answer") or
            values.get("generated_answer") or
            "No answer generated."
        )

        result = {
            "query": query,
            "final_answer": final_answer,
            "agent_decisions": values.get("agent_decisions", []),
            "retrieved_documents": [
                {
                    "content": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page_number", "N/A"),
                    "relevance_score": doc.metadata.get("relevance_score", 0),
                    "retrieval_score": doc.metadata.get("retrieval_score", 0)
                }
                for doc in values.get("validated_documents", [])
            ],
            "reflection": values.get("reflection_result", {}),
            "iterations": values.get("iteration_count", 0),
            "routing_strategy": values.get("routing_decision", {}).get("strategy", "unknown")
        }

        print(f"\n{'='*60}")
        print(f"Pipeline Complete — Iterations: {result['iterations']}, "
              f"Docs used: {len(result['retrieved_documents'])}")
        print(f"{'='*60}\n")

        return result


class TraditionalRAGPipeline:
    """
    Simple baseline RAG without agentic features.
    Used for comparison to demonstrate agentic improvements.

    Uses the shared embedding instance from RetrievalAgent to avoid
    loading the model twice.
    """

    def __init__(self):
        from agents.retrieval import _get_vector_store
        from langchain_groq import ChatGroq

        # Reuse the shared vector store (avoids duplicate embedding model loading)
        self.vector_store = _get_vector_store()

        self.llm = ChatGroq(
            model=Config.LLM_MODEL,
            temperature=0.1,
            api_key=Config.GROQ_API_KEY
        )

    def run(self, query: str) -> Dict[str, Any]:
        """Simple retrieve-then-generate pipeline with a proper system prompt."""
        from langchain_core.messages import SystemMessage, HumanMessage

        docs = self.vector_store.similarity_search(query, k=5)

        if not docs:
            return {
                "query": query,
                "answer": "No relevant documents found for this query.",
                "documents_retrieved": 0,
                "method": "traditional"
            }

        # Build context
        context_parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page_number", "N/A")
            context_parts.append(f"[Source {i+1}] {source} (page {page}):\n{doc.page_content}")

        context = "\n\n---\n\n".join(context_parts)

        system_prompt = (
            "You are a helpful assistant. Answer the user's question based strictly on the "
            "provided documents. Be accurate, complete, and cite sources as [Source N]. "
            "If the documents don't contain the answer, say so clearly."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}")
        ]

        answer = self.llm.invoke(messages).content

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
