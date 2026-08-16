"""
state.py
Graph State Definition

Shared state that flows through all agents.
"""

from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
    """
    Shared state for the Agentic RAG graph.

    Pipeline flow:
      query -> understand -> route -> rewrite -> retrieve -> validate -> generate -> reflect -> END
    """

    # ── Input ────────────────────────────────────────────────────────────────
    query: str
    conversation_history: List[Dict[str, str]]

    # ── Agent outputs ────────────────────────────────────────────────────────
    query_understanding: Optional[Dict[str, Any]]
    routing_decision: Optional[Dict[str, Any]]
    query_rewrite: Optional[Dict[str, Any]]
    retrieved_documents: List[Document]
    validated_documents: List[Document]
    generated_answer: str
    reflection_result: Optional[Dict[str, Any]]
    generation_metadata: Optional[Dict[str, Any]]
    validation_result: Optional[Dict[str, Any]]
    retrieval_metadata: Optional[Dict[str, Any]]

    # ── Control flow ─────────────────────────────────────────────────────────
    needs_regeneration: bool
    # Feedback from ReflectionAgent passed into QueryRewritingAgent on iteration 2
    improvement_feedback: str
    iteration_count: int

    # ── Final output ─────────────────────────────────────────────────────────
    final_answer: str
    agent_decisions: List[Dict[str, Any]]
    error: Optional[str]
