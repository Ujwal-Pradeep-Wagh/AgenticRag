"""
state.py
Graph State Definition

Defines the shared state that flows through all agents in the graph.
Each agent reads from and writes to this state.
"""

from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
    """
    Shared state for the Agentic RAG graph.

    Fields flow through the pipeline:
    1. query -> user input
    2. query_understanding -> from QueryUnderstandingAgent
    3. routing_decision -> from QueryRoutingAgent
    4. query_rewrite -> from QueryRewritingAgent
    5. retrieved_documents -> from RetrievalAgent
    6. validated_documents -> from ValidationAgent
    7. generated_answer -> from ResponseGenerationAgent
    8. reflection_result -> from ReflectionAgent
    9. final_answer -> final output after reflection loop
    """

    # Input
    query: str
    conversation_history: List[Dict[str, str]]

    # Agent outputs
    query_understanding: Optional[Dict[str, Any]]
    routing_decision: Optional[Dict[str, Any]]
    query_rewrite: Optional[Dict[str, Any]]
    retrieved_documents: Optional[List[Document]]
    validated_documents: Optional[List[Document]]
    generated_answer: Optional[str]
    reflection_result: Optional[Dict[str, Any]]

    # Control flow
    needs_regeneration: bool
    iteration_count: int
    error: Optional[str]

    # Final output
    final_answer: Optional[str]
    agent_decisions: List[Dict[str, Any]]
