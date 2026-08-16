"""
graph.py
LangGraph Workflow Definition

Defines the complete agent graph with conditional routing.
"""

from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
from agents.query_understanding import QueryUnderstandingAgent
from agents.query_routing import QueryRoutingAgent
from agents.query_rewriting import QueryRewritingAgent
from agents.retrieval import RetrievalAgent
from agents.validation import ValidationAgent
from agents.response_generation import ResponseGenerationAgent
from agents.reflection import ReflectionAgent
from config import Config


# Initialize agents once at module load (shared across requests)
query_understanding_agent = QueryUnderstandingAgent()
query_routing_agent = QueryRoutingAgent()
query_rewriting_agent = QueryRewritingAgent()
retrieval_agent = RetrievalAgent()
validation_agent = ValidationAgent()
response_generation_agent = ResponseGenerationAgent()
reflection_agent = ReflectionAgent()


def _track(state: AgentState, agent_name: str, decision: Dict) -> AgentState:
    """Append agent decision to the audit trail."""
    decisions = list(state.get("agent_decisions", []))
    decisions.append({"agent": agent_name, "decision": decision})
    return {**state, "agent_decisions": decisions}


def node_query_understanding(state: AgentState) -> AgentState:
    result = query_understanding_agent.run(state)
    return _track(result, "QueryUnderstanding", result.get("query_understanding", {}))


def node_query_routing(state: AgentState) -> AgentState:
    result = query_routing_agent.run(state)
    return _track(result, "QueryRouting", result.get("routing_decision", {}))


def node_query_rewriting(state: AgentState) -> AgentState:
    result = query_rewriting_agent.run(state)
    return _track(result, "QueryRewriting", result.get("query_rewrite", {}))


def node_retrieval(state: AgentState) -> AgentState:
    result = retrieval_agent.run(state)
    return _track(result, "Retrieval", result.get("retrieval_metadata", {}))


def node_validation(state: AgentState) -> AgentState:
    result = validation_agent.run(state)
    return _track(result, "Validation", result.get("validation_result", {}))


def node_response_generation(state: AgentState) -> AgentState:
    result = response_generation_agent.run(state)
    return _track(result, "ResponseGeneration", result.get("generation_metadata", {}))


def node_reflection(state: AgentState) -> AgentState:
    result = reflection_agent.run(state)
    return _track(result, "Reflection", result.get("reflection_result", {}))


# ── Conditional edge functions ──────────────────────────────────────────────

def should_retrieve(state: AgentState) -> Literal["rewrite", "generate"]:
    """Skip retrieval only for direct_answer strategy."""
    routing = state.get("routing_decision", {})
    if routing.get("strategy") == "direct_answer":
        return "generate"
    return "rewrite"


def should_reretrieve(state: AgentState) -> Literal["rewrite", "generate"]:
    """
    Re-retrieve only if validation says quality is low AND
    we haven't exceeded iteration limit (prevents infinite loops).
    """
    if state.get("iteration_count", 0) >= Config.MAX_ITERATIONS:
        return "generate"
    validation = state.get("validation_result", {})
    if validation.get("needs_reretrieval", False):
        return "rewrite"
    return "generate"


def should_regenerate(state: AgentState) -> Literal["regenerate", "finalize"]:
    """
    Regenerate if reflection found problems AND iteration limit not reached.
    """
    if state.get("iteration_count", 0) >= Config.MAX_ITERATIONS:
        return "finalize"
    reflection = state.get("reflection_result", {})
    if reflection.get("needs_regeneration", False):
        return "regenerate"
    return "finalize"


# ── Build the graph ──────────────────────────────────────────────────────────

workflow = StateGraph(AgentState)

workflow.add_node("understand", node_query_understanding)
workflow.add_node("route", node_query_routing)
workflow.add_node("rewrite", node_query_rewriting)
workflow.add_node("retrieve", node_retrieval)
workflow.add_node("validate", node_validation)
workflow.add_node("generate", node_response_generation)
workflow.add_node("reflect", node_reflection)

workflow.set_entry_point("understand")
workflow.add_edge("understand", "route")

workflow.add_conditional_edges(
    "route",
    should_retrieve,
    {"rewrite": "rewrite", "generate": "generate"}
)

workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("retrieve", "validate")

workflow.add_conditional_edges(
    "validate",
    should_reretrieve,
    {"rewrite": "rewrite", "generate": "generate"}
)

workflow.add_edge("generate", "reflect")

workflow.add_conditional_edges(
    "reflect",
    should_regenerate,
    {"regenerate": "rewrite", "finalize": END}
)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)


if __name__ == "__main__":
    print("LangGraph workflow compiled successfully!")
    print("Nodes:", list(workflow.nodes.keys()))
