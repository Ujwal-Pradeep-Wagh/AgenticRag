"""
graph.py
LangGraph Workflow Definition

Defines the complete agent graph with nodes, edges, and conditional routing.
This is the orchestration layer that connects all agents.
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


# Initialize agents
query_understanding_agent = QueryUnderstandingAgent()
query_routing_agent = QueryRoutingAgent()
query_rewriting_agent = QueryRewritingAgent()
retrieval_agent = RetrievalAgent()
validation_agent = ValidationAgent()
response_generation_agent = ResponseGenerationAgent()
reflection_agent = ReflectionAgent()


def track_decision(state: AgentState, agent_name: str, decision: Dict) -> AgentState:
    """Track agent decisions for transparency."""
    decisions = state.get("agent_decisions", [])
    decisions.append({"agent": agent_name, "decision": decision})
    return {**state, "agent_decisions": decisions}


def node_query_understanding(state: AgentState) -> AgentState:
    """Node: Query Understanding"""
    result = query_understanding_agent.run(state)
    return track_decision(result, "QueryUnderstanding", result.get("query_understanding", {}))


def node_query_routing(state: AgentState) -> AgentState:
    """Node: Query Routing"""
    result = query_routing_agent.run(state)
    return track_decision(result, "QueryRouting", result.get("routing_decision", {}))


def node_query_rewriting(state: AgentState) -> AgentState:
    """Node: Query Rewriting"""
    result = query_rewriting_agent.run(state)
    return track_decision(result, "QueryRewriting", result.get("query_rewrite", {}))


def node_retrieval(state: AgentState) -> AgentState:
    """Node: Document Retrieval"""
    result = retrieval_agent.run(state)
    return track_decision(result, "Retrieval", result.get("retrieval_metadata", {}))


def node_validation(state: AgentState) -> AgentState:
    """Node: Context Validation"""
    result = validation_agent.run(state)
    return track_decision(result, "Validation", result.get("validation_result", {}))


def node_response_generation(state: AgentState) -> AgentState:
    """Node: Response Generation"""
    result = response_generation_agent.run(state)
    return track_decision(result, "ResponseGeneration", result.get("generation_metadata", {}))


def node_reflection(state: AgentState) -> AgentState:
    """Node: Reflection"""
    result = reflection_agent.run(state)
    return track_decision(result, "Reflection", result.get("reflection_result", {}))


# Conditional edge functions
def should_retrieve(state: AgentState) -> Literal["rewrite", "generate"]:
    """
    Decide if retrieval is needed based on routing decision.
    """
    routing = state.get("routing_decision", {})
    if routing.get("strategy") == "direct_answer":
        return "generate"
    return "rewrite"


def should_reretrieve(state: AgentState) -> Literal["rewrite", "generate"]:
    """
    Decide if re-retrieval is needed based on validation.
    """
    validation = state.get("validation_result", {})
    if validation.get("needs_reretrieval", False):
        return "rewrite"
    return "generate"


def should_regenerate(state: AgentState) -> Literal["regenerate", "finalize"]:
    """
    Decide if answer needs regeneration based on reflection.
    """
    # Limit iterations to prevent infinite loops
    if state.get("iteration_count", 0) >= 2:
        return "finalize"

    reflection = state.get("reflection_result", {})
    if reflection.get("needs_regeneration", False):
        return "regenerate"
    return "finalize"


# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("understand", node_query_understanding)
workflow.add_node("route", node_query_routing)
workflow.add_node("rewrite", node_query_rewriting)
workflow.add_node("retrieve", node_retrieval)
workflow.add_node("validate", node_validation)
workflow.add_node("generate", node_response_generation)
workflow.add_node("reflect", node_reflection)

# Add edges
workflow.set_entry_point("understand")
workflow.add_edge("understand", "route")

# Conditional routing after routing decision
workflow.add_conditional_edges(
    "route",
    should_retrieve,
    {
        "rewrite": "rewrite",
        "generate": "generate"
    }
)

workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("retrieve", "validate")

# Conditional routing after validation
workflow.add_conditional_edges(
    "validate",
    should_reretrieve,
    {
        "rewrite": "rewrite",
        "generate": "generate"
    }
)

workflow.add_edge("generate", "reflect")

# Conditional routing after reflection (self-correction loop)
workflow.add_conditional_edges(
    "reflect",
    should_regenerate,
    {
        "regenerate": "rewrite",
        "finalize": END
    }
)

# Compile with memory checkpointing
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)


if __name__ == "__main__":
    print("LangGraph workflow compiled successfully!")
    print("Nodes:", list(workflow.nodes.keys()))
