"""
tests/test_agents.py
Unit tests for all agents.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.query_understanding import QueryUnderstandingAgent
from agents.query_routing import QueryRoutingAgent
from agents.query_rewriting import QueryRewritingAgent
from agents.validation import ValidationAgent
from agents.reflection import ReflectionAgent
from agents.response_generation import ResponseGenerationAgent
from langchain_core.documents import Document


def test_query_understanding():
    """Test Query Understanding Agent."""
    agent = QueryUnderstandingAgent()
    state = {"query": "What is the remote work policy?"}
    result = agent.run(state)

    assert "query_understanding" in result
    assert result["query_understanding"]["intent"] is not None
    print("✅ Query Understanding Agent passed")


def test_query_routing():
    """Test Query Routing Agent."""
    agent = QueryRoutingAgent()
    state = {
        "query": "What is the remote work policy?",
        "query_understanding": {
            "intent": "factual",
            "complexity": "simple",
            "needs_retrieval": True
        }
    }
    result = agent.run(state)

    assert "routing_decision" in result
    assert result["routing_decision"]["strategy"] in [
        "direct_answer", "vector_search", "hybrid_search", "multi_query"
    ]
    print("✅ Query Routing Agent passed")


def test_query_rewriting():
    """Test Query Rewriting Agent."""
    agent = QueryRewritingAgent()
    state = {
        "query": "What's the deal with PTO?",
        "query_understanding": {"intent": "factual", "entities": ["PTO"]},
        "routing_decision": {"strategy": "vector_search"}
    }
    result = agent.run(state)

    assert "query_rewrite" in result
    assert "rewritten_query" in result["query_rewrite"]
    print("✅ Query Rewriting Agent passed")


def test_validation():
    """Test Validation Agent."""
    agent = ValidationAgent()
    state = {
        "query": "What is the remote work policy?",
        "retrieved_documents": [
            Document(page_content="Remote work policy allows 2 days WFH.", 
                     metadata={"source": "policy.pdf"})
        ]
    }
    result = agent.run(state)

    assert "validated_documents" in result
    assert "validation_result" in result
    print("✅ Validation Agent passed")


def test_reflection():
    """Test Reflection Agent."""
    agent = ReflectionAgent()
    state = {
        "query": "What is the remote work policy?",
        "generated_answer": "Employees can work remotely 2 days per week.",
        "validated_documents": [
            Document(page_content="Remote work: 2 days WFH per week.", 
                     metadata={"source": "policy.pdf"})
        ],
        "iteration_count": 0
    }
    result = agent.run(state)

    assert "reflection_result" in result
    assert "needs_regeneration" in result
    print("✅ Reflection Agent passed")


def test_response_generation():
    """Test Response Generation Agent."""
    agent = ResponseGenerationAgent()
    state = {
        "query": "What is the remote work policy?",
        "validated_documents": [
            Document(
                page_content="Remote work policy: 2 days WFH per week.",
                metadata={"source": "policy.pdf", "page_number": 1, "relevance_score": 0.9}
            )
        ],
        "query_understanding": {"intent": "factual"}
    }
    result = agent.run(state)

    assert "generated_answer" in result
    assert len(result["generated_answer"]) > 0
    print("✅ Response Generation Agent passed")


if __name__ == "__main__":
    print("Running agent tests...")
    test_query_understanding()
    test_query_routing()
    test_query_rewriting()
    test_validation()
    test_reflection()
    test_response_generation()
    print("\n🎉 All tests passed!")
