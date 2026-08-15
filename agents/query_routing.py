"""
agents/query_routing.py
Query Routing Agent

Responsibilities:
- Decide retrieval strategy based on query understanding
- Choose between: direct_answer, vector_search, hybrid_search, no_retrieval
- Set retrieval parameters (top_k, filters)

Why this is needed:
Not all queries need the same retrieval approach. Simple factual queries
might need fewer documents. Complex analytical queries need more context.
This agent optimizes the retrieval strategy before execution.
"""
import json
from typing import Dict, Any
from agents.base import BaseAgent


class QueryRoutingAgent(BaseAgent):
    """
    Agent that decides the optimal retrieval strategy.

    Inputs:
        - query: str
        - query_understanding: Dict with intent, complexity, etc.

    Outputs:
        - strategy: str - Retrieval strategy to use
        - top_k: int - Number of documents to retrieve
        - filters: Dict - Metadata filters for retrieval
        - reasoning: str - Why this strategy was chosen
    """

    SYSTEM_PROMPT = """You are a Query Routing Agent in an Agentic RAG system.
Your job is to decide the best retrieval strategy for a given query.

Available strategies:
- "direct_answer": Query is conversational or general knowledge, no document retrieval needed
- "vector_search": Standard semantic search over documents
- "hybrid_search": Combine semantic + keyword search (for queries with specific terms)
- "multi_query": Generate multiple search queries for complex questions

Based on the query understanding, decide:
1. Which strategy to use
2. How many documents to retrieve (top_k: 3-10)
3. Any metadata filters to apply

Return ONLY a JSON object:
{
    "strategy": "vector_search",
    "top_k": 5,
    "filters": {},
    "reasoning": "Brief explanation of why this strategy was chosen"
}"""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route the query to the appropriate retrieval strategy.

        Args:
            state: Current graph state with query and understanding

        Returns:
            Updated state with 'routing_decision'
        """
        query = state.get("query", "")
        understanding = state.get("query_understanding", {})

        # Build context for routing decision
        context = f"""Query: "{query}"

Query Understanding:
- Intent: {understanding.get('intent', 'unknown')}
- Complexity: {understanding.get('complexity', 'medium')}
- Needs Retrieval: {understanding.get('needs_retrieval', True)}
- Entities: {understanding.get('entities', [])}
- Domain Hints: {understanding.get('domain_hints', [])}
"""

        try:
            response = self._invoke_llm(context, self.SYSTEM_PROMPT)
            decision = json.loads(response.strip())

            # Validate and set defaults
            valid_strategies = ["direct_answer", "vector_search", "hybrid_search", "multi_query"]
            strategy = decision.get("strategy", "vector_search")
            if strategy not in valid_strategies:
                strategy = "vector_search"

            decision["strategy"] = strategy
            decision.setdefault("top_k", 5)
            decision.setdefault("filters", {})
            decision.setdefault("reasoning", "Default routing")

            print(f"[Routing] strategy={decision['strategy']}, top_k={decision['top_k']}")
            print(f"   Reasoning: {decision['reasoning']}")

            return {**state, "routing_decision": decision}

        except Exception as e:
            # Fallback to safe defaults
            fallback = {
                "strategy": "vector_search",
                "top_k": 5,
                "filters": {},
                "reasoning": f"Fallback due to error: {str(e)}"
            }
            return {**state, "routing_decision": fallback}


if __name__ == "__main__":
    agent = QueryRoutingAgent()
    test_state = {
        "query": "Compare our 2023 and 2024 revenue figures",
        "query_understanding": {
            "intent": "comparative",
            "complexity": "complex",
            "needs_retrieval": True
        }
    }
    result = agent.run(test_state)
    print(json.dumps(result["routing_decision"], indent=2))
