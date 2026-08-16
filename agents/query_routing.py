"""
agents/query_routing.py
Query Routing Agent

Decides retrieval strategy and parameters based on query understanding.
"""
import json
from typing import Dict, Any
from agents.base import BaseAgent
from config import Config


class QueryRoutingAgent(BaseAgent):
    """
    Decides the optimal retrieval strategy.

    Outputs:
        - strategy: vector_search | multi_query | hybrid_search | direct_answer
        - top_k: number of documents to retrieve
        - filters: metadata filters
        - reasoning: explanation
    """

    SYSTEM_PROMPT = """You are a Query Routing Agent in an Agentic RAG system.
Decide the retrieval strategy for the query.

IMPORTANT: Default to vector_search. Only use direct_answer for genuinely trivial 
greetings ("hello", "thanks") or obvious general knowledge that has NOTHING to do 
with company/enterprise documents.

Strategies:
- "vector_search": Standard semantic search (default for most queries)
- "multi_query": Use for complex or multi-part questions needing broader coverage
- "hybrid_search": Semantic + keyword (for queries with very specific terms/names/IDs)
- "direct_answer": ONLY for greetings or pure general knowledge unrelated to any documents

top_k guidance:
- simple factual: 4
- medium: 6  
- complex/comparative: 8

Return ONLY valid JSON, no markdown:
{
    "strategy": "vector_search",
    "top_k": 6,
    "filters": {},
    "reasoning": "Standard semantic search for factual policy query"
}"""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("query", "")
        understanding = state.get("query_understanding", {})

        context = f"""Query: "{query}"

Query Understanding:
- Intent: {understanding.get('intent', 'factual')}
- Complexity: {understanding.get('complexity', 'medium')}
- Needs Retrieval: {understanding.get('needs_retrieval', True)}
- Entities: {understanding.get('entities', [])}
- Domain Hints: {understanding.get('domain_hints', [])}

Choose the best retrieval strategy."""

        try:
            response = self._invoke_llm(context, self.SYSTEM_PROMPT)
            raw = self._extract_json(response)
            decision = json.loads(raw)

            valid_strategies = ["direct_answer", "vector_search", "hybrid_search", "multi_query"]
            strategy = decision.get("strategy", "vector_search")
            if strategy not in valid_strategies:
                strategy = "vector_search"

            # Safety override: if query_understanding says retrieval is needed,
            # don't allow direct_answer routing
            if understanding.get("needs_retrieval", True) and strategy == "direct_answer":
                print(f"[Routing] Overriding direct_answer: query_understanding says retrieval needed")
                strategy = "vector_search"

            decision["strategy"] = strategy
            decision.setdefault("top_k", Config.TOP_K_RETRIEVAL)
            decision.setdefault("filters", {})
            decision.setdefault("reasoning", "Default routing")

            # Cap top_k to reasonable range
            decision["top_k"] = max(3, min(10, decision["top_k"]))

            print(f"[Routing] strategy={decision['strategy']}, top_k={decision['top_k']}")

            return {**state, "routing_decision": decision}

        except Exception as e:
            print(f"[Routing] Error: {e}. Using fallback.")
            fallback = {
                "strategy": "vector_search",
                "top_k": Config.TOP_K_RETRIEVAL,
                "filters": {},
                "reasoning": f"Fallback: {str(e)}"
            }
            return {**state, "routing_decision": fallback}
