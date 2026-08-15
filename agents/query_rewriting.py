"""
agents/query_rewriting.py
Query Rewriting Agent

Responsibilities:
- Improve the original query for better retrieval
- Expand acronyms and ambiguous terms
- Generate multiple query variations for better coverage
- Handle conversational context

Why this is needed:
User queries are often vague, ambiguous, or conversational. Rewriting them
into search-optimized forms dramatically improves retrieval quality.
This is one of the highest-impact optimizations in RAG.
"""
import json
from typing import Dict, Any, List
from agents.base import BaseAgent


class QueryRewritingAgent(BaseAgent):
    """
    Agent that rewrites queries for optimal retrieval.

    Inputs:
        - query: str - Original user query
        - query_understanding: Dict - Understanding from previous agent
        - routing_decision: Dict - Strategy from router
        - conversation_history: List[Dict] - Previous turns (optional)

    Outputs:
        - rewritten_query: str - Optimized search query
        - query_variations: List[str] - Alternative formulations
        - expansion_terms: List[str] - Added context/terms
    """

    SYSTEM_PROMPT = """You are a Query Rewriting Agent in an Agentic RAG system.
Your job is to transform user queries into search-optimized forms.

Rules:
1. Keep the query simple and focused on KEY TERMS
2. Remove conversational fluff ("Can you tell me", "I want to know", "What is")
3. Expand well-known acronyms (e.g., "HR" to "Human Resources")
4. Add missing context from conversation history if available
5. DO NOT add extra qualifying words like "definition", "explanation", "details" unless explicitly in the original query
6. DO NOT over-complicate - simpler is often better for vector search
7. If the query is already clear and concise, return it as-is or with minimal changes

Return ONLY a JSON object:
{
    "rewritten_query": "The optimized search query (keep it simple!)",
    "query_variations": ["Alternative 1", "Alternative 2"],
    "expansion_terms": ["term1", "term2"],
    "reasoning": "Brief explanation of changes made"
}

Example:
Input: "What is the company's remote work policy?"
Good output: "remote work policy"
Bad output: "company remote work policy definition guidelines"
"""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rewrite the query for better retrieval.

        Args:
            state: Current graph state

        Returns:
            Updated state with rewritten query information
        """
        query = state.get("query", "")
        understanding = state.get("query_understanding", {})
        routing = state.get("routing_decision", {})
        history = state.get("conversation_history", [])

        # Build context
        context = f"""Original Query: "{query}"

Query Understanding:
- Intent: {understanding.get('intent', 'unknown')}
- Entities: {understanding.get('entities', [])}
- Complexity: {understanding.get('complexity', 'medium')}

Routing Strategy: {routing.get('strategy', 'vector_search')}

"""

        if history:
            context += "\nConversation History:\n"
            for i, turn in enumerate(history[-3:]):  # Last 3 turns
                context += f"{turn.get('role', 'user')}: {turn.get('content', '')}\n"

        try:
            response = self._invoke_llm(context, self.SYSTEM_PROMPT)
            rewrite_info = json.loads(response.strip())

            # Validate
            rewrite_info.setdefault("rewritten_query", query)
            rewrite_info.setdefault("query_variations", [query])
            rewrite_info.setdefault("expansion_terms", [])
            rewrite_info.setdefault("reasoning", "No changes needed")

            # Use rewritten query as primary, but keep original
            print(f"[Query Rewriting] Original: {query}")
            print(f"   Rewritten: {rewrite_info['rewritten_query']}")
            if rewrite_info['query_variations']:
                print(f"   Variations: {rewrite_info['query_variations']}")

            return {**state, "query_rewrite": rewrite_info}

        except Exception as e:
            # Fallback: use original query
            fallback = {
                "rewritten_query": query,
                "query_variations": [query],
                "expansion_terms": [],
                "reasoning": f"Fallback due to error: {str(e)}"
            }
            return {**state, "query_rewrite": fallback}


if __name__ == "__main__":
    agent = QueryRewritingAgent()
    test_state = {
        "query": "What's the deal with our PTO policy?",
        "query_understanding": {
            "intent": "factual",
            "entities": ["PTO", "policy"],
            "complexity": "simple"
        },
        "routing_decision": {"strategy": "vector_search"}
    }
    result = agent.run(test_state)
    print(json.dumps(result["query_rewrite"], indent=2))
