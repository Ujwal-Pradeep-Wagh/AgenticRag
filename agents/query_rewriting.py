"""
agents/query_rewriting.py
Query Rewriting Agent

Optimizes user queries for vector search retrieval.
"""
import json
from typing import Dict, Any
from agents.base import BaseAgent


class QueryRewritingAgent(BaseAgent):
    """
    Rewrites queries to improve retrieval quality.

    Inputs:
        - query: original user query
        - query_understanding: analysis from understanding agent
        - routing_decision: chosen strategy
        - conversation_history: previous turns
        - improvement_feedback: feedback from reflection agent (if iterating)

    Outputs:
        - rewritten_query: optimized primary query
        - query_variations: 2-3 alternatives for multi_query strategy
        - expansion_terms: additional context terms
        - reasoning: explanation
    """

    SYSTEM_PROMPT = """You are a Query Rewriting Agent in an Agentic RAG system.
Transform user queries into search-optimized forms for vector retrieval.

Rules:
1. Keep the core meaning — DO NOT change what is being asked
2. Remove conversational filler ("Can you tell me", "I want to know", "What is")
3. Expand known acronyms (PTO -> paid time off, HR -> human resources)
4. Use the key noun phrases that would appear in the source document
5. DO NOT add qualifiers like "definition", "explanation", "overview" unless in original
6. DO NOT over-complicate — simpler is better for semantic search
7. If query is clear already, return it with minimal changes
8. If improvement_feedback is provided, use it to fix the previous query's weaknesses

Return ONLY valid JSON, no markdown:
{
    "rewritten_query": "primary search query",
    "query_variations": ["variation 1", "variation 2"],
    "expansion_terms": ["term1", "term2"],
    "reasoning": "brief explanation"
}

Example:
Input: "What's the deal with our PTO policy?"
Output: {"rewritten_query": "paid time off PTO policy", "query_variations": ["vacation days allowance", "time off benefits"], ...}"""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("query", "")
        understanding = state.get("query_understanding", {})
        routing = state.get("routing_decision", {})
        history = state.get("conversation_history", [])
        # Feedback from reflection agent on previous iteration
        improvement_feedback = state.get("improvement_feedback", "")

        context = f'Original Query: "{query}"\n\n'
        context += f"Intent: {understanding.get('intent', 'factual')}\n"
        context += f"Key Entities: {understanding.get('entities', [])}\n"
        context += f"Strategy: {routing.get('strategy', 'vector_search')}\n"

        if improvement_feedback:
            context += f"\nIMPORTANT - Previous answer was weak. Improvement feedback:\n{improvement_feedback}\n"
            context += "Rewrite the query to address this feedback and retrieve better documents.\n"

        if history:
            context += "\nRecent conversation:\n"
            for turn in history[-2:]:
                context += f"  {turn.get('role', 'user')}: {turn.get('content', '')[:100]}\n"

        try:
            response = self._invoke_llm(context, self.SYSTEM_PROMPT)
            raw = self._extract_json(response)
            rewrite_info = json.loads(raw)

            rewrite_info.setdefault("rewritten_query", query)
            rewrite_info.setdefault("query_variations", [])
            rewrite_info.setdefault("expansion_terms", [])
            rewrite_info.setdefault("reasoning", "No changes needed")

            # If rewrite is empty or very short, fall back to original
            if len(rewrite_info["rewritten_query"].strip()) < 3:
                rewrite_info["rewritten_query"] = query

            print(f"[Query Rewriting] '{query}' -> '{rewrite_info['rewritten_query']}'")

            return {**state, "query_rewrite": rewrite_info}

        except Exception as e:
            print(f"[Query Rewriting] Error: {e}. Using original query.")
            return {**state, "query_rewrite": {
                "rewritten_query": query,
                "query_variations": [query],
                "expansion_terms": [],
                "reasoning": f"Fallback: {str(e)}"
            }}
