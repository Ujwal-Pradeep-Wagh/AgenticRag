"""
agents/query_understanding.py
Query Understanding Agent

Responsibilities:
- Analyze user query intent
- Detect query type (factual, analytical, comparative, etc.)
- Extract entities and key terms
- Determine if query needs document retrieval

Why this is needed:
Before routing or rewriting, we need to understand WHAT the user is asking.
This prevents wasting resources on trivial queries and helps downstream agents
make better decisions.
"""
import json
from typing import Dict, Any, List
from agents.base import BaseAgent


class QueryUnderstandingAgent(BaseAgent):
    """
    Agent that analyzes and understands the user's query.

    Inputs:
        - query: str - The raw user question

    Outputs:
        - intent: str - Query intent classification
        - entities: List[str] - Key entities mentioned
        - needs_retrieval: bool - Whether retrieval is needed
        - complexity: str - simple/medium/complex
        - domain_hints: List[str] - Potential document domains
    """

    SYSTEM_PROMPT = """You are a Query Understanding Agent in an Agentic RAG system.
Your job is to analyze user queries and extract structured information.

Analyze the query and return a JSON object with these fields:
- intent: The primary intent (factual, analytical, comparative, procedural, definitional, opinion)
- entities: List of key entities, names, terms, or concepts mentioned
- needs_retrieval: true if the query requires searching documents, false if it is conversational/general knowledge
- complexity: "simple" (single fact), "medium" (requires synthesis), or "complex" (multi-step reasoning)
- domain_hints: List of likely document domains (e.g., ["finance", "hr_policy"])
- confidence: Your confidence in this analysis (0.0 to 1.0)

Respond ONLY with valid JSON. No markdown, no explanations."""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the query and update state with understanding.

        Args:
            state: Current graph state containing 'query'

        Returns:
            Updated state with 'query_understanding' field
        """
        query = state.get("query", "")

        if not query:
            return {**state, "query_understanding": None, "error": "No query provided"}

        prompt = f"Analyze this query: \"{query}\""

        try:
            response = self._invoke_llm(prompt, self.SYSTEM_PROMPT)
            # Parse JSON response
            understanding = json.loads(response.strip())

            # Validate required fields
            understanding.setdefault("intent", "factual")
            understanding.setdefault("entities", [])
            understanding.setdefault("needs_retrieval", True)
            understanding.setdefault("complexity", "medium")
            understanding.setdefault("domain_hints", [])
            understanding.setdefault("confidence", 0.8)

            print(f"[Query Understanding] intent={understanding['intent']}, "
                  f"needs_retrieval={understanding['needs_retrieval']}, "
                  f"complexity={understanding['complexity']}")

            return {**state, "query_understanding": understanding}

        except json.JSONDecodeError:
            print("Warning: Failed to parse JSON, using fallback analysis")
            fallback = {
                "intent": "factual",
                "entities": query.split(),
                "needs_retrieval": True,
                "complexity": "medium",
                "domain_hints": [],
                "confidence": 0.5
            }
            return {**state, "query_understanding": fallback}
        except Exception as e:
            return {**state, "query_understanding": None, "error": str(e)}


# Example usage
if __name__ == "__main__":
    agent = QueryUnderstandingAgent()
    test_state = {"query": "What is the company's remote work policy for software engineers?"}
    result = agent.run(test_state)
    print(json.dumps(result["query_understanding"], indent=2))
