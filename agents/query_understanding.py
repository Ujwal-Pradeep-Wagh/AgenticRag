"""
agents/query_understanding.py
Query Understanding Agent

Analyzes user intent, extracts entities, and determines retrieval needs.
"""
import json
from typing import Dict, Any
from agents.base import BaseAgent


class QueryUnderstandingAgent(BaseAgent):
    """
    Analyzes the user query and extracts structured understanding.

    Outputs:
        - intent: factual | analytical | comparative | procedural | definitional | opinion
        - entities: key terms/names
        - needs_retrieval: bool
        - complexity: simple | medium | complex
        - domain_hints: likely document domains
        - confidence: 0.0 - 1.0
    """

    SYSTEM_PROMPT = """You are a Query Understanding Agent in an Agentic RAG system.
Analyze user queries and return structured JSON.

Return ONLY valid JSON, no markdown fences, no explanation:
{
    "intent": "factual",
    "entities": ["list", "of", "key", "terms"],
    "needs_retrieval": true,
    "complexity": "simple",
    "domain_hints": ["hr_policy", "benefits"],
    "confidence": 0.9
}

Guidelines:
- intent: factual | analytical | comparative | procedural | definitional | opinion
- needs_retrieval: true for almost all non-trivial questions about documents
- complexity: simple (single fact), medium (requires synthesis), complex (multi-step)
- domain_hints: infer from context (hr_policy, finance, it_support, legal, onboarding, etc.)"""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("query", "")

        if not query:
            return {**state, "query_understanding": None, "error": "No query provided"}

        try:
            response = self._invoke_llm(
                f'Analyze this query: "{query}"',
                self.SYSTEM_PROMPT
            )
            raw = self._extract_json(response)
            understanding = json.loads(raw)

            # Ensure all fields exist
            understanding.setdefault("intent", "factual")
            understanding.setdefault("entities", query.split()[:5])
            understanding.setdefault("needs_retrieval", True)
            understanding.setdefault("complexity", "medium")
            understanding.setdefault("domain_hints", [])
            understanding.setdefault("confidence", 0.8)

            # Safety: always attempt retrieval unless very clearly general knowledge
            # Prevents routing agent from incorrectly skipping document search
            if understanding.get("intent") in ("opinion",) and not understanding.get("domain_hints"):
                understanding["needs_retrieval"] = False

            print(f"[Query Understanding] intent={understanding['intent']}, "
                  f"needs_retrieval={understanding['needs_retrieval']}, "
                  f"complexity={understanding['complexity']}")

            return {**state, "query_understanding": understanding}

        except (json.JSONDecodeError, ValueError):
            print("[Query Understanding] Warning: Failed to parse JSON, using fallback.")
            fallback = {
                "intent": "factual",
                "entities": query.split()[:5],
                "needs_retrieval": True,
                "complexity": "medium",
                "domain_hints": [],
                "confidence": 0.5
            }
            return {**state, "query_understanding": fallback}

        except Exception as e:
            print(f"[Query Understanding] Error: {e}")
            fallback = {
                "intent": "factual",
                "entities": [],
                "needs_retrieval": True,
                "complexity": "medium",
                "domain_hints": [],
                "confidence": 0.3
            }
            return {**state, "query_understanding": fallback, "error": str(e)}
