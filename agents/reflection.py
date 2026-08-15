"""
agents/reflection.py
Reflection Agent

Responsibilities:
- Evaluate the generated answer for quality
- Detect hallucinations, incompleteness, or irrelevance
- Decide if the answer needs regeneration
- Provide feedback for improvement

Why this is needed:
This is the self-correction mechanism. After generation, the system
reflects on its own output and decides if it's good enough or needs
another iteration. This dramatically reduces hallucinations.
"""
import json
from typing import Dict, Any
from agents.base import BaseAgent


class ReflectionAgent(BaseAgent):
    """
    Agent that reflects on and evaluates generated answers.

    Inputs:
        - query: str
        - validated_documents: List[Document]
        - generated_answer: str
        - generation_metadata: Dict

    Outputs:
        - reflection_result: Dict with evaluation scores
        - needs_regeneration: bool
        - improvement_feedback: str
    """

    SYSTEM_PROMPT = """You are a Reflection Agent in an Agentic RAG system.
Your job is to critically evaluate answers and detect problems.

Evaluate on these criteria (score 0.0 to 1.0):
1. factual_accuracy: Is the answer supported by the provided documents?
2. completeness: Does it fully answer the query?
3. relevance: Is the answer relevant to the query?
4. clarity: Is the answer clear and well-structured?

Also provide:
- overall_score: Average of the above
- is_hallucinated: true if the answer contains unsupported claims
- is_incomplete: true if the answer misses key information
- feedback: Specific suggestions for improvement
- needs_regeneration: true if overall_score < 0.7 or is_hallucinated

Return ONLY JSON:
{
    "scores": {
        "factual_accuracy": 0.9,
        "completeness": 0.8,
        "relevance": 0.9,
        "clarity": 0.85
    },
    "overall_score": 0.86,
    "is_hallucinated": false,
    "is_incomplete": false,
    "feedback": "The answer is good but could include more details about...",
    "needs_regeneration": false
}"""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on the generated answer.

        Args:
            state: Current graph state

        Returns:
            Updated state with reflection results
        """
        query = state.get("query", "")
        answer = state.get("generated_answer", "")
        documents = state.get("validated_documents", [])

        if not answer:
            return {**state, "reflection_result": {
                "overall_score": 0.0,
                "needs_regeneration": True,
                "feedback": "No answer generated"
            }}

        # Build context
        doc_context = "\n\n".join([
            f"Source {i}: {doc.page_content[:400]}..."
            for i, doc in enumerate(documents[:3])
        ])

        prompt = f"""Query: "{query}"

Generated Answer:
{answer}

Source Documents:
{doc_context}

Evaluate the answer critically."""

        try:
            response = self._invoke_llm(prompt, self.SYSTEM_PROMPT)
            reflection = json.loads(response.strip())

            scores = reflection.get("scores", {})
            overall = reflection.get("overall_score", 0.0)
            needs_regen = reflection.get("needs_regeneration", False)

            print(f"[Reflection] overall_score={overall:.2f}, "
                  f"hallucinated={reflection.get('is_hallucinated', False)}, "
                  f"needs_regen={needs_regen}")

            if needs_regen:
                print(f"   Feedback: {reflection.get('feedback', 'N/A')}")

            return {
                **state,
                "reflection_result": reflection,
                "needs_regeneration": needs_regen,
                "iteration_count": state.get("iteration_count", 0) + 1
            }

        except Exception as e:
            print(f"Warning: Reflection error: {str(e)}")
            return {
                **state,
                "reflection_result": {
                    "overall_score": 0.7,
                    "needs_regeneration": False,
                    "error": str(e)
                }
            }


if __name__ == "__main__":
    agent = ReflectionAgent()
    from langchain_core.documents import Document
    test_state = {
        "query": "What is the remote work policy?",
        "generated_answer": "Employees can work remotely 2 days per week with manager approval.",
        "validated_documents": [
            Document(page_content="Remote work policy: 2 days WFH per week with manager approval.", 
                     metadata={"source": "policy.pdf"})
        ],
        "iteration_count": 0
    }
    result = agent.run(test_state)
    print(json.dumps(result["reflection_result"], indent=2))
