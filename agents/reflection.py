"""
agents/reflection.py
Reflection Agent

Evaluates the generated answer and provides improvement feedback.
The feedback is stored in state so the query rewriting agent can use it
during the next iteration.
"""
import json
from typing import Dict, Any
from agents.base import BaseAgent
from config import Config


class ReflectionAgent(BaseAgent):
    """
    Critically evaluates the generated answer.

    On a weak answer:
    - Sets needs_regeneration = True
    - Writes specific improvement_feedback to state
    - The graph loops back to query rewriting, which uses the feedback

    Scores each dimension 0.0 to 1.0:
    - factual_accuracy: supported by documents
    - completeness: fully answers the question
    - relevance: on-topic
    - clarity: well-structured
    """

    SYSTEM_PROMPT = """You are a Reflection Agent evaluating a RAG system's answer.

Score the answer on:
- factual_accuracy (0-1): Is every claim supported by the provided source documents?
- completeness (0-1): Does it fully answer all parts of the question?
- relevance (0-1): Is the answer focused on what was asked?
- clarity (0-1): Is it well-organized and clear?

Set needs_regeneration = true if overall_score < 0.65 OR answer contains unsupported claims.

Return ONLY valid JSON, no markdown:
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
    "feedback": "Specific actionable suggestions for improvement",
    "needs_regeneration": false
}"""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("query", "")
        answer = state.get("generated_answer", "")
        documents = state.get("validated_documents", [])
        iteration = state.get("iteration_count", 0)

        if not answer:
            return {**state,
                    "reflection_result": {"overall_score": 0.0, "needs_regeneration": True, "feedback": "No answer generated"},
                    "needs_regeneration": True,
                    "improvement_feedback": "No answer was generated. Try a broader query.",
                    "iteration_count": iteration + 1}

        # Build concise context (enough for evaluation, not truncated arbitrarily)
        doc_context = "\n\n".join([
            f"[Source {i+1}]: {doc.page_content[:600]}"
            for i, doc in enumerate(documents[:4])
        ])

        prompt = f"""Query: "{query}"

Generated Answer:
{answer}

Source Documents (what the answer should be based on):
{doc_context}

Evaluate the answer."""

        try:
            response = self._invoke_llm(prompt, self.SYSTEM_PROMPT)
            raw = self._extract_json(response)
            reflection = json.loads(raw)

            # Compute overall_score if not provided
            scores = reflection.get("scores", {})
            if scores and "overall_score" not in reflection:
                reflection["overall_score"] = round(
                    sum(scores.values()) / len(scores), 3
                )

            overall = reflection.get("overall_score", 0.75)
            needs_regen = reflection.get("needs_regeneration", False)
            feedback = reflection.get("feedback", "")

            print(f"[Reflection] score={overall:.2f}, "
                  f"hallucinated={reflection.get('is_hallucinated', False)}, "
                  f"needs_regen={needs_regen}")

            if needs_regen:
                print(f"   Feedback: {feedback}")

            return {
                **state,
                "reflection_result": reflection,
                "needs_regeneration": needs_regen,
                # Store feedback so query_rewriting can use it next iteration
                "improvement_feedback": feedback if needs_regen else "",
                "iteration_count": iteration + 1,
                # Update final_answer only when we are NOT regenerating
                "final_answer": state.get("generated_answer", "") if not needs_regen else state.get("final_answer", "")
            }

        except Exception as e:
            print(f"[Reflection] Error: {e}. Accepting answer as-is.")
            return {
                **state,
                "reflection_result": {"overall_score": 0.75, "needs_regeneration": False, "error": str(e)},
                "needs_regeneration": False,
                "improvement_feedback": "",
                "iteration_count": iteration + 1,
                "final_answer": state.get("generated_answer", "")
            }
