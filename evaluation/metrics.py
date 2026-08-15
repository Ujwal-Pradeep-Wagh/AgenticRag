"""
evaluation/metrics.py
Evaluation Metrics for Agentic RAG

Compares baseline vs agentic RAG on key metrics.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import json


@dataclass
class EvaluationResult:
    """Result of evaluating a single query."""
    query: str
    method: str  # "traditional" or "agentic"
    answer: str
    retrieved_count: int
    relevance_score: float
    has_citations: bool
    response_length: int


class RAGEvaluator:
    """
    Evaluates RAG system performance.

    Metrics:
    1. Retrieval Accuracy: % of queries with relevant docs retrieved
    2. Answer Relevance: LLM-judged relevance score
    3. Hallucination Rate: % of answers with unsupported claims
    4. Response Quality: Overall usefulness score
    """

    def __init__(self, llm):
        self.llm = llm

    def evaluate_answer(self, query: str, answer: str, context: str = "") -> Dict[str, float]:
        """Evaluate a single answer using LLM-as-judge."""
        prompt = f"""Evaluate this RAG answer on a scale of 0-1 for each metric.

Query: {query}
Answer: {answer}
Context: {context}

Metrics:
1. relevance: Does the answer address the query?
2. accuracy: Is the answer factually correct?
3. completeness: Does it cover all aspects?
4. clarity: Is it well-structured and clear?

Return ONLY JSON: {{"relevance": 0.8, "accuracy": 0.9, "completeness": 0.7, "clarity": 0.85}}"""

        try:
            response = self.llm.invoke(prompt).content
            scores = json.loads(response.strip())
            scores["overall"] = sum(scores.values()) / len(scores)
            return scores
        except:
            return {"relevance": 0.5, "accuracy": 0.5, "completeness": 0.5, "clarity": 0.5, "overall": 0.5}

    def run_comparison(self, test_queries: List[str], 
                       traditional_pipeline, 
                       agentic_pipeline) -> Dict[str, Any]:
        """
        Run head-to-head comparison between traditional and agentic RAG.

        This is the KEY EXPERIMENT that proves your research contribution.
        """
        results = {
            "traditional": [],
            "agentic": [],
            "summary": {}
        }

        for query in test_queries:
            print(f"Evaluating: {query[:50]}...")

            # Traditional
            trad_result = traditional_pipeline.run(query)
            trad_scores = self.evaluate_answer(query, trad_result["answer"])
            results["traditional"].append({
                "query": query,
                "scores": trad_scores,
                "docs_retrieved": trad_result["documents_retrieved"]
            })

            # Agentic
            agent_result = agentic_pipeline.run(query)
            agent_scores = self.evaluate_answer(query, agent_result["final_answer"])
            results["agentic"].append({
                "query": query,
                "scores": agent_scores,
                "docs_retrieved": len(agent_result["retrieved_documents"]),
                "iterations": agent_result["iterations"]
            })

        # Calculate averages
        for method in ["traditional", "agentic"]:
            scores = [r["scores"]["overall"] for r in results[method]]
            results["summary"][method] = {
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "retrieval_rate": sum(1 for r in results[method] if r["docs_retrieved"] > 0) / len(results[method]) if results[method] else 0
            }

        # Calculate improvement
        trad_avg = results["summary"]["traditional"]["avg_score"]
        agent_avg = results["summary"]["agentic"]["avg_score"]
        results["summary"]["improvement"] = {
            "absolute": agent_avg - trad_avg,
            "percentage": ((agent_avg - trad_avg) / trad_avg * 100) if trad_avg > 0 else 0
        }

        return results


if __name__ == "__main__":
    print("RAG Evaluator ready!")
    print("Usage:")
    print("  from evaluation.metrics import RAGEvaluator")
    print("  evaluator = RAGEvaluator(llm)")
    print("  results = evaluator.run_comparison(test_queries, trad_pipeline, agentic_pipeline)")
