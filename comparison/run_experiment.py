"""
comparison/run_experiment.py
Run the baseline vs agentic comparison experiment.

This is the KEY experiment that demonstrates measurable improvement.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import AgenticRAGPipeline, TraditionalRAGPipeline
from evaluation.metrics import RAGEvaluator
from config import Config
from langchain_groq import ChatGroq


# Sample test queries for enterprise documents
TEST_QUERIES = [
    "What is the company's remote work policy?",
    "How many vacation days do employees get per year?",
    "What is the procedure for requesting time off?",
    "Compare the benefits package between full-time and part-time employees.",
    "What are the security requirements for handling confidential data?",
    "Explain the code of conduct for employees.",
    "What is the reimbursement policy for business travel?",
    "How does the performance review process work?",
    "What are the IT support contact details?",
    "Describe the onboarding process for new hires."
]


def run_experiment():
    """Run the full comparison experiment."""
    print("="*70)
    print("AGENTIC RAG vs TRADITIONAL RAG - COMPARISON EXPERIMENT")
    print("="*70)

    # Initialize pipelines
    print("\n[1] Initializing pipelines...")
    agentic = AgenticRAGPipeline()
    traditional = TraditionalRAGPipeline()

    # Initialize evaluator
    llm = ChatGroq(
        model=Config.LLM_MODEL,
        temperature=0.0,
        api_key=Config.GROQ_API_KEY
    )
    evaluator = RAGEvaluator(llm)

    # Run comparison
    print("\n[2] Running comparison on test queries...")
    results = evaluator.run_comparison(TEST_QUERIES, traditional, agentic)

    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    for method in ["traditional", "agentic"]:
        summary = results["summary"][method]
        print(f"\n{method.upper()} RAG:")
        print(f"  Average Score: {summary['avg_score']:.3f}")
        print(f"  Retrieval Rate: {summary['retrieval_rate']:.1%}")

    improvement = results["summary"]["improvement"]
    print(f"\n📈 IMPROVEMENT:")
    print(f"  Absolute: +{improvement['absolute']:.3f}")
    print(f"  Percentage: +{improvement['percentage']:.1f}%")

    # Save results
    import json
    os.makedirs("./comparison/results", exist_ok=True)
    with open("./comparison/results/experiment_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Results saved to: ./comparison/results/experiment_results.json")
    print("="*70)

    return results


if __name__ == "__main__":
    run_experiment()
