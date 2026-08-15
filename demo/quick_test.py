"""
demo/quick_test.py
Quick test script to verify the Agentic RAG system works end-to-end.

Usage:
    python demo/quick_test.py

This script:
1. Ingests the demo PDF
2. Runs test queries through both Agentic and Traditional RAG
3. Compares results side-by-side
4. Generates a summary report
"""

import os
import sys
import json
import time

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import AgenticRAGPipeline, TraditionalRAGPipeline
from ingestion.pipeline import DocumentIngestionPipeline
from config import Config


def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def print_section(text):
    print(f"\n{'─'*70}")
    print(f"  {text}")
    print("─"*70)


def main():
    print_header("AGENTIC RAG - QUICK TEST")

    # Step 1: Check configuration
    print_section("STEP 1: Checking Configuration")
    try:
        Config.validate()
        print(f"  ✅ API Key configured")
        print(f"  📌 Model: {Config.LLM_MODEL}")
        print(f"  📌 Embeddings: {Config.EMBEDDING_MODEL}")
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        print("  Please set GROQ_API_KEY in your .env file")
        print("  Get a free key at: https://console.groq.com/")
        return

    # Step 2: Ingest demo document
    print_section("STEP 2: Ingesting Demo Document")
    demo_pdf = os.path.join(os.path.dirname(__file__), "company_handbook.pdf")

    if not os.path.exists(demo_pdf):
        print(f"  ❌ Demo PDF not found: {demo_pdf}")
        return

    try:
        pipeline = DocumentIngestionPipeline()
        result = pipeline.ingest(demo_pdf)
        print(f"  ✅ Ingested: {result['pages_loaded']} pages → {result['chunks_created']} chunks")
        print(f"  📊 Total documents in DB: {result['collection_count']}")
    except Exception as e:
        print(f"  ❌ Ingestion error: {e}")
        return

    # Step 3: Load test queries
    print_section("STEP 3: Loading Test Queries")
    queries_file = os.path.join(os.path.dirname(__file__), "test_queries.json")
    with open(queries_file, 'r') as f:
        test_data = json.load(f)

    # Use first 5 queries for quick test
    test_queries = test_data["test_cases"][:5]
    print(f"  ✅ Loaded {len(test_queries)} test queries")

    # Step 4: Run comparison
    print_section("STEP 4: Running Agentic vs Traditional RAG")

    agentic = AgenticRAGPipeline()
    traditional = TraditionalRAGPipeline()

    results = []

    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        print(f"\n  Query {i}/{len(test_queries)}: {query}")

        # Agentic RAG
        start = time.time()
        try:
            agentic_result = agentic.run(query)
            agentic_time = time.time() - start
            agentic_answer = agentic_result["final_answer"]
            agentic_iters = agentic_result["iterations"]
            agentic_docs = len(agentic_result["retrieved_documents"])
        except Exception as e:
            agentic_answer = f"ERROR: {str(e)}"
            agentic_time = 0
            agentic_iters = 0
            agentic_docs = 0

        # Traditional RAG
        start = time.time()
        try:
            trad_result = traditional.run(query)
            trad_time = time.time() - start
            trad_answer = trad_result["answer"]
            trad_docs = trad_result["documents_retrieved"]
        except Exception as e:
            trad_answer = f"ERROR: {str(e)}"
            trad_time = 0
            trad_docs = 0

        results.append({
            "query": query,
            "agentic_answer": agentic_answer[:200] + "..." if len(agentic_answer) > 200 else agentic_answer,
            "traditional_answer": trad_answer[:200] + "..." if len(trad_answer) > 200 else trad_answer,
            "agentic_time": round(agentic_time, 2),
            "traditional_time": round(trad_time, 2),
            "agentic_docs": agentic_docs,
            "traditional_docs": trad_docs,
            "agentic_iterations": agentic_iters
        })

        print(f"    🤖 Agentic: {agentic_time:.1f}s | {agentic_docs} docs | {agentic_iters} iterations")
        print(f"    📚 Traditional: {trad_time:.1f}s | {trad_docs} docs")

    # Step 5: Print summary
    print_section("STEP 5: Summary Report")

    total_agentic_time = sum(r["agentic_time"] for r in results)
    total_trad_time = sum(r["traditional_time"] for r in results)
    total_agentic_docs = sum(r["agentic_docs"] for r in results)
    total_trad_docs = sum(r["traditional_docs"] for r in results)

    print(f"\n  📊 PERFORMANCE:")
    print(f"    Total Agentic Time: {total_agentic_time:.1f}s (avg: {total_agentic_time/len(results):.1f}s/query)")
    print(f"    Total Traditional Time: {total_trad_time:.1f}s (avg: {total_trad_time/len(results):.1f}s/query)")
    if total_trad_time > 0:
        print(f"    Speed Difference: {((total_agentic_time - total_trad_time) / total_trad_time * 100):+.0f}%")

    print(f"\n  📄 RETRIEVAL:")
    print(f"    Agentic docs retrieved: {total_agentic_docs}")
    print(f"    Traditional docs retrieved: {total_trad_docs}")

    print(f"\n  🔄 SELF-CORRECTION:")
    total_iterations = sum(r["agentic_iterations"] for r in results)
    print(f"    Total regeneration loops: {total_iterations}")
    print(f"    Avg iterations per query: {total_iterations/len(results):.1f}")

    # Save results
    output_file = os.path.join(os.path.dirname(__file__), "test_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  💾 Results saved to: {output_file}")

    # Step 6: Show sample answers
    print_section("STEP 6: Sample Answer Comparison")
    sample = results[0]
    print(f"\n  Query: {sample['query']}")
    print(f"\n  🤖 AGENTIC RAG:")
    print(f"    {sample['agentic_answer']}")
    print(f"\n  📚 TRADITIONAL RAG:")
    print(f"    {sample['traditional_answer']}")

    print("\n" + "="*70)
    print("  ✅ QUICK TEST COMPLETE")
    print("="*70)
    print("\n  Next steps:")
    print("    1. Run full experiment: python comparison/run_experiment.py")
    print("    2. Launch UI: streamlit run frontend/app.py")
    print("    3. Run API: uvicorn app:app --reload")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
