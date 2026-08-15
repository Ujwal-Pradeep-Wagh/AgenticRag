"""
demo/simple_test.py
Simple quick test to verify the system works without full comparison.

Usage:
    python demo/simple_test.py
"""

import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import AgenticRAGPipeline
from ingestion.pipeline import DocumentIngestionPipeline
from config import Config


def main():
    print("\n" + "="*70)
    print("  AGENTIC RAG - SIMPLE TEST")
    print("="*70)

    # Check configuration
    print("\n[1/4] Checking Configuration...")
    try:
        Config.validate()
        print(f"  ✅ API Key configured")
        print(f"  📌 Model: {Config.LLM_MODEL}")
        print(f"  📌 Embeddings: {Config.EMBEDDING_MODEL}")
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return

    # Ingest demo document
    print("\n[2/4] Ingesting Demo Document...")
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
        import traceback
        traceback.print_exc()
        return

    # Test a simple query
    print("\n[3/4] Testing Agentic RAG Pipeline...")
    query = "What is the company's remote work policy?"
    print(f"  Query: {query}")

    try:
        agentic = AgenticRAGPipeline()
        result = agentic.run(query)
        
        print(f"\n  ✅ Pipeline completed successfully!")
        print(f"  📄 Documents retrieved: {len(result['retrieved_documents'])}")
        print(f"  🔄 Iterations: {result['iterations']}")
        print(f"\n  Answer Preview:")
        answer = result['final_answer']
        if len(answer) > 300:
            answer = answer[:300] + "..."
        print(f"  {answer}")
        
    except Exception as e:
        print(f"  ❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Summary
    print("\n[4/4] Summary")
    print("="*70)
    print("  ✅ ALL TESTS PASSED!")
    print("="*70)
    print("\n  Next steps:")
    print("    1. Run full comparison: python demo/quick_test.py")
    print("    2. Launch UI: streamlit run frontend/app.py")
    print("    3. Run API: uvicorn app:app --reload")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
