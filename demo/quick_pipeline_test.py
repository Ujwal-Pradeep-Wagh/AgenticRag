"""
Quick pipeline test without unicode characters
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*70)
print("QUICK PIPELINE TEST")
print("="*70)

from pipeline import AgenticRAGPipeline

print("\n[1/2] Initializing pipeline...")
pipeline = AgenticRAGPipeline()

print("\n[2/2] Running query...")
print("Query: What is the remote work policy?")
print("(This takes 30-60 seconds due to LLM API calls...)\n")

try:
    result = pipeline.run("What is the remote work policy?")
    
    print("\n" + "="*70)
    print("SUCCESS!")
    print("="*70)
    
    print(f"\nDocuments retrieved: {len(result.get('retrieved_documents', []))}")
    print(f"Iterations: {result.get('iterations', 0)}")
    print(f"Agent decisions: {len(result.get('agent_decisions', []))}")
    
    print(f"\nFinal Answer:")
    print("-"*70)
    answer = result.get('final_answer', 'No answer')
    if len(answer) > 500:
        print(answer[:500] + "...\n(truncated)")
    else:
        print(answer)
    print("-"*70)
    
    if len(result.get('retrieved_documents', [])) == 0:
        print("\nWARNING: 0 documents in final result")
        print("Check if validation agent filtered them all out")
    else:
        print(f"\nDocuments used in answer:")
        for i, doc in enumerate(result.get('retrieved_documents', [])[:3], 1):
            print(f"  [{i}] {doc.get('source')} - Score: {doc.get('score', 0):.3f}")
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70 + "\n")
