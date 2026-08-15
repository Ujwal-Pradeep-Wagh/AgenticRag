"""
Diagnostic script to trace exactly what happens in the pipeline
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*70)
print("PIPELINE DIAGNOSTIC")
print("="*70)

# Step 1: Check documents
print("\n[1/6] Checking Vector Store...")
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config

embeddings = HuggingFaceEmbeddings(
    model_name=Config.EMBEDDING_MODEL,
    model_kwargs={"device": Config.EMBEDDING_DEVICE},
    encode_kwargs={"normalize_embeddings": True}
)
vector_store = Chroma(
    collection_name=Config.COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=Config.CHROMA_PERSIST_DIR
)

count = vector_store._collection.count()
print(f"  Documents in DB: {count}")

if count == 0:
    print("  ⚠️ No documents! Please ingest first.")
    sys.exit(1)

# Step 2: Test direct search
print("\n[2/6] Testing Direct Search...")
query = "remote work policy"
results = vector_store.similarity_search_with_score(query, k=3)
print(f"  Query: '{query}'")
print(f"  Results: {len(results)} documents")
if results:
    for i, (doc, score) in enumerate(results[:2], 1):
        print(f"    [{i}] Score: {score:.4f}, Content: {doc.page_content[:80]}...")

# Step 3: Test Retrieval Agent
print("\n[3/6] Testing Retrieval Agent...")
from agents.retrieval import RetrievalAgent

agent = RetrievalAgent()
state = {
    "query": "What is the remote work policy?",
    "query_rewrite": {"rewritten_query": "remote work policy"},
    "routing_decision": {"strategy": "vector_search", "top_k": 5, "filters": {}}
}

result = agent.run(state)
retrieved = result.get("retrieved_documents", [])
print(f"  Retrieved: {len(retrieved)} documents")

# Step 4: Test Validation Agent
print("\n[4/6] Testing Validation Agent...")
from agents.validation import ValidationAgent

val_agent = ValidationAgent()
val_state = {
    **state,
    "retrieved_documents": retrieved
}

val_result = val_agent.run(val_state)
validated = val_result.get("validated_documents", [])
print(f"  Validated: {len(validated)} documents (filtered from {len(retrieved)})")

if len(validated) == 0 and len(retrieved) > 0:
    print("  ⚠️ WARNING: All documents were filtered out by validation!")
    print("     This means the LLM judged all docs as having relevance < 0.5")

# Step 5: Test Response Generation
print("\n[5/6] Testing Response Generation...")
from agents.response_generation import ResponseGenerationAgent

gen_agent = ResponseGenerationAgent()
gen_state = {
    **val_state,
    "validated_documents": validated
}

gen_result = gen_agent.run(gen_state)
answer = gen_result.get("generated_answer", "")
print(f"  Answer length: {len(answer)} characters")
print(f"  Answer preview: {answer[:150]}...")

# Step 6: Full Pipeline Test
print("\n[6/6] Testing Full Pipeline...")
print("  (This takes 30-60 seconds...)")

from pipeline import AgenticRAGPipeline

pipeline = AgenticRAGPipeline()
query = "What is the remote work policy?"

try:
    result = pipeline.run(query)
    
    print(f"\n  ✅ Pipeline completed!")
    print(f"  Final answer length: {len(result.get('final_answer', ''))}")
    print(f"  Documents in result: {len(result.get('retrieved_documents', []))}")
    print(f"  Iterations: {result.get('iterations', 0)}")
    print(f"  Agent decisions: {len(result.get('agent_decisions', []))}")
    
    print(f"\n  Final Answer:")
    print(f"  {result.get('final_answer', 'None')[:300]}...")
    
except Exception as e:
    print(f"\n  ❌ Pipeline failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)

print("\n📋 SUMMARY:")
print(f"  - Vector store has {count} documents")
print(f"  - Direct search returns {len(results)} results")
print(f"  - Retrieval agent returns {len(retrieved)} documents")  
print(f"  - Validation agent passes {len(validated)} documents")
print(f"  - Response generated: {'Yes' if len(answer) > 50 else 'No/Short'}")

if len(validated) < len(retrieved):
    print(f"\n⚠️  ISSUE DETECTED:")
    print(f"  Validation agent is filtering out documents!")
    print(f"  This is likely why Streamlit shows 0 results.")
    print(f"  The validation LLM might be too strict or failing.")
    print(f"\n  SOLUTIONS:")
    print(f"  1. Check validation agent logs above")
    print(f"  2. Lower the relevance threshold in agents/validation.py")
    print(f"  3. Improve document quality/relevance")
    print(f"  4. Use better query rewrites")

print("\n" + "="*70 + "\n")
