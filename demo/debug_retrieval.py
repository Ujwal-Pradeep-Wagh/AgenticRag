"""
Debug script to test vector store retrieval directly
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config

print("="*70)
print("DEBUG: Vector Store Retrieval Test")
print("="*70)

# Initialize embeddings and vector store
print("\n[1] Initializing embeddings and vector store...")
embeddings = HuggingFaceEmbeddings(
    model_name=Config.EMBEDDING_MODEL,
    model_kwargs={"device": Config.EMBEDDING_DEVICE},
    encode_kwargs={"normalize_embeddings": True}
)
print(f"  ✅ Embeddings model: {Config.EMBEDDING_MODEL}")

vector_store = Chroma(
    collection_name=Config.COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=Config.CHROMA_PERSIST_DIR
)
print(f"  ✅ Vector store connected")
print(f"  📂 Persist directory: {Config.CHROMA_PERSIST_DIR}")
print(f"  📚 Collection name: {Config.COLLECTION_NAME}")

# Check collection stats
print("\n[2] Checking collection...")
try:
    collection = vector_store._collection
    count = collection.count()
    print(f"  📊 Total documents in collection: {count}")
    
    if count == 0:
        print("  ⚠️  Collection is empty! Need to ingest documents first.")
    else:
        # Sample a few documents
        results = collection.get(limit=3, include=['documents', 'metadatas'])
        print(f"\n[3] Sample documents:")
        for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas']), 1):
            print(f"\n  Document {i}:")
            print(f"    Content: {doc[:100]}...")
            print(f"    Metadata: {meta}")
            
except Exception as e:
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test search
if count > 0:
    print("\n[4] Testing similarity search...")
    test_queries = [
        "remote work policy",
        "vacation days",
        "time off",
    ]
    
    for query in test_queries:
        print(f"\n  Query: '{query}'")
        try:
            results = vector_store.similarity_search_with_score(query, k=3)
            print(f"    Found {len(results)} documents")
            for i, (doc, score) in enumerate(results, 1):
                print(f"    [{i}] Score: {score:.4f}, Content: {doc.page_content[:80]}...")
        except Exception as e:
            print(f"    ❌ Error: {e}")

print("\n" + "="*70)
print("DEBUG COMPLETE")
print("="*70 + "\n")
