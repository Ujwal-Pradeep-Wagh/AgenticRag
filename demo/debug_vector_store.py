"""
Debug why vector store returns different results
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config

print("="*70)
print("VECTOR STORE DEBUG")
print("="*70)

# Test 1: Direct vector store
print("\n[Test 1] Direct Vector Store Initialization")
embeddings1 = HuggingFaceEmbeddings(
    model_name=Config.EMBEDDING_MODEL,
    model_kwargs={"device": Config.EMBEDDING_DEVICE},
    encode_kwargs={"normalize_embeddings": True}
)
vs1 = Chroma(
    collection_name=Config.COLLECTION_NAME,
    embedding_function=embeddings1,
    persist_directory=Config.CHROMA_PERSIST_DIR
)

count1 = vs1._collection.count()
print(f"  Collection count: {count1}")

queries_to_test = [
    "remote work policy",
    "What is the remote work policy?",
    "company remote work policy",
    "vacation days"
]

for query in queries_to_test:
    results = vs1.similarity_search_with_score(query, k=5)
    print(f"  Query: '{query}' -> {len(results)} results")
    if results:
        print(f"    Best score: {results[0][1]:.4f}")

# Test 2: Through Retrieval Agent
print("\n[Test 2] Through Retrieval Agent")
from agents.retrieval import RetrievalAgent

agent = RetrievalAgent()
print(f"  Agent vector store count: {agent.vector_store._collection.count()}")

for query in queries_to_test[:2]:
    state = {
        "query": query,
        "query_rewrite": {"rewritten_query": query},
        "routing_decision": {"strategy": "vector_search", "top_k": 5, "filters": {}}
    }
    result = agent.run(state)
    docs = result.get("retrieved_documents", [])
    print(f"  Query: '{query}' -> {len(docs)} documents")

# Test 3: Check collection details
print("\n[Test 3] Collection Details")
collection = vs1._collection
print(f"  Name: {collection.name}")
print(f"  Count: {collection.count()}")

# Get some sample IDs
sample = collection.get(limit=3, include=['documents', 'metadatas'])
print(f"  Sample documents: {len(sample['documents'])}")
if sample['documents']:
    print(f"  First doc preview: {sample['documents'][0][:100]}...")

print("\n" + "="*70 + "\n")
