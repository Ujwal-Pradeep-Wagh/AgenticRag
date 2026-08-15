"""
Test the retrieval agent directly
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.retrieval import RetrievalAgent

print("="*70)
print("TESTING RETRIEVAL AGENT DIRECTLY")
print("="*70)

# Initialize agent
print("\n[1] Initializing Retrieval Agent...")
agent = RetrievalAgent()
print("  ✅ Agent initialized")

# Test state
test_state = {
    "query": "What is the company's remote work policy?",
    "query_rewrite": {
        "rewritten_query": "company remote work policy",
        "query_variations": ["remote work policy", "work from home policy"],
        "expansion_terms": [],
        "reasoning": "Simplified query"
    },
    "routing_decision": {
        "strategy": "vector_search",
        "top_k": 5,
        "filters": {},
        "reasoning": "Standard vector search"
    }
}

print("\n[2] Running retrieval...")
print(f"  Query: {test_state['query']}")
print(f"  Rewritten: {test_state['query_rewrite']['rewritten_query']}")
print(f"  Strategy: {test_state['routing_decision']['strategy']}")
print(f"  Top K: {test_state['routing_decision']['top_k']}")

result = agent.run(test_state)

print("\n[3] Results:")
documents = result.get("retrieved_documents", [])
metadata = result.get("retrieval_metadata", {})

print(f"  📊 Documents retrieved: {len(documents)}")
print(f"  📋 Metadata: {metadata}")

if documents:
    print(f"\n[4] Sample documents:")
    for i, doc in enumerate(documents[:3], 1):
        score = doc.metadata.get("retrieval_score", "N/A")
        content = doc.page_content[:100].replace("\n", " ")
        print(f"\n  Document {i}:")
        print(f"    Score: {score}")
        print(f"    Content: {content}...")
else:
    print("\n  ⚠️  No documents retrieved!")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70 + "\n")
