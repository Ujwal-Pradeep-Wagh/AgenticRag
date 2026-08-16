"""
agents/validation.py
Validation Agent

Grades retrieved documents for relevance and filters low-quality context.
Uses a fast heuristic approach (no extra LLM call) combined with retrieval scores
to avoid the latency and accuracy problems of LLM-based grading.
"""
from typing import Dict, Any, List
from langchain_core.documents import Document
from agents.base import BaseAgent
from config import Config


class ValidationAgent(BaseAgent):
    """
    Validates retrieved document quality without an extra LLM call.

    Strategy:
    1. Use the retrieval_score already attached by RetrievalAgent
    2. Boost score if query terms appear in document text (keyword overlap)
    3. Filter documents below VALIDATION_THRESHOLD
    4. Always keep at least 2 documents as fallback

    This removes a full LLM round-trip (~4-6 sec) and produces more consistent
    results than asking the LLM to grade truncated 500-char document snippets.
    """

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("query", "")
        documents: List[Document] = state.get("retrieved_documents", [])

        if not documents:
            return {**state,
                    "validated_documents": [],
                    "validation_result": {
                        "overall_quality": "none",
                        "needs_reretrieval": True,
                        "document_count": 0,
                        "reason": "No documents retrieved"
                    }}

        threshold = Config.VALIDATION_THRESHOLD
        query_terms = set(query.lower().split())
        # Remove common stop words from overlap scoring
        stop_words = {"what", "is", "the", "how", "are", "does", "do", "a", "an",
                      "of", "for", "in", "to", "and", "or", "it", "its", "with",
                      "can", "tell", "me", "i", "want", "know", "about", "our"}
        query_terms -= stop_words

        scored_docs = []
        for doc in documents:
            base_score = float(doc.metadata.get("retrieval_score", 0.5))

            # Keyword overlap boost: up to +0.15
            if query_terms:
                doc_text_lower = doc.page_content.lower()
                matched = sum(1 for t in query_terms if t in doc_text_lower)
                overlap_ratio = matched / len(query_terms)
                boost = overlap_ratio * 0.15
            else:
                boost = 0.0

            final_score = min(1.0, base_score + boost)
            doc.metadata["relevance_score"] = round(final_score, 4)
            scored_docs.append((doc, final_score))

        # Sort by combined score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Filter below threshold
        validated = [doc for doc, score in scored_docs if score >= threshold]

        # Always keep at least 2 documents (top-ranked) as fallback
        if len(validated) < 2:
            validated = [doc for doc, _ in scored_docs[:2]]
            print(f"[Validation] Fallback: keeping top 2 docs (all scored below {threshold})")

        avg_score = sum(doc.metadata.get("relevance_score", 0) for doc in validated) / len(validated)
        overall_quality = "high" if avg_score >= 0.65 else "medium" if avg_score >= 0.45 else "low"
        # Only trigger re-retrieval if quality is genuinely low AND we have very few docs
        needs_reretrieval = (overall_quality == "low" and len(validated) < 2)

        print(f"[Validation] quality={overall_quality}, avg_score={avg_score:.2f}, "
              f"valid={len(validated)}/{len(documents)}, reretrieval={needs_reretrieval}")

        return {
            **state,
            "validated_documents": validated,
            "validation_result": {
                "overall_quality": overall_quality,
                "avg_relevance_score": round(avg_score, 3),
                "needs_reretrieval": needs_reretrieval,
                "document_count": len(validated),
                "total_retrieved": len(documents)
            }
        }
