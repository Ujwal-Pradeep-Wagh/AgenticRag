"""
agents/response_generation.py
Response Generation Agent

Generates the final answer from validated context documents.
"""
from typing import Dict, Any, List
from langchain_core.documents import Document
from agents.base import BaseAgent


class ResponseGenerationAgent(BaseAgent):
    """
    Generates a grounded, cited answer from validated documents.

    Inputs:
        - query: user question
        - validated_documents: filtered relevant documents
        - query_understanding: intent and complexity
        - improvement_feedback: optional feedback from reflection (iteration 2+)
    """

    SYSTEM_PROMPT = """You are a Response Generation Agent in an Agentic RAG system.
Generate accurate, well-structured answers strictly from the provided source documents.

Rules:
1. Base your answer ONLY on the provided documents — never invent information
2. Cite sources inline using [Source 1], [Source 2], etc.
3. If documents don't contain the answer, clearly say: "The available documents don't contain specific information about [topic]."
4. Be complete — cover all relevant points from the documents
5. Structure with bullet points or headings when the answer has multiple parts
6. Be concise — don't repeat the same information multiple times
7. If documents conflict with each other, acknowledge it
8. End with a brief summary for complex answers"""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("query", "")
        documents: List[Document] = state.get("validated_documents", [])
        understanding = state.get("query_understanding", {})
        improvement_feedback = state.get("improvement_feedback", "")

        if not documents:
            answer = (
                "I couldn't find relevant information in the available documents to answer your question. "
                "Please try rephrasing your query, or upload documents that contain the relevant information."
            )
            return {**state,
                    "generated_answer": answer,
                    "final_answer": answer,
                    "generation_metadata": {
                        "sources_used": 0,
                        "confidence": 0.0,
                        "strategy": "no_documents"
                    }}

        # Build context with full document content (not truncated)
        context_parts = []
        for i, doc in enumerate(documents):
            source_info = (f"[Source {i+1}] "
                           f"{doc.metadata.get('source', 'unknown')} "
                           f"(page {doc.metadata.get('page_number', 'N/A')})")
            context_parts.append(f"{source_info}:\n{doc.page_content}")

        context = "\n\n---\n\n".join(context_parts)

        # Include feedback if this is a regeneration iteration
        feedback_section = ""
        if improvement_feedback:
            feedback_section = (
                f"\n\nIMPORTANT: The previous answer was inadequate. "
                f"Address this feedback in your new answer:\n{improvement_feedback}\n"
            )

        prompt = f"""Question: {query}

Source Documents:
{context}
{feedback_section}
Provide a thorough, accurate answer with source citations."""

        try:
            answer = self._invoke_llm(prompt, self.SYSTEM_PROMPT)

            avg_score = (
                sum(doc.metadata.get("relevance_score", 0.5) for doc in documents) / len(documents)
            )

            print(f"[Response Generation] {len(answer)} chars, "
                  f"sources={len(documents)}, confidence={avg_score:.2f}")

            return {
                **state,
                "generated_answer": answer,
                "final_answer": answer,
                "generation_metadata": {
                    "sources_used": len(documents),
                    "confidence": round(avg_score, 3),
                    "intent": understanding.get("intent", "factual"),
                    "document_sources": [
                        {
                            "file": doc.metadata.get("source", "unknown"),
                            "page": doc.metadata.get("page_number", "N/A"),
                            "relevance_score": doc.metadata.get("relevance_score", 0),
                            "retrieval_score": doc.metadata.get("retrieval_score", 0)
                        }
                        for doc in documents
                    ]
                }
            }

        except Exception as e:
            print(f"[Response Generation] Error: {e}")
            error_answer = f"Error generating response: {str(e)}"
            return {**state,
                    "generated_answer": error_answer,
                    "final_answer": error_answer,
                    "generation_metadata": {"error": str(e)}}
