"""
agents/response_generation.py
Response Generation Agent

Responsibilities:
- Generate final answer using validated context
- Synthesize information from multiple documents
- Cite sources appropriately
- Handle cases where no relevant documents were found

Why this is needed:
This is the final output agent. It takes all the processed information
and produces a coherent, accurate, and well-cited answer for the user.
"""
from typing import Dict, Any, List
from langchain_core.documents import Document
from agents.base import BaseAgent


class ResponseGenerationAgent(BaseAgent):
    """
    Agent that generates the final response.

    Inputs:
        - query: str
        - validated_documents: List[Document]
        - query_understanding: Dict

    Outputs:
        - generated_answer: str
        - generation_metadata: Dict
    """

    SYSTEM_PROMPT = """You are a Response Generation Agent in an Agentic RAG system.
Your job is to generate accurate, helpful answers based on retrieved documents.

Rules:
1. Base your answer ONLY on the provided documents
2. If the documents don't contain the answer, say so clearly
3. Cite sources using [Source X] format
4. Be concise but complete
5. Structure the answer clearly with headings or bullet points when helpful
6. Do NOT make up information not in the documents
7. If documents conflict, acknowledge the discrepancy"""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate the final answer.

        Args:
            state: Current graph state

        Returns:
            Updated state with generated answer
        """
        query = state.get("query", "")
        documents = state.get("validated_documents", [])
        understanding = state.get("query_understanding", {})

        if not documents:
            # No relevant documents found
            answer = (
                "I couldn't find relevant information in the available documents "
                "to answer your question. Please try rephrasing your query or "
                "upload documents that might contain this information."
            )
            return {
                **state,
                "generated_answer": answer,
                "generation_metadata": {
                    "sources_used": 0,
                    "strategy": "no_documents",
                    "confidence": 0.0
                }
            }

        # Build context from documents
        context_parts = []
        for i, doc in enumerate(documents):
            source_ref = f"[Source {i+1}]"
            context_parts.append(
                f"{source_ref} (from {doc.metadata.get('source', 'unknown')}, "
                f"page {doc.metadata.get('page_number', 'N/A')}):\n"
                f"{doc.page_content}"
            )

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""Answer the following question based on the provided documents.

Question: {query}

Documents:
{context}

Provide a clear, accurate answer with source citations."""

        try:
            answer = self._invoke_llm(prompt, self.SYSTEM_PROMPT)

            # Calculate confidence based on document scores
            avg_score = sum(
                doc.metadata.get("relevance_score", 0.5) 
                for doc in documents
            ) / len(documents) if documents else 0

            print(f"[Response Generation] Generated answer ({len(answer)} chars, "
                  f"confidence={avg_score:.2f})")

            return {
                **state,
                "generated_answer": answer,
                "generation_metadata": {
                    "sources_used": len(documents),
                    "strategy": understanding.get("intent", "factual"),
                    "confidence": avg_score,
                    "document_sources": [
                        {
                            "file": doc.metadata.get("source", "unknown"),
                            "page": doc.metadata.get("page_number", "N/A"),
                            "score": doc.metadata.get("relevance_score", 0)
                        }
                        for doc in documents
                    ]
                }
            }

        except Exception as e:
            return {
                **state,
                "generated_answer": f"Error generating response: {str(e)}",
                "generation_metadata": {"error": str(e)}
            }


if __name__ == "__main__":
    agent = ResponseGenerationAgent()
    from langchain_core.documents import Document
    test_state = {
        "query": "What is the remote work policy?",
        "validated_documents": [
            Document(
                page_content="The remote work policy allows employees to work from home up to 2 days per week with prior manager approval. All remote work arrangements must be documented in the HR system.",
                metadata={"source": "hr_policy.pdf", "page_number": 5, "relevance_score": 0.95}
            )
        ],
        "query_understanding": {"intent": "factual"}
    }
    result = agent.run(test_state)
    print(result["generated_answer"])
