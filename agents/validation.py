"""
agents/validation.py
Validation Agent

Responsibilities:
- Grade retrieved documents for relevance to the query
- Filter out irrelevant chunks
- Calculate overall context quality score
- Decide if re-retrieval is needed

Why this is needed:
Poor retrieval leads to hallucinated or irrelevant answers. This agent
acts as a gatekeeper, ensuring only high-quality context reaches the
generation stage.
"""
import json
from typing import Dict, Any, List
from langchain_core.documents import Document
from agents.base import BaseAgent


class ValidationAgent(BaseAgent):
    """
    Agent that validates retrieved document quality.

    Inputs:
        - query: str - Original user query
        - retrieved_documents: List[Document]
        - query_understanding: Dict

    Outputs:
        - validated_documents: List[Document] - Filtered relevant docs
        - validation_result: Dict with scores and decisions
        - needs_reretrieval: bool - Whether to trigger re-retrieval
    """

    SYSTEM_PROMPT = """You are a Validation Agent in an Agentic RAG system.
Your job is to grade whether retrieved documents are relevant to the user's query.

For EACH document, assign:
- relevance_score: 0.0 to 1.0 (1.0 = highly relevant)
- reasoning: Brief explanation

Also provide:
- overall_quality: "high", "medium", or "low"
- needs_reretrieval: true if overall quality is poor (< 0.5 average)
- suggestions: How to improve retrieval if needed

Return ONLY a JSON object:
{
    "document_grades": [
        {"index": 0, "relevance_score": 0.9, "reasoning": "..."},
        ...
    ],
    "overall_quality": "high",
    "needs_reretrieval": false,
    "suggestions": ""
}"""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate retrieved documents.

        Args:
            state: Current graph state

        Returns:
            Updated state with validation results
        """
        query = state.get("query", "")
        documents = state.get("retrieved_documents", [])

        if not documents:
            return {**state, "validated_documents": [], "validation_result": {
                "overall_quality": "none",
                "needs_reretrieval": True,
                "reason": "No documents retrieved"
            }}

        # Build document context
        doc_context = "\n\n".join([
            f"Document {i}:\n{doc.page_content[:500]}..."
            for i, doc in enumerate(documents)
        ])

        prompt = f"""Query: "{query}"

Retrieved Documents:
{doc_context}

Grade each document's relevance to the query."""

        try:
            response = self._invoke_llm(prompt, self.SYSTEM_PROMPT)
            validation = json.loads(response.strip())

            # Extract grades
            grades = validation.get("document_grades", [])

            # Filter documents: keep only those with score >= 0.5
            validated_docs = []
            for i, doc in enumerate(documents):
                if i < len(grades):
                    score = grades[i].get("relevance_score", 0.0)
                    doc.metadata["relevance_score"] = score
                    if score >= 0.5:
                        validated_docs.append(doc)
                else:
                    # If no grade provided, default to keeping it with score 0.6
                    doc.metadata["relevance_score"] = 0.6
                    validated_docs.append(doc)

            # If ALL documents were filtered out, keep top 2 anyway (fallback)
            if not validated_docs and documents:
                print("[Validation] Warning: All docs filtered. Keeping top 2 as fallback.")
                for doc in documents[:2]:
                    doc.metadata["relevance_score"] = 0.5
                    validated_docs.append(doc)

            needs_reretrieval = validation.get("needs_reretrieval", False)
            overall = validation.get("overall_quality", "medium")

            print(f"[Validation] quality={overall}, "
                  f"valid_docs={len(validated_docs)}/{len(documents)}, "
                  f"reretrieval={needs_reretrieval}")

            return {
                **state,
                "validated_documents": validated_docs,
                "validation_result": {
                    "overall_quality": overall,
                    "needs_reretrieval": needs_reretrieval,
                    "document_count": len(validated_docs),
                    "suggestions": validation.get("suggestions", "")
                }
            }

        except Exception as e:
            # Fallback: accept all documents with warning score
            print(f"[Validation] Error: {str(e)}. Accepting all documents as fallback.")
            for doc in documents:
                doc.metadata["relevance_score"] = 0.6
            return {
                **state,
                "validated_documents": documents,
                "validation_result": {
                    "overall_quality": "medium",
                    "needs_reretrieval": False,
                    "document_count": len(documents),
                    "error": str(e)
                }
            }


if __name__ == "__main__":
    agent = ValidationAgent()
    from langchain_core.documents import Document
    test_state = {
        "query": "What is the remote work policy?",
        "retrieved_documents": [
            Document(page_content="The remote work policy allows employees to work from home 2 days per week.", 
                     metadata={"source": "policy.pdf", "page_number": 1}),
            Document(page_content="The cafeteria serves lunch from 11am to 2pm daily.", 
                     metadata={"source": "facilities.pdf", "page_number": 3})
        ]
    }
    result = agent.run(test_state)
    print(json.dumps(result["validation_result"], indent=2))
