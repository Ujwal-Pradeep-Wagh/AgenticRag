"""
frontend/app.py
Streamlit UI for the Agentic RAG System

Features:
- Bulk PDF upload (multiple files at once)
- Ask questions with agent trace
- Side-by-side comparison with Traditional RAG
- Document stats in sidebar
"""

import streamlit as st
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import AgenticRAGPipeline, TraditionalRAGPipeline
from ingestion.pipeline import DocumentIngestionPipeline

st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="robot",
    layout="wide"
)

st.title("Agentic RAG System")
st.markdown(
    "Multi-agent pipeline: **Query Understanding** -> **Routing** -> "
    "**Rewriting** -> **Retrieval** -> **Validation** -> **Generation** -> **Reflection**"
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Document Upload")

    # Bulk upload: accept_multiple_files=True
    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can select multiple PDF files at once"
    )

    if uploaded_files:
        upload_dir = "./data/uploads"
        os.makedirs(upload_dir, exist_ok=True)

        if st.button("Ingest Selected Files", type="primary"):
            ingestion = DocumentIngestionPipeline()
            saved_paths = []

            # Save all uploaded files to disk first
            for uf in uploaded_files:
                dest = os.path.join(upload_dir, uf.name)
                with open(dest, "wb") as f:
                    f.write(uf.getvalue())
                saved_paths.append(dest)

            # Bulk ingest
            with st.spinner(f"Ingesting {len(saved_paths)} file(s)..."):
                results = ingestion.ingest_multiple(saved_paths)

            for r in results:
                fname = os.path.basename(r["file"])
                if r["status"] == "success":
                    st.success(f"{fname}: {r['chunks_created']} chunks from {r['pages_loaded']} pages")
                elif r["status"] == "skipped":
                    st.info(f"{fname}: already in database (skipped)")
                else:
                    st.error(f"{fname}: {r.get('error', 'unknown error')}")

            st.rerun()

    st.divider()
    st.header("Settings")
    show_agent_decisions = st.checkbox("Show Agent Decisions", value=True)
    show_retrieved_chunks = st.checkbox("Show Retrieved Chunks", value=True)
    compare_baseline = st.checkbox("Compare with Traditional RAG", value=False)

    st.divider()
    st.header("DB Stats")
    try:
        stats = DocumentIngestionPipeline().get_stats()
        st.metric("Chunks in DB", stats["total_documents"])
        st.caption(f"Chunk size: {stats['chunk_size']} | Overlap: {stats['chunk_overlap']}")
        st.caption(f"Model: {stats['embedding_model']}")
        if stats["total_documents"] == 0:
            st.warning("No documents ingested yet. Upload PDFs above.")
    except Exception as e:
        st.error(f"DB error: {e}")

# ── Main Chat Area ────────────────────────────────────────────────────────────

st.header("Ask a Question")

query = st.text_input(
    "Your question:",
    placeholder="e.g. What is the company's remote work policy?",
    key="query_input"
)

run_button = st.button("Search", type="primary")

if run_button and query:

    if compare_baseline:
        col_agentic, col_traditional = st.columns(2)
    else:
        col_agentic = st.container()

    # ── Agentic RAG ──────────────────────────────────────────────────────────
    with st.spinner("Running Agentic RAG pipeline..."):
        try:
            agentic = AgenticRAGPipeline()
            agentic_result = agentic.run(query)
        except Exception as e:
            import traceback
            st.error(f"Agentic RAG Error: {e}")
            with st.expander("Full traceback"):
                st.code(traceback.format_exc())
            agentic_result = None

    if agentic_result:
        with col_agentic:
            st.subheader("Agentic RAG Answer")

            final_answer = agentic_result.get("final_answer", "")
            if final_answer:
                st.success(final_answer)
            else:
                st.warning("No answer generated. Check agent decisions below.")

            # Metrics row
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Iterations", agentic_result.get("iterations", 0))
            mc2.metric("Docs Used", len(agentic_result.get("retrieved_documents", [])))
            mc3.metric("Agents Run", len(agentic_result.get("agent_decisions", [])))
            mc4.metric("Strategy", agentic_result.get("routing_strategy", "—"))

            # Reflection score
            reflection = agentic_result.get("reflection", {})
            if reflection.get("overall_score"):
                score_pct = int(reflection["overall_score"] * 100)
                st.progress(score_pct / 100, text=f"Answer quality: {score_pct}%")

            # Agent decisions
            if show_agent_decisions:
                with st.expander("Agent Decisions (full trace)", expanded=False):
                    for d in agentic_result.get("agent_decisions", []):
                        st.markdown(f"**{d.get('agent', '?')}**")
                        st.json(d.get("decision", {}))
                        st.divider()

            # Retrieved chunks
            if show_retrieved_chunks:
                with st.expander("Retrieved & Validated Chunks", expanded=False):
                    docs = agentic_result.get("retrieved_documents", [])
                    if docs:
                        for i, doc in enumerate(docs):
                            rel = doc.get("relevance_score", 0)
                            ret = doc.get("retrieval_score", 0)
                            with st.container():
                                st.markdown(
                                    f"**Chunk {i+1}** — "
                                    f"`{doc.get('source', '?')}` p.{doc.get('page', '?')} | "
                                    f"relevance={rel:.3f} retrieval={ret:.3f}"
                                )
                                st.caption(doc.get("content", ""))
                                st.divider()
                    else:
                        st.warning("No chunks retrieved — try re-ingesting documents or rephrasing the query.")

    # ── Traditional RAG (optional) ───────────────────────────────────────────
    if compare_baseline:
        with st.spinner("Running Traditional RAG..."):
            try:
                trad = TraditionalRAGPipeline()
                trad_result = trad.run(query)
            except Exception as e:
                st.error(f"Traditional RAG Error: {e}")
                trad_result = None

        if trad_result:
            with col_traditional:
                st.subheader("Traditional RAG Answer")
                st.info(trad_result["answer"])
                st.metric("Docs Retrieved", trad_result["documents_retrieved"])

elif run_button and not query:
    st.warning("Please enter a question first.")
