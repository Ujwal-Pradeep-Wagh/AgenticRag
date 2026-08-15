"""
frontend/app.py
Simple Streamlit UI for Agentic RAG

Features:
- Upload PDF documents
- Ask questions
- View retrieved chunks
- View agent decisions
- View final answer
- Compare with Traditional RAG
"""

import streamlit as st
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import AgenticRAGPipeline, TraditionalRAGPipeline
from ingestion.pipeline import DocumentIngestionPipeline


st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agentic RAG System")
st.markdown("""
This system uses multiple AI agents to collaboratively improve information retrieval:
- **Query Understanding**: Analyzes intent and complexity
- **Query Routing**: Decides optimal retrieval strategy  
- **Query Rewriting**: Optimizes queries for search
- **Retrieval**: Fetches relevant documents
- **Validation**: Filters low-quality context
- **Reflection**: Self-corrects weak answers
""")

# Sidebar
with st.sidebar:
    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        os.makedirs("./data/uploads", exist_ok=True)
        file_path = f"./data/uploads/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        with st.spinner("Processing document..."):
            try:
                pipeline = DocumentIngestionPipeline()
                result = pipeline.ingest(file_path)
                st.success(f"✅ Ingested: {result['chunks_created']} chunks from {result['pages_loaded']} pages")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    st.header("⚙️ Settings")
    show_agent_decisions = st.checkbox("Show Agent Decisions", value=True)
    show_retrieved_chunks = st.checkbox("Show Retrieved Chunks", value=True)
    compare_baseline = st.checkbox("Compare with Traditional RAG", value=False)

    st.header("📊 Stats")
    try:
        stats = DocumentIngestionPipeline().get_stats()
        st.metric("Total Documents", stats["total_documents"])
    except:
        st.info("No documents ingested yet")

# Main area
st.header("💬 Ask a Question")

query = st.text_input("Enter your question:", placeholder="What is the company's remote work policy?")

if st.button("🔍 Search", type="primary") and query:
    if compare_baseline:
        col1, col2 = st.columns(2)
    else:
        col1 = st.container()

    # Progress tracking
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    progress_text.text("Initializing pipeline...")
    progress_bar.progress(10)
    
    with st.spinner("Agentic RAG processing..."):
        try:
            progress_text.text("Running multi-agent pipeline...")
            progress_bar.progress(30)
            
            agentic = AgenticRAGPipeline()
            agentic_result = agentic.run(query)
            
            progress_bar.progress(100)
            progress_text.text("✅ Complete!")
            
        except Exception as e:
            progress_bar.empty()
            progress_text.empty()
            st.error(f"Agentic RAG Error: {str(e)}")
            import traceback
            with st.expander("Show Error Details"):
                st.code(traceback.format_exc())
            agentic_result = None

    # Clear progress indicators
    progress_bar.empty()
    progress_text.empty()

    if agentic_result:
        with col1:
            st.subheader("🤖 Agentic RAG Answer")
            
            # Display answer
            if agentic_result.get("final_answer"):
                st.info(agentic_result["final_answer"])
            else:
                st.warning("No answer generated. Check agent decisions below for details.")

            # Metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Iterations", agentic_result.get("iterations", 0))
            with col_b:
                st.metric("Documents Used", len(agentic_result.get("retrieved_documents", [])))
            with col_c:
                agents_run = len(agentic_result.get("agent_decisions", []))
                st.metric("Agents Executed", agents_run)

            if show_agent_decisions:
                st.subheader("🧠 Agent Decisions")
                decisions = agentic_result.get("agent_decisions", [])
                if decisions:
                    for decision in decisions:
                        agent_name = decision.get('agent', 'Unknown')
                        with st.expander(f"📌 {agent_name}"):
                            st.json(decision.get("decision", {}))
                else:
                    st.info("No agent decisions recorded")

            if show_retrieved_chunks:
                st.subheader("📄 Retrieved Chunks")
                docs = agentic_result.get("retrieved_documents", [])
                if docs:
                    for i, doc in enumerate(docs):
                        score = doc.get('score', 0)
                        with st.expander(f"Chunk {i+1} (Relevance: {score:.3f})"):
                            st.write(f"**Source:** {doc.get('source', 'unknown')} (Page {doc.get('page', 'N/A')})")
                            st.write(doc.get("content", "No content"))
                else:
                    st.warning("⚠️ No documents retrieved or all filtered out by validation agent")
                    st.info("This can happen if:\n- Vector store is empty\n- Validation agent filtered all docs (relevance < 0.5)\n- Query doesn't match any documents")

    if compare_baseline:
        with st.spinner("Traditional RAG processing..."):
            try:
                traditional = TraditionalRAGPipeline()
                trad_result = traditional.run(query)
            except Exception as e:
                st.error(f"Traditional RAG Error: {str(e)}")
                trad_result = None

        if trad_result:
            with col2:
                st.subheader("📚 Traditional RAG Answer")
                st.warning(trad_result["answer"])
                st.metric("Documents Retrieved", trad_result["documents_retrieved"])
