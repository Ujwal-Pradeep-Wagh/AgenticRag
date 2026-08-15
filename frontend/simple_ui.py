"""
Simple Streamlit UI for quick testing
"""
import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import AgenticRAGPipeline
from ingestion.pipeline import DocumentIngestionPipeline

st.set_page_config(page_title="Agentic RAG - Simple", page_icon="🤖", layout="wide")

st.title("🤖 Agentic RAG - Simple Interface")

# Sidebar
with st.sidebar:
    st.header("📊 System Status")
    
    try:
        pipeline = DocumentIngestionPipeline()
        stats = pipeline.get_stats()
        st.metric("Documents in DB", stats["total_documents"])
        
        if stats["total_documents"] == 0:
            st.warning("⚠️ No documents yet!")
            st.info("Upload a PDF to get started")
    except Exception as e:
        st.error(f"Error: {e}")
    
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    
    if uploaded_file:
        os.makedirs("./data/uploads", exist_ok=True)
        file_path = f"./data/uploads/{uploaded_file.name}"
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        with st.spinner("Processing..."):
            try:
                result = pipeline.ingest(file_path)
                st.success(f"✅ Added {result['chunks_created']} chunks!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

# Main area
st.header("💬 Ask a Question")

query = st.text_area(
    "Your question:",
    placeholder="What is the company's remote work policy?",
    height=100
)

if st.button("🔍 Search", type="primary", disabled=not query):
    with st.spinner("🤖 Running multi-agent pipeline... (this may take 30-60 seconds)"):
        try:
            pipeline = AgenticRAGPipeline()
            result = pipeline.run(query)
            
            st.success("✅ Complete!")
            
            # Show answer
            st.subheader("📝 Answer")
            st.info(result.get("final_answer", "No answer generated"))
            
            # Show metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Documents Retrieved", len(result.get("retrieved_documents", [])))
            with col2:
                st.metric("Iterations", result.get("iterations", 0))
            with col3:
                agents = len(result.get("agent_decisions", []))
                st.metric("Agents Run", agents)
            
            # Show documents
            with st.expander("📄 View Retrieved Documents"):
                docs = result.get("retrieved_documents", [])
                if docs:
                    for i, doc in enumerate(docs, 1):
                        st.markdown(f"**Document {i}** (Score: {doc.get('score', 0):.3f})")
                        st.text(f"Source: {doc.get('source')} - Page {doc.get('page')}")
                        st.text(doc.get('content', 'No content')[:200] + "...")
                        st.divider()
                else:
                    st.warning("No documents in final result")
                    st.info("Documents may have been filtered by validation agent or retrieval returned 0 results")
            
            # Show agent decisions
            with st.expander("🧠 View Agent Decisions"):
                for decision in result.get("agent_decisions", []):
                    st.markdown(f"**{decision.get('agent')}**")
                    st.json(decision.get('decision', {}))
                    st.divider()
                    
        except Exception as e:
            st.error(f"❌ Error: {e}")
            with st.expander("Show Full Error"):
                import traceback
                st.code(traceback.format_exc())

# Footer
st.divider()
st.caption("Multi-Agent RAG System | LangGraph + Groq")
