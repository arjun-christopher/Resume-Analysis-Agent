# app/streamlit_app.py - Focused UI: Upload (auto-index) + Chat
from __future__ import annotations
import time
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
import pandas as pd

from utils import human_size, list_supported_files, clear_dir, safe_listdir
from parsing import extract_docs_to_chunks_and_records
from fast_semantic_rag import create_fast_semantic_rag

load_dotenv()
st.set_page_config(page_title="RAG - Resume", layout="wide")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR = DATA_DIR / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

if "agent" not in st.session_state:
    st.session_state.agent = create_fast_semantic_rag(str(INDEX_DIR))
if "manifest" not in st.session_state:
    st.session_state.manifest = {}
if "history" not in st.session_state:
    st.session_state.history = []

st.title("Resume Analysis Agent — RAG")

# ---- Sidebar (upload only PDFs & DOC/DOCX) ----
with st.sidebar:
    st.header("Upload Resumes")
    uploaded = st.file_uploader(
        "Drop PDF or Word files here", type=["pdf","doc","docx"], accept_multiple_files=True,
        help="Only PDFs and Word documents are supported"
    )

    if uploaded:
        start = time.time()
        saved_files = []
        for uf in uploaded:
            dest = UPLOAD_DIR / uf.name
            dest.write_bytes(uf.read())
            saved_files.append(dest)

        # Auto-index immediately — no button
        with st.spinner(f"Indexing {len(saved_files)} file(s)..."):
            chunks, metas, records = extract_docs_to_chunks_and_records(saved_files)
            if chunks:
                # Convert to the format expected by fast semantic RAG
                documents = [chunk.page_content if hasattr(chunk, 'page_content') else str(chunk) for chunk in chunks]
                metadata = [meta for meta in metas]
                st.session_state.agent.add_documents(documents, metadata)
        st.success(f"Uploaded & indexed {len(saved_files)} file(s) in {time.time()-start:.2f}s")

    st.markdown("---")
    files = list_supported_files(UPLOAD_DIR)
    sizes = sum(p.stat().st_size for p in files)
    st.caption(f"Session: {len(files)} files • {human_size(sizes)}")
    if files:
        with st.expander("Files", expanded=False):
            for p in files[:20]:
                st.write(f"• {p.name} — {human_size(p.stat().st_size)}")
        if len(files) > 20:
            st.caption(f"...and {len(files)-20} more")

    if st.button("Clear Session", use_container_width=True):
        clear_dir(UPLOAD_DIR); clear_dir(INDEX_DIR)
        st.session_state.agent = create_fast_semantic_rag(str(INDEX_DIR))
        st.session_state.history.clear()
        st.success("Cleared all files and index.")

# ---- Main: Chat ----
st.subheader("AI-Powered Resume Analysis")

if not st.session_state.history:
    st.info("Upload PDFs or Word documents to build a searchable knowledge base, then ask questions about candidates.")
    
    # Example queries
    with st.expander("Example Questions", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Analysis & Ranking:**")
            examples1 = [
                "Rank candidates by Python and AWS experience",
                "Who has the most leadership experience?",
                "Compare candidates' machine learning skills",
                "Find candidates with 5+ years in data science",
                "Which candidate is best for a senior developer role?"
            ]
            for ex in examples1:
                if st.button(f"{ex}", key=f"ex1_{ex[:20]}"):
                    st.session_state.example_query = ex
        
        with col2:
            st.markdown("**Information Extraction:**")
            examples2 = [
                "List all email addresses and LinkedIn profiles",
                "Show candidates' educational backgrounds",
                "Extract all certifications mentioned",
                "What companies have candidates worked at?",
                "Perform EDA analysis on the resume corpus"
            ]
            for ex in examples2:
                if st.button(f"{ex}", key=f"ex2_{ex[:20]}"):
                    st.session_state.example_query = ex

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])

# Handle example query selection
if hasattr(st.session_state, 'example_query'):
    prompt = st.session_state.example_query
    del st.session_state.example_query
else:
    prompt = st.chat_input("Ask about the uploaded resumes…")

if prompt:
    st.session_state.history.append({"role":"user","text":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing resumes with advanced RAG..."):
            # Use advanced RAG system
            result = st.session_state.agent.query(prompt)
            
            # Display main response
            st.markdown(result["answer"])
            
            # Show processing stats
            processing_time = result.get("processing_time", 0)
            method = result.get("method", "unknown")
            source_docs = result.get("source_documents", [])
            
            if source_docs:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Documents Found", len(source_docs))
                with col2:
                    st.metric("Processing Time", f"{processing_time:.2f}s")
                with col3:
                    st.metric("Method", method.title())
            
            # Show source documents if available
            if source_docs:
                with st.expander("Source Documents", expanded=False):
                    for i, doc in enumerate(source_docs[:5]):
                        st.markdown(f"**Document {i+1}:**")
                        # Handle both dict and object formats
                        if isinstance(doc, dict):
                            content = doc.get('content', '')
                            metadata = doc.get('metadata', {})
                            relevance_score = doc.get('relevance_score', 0.0)
                        else:
                            content = getattr(doc, 'page_content', '')
                            metadata = getattr(doc, 'metadata', {})
                            relevance_score = getattr(doc, 'relevance_score', 0.0)
                        
                        st.text(content[:500] + "..." if len(content) > 500 else content)
                        if metadata:
                            st.json(metadata)
                        if relevance_score > 0:
                            st.caption(f"Relevance Score: {relevance_score:.3f}")
                        st.markdown("---")
            
            # Show system stats
            with st.expander("System Information", expanded=False):
                stats = st.session_state.agent.get_system_stats()
                st.json(stats)
    
    st.session_state.history.append({"role":"assistant","text":result["answer"]})
