# app/streamlit_app.py - Focused UI: Upload (auto-index) + Chat
from __future__ import annotations
import time
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
import pandas as pd

from utils import human_size, list_supported_files, clear_dir, safe_listdir
from parser import extract_docs_to_chunks_and_records
from rag_engine import create_advanced_rag_engine

load_dotenv()
st.set_page_config(page_title="Resume Analysis Agent", layout="wide")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR = DATA_DIR / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

if "agent" not in st.session_state:
    st.session_state.agent = create_advanced_rag_engine(str(INDEX_DIR))
if "manifest" not in st.session_state:
    st.session_state.manifest = {}
if "history" not in st.session_state:
    st.session_state.history = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

# Check if index is already loaded at startup
if st.session_state.agent.has_index() and not st.session_state.indexed_files:
    # Index was loaded from disk, populate indexed_files from uploads directory
    existing_files = list_supported_files(UPLOAD_DIR)
    st.session_state.indexed_files = {f.name for f in existing_files}
    if existing_files:
        st.info(f"📂 Loaded existing index with {len(existing_files)} files from previous session")

st.title("Resume Analysis Agent")

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
        new_files = []
        
        for uf in uploaded:
            dest = UPLOAD_DIR / uf.name
            dest.write_bytes(uf.read())
            saved_files.append(dest)
            
            # Check if this file is new (not already indexed)
            if uf.name not in st.session_state.indexed_files:
                new_files.append(dest)
                st.session_state.indexed_files.add(uf.name)

        # Only index new files
        if new_files:
            with st.spinner(f"Indexing {len(new_files)} new file(s)..."):
                chunks, metas, records = extract_docs_to_chunks_and_records(new_files)
                if chunks:
                    # Convert to the format expected by fast semantic RAG
                    documents = [chunk.page_content if hasattr(chunk, 'page_content') else str(chunk) for chunk in chunks]
                    metadata = [meta for meta in metas]
                    st.session_state.agent.add_documents(documents, metadata)
            st.success(f"Indexed {len(new_files)} new file(s) in {time.time()-start:.2f}s")
        else:
            st.info(f"All {len(saved_files)} file(s) already indexed, skipping re-indexing")

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

    # Single-click Clear Session - resets everything immediately
    if st.button("Clear Session", use_container_width=True, type="primary"):
        # Clear the agent's index
        if hasattr(st.session_state, 'agent'):
            st.session_state.agent.clear_index()
        
        # Clear directories completely
        clear_dir(UPLOAD_DIR)
        clear_dir(INDEX_DIR)
        
        # Clear ALL session state variables
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Reinitialize essential state from scratch
        st.session_state.agent = create_advanced_rag_engine(str(INDEX_DIR))
        st.session_state.history = []
        st.session_state.manifest = {}
        st.session_state.indexed_files = set()
        
        # Force complete UI refresh
        st.rerun()

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
                    st.rerun()
        
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
                    st.rerun()

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])

# Handle example query selection first
example_prompt = None
if hasattr(st.session_state, 'example_query'):
    example_prompt = st.session_state.example_query
    del st.session_state.example_query

# Always show the chat input field - this ensures it's persistent
user_input = st.chat_input("Ask about the uploaded resumes…")

# Use example prompt if available, otherwise use user input
prompt = example_prompt or user_input

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
    
    st.session_state.history.append({"role":"assistant","text":result["answer"]})
