# app/streamlit_app.py - Focused UI: Upload (auto-index) + Chat
from __future__ import annotations

import sys
import os
from pathlib import Path
import time
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

# File limits for indexing
MAX_FILES_PER_SESSION = 5
MAX_FILES_PER_BATCH = 5

if "agent" not in st.session_state:
    st.session_state.agent = create_advanced_rag_engine(str(INDEX_DIR))
if "manifest" not in st.session_state:
    st.session_state.manifest = {}
if "history" not in st.session_state:
    st.session_state.history = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Check if index is already loaded at startup
if st.session_state.agent.has_index() and not st.session_state.indexed_files:
    # Index was loaded from disk, populate indexed_files from uploads directory
    existing_files = list_supported_files(UPLOAD_DIR)
    st.session_state.indexed_files = {f.name for f in existing_files}
    if existing_files:
        st.info(f"Loaded existing index with {len(existing_files)} files from previous session")

st.title("Resume Analysis Agent")

# ---- Sidebar (upload only PDFs & DOC/DOCX) ----
with st.sidebar:
    st.header("Upload Resumes")
    
    # Display file limit warning
    current_file_count = len(st.session_state.indexed_files)
    files_remaining = MAX_FILES_PER_SESSION - current_file_count
    
    if files_remaining > 0:
        st.info(f"Session limit: {current_file_count}/{MAX_FILES_PER_SESSION} files indexed | {files_remaining} remaining")
    else:
        st.warning(f"Session limit reached: {MAX_FILES_PER_SESSION}/{MAX_FILES_PER_SESSION} files indexed")
        st.caption("Clear session to upload new files")
    
    uploaded = st.file_uploader(
        "Drop PDF or Word files here", type=["pdf","doc","docx"], accept_multiple_files=True,
        help=f"Maximum {MAX_FILES_PER_SESSION} files per session. Only PDFs and Word documents are supported.",
        key=f"file_uploader_{st.session_state.uploader_key}",
        disabled=(files_remaining <= 0)
    )

    if uploaded:
        # Check if adding these files would exceed the session limit
        current_count = len(st.session_state.indexed_files)
        new_file_names = [uf.name for uf in uploaded if uf.name not in st.session_state.indexed_files]
        
        if current_count + len(new_file_names) > MAX_FILES_PER_SESSION:
            allowed_count = MAX_FILES_PER_SESSION - current_count
            st.error(f"Cannot upload {len(new_file_names)} file(s). Session limit: {MAX_FILES_PER_SESSION} files maximum.")
            st.error(f"You can only add {allowed_count} more file(s). Currently indexed: {current_count}/{MAX_FILES_PER_SESSION}")
            st.info("Click 'Clear Session' to reset and upload new files.")
        else:
            start = time.time()
            saved_files = []
            new_files = []
            
            # Enforce per-batch limit
            files_to_process = uploaded[:MAX_FILES_PER_BATCH]
            if len(uploaded) > MAX_FILES_PER_BATCH:
                st.warning(f"Processing only first {MAX_FILES_PER_BATCH} files in this batch.")
            
            for uf in files_to_process:
                dest = UPLOAD_DIR / uf.name
                dest.write_bytes(uf.read())
                saved_files.append(dest)
                
                # Check if this file is new (not already indexed)
                if uf.name not in st.session_state.indexed_files:
                    new_files.append(dest)
                    st.session_state.indexed_files.add(uf.name)

            # Only index new files
            if new_files:
                try:
                    with st.spinner(f"Indexing {len(new_files)} new file(s)..."):
                        chunks, metas, records = extract_docs_to_chunks_and_records(new_files)
                        
                        if chunks:
                            # Convert to the format expected by fast semantic RAG
                            documents = [chunk.page_content if hasattr(chunk, 'page_content') else str(chunk) for chunk in chunks]
                            metadata = [meta for meta in metas]
                            
                            st.session_state.agent.add_documents(documents, metadata)
                    
                    processing_time = time.time() - start
                    st.success(f"Indexed {len(new_files)} new file(s) in {processing_time:.2f}s")
                    st.info(f"Total: {len(st.session_state.indexed_files)}/{MAX_FILES_PER_SESSION} files in session")
                except Exception as e:
                    st.error(f"Error indexing files: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
                    # Remove failed files from indexed set
                    for nf in new_files:
                        st.session_state.indexed_files.discard(nf.name)
            else:
                st.info(f"All {len(saved_files)} file(s) already indexed, skipping re-indexing")

    st.markdown("---")
    files = list_supported_files(UPLOAD_DIR)
    sizes = sum(p.stat().st_size for p in files)
    st.caption(f"Session: {len(files)} files • {human_size(sizes)}")
    
    # Show indexed resumes with their details
    if st.session_state.agent.has_index():
        resume_list = st.session_state.agent.get_resume_list()
        if resume_list:
            st.markdown("### Indexed Resumes")
            st.caption(f"{len(resume_list)} unique candidates indexed")
            
            # Display candidate names in the sidebar
            with st.expander("Uploaded Resumes", expanded=False):
                for idx, resume_info in enumerate(resume_list, 1):
                    candidate_name = resume_info.get('candidate_name', 'Unknown')
                    resume_name = resume_info.get('resume_name', 'Unknown')

                    st.write(f"{idx}. **{candidate_name}** ({resume_name})")

    # Single-click Clear Session - resets everything immediately
    if st.button("Clear Session", use_container_width=True, type="primary"):
        # Clear the agent's index
        if hasattr(st.session_state, 'agent'):
            st.session_state.agent.clear_index()
        
        # Clear directories completely
        clear_dir(UPLOAD_DIR)
        clear_dir(INDEX_DIR)
        
        # Increment uploader key to reset the file uploader widget
        st.session_state.uploader_key += 1
        
        # Clear ALL session state variables except uploader_key
        keys_to_keep = ['uploader_key']
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
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
        try:
            with st.spinner("Analyzing resume(s)..."):
                # Use advanced RAG system
                result = st.session_state.agent.query(prompt)
                
                # Display main response
                st.markdown(result["answer"])
                
                # Show additional metadata if multi-resume query
                if result.get('grouped_by_resume'):
                    resume_count = result.get('resume_count', 0)
                    st.info(f"ℹ️ Analysis covers **{resume_count} candidates**")
                
                # Show source information in expandable section
                if result.get('source_documents'):
                    with st.expander("Source Information", expanded=False):
                        # Group sources by candidate/resume
                        sources_by_candidate = {}
                        for doc in result['source_documents'][:15]:  # Show more sources
                            candidate_name = doc.metadata.get('candidate_name')
                            resume_name = doc.metadata.get('resume_name', 'Unknown')
                            
                            # Use candidate name as primary key, fallback to resume name
                            display_key = candidate_name if candidate_name else resume_name
                            
                            if display_key not in sources_by_candidate:
                                sources_by_candidate[display_key] = {
                                    'candidate_name': candidate_name,
                                    'resume_name': resume_name,
                                    'sources': []
                                }
                            
                            sources_by_candidate[display_key]['sources'].append({
                                'section': doc.metadata.get('section_type', 'unknown'),
                                'score': doc.metadata.get('relevance_score', 0),
                                'text': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
                            })
                        
                        # Display grouped sources with candidate names
                        for display_key, data in sources_by_candidate.items():
                            candidate_name = data['candidate_name']
                            resume_name = data['resume_name']
                            
                            if candidate_name:
                                st.markdown(f"### **{candidate_name}**")
                                st.caption(f"File: {resume_name}")
                            else:
                                st.markdown(f"### **{resume_name}**")
                            
                            for source in data['sources']:
                                st.markdown(f"- **[{source['section'].title()}]** (relevance: {source['score']:.3f})")
                                st.caption(f"  {source['text']}")
                            
                            st.markdown("---")
            
            st.session_state.history.append({"role":"assistant","text":result["answer"]})
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            st.error(error_msg)
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
            st.session_state.history.append({"role":"assistant","text":error_msg})
