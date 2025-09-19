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
from rag_engine import ResumeRAGAgent

load_dotenv()
st.set_page_config(page_title="Resume Analysis Agent (RAG)", layout="wide")

DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = DATA_DIR / "uploads"; UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR = DATA_DIR / "index";   INDEX_DIR.mkdir(parents=True, exist_ok=True)

if "agent" not in st.session_state:
    st.session_state.agent = ResumeRAGAgent(str(INDEX_DIR))
if "manifest" not in st.session_state:
    st.session_state.manifest = {}
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, Any]] = []

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
                st.session_state.agent.build_or_update_index(chunks, metas)
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
        st.session_state.agent = ResumeRAGAgent(str(INDEX_DIR))
        st.session_state.history.clear()
        st.success("Cleared all files and index.")

# ---- Main: Chat ----
st.subheader("Chat")
if not st.session_state.history:
    st.info("Upload PDFs or Word documents from the sidebar, then ask questions like:\n"
            "- Rank candidates by Python & AWS experience\n"
            "- Who has ML + leadership?\n"
            "- List LinkedIn/GitHub links per candidate")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])

prompt = st.chat_input("Ask about the uploaded resumes…")
if prompt:
    st.session_state.history.append({"role":"user","text":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = st.session_state.agent.answer(prompt)
            st.markdown(result["text"])
            # Show provenance table
            if result.get("hits_meta"):
                st.markdown("**Sources**")
                df = pd.DataFrame(result["hits_meta"])
                st.dataframe(df, use_container_width=True)
    st.session_state.history.append({"role":"assistant","text":result["text"]})
