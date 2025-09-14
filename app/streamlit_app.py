import os
from pathlib import Path
from zipfile import ZipFile, is_zipfile

import streamlit as st
from dotenv import load_dotenv

from utils import human_size, safe_listdir
from parsing import extract_docs_to_chunks_and_records, file_sha256
from rag_engine import AutoHybridRAG

# ---------------- Boot
load_dotenv()
st.set_page_config(page_title="RAG-Resume — Auto • Hybrid • Robust", page_icon="📄", layout="wide")

DATA_DIR   = Path("data"); DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = DATA_DIR / "uploads"; UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR  = DATA_DIR / "index"; INDEX_DIR.mkdir(parents=True, exist_ok=True)

# State
if "manifest" not in st.session_state:
    st.session_state["manifest"] = {}   # {filepath: sha256}
if "rag" not in st.session_state:
    st.session_state["rag"] = AutoHybridRAG(str(INDEX_DIR))
if "history" not in st.session_state:
    st.session_state["history"] = []    # [(role, text)]

# ---------------- Header
st.title("📄 RAG-Resume — Auto • Hybrid • Robust")
st.caption("Upload PDF/DOCX/PNG/JPG/ZIP (≤50 per session). Indexing happens automatically. Ask any question about the resumes. No DB. No knobs. No blockchain.")

# ---------------- Sidebar
with st.sidebar:
    if st.button("🧹 Clear session (uploads + index)"):
        for p in safe_listdir(UPLOAD_DIR): p.unlink(missing_ok=True)
        for p in safe_listdir(INDEX_DIR):  p.unlink(missing_ok=True)
        st.session_state["manifest"].clear()
        st.session_state["history"].clear()
        st.session_state["rag"] = AutoHybridRAG(str(INDEX_DIR))
        st.success("Cleared.")

    st.markdown("### Current files")
    files = [p for p in safe_listdir(UPLOAD_DIR) if p.suffix.lower() in {".pdf",".docx",".png",".jpg",".jpeg"}]
    if files:
        for p in files:
            st.write(f"- {p.name}  `{human_size(p.stat().st_size)}`")
    else:
        st.write("_None yet_")

# ---------------- Upload (auto-index)
st.subheader("Upload resumes")
u = st.file_uploader("Drop files here (PDF, DOCX, PNG, JPG, or ZIP)", accept_multiple_files=True,
                     type=["pdf","docx","png","jpg","jpeg","zip"])

def _save_and_expand(files):
    saved = []
    for f in files:
        dest = UPLOAD_DIR / f.name
        dest.write_bytes(f.read())
        saved.append(dest)
    # unzip
    for p in list(UPLOAD_DIR.iterdir()):
        if p.suffix.lower() == ".zip" and is_zipfile(p):
            with ZipFile(p, "r") as z:
                z.extractall(UPLOAD_DIR)
            p.unlink(missing_ok=True)
    # supported
    return [p for p in UPLOAD_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".pdf",".docx",".png",".jpg",".jpeg"}]

def _auto_index(new_paths):
    # cap: 50 per session
    if len(list(UPLOAD_DIR.glob("*"))) > 50:
        st.error("Maximum of 50 files per session. Please remove some files.")
        return

    changed = []
    for p in new_paths:
        h = file_sha256(p)
        if st.session_state["manifest"].get(str(p)) != h:
            changed.append(p)

    if not changed: return
    with st.spinner(f"Indexing {len(changed)} file(s)…"):
        chunks, metas, _records = extract_docs_to_chunks_and_records(changed)  # records computed but not stored
        if chunks:
            st.session_state["rag"].build_or_update_index(chunks, metas)
        for p in changed:
            st.session_state["manifest"][str(p)] = file_sha256(p)
    st.success("Index updated.")

if u:
    paths = _save_and_expand(u)
    _auto_index(paths)

# ---------------- Chat UI
st.subheader("Chat with your resumes")
for role, msg in st.session_state["history"]:
    st.chat_message(role).write(msg)

prompt = st.chat_input("Ask anything (e.g., 'Rank candidates with Python + AWS; cite file & page').")
if prompt:
    st.session_state["history"].append(("user", prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            ans = st.session_state["rag"].answer(prompt)
        st.session_state["history"].append(("assistant", ans["text"]))
        st.write(ans["text"])
