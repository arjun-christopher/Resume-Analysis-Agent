import os, json
from pathlib import Path
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import numpy as np

from .config import session_paths
from .embeddings import ensure_embedder

# Paths: faiss + store.pkl under vectors_dir
def _index_paths(session_id: str):
    p = session_paths(session_id).vectors_dir
    return p/"faiss.index", p/"store.pkl"

def load_or_create_store(session_id: str, embedder=None) -> FAISS:
    embedder = embedder or ensure_embedder()
    idx_path, store_path = _index_paths(session_id)
    if idx_path.exists() and store_path.exists():
        return FAISS.load_local(folder_path=str(session_paths(session_id).vectors_dir),
                                embeddings=embedder, allow_dangerous_deserialization=True)
    return FAISS.from_documents(documents=[], embedding=embedder)

def add_chunks_to_store(store: FAISS, candidate_id: str, chunks: List[Dict[str,Any]]) -> int:
    docs = [
        Document(
            page_content=c["text"],
            metadata={"candidate_id": candidate_id, "section": c["section"], "chunk_id": c["id"]}
        )
        for c in chunks
    ]
    if docs:
        store.add_documents(docs)
    return len(docs)

# We'll provide a wrapper saving function
def save_store(session_id: str, store: FAISS):
    store.save_local(folder_path=str(session_paths(session_id).vectors_dir))

def search_topk(store: FAISS, query: str, k: int, embedder=None):
    embedder = embedder or ensure_embedder()
    docs = store.similarity_search_with_score(query, k=k)
    out = []
    for doc, score in docs:
        out.append({
            "score": 1.0/(1.0+score) if isinstance(score,(float,int)) else 0.0,  # invert distance-ish
            "preview": doc.page_content[:400],
            "metadata": doc.metadata
        })
    return out
