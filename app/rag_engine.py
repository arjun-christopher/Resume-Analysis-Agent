# app/rag_engine.py - FAISS retriever + EDA + semantic sentence builder + structured answers
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time, math, re, json
import numpy as np
from collections import Counter, defaultdict

from sentence_transformers import SentenceTransformer
import faiss

# ---------- Embeddings ----------
_EMB_MODEL = None
def get_model():
    global _EMB_MODEL
    if _EMB_MODEL is None:
        _EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMB_MODEL

# ---------- EDA: term relationships ----------
TOKEN_PAT = re.compile(r"[A-Za-z][A-Za-z0-9_+#.\-]+")
STOP = set("""
a an and are as at be by for from has have in into is it its of on or that the this to was were will with your you
""".split())

def _tokenize(text: str) -> List[str]:
    toks = [t.lower() for t in TOKEN_PAT.findall(text)]
    return [t for t in toks if t not in STOP and len(t) > 1]

def _bigrams(tokens: List[str]) -> List[Tuple[str,str]]:
    return list(zip(tokens, tokens[1:]))

def eda_relationships(chunks: List[str]) -> Dict[str, Any]:
    tokens = []
    for ch in chunks:
        tokens.extend(_tokenize(ch))
    counts = Counter(tokens)
    bigrams = Counter(_bigrams(tokens))
    # simple co-occurrence: within-window pairs
    window = 8
    co = defaultdict(int)
    for ch in chunks:
        toks = _tokenize(ch)
        for i,t in enumerate(toks):
            for j in range(i+1, min(i+1+window, len(toks))):
                u = toks[j]
                if t != u:
                    key = tuple(sorted((t,u)))
                    co[key] += 1
    top_terms = [w for w,_ in counts.most_common(30)]
    top_pairs = [(*k,v) for k,v in sorted(co.items(), key=lambda kv: kv[1], reverse=True)[:30]]
    top_bi = bigrams.most_common(30)
    return {"top_terms": top_terms, "top_pairs": top_pairs, "top_bigrams": top_bi}

# ---------- Semantic sentence builder ----------
def semantic_sentences_from_eda(eda: Dict[str, Any]) -> List[str]:
    out = []
    terms = eda.get("top_terms", [])[:8]
    if terms:
        out.append(f"The corpus frequently mentions: {', '.join(terms)}.")
    for (a,b,cnt) in eda.get("top_pairs", [])[:6]:
        out.append(f"Terms '{a}' and '{b}' co-occur {cnt} times, indicating a relationship.")
    for ((a,b), cnt) in eda.get("top_bigrams", [])[:6]:
        out.append(f"Common phrase: '{a} {b}' ({cnt}).")
    return out

# ---------- Vector Index ----------
class VectorIndex:
    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.emb_dim = 384  # all-MiniLM-L6-v2
        self.index = faiss.IndexFlatIP(self.emb_dim)
        self.texts: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self.normed = False
        self.session_stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "industries_detected": set(),
            "unique_skills": set()
        }

    def _save(self):
        faiss.write_index(self.index, str(self.index_dir / "faiss.index"))
        with open(self.index_dir / "store.json", "w", encoding="utf-8") as f:
            json.dump({"texts": self.texts, "metas": self.metas}, f)

    def _load_if_exists(self):
        idx_path = self.index_dir / "faiss.index"
        js_path  = self.index_dir / "store.json"
        if idx_path.exists() and js_path.exists():
            self.index = faiss.read_index(str(idx_path))
            obj = json.loads((js_path).read_text(encoding="utf-8"))
            self.texts = obj["texts"]; self.metas = obj["metas"]
            self.normed = True

    def build_or_update_index(self, chunks: List[str], metas: List[Dict[str, Any]]):
        self._load_if_exists()
        model = get_model()
        X = model.encode(chunks, normalize_embeddings=True, convert_to_numpy=True)
        self.index.add(X)
        self.texts.extend(chunks)
        self.metas.extend(metas)
        self.normed = True

        # Stats
        self.session_stats["total_chunks"] = len(self.texts)
        # Skills
        for m in metas:
            for s in m.get("skills", []):
                self.session_stats["unique_skills"].add(s)

        self._save()

    def search(self, query: str, k: int = 8) -> Tuple[List[str], List[Dict[str,Any]], List[float]]:
        if not self.texts:
            return [], [], []
        model = get_model()
        q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        D, I = self.index.search(q, min(k, len(self.texts)))
        scores = D[0].tolist()
        idxs = I[0].tolist()
        hits_text = [self.texts[i] for i in idxs]
        hits_meta = [self.metas[i] for i in idxs]
        return hits_text, hits_meta, scores

    def get_session_summary(self) -> Dict[str, Any]:
        return {
            "total_documents": len(set(m["path"] for m in self.metas)) if self.metas else 0,
            "total_chunks": len(self.texts),
            "industries_detected": [],  # placeholder (not used here)
            "unique_skills_count": len(self.session_stats["unique_skills"]),
            "top_skills": sorted(list(self.session_stats["unique_skills"]))[:25],
            "documents": []
        }

# ---------- High-level Agent ----------
class ResumeRAGAgent:
    def __init__(self, index_dir: str):
        self.index = VectorIndex(index_dir)

    def build_or_update_index(self, chunks: List[str], metas: List[Dict[str, Any]]):
        self.index.build_or_update_index(chunks, metas)

    def _format_sources(self, hits_meta: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for m in hits_meta:
            out.append({
                "file": m.get("file"),
                "page": m.get("page", 1),
                "chunk_index": m.get("chunk_index", -1),
                "skills": ", ".join(m.get("skills", [])[:8]),
                "names": ", ".join(m.get("names", [])[:3]),
                "emails": ", ".join(m.get("emails", [])[:3]),
                "social_links": ", ".join(m.get("social_links", [])[:3]),
            })
        return out

    def answer(self, query: str) -> Dict[str, Any]:
        t0 = time.time()
        hits_text, hits_meta, scores = self.index.search(query, k=12)
        if not hits_text:
            return {"text": "No indexed resumes yet. Please upload PDFs or Word documents."}

        # EDA-based semantic context
        eda = eda_relationships(hits_text)
        sem_sentences = semantic_sentences_from_eda(eda)

        # Scored bullet points
        bullets = []
        for t, m, s in zip(hits_text, hits_meta, scores):
            who = ", ".join(m.get("names", [])[:1]) or "Candidate"
            page = m.get("page", 1); file = m.get("file","")
            bullets.append(
                f"- **{who}** — score {s:.3f} — _{file}, p.{page}_\n"
                f"  - Skills: {', '.join(m.get('skills', [])[:8]) or '—'}\n"
                f"  - Contact: {', '.join(m.get('emails', [])[:2]) or '—'}\n"
                f"  - Social: {', '.join(m.get('social_links', [])[:2]) or '—'}"
            )

        text = "### Answer\n" + "\n".join(bullets[:8])
        if sem_sentences:
            text += "\n\n### Corpus Semantics\n" + "\n".join([f"- {s}" for s in sem_sentences])

        return {
            "text": text,
            "hits_meta": self._format_sources(hits_meta[:8]),
            "retrieval_stats": {
                "total_hits": len(hits_text),
                "avg_score": float(np.mean(scores)) if scores else 0.0
            }
        }
