import os
import json
import hashlib
from typing import List, Dict, Any, Tuple
from pathlib import Path
from collections import defaultdict

# Local retrieval stack
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS as LCFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import numpy as np

# Optional accuracy mode
_HAS_RA = False
try:
    from raganything import RAGAnything, RAGAnythingConfig
    from lightrag.llm.openai import openai_complete_if_cache  # we'll still use local embeddings
    from lightrag.utils import EmbeddingFunc
    _HAS_RA = True
except Exception:
    _HAS_RA = False

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _tokenize(text: str) -> List[str]:
    return [t for t in ''.join([c.lower() if c.isalnum() else ' ' for c in text]).split() if t]

def _rrf(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)

# Extended skill vocab (industry-level, multi-domain)
DEFAULT_SKILLS = {
    # Programming Languages
    "python","java","c","c++","c#","go","rust","ruby","perl","scala","php","r","swift","objective-c",
    "javascript","typescript","kotlin","dart","matlab","shell","bash","powershell","haskell","fortran",
    
    # Web Development
    "html","css","sass","less","jquery","react","angular","vue","svelte","next.js","nuxt.js",
    "node","express","fastapi","nestjs","spring boot","flask","django","rails","laravel",
    
    # Mobile Development
    "android","ios","react native","flutter","ionic","xamarin","swiftui","kotlin multiplatform",
    
    # Databases & Data Engineering
    "sql","mysql","postgres","oracle","mssql","mongodb","cassandra","couchdb","dynamodb","elasticsearch",
    "redis","neo4j","janusgraph","snowflake","redshift","bigquery","hive","hbase","presto","trino",
    "kafka","spark","hadoop","beam","flink","airflow","databricks","luigi","storm",
    
    # Cloud Platforms
    "aws","azure","gcp","ibm cloud","oracle cloud","digitalocean","heroku","cloudflare","vercel",
    
    # DevOps, CI/CD, Infrastructure
    "docker","kubernetes","openshift","terraform","ansible","puppet","chef","jenkins","github actions",
    "gitlab ci","circleci","argo cd","helm","prometheus","grafana","nagios","splunk","elastic stack",
    
    # Operating Systems & Tools
    "linux","unix","windows","macos","vim","emacs","git","svn","mercurial","jira","confluence",
    
    # AI / ML / Data Science
    "pandas","numpy","scipy","matplotlib","seaborn","plotly","statsmodels",
    "scikit-learn","xgboost","lightgbm","catboost",
    "pytorch","tensorflow","keras","jax","huggingface","transformers","spacy","nltk","gensim",
    "mlflow","dvc","ray","optuna","onnx","openvino","accelerate",
    "nlp","llm","chatgpt","computer vision","opencv","stable diffusion","deep learning",
    "reinforcement learning","recommendation systems","time series","anomaly detection",
    
    # BI & Analytics
    "tableau","power bi","qlikview","looker","superset","metabase","mode","excel","sas","stata",
    
    # Blockchain & Web3
    "solidity","ethereum","polygon","hyperledger","web3","truffle","ganache","hardhat",
    "ipfs","smart contracts","nft","defi","rust (solana)",
    
    # Cybersecurity
    "penetration testing","ethical hacking","burpsuite","metasploit","wireshark","nmap","nessus",
    "splunk security","siem","firewalls","ids","ips","zero trust","owasp","iso27001","hipaa","gdpr",
    
    # Testing & QA
    "selenium","cypress","pytest","junit","mocha","chai","jest","playwright","postman","soapui",
    "appium","loadrunner","jmeter","rest assured","karate","robot framework",
    
    # Project / Product / Agile
    "scrum","kanban","agile","safe","prince2","pmp","okrs","roadmapping","stakeholder management",
    
    # UI / UX / Design
    "figma","adobe xd","sketch","photoshop","illustrator","canva","invision","zeplin",
    "wireframing","prototyping","usability testing","design thinking","accessibility","hci",
    
    # Emerging Tech
    "iot","edge computing","digital twins","vr","ar","xr","quantum computing","robotics","5g",
    
    # Soft Skills (in JD keywords)
    "communication","teamwork","leadership","problem solving","critical thinking",
    "collaboration","adaptability","creativity","mentoring","time management"
}

# ---------------- LLM fallback manager ----------------
class LLMFallback:
    """
    Tries: OpenAI -> Gemini -> Transformers (free local) -> Ollama -> None
    Returns plain string or None.
    """
    def __init__(self):
        self.openai_api = os.getenv("OPENAI_API_KEY")
        self.gemini_api = os.getenv("GEMINI_API_KEY")
        self.hf_model   = os.getenv("HF_LOCAL_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.ollama_url = os.getenv("OLLAMA_BASE_URL")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3")

    def _format_prompt(self, question: str, context: str, style: str = "bullet points"):
        sys = ("You are a precise resume analysis assistant. Answer ONLY from the given context. "
               "If the answer isn't in the context, reply exactly: 'Not found in the uploaded resumes.' "
               "Always include an **Evidence** section with filename and page for each claim.")
        user = f"Question:\n{question}\n\nContext:\n{context}\n\nReturn a structured answer in {style}."
        return sys, user

    def try_openai(self, question: str, context: str, style: str):
        if not self.openai_api: return None
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api)
            sys, user = self._format_prompt(question, context, style)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":sys},{"role":"user","content":user}],
                temperature=0.1,
            )
            return resp.choices[0].message.content
        except Exception:
            return None

    def try_gemini(self, question: str, context: str, style: str):
        if not self.gemini_api: return None
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_api)
            model = genai.GenerativeModel("gemini-1.5-flash")
            sys, user = self._format_prompt(question, context, style)
            resp = model.generate_content([sys, user])
            return resp.text
        except Exception:
            return None

    def try_transformers(self, question: str, context: str, style: str):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            prompt = (
                "### System:\n"
                "You answer strictly from the context. If not present, say: Not found in the uploaded resumes.\n"
                "Always add an Evidence section with filename and page.\n\n"
                "### Context:\n" + context + "\n\n"
                "### User:\n" + question + f"\n\nReturn a structured answer in {style}.\n"
            )
            pipe = pipeline(
                "text-generation",
                model=self.hf_model,
                torch_dtype="auto",
                device_map="auto",
                max_new_tokens=600,
                do_sample=False,
            )
            out = pipe(prompt, max_new_tokens=600)[0]["generated_text"]
            return out.split("### User:")[-1].strip() if out else out
        except Exception:
            return None

    def try_ollama(self, question: str, context: str, style: str):
        if not self.ollama_url: return None
        try:
            import ollama
            sys, user = self._format_prompt(question, context, style)
            r = ollama.chat(model=self.ollama_model, messages=[
                {"role":"system","content":sys},
                {"role":"user","content":user},
            ])
            return r["message"]["content"]
        except Exception:
            return None

    def generate(self, question: str, context: str, style: str = "bullet points"):
        return (
            self.try_openai(question, context, style)
            or self.try_gemini(question, context, style)
            or self.try_transformers(question, context, style)
            or self.try_ollama(question, context, style)
        )

# ---------------- Accuracy mode (RAG-Anything) ----------------
class RAGAnythingEngine:
    """Uses RAG-Anything with LOCAL embeddings to avoid rate limits. Optional."""
    def __init__(self, working_dir: str):
        from sentence_transformers import SentenceTransformer
        from lightrag.utils import EmbeddingFunc

        self.working_dir = working_dir
        self.ra_cfg = RAGAnythingConfig(
            working_dir=working_dir,
            parser="mineru",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=False,
        )
        # local HF embeddings
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self._hf = SentenceTransformer(model_name)
        dim = self._hf.get_sentence_embedding_dimension()
        def _embed(texts):
            vec = self._hf.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return [v.tolist() for v in np.asarray(vec)]
        self.emb = EmbeddingFunc(embedding_dim=dim, max_token_size=8192, func=_embed)

        # LLM wrapper for RAG-Anything (we'll let our LLMFallback handle generation later;
        # here we only need a stub to allow ingestion/query if RA requests)
        def _llm(prompt, system_prompt=None, history_messages=[], **kw):
            return prompt  # RA sometimes calls this during parsing; return prompt as stub

        self.ra = RAGAnything(config=self.ra_cfg, llm_model_func=_llm, embedding_func=self.emb)

    def ingest_files(self, file_paths: List[str]):
        import asyncio
        async def _run():
            for fp in file_paths:
                try:
                    await self.ra.process_document_complete(
                        file_path=fp, output_dir=self.working_dir,
                        parse_method="auto", parser="mineru", display_stats=False
                    )
                except Exception:
                    pass
        asyncio.run(_run())

    def retrieve(self, query: str, top_k: int = 8) -> List[Dict[str,Any]]:
        # Use RA to find contexts; we will generate with our fallback LLM
        import asyncio
        async def _aq():
            return await self.ra.aquery(query, mode="hybrid")
        try:
            res = asyncio.run(_aq())
        except Exception:
            res = self.ra.query(query, mode="hybrid")
        ctxs = res.get("contexts") or res.get("chunks") or []
        out = []
        for c in ctxs[:top_k]:
            md = c.get("metadata") or {}
            out.append({
                "text": c.get("text") or c.get("content") or "",
                "file": md.get("file") or md.get("file_path") or md.get("source"),
                "page": md.get("page") or md.get("page_idx"),
            })
        return out

# ---------------- Local hybrid engine ----------------
class LocalHybridEngine:
    def __init__(self, index_dir: str, embed_model: str):
        self.index_dir = index_dir
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        self.emb = HuggingFaceEmbeddings(model_name=embed_model)
        self.docs: List[Document] = []
        self.bm25 = None
        self.doc_tokens: List[List[str]] = []
        self.lc_store = None
        self.reranker = None
        try:
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            self.reranker = None
        # Try load
        try:
            self.lc_store = LCFAISS.load_local(self.index_dir, self.emb, allow_dangerous_deserialization=True)
        except Exception:
            self.lc_store = None

    def build_or_update(self, chunks: List[str], metas: List[Dict[str,Any]]):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        new_docs = []
        for i, t in enumerate(chunks):
            md = metas[i]
            parts = splitter.split_text(t) or [t]
            for pt in parts:
                new_docs.append(Document(page_content=pt, metadata=md))
        self.docs.extend(new_docs)
        if self.docs:
            self.lc_store = LCFAISS.from_documents(self.docs, self.emb)
            self.lc_store.save_local(self.index_dir)
        self.doc_tokens = [_tokenize(d.page_content) for d in self.docs]
        self.bm25 = BM25Okapi(self.doc_tokens) if self.doc_tokens else None

    def retrieve(self, query: str, top_k: int = 8) -> List[Dict[str,Any]]:
        dense = self._dense(query, 40)
        sparse = self._sparse(query, 40)
        fused = self._fuse(query, dense, sparse)
        pool = fused[: max(50, top_k)]
        if self.reranker:
            try:
                texts = [d.page_content for _, d in pool]
                scores = self.reranker.predict([[query, t] for t in texts]).tolist()
                pool = sorted(zip(scores, [d for _, d in pool]), key=lambda x: x[0], reverse=True)
            except Exception:
                pass
        top = pool[:top_k]
        out = []
        for s, d in top:
            out.append({"text": d.page_content, "file": d.metadata.get("file"), "page": d.metadata.get("page")})
        return out

    def _dense(self, q, k):
        if not self.lc_store: return []
        docs = self.lc_store.similarity_search(q, k=k)
        return [(1.0 - i*0.001, d) for i, d in enumerate(docs)]
    def _sparse(self, q, k):
        if not self.bm25 or not self.doc_tokens: return []
        toks = _tokenize(q); scores = self.bm25.get_scores(toks)
        idxs = np.argsort(scores)[::-1][:k]
        return [(float(scores[i]), self.docs[i]) for i in idxs]
    def _fuse(self, q, dense, sparse):
        d_rank = {id(doc): r for r, (_, doc) in enumerate([(-s, d) for s, d in dense], start=1)}
        s_rank = {id(doc): r for r, (_, doc) in enumerate([(-s, d) for s, d in sparse], start=1)}
        all_ids = set(list(d_rank.keys()) + list(s_rank.keys()))
        fused = defaultdict(float)
        id_to_doc = {id(d): d for _, d in dense + sparse}
        ql = q.lower()
        for did in all_ids:
            if did in d_rank: fused[did] += _rrf(d_rank[did])
            if did in s_rank: fused[did] += _rrf(s_rank[did])
            d = id_to_doc[did]
            skills = set([s.lower() for s in (d.metadata.get("skills") or [])])
            overlap = sum(1 for s in skills if s in ql)
            if overlap: fused[did] += 0.6 * overlap
        ranked = sorted(all_ids, key=lambda x: fused[x], reverse=True)
        return [(fused[i], id_to_doc[i]) for i in ranked]

# ---------------- Facade: Auto-hybrid, zero knobs ----------------
class AutoHybridRAG:
    """
    Chooses RAG-Anything (if installed) for ingestion/retrieval with local embeddings,
    else uses LocalHybridEngine. LLM fallback order:
    OpenAI -> Gemini -> Transformers (free) -> Ollama -> retrieval-only.
    """
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.embed_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.any_engine = RAGAnythingEngine(index_dir) if _HAS_RA else None
        self.local_engine = LocalHybridEngine(index_dir, self.embed_model)
        self.llm = LLMFallback()

        # Keep a list of ingested file paths for RA
        self._files: List[str] = []

    # called by app when new text chunks are ready (local engine path)
    def build_or_update_index(self, chunks: List[str], metas: List[Dict[str,Any]]):
        self.local_engine.build_or_update(chunks, metas)

    def _ensure_anything_ingest(self):
        if not self.any_engine or not self._files: return
        # nothing more to do; ingestion happens on upload via parsing module; we ingest via app only when files exist
        pass

    def answer(self, question: str, style: str = "bullet points") -> Dict[str,Any]:
        # Prefer RA retrieval if available & already has working_dir data; else local retrieval
        contexts = []
        if self.any_engine:
            try:
                contexts = self.any_engine.retrieve(question, top_k=8)
            except Exception:
                contexts = []
        if not contexts:
            contexts = self.local_engine.retrieve(question, top_k=8)

        context_text = "\n\n".join([
            f"[{i+1}] {c['text']}\n(Source: {c.get('file')} p.{c.get('page')})"
            for i, c in enumerate(contexts)
        ])

        out = self.llm.generate(question, context_text, style)
        if not out:
            # retrieval-only fallback
            bullets = []
            for i, c in enumerate(contexts, start=1):
                snip = c["text"].replace("\n"," ")
                if len(snip) > 320: snip = snip[:320] + "…"
                bullets.append(f"- [{i}] **{c.get('file')}** p.{c.get('page')}: {snip}")
            out = ("## Answer (retrieval summary)\n"
                   "LLM unavailable; here are the most relevant snippets:\n" +
                   "\n".join(bullets) +
                   "\n\n**Evidence**\n" + "\n".join([f"- {c.get('file')} (p.{c.get('page')})" for c in contexts[:8]]))

        return {
            "text": out,
            "hits_meta": [{"file": c.get("file"), "page": c.get("page")} for c in contexts],
            "answer_hash": _sha256(out),
        }
