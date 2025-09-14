# app/rag_engine.py - Enhanced with better hybrid retrieval and industry-agnostic capabilities
import os
import json
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

# Core retrieval dependencies
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS as LCFAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Optional accuracy mode
_HAS_RA = False
try:
    from raganything import RAGAnything, RAGAnythingConfig
    _HAS_RA = True
except ImportError:
    _HAS_RA = False

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _tokenize(text: str) -> List[str]:
    """Enhanced tokenization with better preprocessing"""
    # Remove special characters, convert to lowercase, split on whitespace
    processed = ''.join([c.lower() if c.isalnum() else ' ' for c in text])
    tokens = [t for t in processed.split() if len(t) > 2]  # Filter short tokens
    return tokens

def _rrf(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion scoring"""
    return 1.0 / (k + rank)

def _calculate_semantic_similarity(query_embedding: np.ndarray, doc_embedding: np.ndarray) -> float:
    """Calculate cosine similarity between embeddings"""
    dot_product = np.dot(query_embedding, doc_embedding)
    norms = np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
    return dot_product / norms if norms > 0 else 0.0

class IndustryClassifier:
    """Simple industry classification based on keywords"""
    
    INDUSTRY_KEYWORDS = {
        "technology": [
            "software", "developer", "engineer", "python", "java", "javascript", "react", 
            "aws", "cloud", "api", "database", "frontend", "backend", "devops", "agile"
        ],
        "finance": [
            "financial", "banking", "investment", "trading", "portfolio", "risk", "audit",
            "accounting", "bloomberg", "excel", "sql", "derivatives", "equity", "bonds"
        ],
        "healthcare": [
            "medical", "healthcare", "clinical", "patient", "hospital", "nurse", "doctor",
            "pharmaceutical", "biotech", "fhir", "hl7", "epic", "cerner", "clinical trials"
        ],
        "marketing": [
            "marketing", "digital marketing", "seo", "sem", "social media", "content",
            "campaign", "brand", "analytics", "google analytics", "facebook ads", "crm"
        ],
        "sales": [
            "sales", "business development", "account management", "lead generation",
            "salesforce", "crm", "quota", "pipeline", "customer acquisition"
        ],
        "design": [
            "design", "ui", "ux", "user experience", "figma", "adobe", "photoshop",
            "wireframe", "prototype", "visual design", "graphic design"
        ],
        "data_science": [
            "data science", "machine learning", "ai", "artificial intelligence", "python",
            "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "statistics"
        ],
        "project_management": [
            "project management", "scrum", "agile", "pmp", "jira", "confluence",
            "stakeholder", "timeline", "budget", "resource planning"
        ],
        "education": [
            "teacher", "education", "curriculum", "training", "learning", "instruction",
            "academic", "university", "college", "student"
        ],
        "manufacturing": [
            "manufacturing", "production", "quality", "lean", "six sigma", "supply chain",
            "operations", "process improvement", "autocad", "solidworks"
        ]
    }
    
    @classmethod
    def classify_text(cls, text: str) -> Dict[str, float]:
        """Classify text into industries with confidence scores"""
        text_lower = text.lower()
        scores = {}
        
        for industry, keywords in cls.INDUSTRY_KEYWORDS.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            scores[industry] = matches / len(keywords) if keywords else 0
        
        return scores
    
    @classmethod
    def get_primary_industry(cls, text: str) -> str:
        """Get the most likely industry for given text"""
        scores = cls.classify_text(text)
        if not scores:
            return "general"
        
        max_industry = max(scores.items(), key=lambda x: x[1])
        return max_industry[0] if max_industry[1] > 0.1 else "general"

class LLMFallback:
    """Enhanced LLM fallback with better prompt engineering and industry awareness"""
    
    def __init__(self):
        self.openai_api = os.getenv("OPENAI_API_KEY")
        self.gemini_api = os.getenv("GEMINI_API_KEY")
        self.anthropic_api = os.getenv("ANTHROPIC_API_KEY")
        self.hf_model = os.getenv("HF_LOCAL_MODEL", "microsoft/DialoGPT-medium")
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3")

    def _format_enhanced_prompt(self, question: str, context: str, style: str = "bullet points") -> Tuple[str, str]:
        """Enhanced prompt formatting with better context handling"""
        
        # Detect question type and industry context
        question_lower = question.lower()
        industry = IndustryClassifier.get_primary_industry(context)
        
        # Determine question type
        question_type = "general"
        if any(word in question_lower for word in ["rank", "compare", "best", "top"]):
            question_type = "ranking"
        elif any(word in question_lower for word in ["skills", "experience", "qualification"]):
            question_type = "skills_analysis"
        elif any(word in question_lower for word in ["suitable", "fit", "match"]):
            question_type = "candidate_matching"
        
        system_prompt = f"""You are an expert resume analyst specializing in {industry} industry recruitment and talent assessment.

ANALYSIS GUIDELINES:
- Provide precise, evidence-based answers using ONLY the given resume context
- Always cite specific files and page numbers for each claim
- For ranking questions: provide clear ranking criteria and scores
- For skills analysis: categorize by proficiency levels (Expert/Advanced/Intermediate/Beginner)
- For candidate matching: explain fit percentage with reasoning

RESPONSE FORMAT:
- Use {style} format for main content
- Always include an **Evidence** section at the end
- If information is not in the context, state: "Not found in the uploaded resumes"
- Be specific about years of experience, skill levels, and achievements

INDUSTRY CONTEXT: {industry.replace('_', ' ').title()}"""

        user_prompt = f"""QUESTION: {question}

RESUME CONTEXT:
{context}

Please provide a comprehensive analysis following the guidelines above."""

        return system_prompt, user_prompt

    def try_openai(self, question: str, context: str, style: str) -> Optional[str]:
        if not self.openai_api:
            return None
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api)
            sys, user = self._format_enhanced_prompt(question, context, style)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI failed: {e}")
            return None

    def try_gemini(self, question: str, context: str, style: str) -> Optional[str]:
        if not self.gemini_api:
            return None
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_api)
            model = genai.GenerativeModel("gemini-1.5-flash")
            sys, user = self._format_enhanced_prompt(question, context, style)
            
            response = model.generate_content([f"{sys}\n\n{user}"])
            return response.text
        except Exception as e:
            print(f"Gemini failed: {e}")
            return None

    def try_anthropic(self, question: str, context: str, style: str) -> Optional[str]:
        if not self.anthropic_api:
            return None
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.anthropic_api)
            sys, user = self._format_enhanced_prompt(question, context, style)
            
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1500,
                temperature=0.1,
                system=sys,
                messages=[{"role": "user", "content": user}]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Anthropic failed: {e}")
            return None

    def try_ollama(self, question: str, context: str, style: str) -> Optional[str]:
        try:
            import requests
            sys, user = self._format_enhanced_prompt(question, context, style)
            
            payload = {
                "model": self.ollama_model,
                "messages": [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user}
                ],
                "stream": False
            }
            
            response = requests.post(f"{self.ollama_url}/api/chat", json=payload)
            if response.status_code == 200:
                return response.json()["message"]["content"]
        except Exception as e:
            print(f"Ollama failed: {e}")
            return None

    def generate(self, question: str, context: str, style: str = "bullet points") -> Optional[str]:
        """Try multiple LLM providers in order of preference"""
        return (
            self.try_openai(question, context, style) or
            self.try_anthropic(question, context, style) or
            self.try_gemini(question, context, style) or
            self.try_ollama(question, context, style)
        )

class EnhancedHybridRetriever:
    """Advanced hybrid retrieval with multiple ranking strategies"""
    
    def __init__(self, index_dir: str, embed_model: str):
        self.index_dir = index_dir
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        self.embed_model_name = embed_model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize components
        self.documents: List[Document] = []
        self.doc_tokens: List[List[str]] = []
        self.bm25 = None
        self.vector_store = None
        self.reranker = None
        
        # Load reranker if available
        try:
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            print("Reranker not available, using basic ranking")
        
        # Try to load existing index
        self._load_existing_index()
    
    def _load_existing_index(self):
        """Load existing FAISS index if available"""
        try:
            self.vector_store = LCFAISS.load_local(
                self.index_dir, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            print("Loaded existing vector index")
        except Exception:
            print("No existing vector index found")
    
    def build_or_update_index(self, chunks: List[str], metas: List[Dict[str, Any]]):
        """Build or update the search index with new documents"""
        if not chunks:
            return
        
        # Create documents
        new_documents = []
        for i, chunk in enumerate(chunks):
            metadata = metas[i] if i < len(metas) else {}
            doc = Document(page_content=chunk, metadata=metadata)
            new_documents.append(doc)
        
        self.documents.extend(new_documents)
        
        # Update vector store
        if self.vector_store is None:
            self.vector_store = LCFAISS.from_documents(new_documents, self.embeddings)
        else:
            # Add new documents to existing store
            new_store = LCFAISS.from_documents(new_documents, self.embeddings)
            self.vector_store.merge_from(new_store)
        
        # Save updated index
        self.vector_store.save_local(self.index_dir)
        
        # Update BM25 index
        self.doc_tokens = [_tokenize(doc.page_content) for doc in self.documents]
        self.bm25 = BM25Okapi(self.doc_tokens) if self.doc_tokens else None
        
        print(f"Updated index with {len(new_documents)} new documents. Total: {len(self.documents)}")
    
    def _dense_retrieval(self, query: str, k: int = 50) -> List[Tuple[float, Document]]:
        """Dense vector-based retrieval"""
        if not self.vector_store:
            return []
        
        try:
            # Get similar documents with scores
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
            # Convert to (score, doc) format with higher score = better match
            return [(1.0 / (1.0 + score), doc) for doc, score in docs_with_scores]
        except Exception as e:
            print(f"Dense retrieval failed: {e}")
            return []
    
    def _sparse_retrieval(self, query: str, k: int = 50) -> List[Tuple[float, Document]]:
        """Sparse BM25-based retrieval"""
        if not self.bm25 or not self.doc_tokens:
            return []
        
        try:
            query_tokens = _tokenize(query)
            scores = self.bm25.get_scores(query_tokens)
            
            # Get top k documents
            top_indices = np.argsort(scores)[::-1][:k]
            return [(float(scores[i]), self.documents[i]) for i in top_indices if scores[i] > 0]
        except Exception as e:
            print(f"Sparse retrieval failed: {e}")
            return []
    
    def _skill_based_boosting(self, query: str, docs: List[Tuple[float, Document]]) -> List[Tuple[float, Document]]:
        """Boost documents based on skill matching"""
        query_lower = query.lower()
        boosted_docs = []
        
        for score, doc in docs:
            boost = 0.0
            metadata = doc.metadata
            
            # Boost based on skill matches
            doc_skills = metadata.get("skills", [])
            skill_matches = sum(1 for skill in doc_skills if skill.lower() in query_lower)
            if skill_matches > 0:
                boost += 0.3 * skill_matches
            
            # Boost based on industry relevance
            doc_text = doc.page_content.lower()
            industry_scores = IndustryClassifier.classify_text(doc_text)
            query_industry = IndustryClassifier.get_primary_industry(query)
            
            if query_industry in industry_scores and industry_scores[query_industry] > 0.1:
                boost += 0.2 * industry_scores[query_industry]
            
            # Boost based on experience indicators
            experience_keywords = ["years", "experience", "expert", "senior", "lead", "manager"]
            if any(keyword in query_lower for keyword in experience_keywords):
                if any(keyword in doc_text for keyword in experience_keywords):
                    boost += 0.1
            
            boosted_docs.append((score + boost, doc))
        
        return boosted_docs
    
    def _hybrid_fusion(self, query: str, dense_results: List[Tuple[float, Document]], 
                      sparse_results: List[Tuple[float, Document]]) -> List[Tuple[float, Document]]:
        """Advanced fusion of dense and sparse retrieval results"""
        
        # Create ranking dictionaries
        dense_ranks = {id(doc): rank for rank, (_, doc) in enumerate(
            sorted(dense_results, key=lambda x: x[0], reverse=True), 1
        )}
        
        sparse_ranks = {id(doc): rank for rank, (_, doc) in enumerate(
            sorted(sparse_results, key=lambda x: x[0], reverse=True), 1
        )}
        
        # Combine all unique documents
        all_doc_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        id_to_doc = {}
        
        for _, doc in dense_results + sparse_results:
            id_to_doc[id(doc)] = doc
        
        # Calculate fusion scores
        fused_scores = {}
        for doc_id in all_doc_ids:
            score = 0.0
            
            # RRF from dense retrieval
            if doc_id in dense_ranks:
                score += _rrf(dense_ranks[doc_id], k=60)
            
            # RRF from sparse retrieval
            if doc_id in sparse_ranks:
                score += _rrf(sparse_ranks[doc_id], k=60)
            
            fused_scores[doc_id] = score
        
        # Apply skill-based boosting
        fusion_results = [(fused_scores[doc_id], id_to_doc[doc_id]) for doc_id in all_doc_ids]
        boosted_results = self._skill_based_boosting(query, fusion_results)
        
        # Sort by final score
        return sorted(boosted_results, key=lambda x: x[0], reverse=True)
    
    def retrieve(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """Main retrieval method with hybrid approach"""
        if not self.documents:
            return []
        
        # Get dense and sparse results
        dense_results = self._dense_retrieval(query, k=min(50, len(self.documents)))
        sparse_results = self._sparse_retrieval(query, k=min(50, len(self.documents)))
        
        # Fusion and ranking
        fused_results = self._hybrid_fusion(query, dense_results, sparse_results)
        
        # Get top candidates for reranking
        candidates = fused_results[:min(50, len(fused_results))]
        
        # Apply reranking if available
        if self.reranker and len(candidates) > 1:
            try:
                texts = [doc.page_content for _, doc in candidates]
                rerank_scores = self.reranker.predict([[query, text] for text in texts])
                
                # Combine fusion scores with rerank scores
                final_results = []
                for i, (fusion_score, doc) in enumerate(candidates):
                    combined_score = 0.6 * rerank_scores[i] + 0.4 * fusion_score
                    final_results.append((combined_score, doc))
                
                candidates = sorted(final_results, key=lambda x: x[0], reverse=True)
            except Exception as e:
                print(f"Reranking failed: {e}")
        
        # Format results
        results = []
        for score, doc in candidates[:top_k]:
            results.append({
                "text": doc.page_content,
                "file": doc.metadata.get("file", "unknown"),
                "page": doc.metadata.get("page", 1),
                "score": float(score),
                "skills": doc.metadata.get("skills", []),
                "chunk_index": doc.metadata.get("chunk_index", 0)
            })
        
        return results

class AutoHybridRAG:
    """Enhanced main RAG system with industry-agnostic capabilities"""
    
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.embed_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize retriever and LLM
        self.retriever = EnhancedHybridRetriever(index_dir, self.embed_model)
        self.llm = LLMFallback()
        
        # Document registry for tracking
        self.document_registry = {}
        self.session_stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "industries_detected": set(),
            "unique_skills": set()
        }
    
    def build_or_update_index(self, chunks: List[str], metas: List[Dict[str, Any]]):
        """Build or update the search index"""
        if not chunks:
            return
        
        # Update retriever
        self.retriever.build_or_update_index(chunks, metas)
        
        # Update session statistics
        self.session_stats["total_chunks"] += len(chunks)
        
        for meta in metas:
            # Track unique files
            file_path = meta.get("path", "")
            if file_path and file_path not in self.document_registry:
                self.document_registry[file_path] = {
                    "file": meta.get("file", "unknown"),
                    "chunks": 0,
                    "skills": set(),
                    "industry": "general"
                }
                self.session_stats["total_documents"] += 1
            
            if file_path:
                self.document_registry[file_path]["chunks"] += 1
                
                # Update skills
                doc_skills = set(meta.get("skills", []))
                self.document_registry[file_path]["skills"].update(doc_skills)
                self.session_stats["unique_skills"].update(doc_skills)
                
                # Detect industry
                chunk_text = chunks[metas.index(meta)] if metas.index(meta) < len(chunks) else ""
                industry = IndustryClassifier.get_primary_industry(chunk_text)
                if industry != "general":
                    self.document_registry[file_path]["industry"] = industry
                    self.session_stats["industries_detected"].add(industry)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session"""
        return {
            "total_documents": self.session_stats["total_documents"],
            "total_chunks": self.session_stats["total_chunks"],
            "industries_detected": list(self.session_stats["industries_detected"]),
            "unique_skills_count": len(self.session_stats["unique_skills"]),
            "top_skills": sorted(list(self.session_stats["unique_skills"]))[:20],
            "documents": list(self.document_registry.values())
        }
    
    def answer(self, question: str, style: str = "bullet points") -> Dict[str, Any]:
        """Generate answer with enhanced context and analysis"""
        
        # Retrieve relevant contexts
        contexts = self.retriever.retrieve(question, top_k=10)
        
        if not contexts:
            return {
                "text": "No relevant information found in the uploaded resumes. Please upload resumes and try again.",
                "hits_meta": [],
                "answer_hash": _sha256("no_results"),
                "industry_context": "general",
                "retrieval_stats": {"total_hits": 0, "avg_score": 0.0}
            }
        
        # Build context text with enhanced formatting
        context_parts = []
        for i, ctx in enumerate(contexts):
            context_part = (
                f"[Document {i+1}] **{ctx.get('file', 'unknown')}** (Page {ctx.get('page', 1)})\n"
                f"Skills: {', '.join(ctx.get('skills', [])[:5])}{'...' if len(ctx.get('skills', [])) > 5 else ''}\n"
                f"Content: {ctx['text']}\n"
                f"---"
            )
            context_parts.append(context_part)
        
        context_text = "\n\n".join(context_parts)
        
        # Detect industry context
        industry_context = IndustryClassifier.get_primary_industry(question + " " + context_text)
        
        # Generate answer
        answer_text = self.llm.generate(question, context_text, style)
        
        # Calculate retrieval statistics
        scores = [ctx.get('score', 0.0) for ctx in contexts]
        retrieval_stats = {
            "total_hits": len(contexts),
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0
        }
        
        # Fallback if no LLM available
        if not answer_text:
            fallback_parts = ["## Retrieved Information\n"]
            fallback_parts.append("LLM services are unavailable. Here are the most relevant resume excerpts:\n")
            
            for i, ctx in enumerate(contexts[:5]):
                snippet = ctx["text"][:400] + "..." if len(ctx["text"]) > 400 else ctx["text"]
                fallback_parts.append(f"**{i+1}. {ctx.get('file')}** (Page {ctx.get('page')})")
                fallback_parts.append(f"Skills: {', '.join(ctx.get('skills', []))}")
                fallback_parts.append(f"{snippet}\n")
            
            fallback_parts.append("\n**Sources:**")
            for ctx in contexts[:8]:
                fallback_parts.append(f"- {ctx.get('file')} (Page {ctx.get('page')})")
            
            answer_text = "\n".join(fallback_parts)
        
        return {
            "text": answer_text,
            "hits_meta": [{"file": ctx.get("file"), "page": ctx.get("page"), "score": ctx.get("score", 0.0)} for ctx in contexts],
            "answer_hash": _sha256(answer_text),
            "industry_context": industry_context,
            "retrieval_stats": retrieval_stats,
            "session_summary": self.get_session_summary()
        }
    
    def clear_index(self):
        """Clear all indexed data"""
        self.retriever = EnhancedHybridRetriever(self.index_dir, self.embed_model)
        self.document_registry.clear()
        self.session_stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "industries_detected": set(),
            "unique_skills": set()
        }
        
        # Clear index files
        index_path = Path(self.index_dir)
        if index_path.exists():
            import shutil
            shutil.rmtree(index_path, ignore_errors=True)
            index_path.mkdir(parents=True, exist_ok=True)