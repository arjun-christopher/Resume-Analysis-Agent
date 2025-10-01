# app/rag_engine.py - Advanced Section-Based RAG with Hybrid Search
"""
Advanced RAG System with:
- Intelligent section-based chunking (variable chunk sizes)
- Hierarchical retrieval (section-level + content-level)
- Hybrid search (semantic + BM25)
- Advanced reranking and metadata enrichment
- Optimized for speed and accuracy
"""

from __future__ import annotations
import logging
import os
import re
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

# Import from existing modules
try:
    from fastembed import TextEmbedding
    _HAS_FASTEMBED = True
except ImportError:
    _HAS_FASTEMBED = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

try:
    from langchain.schema import Document
    from langchain_community.llms import Ollama
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False

# API LLMs
try:
    from langchain_openai import ChatOpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

try:
    from langchain_anthropic import ChatAnthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    _HAS_GOOGLE = True
except ImportError:
    _HAS_GOOGLE = False

try:
    from langchain_groq import ChatGroq
    _HAS_GROQ = True
except ImportError:
    _HAS_GROQ = False

try:
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    _HAS_HUGGINGFACE = True
except ImportError:
    _HAS_HUGGINGFACE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# SECTION-BASED CHUNKING SYSTEM
# ============================================================================

@dataclass
class ResumeSection:
    """Represents a section in a resume with metadata"""
    section_type: str  # e.g., "experience", "education", "skills"
    title: str  # The actual section title found
    content: str  # Full section content
    start_pos: int  # Character position in document
    end_pos: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    subsections: List['ResumeSection'] = field(default_factory=list)


class SectionDetector:
    """Intelligent section detection for resumes"""
    
    # Comprehensive section patterns (ordered by priority)
    SECTION_PATTERNS = {
        'summary': [
            r'\b(?:professional\s+)?summary\b',
            r'\b(?:career\s+)?objective\b',
            r'\bprofile\b',
            r'\babout\s+me\b',
            r'\bexecutive\s+summary\b'
        ],
        'experience': [
            r'\b(?:work\s+)?experience\b',
            r'\b(?:professional\s+)?experience\b',
            r'\bemployment\s+history\b',
            r'\bwork\s+history\b',
            r'\bcareer\s+history\b',
            r'\bprofessional\s+background\b'
        ],
        'education': [
            r'\beducation\b',
            r'\bacademic\s+background\b',
            r'\bacademic\s+qualifications\b',
            r'\beducational\s+background\b'
        ],
        'skills': [
            r'\b(?:technical\s+)?skills\b',
            r'\bcore\s+competencies\b',
            r'\bexpertise\b',
            r'\bproficiencies\b',
            r'\bkey\s+skills\b',
            r'\bareas\s+of\s+expertise\b'
        ],
        'projects': [
            r'\bprojects\b',
            r'\bkey\s+projects\b',
            r'\bnotable\s+projects\b',
            r'\bproject\s+experience\b'
        ],
        'certifications': [
            r'\bcertifications?\b',
            r'\blicenses?\b',
            r'\bprofessional\s+certifications?\b',
            r'\bcertifications?\s+and\s+licenses?\b'
        ],
        'achievements': [
            r'\bachievements?\b',
            r'\baccomplishments?\b',
            r'\bawards?\b',
            r'\bhonors?\b',
            r'\brecognition\b'
        ],
        'publications': [
            r'\bpublications?\b',
            r'\bresearch\b',
            r'\bpapers?\b',
            r'\bpublished\s+works?\b'
        ],
        'activities': [
            r'\bactivities\b',
            r'\bextracurricular\b',
            r'\bvolunteer\s+work\b',
            r'\bvolunteering\b',
            r'\bcommunity\s+service\b'
        ],
        'languages': [
            r'\blanguages?\b',
            r'\blinguistic\s+skills\b'
        ],
        'references': [
            r'\breferences?\b',
            r'\bavailable\s+upon\s+request\b'
        ],
        'contact': [
            r'\bcontact\s+information\b',
            r'\bcontact\s+details\b',
            r'\bpersonal\s+information\b'
        ]
    }
    
    def __init__(self):
        # Compile patterns for speed
        self.compiled_patterns = {}
        for section_type, patterns in self.SECTION_PATTERNS.items():
            self.compiled_patterns[section_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
        
        # Pattern to detect section headers (lines that look like headings)
        self.header_pattern = re.compile(
            r'^[\s]*([A-Z][A-Za-z\s&/]+)[\s]*:?[\s]*$',
            re.MULTILINE
        )
    
    def detect_sections(self, text: str) -> List[ResumeSection]:
        """Detect all sections in resume text"""
        sections = []
        lines = text.split('\n')
        
        # Find potential section headers with their positions
        potential_headers = []
        current_pos = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check if line looks like a section header
            if self._is_section_header(line_stripped):
                section_type = self._classify_section(line_stripped)
                if section_type:
                    potential_headers.append({
                        'line_num': i,
                        'title': line_stripped.rstrip(':'),
                        'type': section_type,
                        'char_pos': current_pos
                    })
            
            current_pos += len(line) + 1  # +1 for newline
        
        # Extract sections based on headers
        for idx, header in enumerate(potential_headers):
            start_line = header['line_num']
            start_pos = header['char_pos']
            
            # Determine end of section (next header or end of document)
            if idx + 1 < len(potential_headers):
                end_line = potential_headers[idx + 1]['line_num']
                end_pos = potential_headers[idx + 1]['char_pos']
            else:
                end_line = len(lines)
                end_pos = len(text)
            
            # Extract section content (skip the header line itself)
            section_lines = lines[start_line + 1:end_line]
            content = '\n'.join(section_lines).strip()
            
            if content:  # Only add if there's actual content
                section = ResumeSection(
                    section_type=header['type'],
                    title=header['title'],
                    content=content,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    metadata={
                        'line_start': start_line,
                        'line_end': end_line,
                        'word_count': len(content.split()),
                        'char_count': len(content)
                    }
                )
                sections.append(section)
        
        # If no sections detected, treat entire document as one section
        if not sections:
            sections.append(ResumeSection(
                section_type='full_document',
                title='Full Resume',
                content=text,
                start_pos=0,
                end_pos=len(text),
                metadata={
                    'word_count': len(text.split()),
                    'char_count': len(text)
                }
            ))
        
        logger.info(f"Detected {len(sections)} sections: {[s.section_type for s in sections]}")
        return sections
    
    def _is_section_header(self, line: str) -> bool:
        """Check if a line looks like a section header"""
        if not line or len(line) > 100:  # Headers are usually short
            return False
        
        # Check various header patterns
        checks = [
            # All caps or title case
            line.isupper() or line.istitle(),
            # Has colon at end
            line.endswith(':'),
            # Short line (< 50 chars)
            len(line) < 50,
            # Matches header pattern
            bool(self.header_pattern.match(line))
        ]
        
        return any(checks)
    
    def _classify_section(self, header_text: str) -> Optional[str]:
        """Classify section type from header text"""
        header_lower = header_text.lower().strip(':').strip()
        
        # Check against all patterns
        for section_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(header_lower):
                    return section_type
        
        return None


class SectionBasedChunker:
    """Create intelligent chunks based on resume sections"""
    
    def __init__(self, max_chunk_size: int = 1000, min_chunk_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.section_detector = SectionDetector()
    
    def chunk_document(self, text: str, doc_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk document based on sections with variable chunk sizes
        
        Returns list of chunks with metadata
        """
        if doc_metadata is None:
            doc_metadata = {}
        
        chunks = []
        sections = self.section_detector.detect_sections(text)
        
        for section in sections:
            # Each section becomes one or more chunks based on size
            section_chunks = self._chunk_section(section, doc_metadata)
            chunks.extend(section_chunks)
        
        logger.info(f"Created {len(chunks)} variable-sized chunks from {len(sections)} sections")
        return chunks
    
    def _chunk_section(self, section: ResumeSection, doc_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk a single section intelligently"""
        chunks = []
        content = section.content
        
        # If section is small enough, keep as single chunk
        if len(content) <= self.max_chunk_size:
            chunk = {
                'text': content,
                'section_type': section.section_type,
                'section_title': section.title,
                'chunk_size': len(content),
                'is_complete_section': True,
                'metadata': {
                    **doc_metadata,
                    **section.metadata,
                    'section_type': section.section_type,
                    'section_title': section.title
                }
            }
            chunks.append(chunk)
        else:
            # Split large section into smaller chunks at natural boundaries
            sub_chunks = self._split_large_section(content, section)
            for i, sub_chunk in enumerate(sub_chunks):
                chunk = {
                    'text': sub_chunk,
                    'section_type': section.section_type,
                    'section_title': section.title,
                    'chunk_size': len(sub_chunk),
                    'is_complete_section': False,
                    'sub_chunk_index': i,
                    'total_sub_chunks': len(sub_chunks),
                    'metadata': {
                        **doc_metadata,
                        **section.metadata,
                        'section_type': section.section_type,
                        'section_title': section.title,
                        'sub_chunk_index': i
                    }
                }
                chunks.append(chunk)
        
        return chunks
    
    def _split_large_section(self, content: str, section: ResumeSection) -> List[str]:
        """Split large section at natural boundaries (paragraphs, sentences)"""
        chunks = []
        
        # Try splitting by double newlines (paragraphs) first
        paragraphs = re.split(r'\n\s*\n', content)
        
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_size = len(para)
            
            # If single paragraph exceeds max size, split by sentences
            if para_size > self.max_chunk_size:
                # Add current chunk if exists
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Split paragraph into sentences
                sentences = re.split(r'[.!?]+\s+', para)
                sent_chunk = []
                sent_size = 0
                
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    
                    if sent_size + len(sent) > self.max_chunk_size and sent_chunk:
                        chunks.append('. '.join(sent_chunk) + '.')
                        sent_chunk = [sent]
                        sent_size = len(sent)
                    else:
                        sent_chunk.append(sent)
                        sent_size += len(sent) + 2  # +2 for ". "
                
                if sent_chunk:
                    chunks.append('. '.join(sent_chunk) + '.')
            
            # Normal case: add paragraph to current chunk
            elif current_size + para_size > self.max_chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size + 2  # +2 for \n\n
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks


# ============================================================================
# HYBRID SEARCH SYSTEM (Semantic + BM25)
# ============================================================================

class HybridSearchEngine:
    """Combines dense (semantic) and sparse (BM25) retrieval"""
    
    def __init__(self, embedding_model: str = "BAAI/bge-small-en-v1.5"):
        self.embedding_model_name = embedding_model
        self.embedder = None
        self.embed_method = None
        self._init_embedder()
        
        # Storage
        self.documents = []
        self.metadata = []
        self.embeddings = None
        self.faiss_index = None
        self.bm25 = None
        self.tokenized_corpus = []
    
    def _init_embedder(self):
        """Initialize embedder with fallback options"""
        # Try FastEmbed first
        if _HAS_FASTEMBED:
            try:
                self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
                self.embed_method = "fastembed"
                logger.info("✅ Initialized FastEmbed for embeddings")
                return
            except Exception as e:
                logger.warning(f"FastEmbed failed: {e}")
        
        # Try SentenceTransformers
        if _HAS_SENTENCE_TRANSFORMERS:
            try:
                os.environ['TOKENIZERS_PARALLELISM'] = 'false'
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
                self.embed_method = "sentence_transformers"
                logger.info("✅ Initialized SentenceTransformer for embeddings")
                return
            except Exception as e:
                logger.warning(f"SentenceTransformer failed: {e}")
        
        # Fallback
        self.embed_method = "dummy"
        logger.warning("⚠️ Using dummy embeddings")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed texts using configured method"""
        if self.embed_method == "fastembed":
            embeddings = list(self.embedder.embed(texts))
            return np.array(embeddings)
        elif self.embed_method == "sentence_transformers":
            return self.embedder.encode(texts, show_progress_bar=False)
        else:
            # Dummy embeddings
            return np.random.random((len(texts), 384)).astype(np.float32)
    
    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]]):
        """Add documents to both dense and sparse indices"""
        if not documents:
            return
        
        # Store documents
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        
        # Generate embeddings for semantic search
        new_embeddings = self.embed_texts(documents)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        # Build FAISS index
        self._build_faiss_index()
        
        # Build BM25 index
        self._build_bm25_index()
        
        logger.info(f"Added {len(documents)} documents. Total: {len(self.documents)}")
    
    def _build_faiss_index(self):
        """Build FAISS index for fast semantic search"""
        if not _HAS_FAISS or self.embeddings is None:
            return
        
        dimension = self.embeddings.shape[1]
        
        # Use IndexFlatIP for cosine similarity (after normalization)
        self.faiss_index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings
        normalized_embeddings = self.embeddings.copy()
        faiss.normalize_L2(normalized_embeddings)
        
        self.faiss_index.add(normalized_embeddings)
    
    def _build_bm25_index(self):
        """Build BM25 index for keyword search"""
        if not _HAS_BM25:
            return
        
        # Tokenize corpus
        self.tokenized_corpus = [doc.lower().split() for doc in self.documents]
        
        # Build BM25
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def search(
        self, 
        query: str, 
        k: int = 10, 
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Hybrid search combining semantic and BM25
        
        Args:
            query: Search query
            k: Number of results
            semantic_weight: Weight for semantic search (0-1)
            bm25_weight: Weight for BM25 search (0-1)
        """
        if not self.documents:
            return []
        
        # Normalize weights
        total_weight = semantic_weight + bm25_weight
        semantic_weight /= total_weight
        bm25_weight /= total_weight
        
        # Get semantic search results
        semantic_results = self._semantic_search(query, k * 2)  # Get more for fusion
        
        # Get BM25 results
        bm25_results = self._bm25_search(query, k * 2)
        
        # Fuse results using reciprocal rank fusion
        fused_results = self._fuse_results(semantic_results, bm25_results, semantic_weight, bm25_weight)
        
        # Return top k
        return fused_results[:k]
    
    def _semantic_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Semantic search using FAISS"""
        if self.faiss_index is None or self.embeddings is None:
            return []
        
        # Embed query
        query_embedding = self.embed_texts([query])[0].reshape(1, -1)
        
        # Normalize
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.faiss_index.search(query_embedding, min(k, len(self.documents)))
        
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
    
    def _bm25_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """BM25 keyword search"""
        if self.bm25 is None:
            return []
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        return [(int(idx), float(scores[idx])) for idx in top_indices]
    
    def _fuse_results(
        self, 
        semantic_results: List[Tuple[int, float]], 
        bm25_results: List[Tuple[int, float]],
        semantic_weight: float,
        bm25_weight: float
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """Fuse semantic and BM25 results using weighted scoring"""
        
        # Normalize scores
        def normalize_scores(results):
            if not results:
                return {}
            scores = [score for _, score in results]
            min_score = min(scores)
            max_score = max(scores)
            if max_score == min_score:
                return {idx: 1.0 for idx, _ in results}
            return {idx: (score - min_score) / (max_score - min_score) for idx, score in results}
        
        semantic_scores = normalize_scores(semantic_results)
        bm25_scores = normalize_scores(bm25_results)
        
        # Combine scores
        combined_scores = {}
        all_indices = set(semantic_scores.keys()) | set(bm25_scores.keys())
        
        for idx in all_indices:
            sem_score = semantic_scores.get(idx, 0.0)
            bm25_score = bm25_scores.get(idx, 0.0)
            combined_scores[idx] = (semantic_weight * sem_score) + (bm25_weight * bm25_score)
        
        # Sort by combined score
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return results with documents
        results = []
        for idx, score in sorted_indices:
            if idx < len(self.documents):
                results.append((
                    self.documents[idx],
                    self.metadata[idx],
                    score
                ))
        
        return results


# ============================================================================
# ADVANCED RAG ENGINE
# ============================================================================

class AdvancedRAGEngine:
    """
    Advanced RAG with:
    - Section-based variable chunking
    - Hybrid search (semantic + BM25)
    - Hierarchical retrieval
    - Smart reranking
    """
    
    def __init__(self, storage_path: str = "data/advanced_rag"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.chunker = SectionBasedChunker(max_chunk_size=1000, min_chunk_size=100)
        self.search_engine = HybridSearchEngine()
        self.llm = self._init_llm()
        self.current_llm_provider = None
        
        # Statistics
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'queries_processed': 0,
            'sections_detected': 0
        }
    
    def _init_llm(self):
        """Initialize LLM with Gemini first, then Ollama fallback"""
        if not _HAS_LANGCHAIN:
            logger.warning("LangChain not available")
            return None
        
        # Prioritize Gemini, then Ollama for low-end systems
        fallback_order = os.getenv('LLM_FALLBACK_ORDER', 'google,ollama').split(',')
        temperature = float(os.getenv('LLM_TEMPERATURE', '0.1'))
        max_tokens = int(os.getenv('LLM_MAX_TOKENS', '2048'))
        timeout = int(os.getenv('LLM_TIMEOUT', '30'))
        
        for provider in fallback_order:
            provider = provider.strip().lower()
            
            if not self._is_provider_enabled(provider):
                continue
            
            try:
                llm = self._init_provider_llm(provider, temperature, max_tokens, timeout)
                if llm:
                    self.current_llm_provider = provider
                    logger.info(f"✅ LLM initialized: {provider}")
                    return llm
            except Exception as e:
                logger.warning(f"Failed to initialize {provider}: {e}")
        
        logger.error("All LLM providers failed")
        return None
    
    def _is_provider_enabled(self, provider: str) -> bool:
        """Check if provider is enabled"""
        return os.getenv(f'ENABLE_{provider.upper()}', 'true').lower() == 'true'
    
    def _init_provider_llm(self, provider: str, temperature: float, max_tokens: int, timeout: int):
        """Initialize specific LLM provider"""
        if provider == 'openai' and _HAS_OPENAI:
            api_key = os.getenv('OPENAI_API_KEY')
            model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
            if api_key and api_key != 'your_openai_api_key_here':
                return ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout, api_key=api_key)
        
        elif provider == 'anthropic' and _HAS_ANTHROPIC:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            model = os.getenv('ANTHROPIC_MODEL', 'claude-3-haiku-20240307')
            if api_key and api_key != 'your_anthropic_api_key_here':
                return ChatAnthropic(model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout, api_key=api_key)
        
        elif provider == 'google' and _HAS_GOOGLE:
            api_key = os.getenv('GOOGLE_API_KEY')
            model = os.getenv('GOOGLE_MODEL', 'gemini-pro')
            if api_key and api_key != 'your_google_api_key_here':
                return ChatGoogleGenerativeAI(model=model, temperature=temperature, max_output_tokens=max_tokens, timeout=timeout, google_api_key=api_key)
        
        elif provider == 'groq' and _HAS_GROQ:
            api_key = os.getenv('GROQ_API_KEY')
            model = os.getenv('GROQ_MODEL', 'llama3-8b-8192')
            if api_key and api_key != 'your_groq_api_key_here':
                return ChatGroq(model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout, api_key=api_key)
        
        elif provider == 'huggingface' and _HAS_HUGGINGFACE:
            api_key = os.getenv('HUGGINGFACE_API_KEY')
            model = os.getenv('HUGGINGFACE_MODEL', 'meta-llama/Llama-2-7b-chat-hf')
            if api_key and api_key != 'your_huggingface_api_key_here':
                endpoint = HuggingFaceEndpoint(repo_id=model, temperature=temperature, max_new_tokens=max_tokens, timeout=timeout, huggingfacehub_api_token=api_key)
                return ChatHuggingFace(llm=endpoint)
        
        elif provider == 'ollama':
            model = os.getenv('OLLAMA_MODEL', 'qwen2.5:7b')
            base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            ollama_timeout = int(os.getenv('OLLAMA_TIMEOUT', '60'))
            return Ollama(model=model, temperature=temperature, num_predict=max_tokens, timeout=ollama_timeout, base_url=base_url)
        
        return None
    
    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]] = None):
        """Add documents with section-based chunking"""
        start_time = time.time()
        
        if metadata is None:
            metadata = [{'source': f'doc_{i}'} for i in range(len(documents))]
        
        all_chunks = []
        all_metadata = []
        total_sections = 0
        
        for doc, meta in zip(documents, metadata):
            # Create section-based chunks
            chunks_data = self.chunker.chunk_document(doc, meta)
            
            for chunk_data in chunks_data:
                all_chunks.append(chunk_data['text'])
                all_metadata.append(chunk_data['metadata'])
            
            # Count sections
            sections = self.chunker.section_detector.detect_sections(doc)
            total_sections += len(sections)
        
        # Add to search engine
        if all_chunks:
            self.search_engine.add_documents(all_chunks, all_metadata)
        
        # Update stats
        self.stats['documents_processed'] += len(documents)
        self.stats['chunks_created'] += len(all_chunks)
        self.stats['sections_detected'] += total_sections
        
        processing_time = time.time() - start_time
        logger.info(f"Processed {len(documents)} docs -> {len(all_chunks)} chunks in {processing_time:.2f}s")
        
        return {
            'documents_added': len(documents),
            'chunks_created': len(all_chunks),
            'sections_detected': total_sections,
            'processing_time': processing_time
        }
    
    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """Query with advanced RAG"""
        start_time = time.time()
        
        try:
            # Adjust weights based on query type
            semantic_weight, bm25_weight = self._determine_search_weights(question)
            
            # Hybrid search
            results = self.search_engine.search(
                question, 
                k=k, 
                semantic_weight=semantic_weight,
                bm25_weight=bm25_weight
            )
            
            if not results:
                return {
                    'answer': "No relevant documents found.",
                    'source_documents': [],
                    'processing_time': time.time() - start_time
                }
            
            # Extract pattern insights
            pattern_insights = self._extract_insights(question, results)
            
            # Generate LLM response
            llm_response = ""
            if self.llm:
                context = '\n\n'.join([f"[{meta.get('section_type', 'unknown')}] {text}" for text, meta, _ in results])
                llm_response = self._generate_response(question, context)
            
            # Combine responses
            final_answer = self._combine_responses(llm_response, pattern_insights, question)
            
            # Prepare source documents
            source_docs = []
            for text, meta, score in results:
                source_docs.append(Document(
                    page_content=text,
                    metadata={**meta, 'relevance_score': score}
                ))
            
            self.stats['queries_processed'] += 1
            
            return {
                'answer': final_answer,
                'source_documents': source_docs,
                'processing_time': time.time() - start_time,
                'method': 'advanced_hybrid_rag',
                'search_weights': {'semantic': semantic_weight, 'bm25': bm25_weight}
            }
        
        except Exception as e:
            logger.error(f"Query error: {e}")
            return {
                'answer': f"Error: {str(e)}",
                'source_documents': [],
                'processing_time': time.time() - start_time
            }
    
    def _determine_search_weights(self, question: str) -> Tuple[float, float]:
        """Determine optimal search weights based on question type"""
        question_lower = question.lower()
        
        # Keyword-heavy queries (favor BM25)
        keyword_indicators = ['name', 'email', 'phone', 'specific', 'exactly', 'list']
        if any(indicator in question_lower for indicator in keyword_indicators):
            return 0.4, 0.6  # Favor BM25
        
        # Semantic queries (favor semantic search)
        semantic_indicators = ['compare', 'analyze', 'describe', 'explain', 'how', 'why', 'best', 'most']
        if any(indicator in question_lower for indicator in semantic_indicators):
            return 0.8, 0.2  # Favor semantic
        
        # Balanced default
        return 0.7, 0.3
    
    def _extract_insights(self, question: str, results: List[Tuple[str, Dict, float]]) -> str:
        """Extract structured insights from results"""
        question_lower = question.lower()
        insights = []
        
        # Aggregate data by section type
        section_data = defaultdict(list)
        for text, metadata, score in results:
            section_type = metadata.get('section_type', 'unknown')
            section_data[section_type].append((text, metadata, score))
        
        # Extract specific data based on question
        if any(word in question_lower for word in ['email', 'contact']):
            emails = set()
            for text, metadata, _ in results:
                if 'emails' in metadata:
                    emails.update(metadata['emails'])
            if emails:
                insights.append(f"**📧 Emails Found:** {', '.join(sorted(emails))}")
        
        if any(word in question_lower for word in ['skill', 'technology']):
            skills = set()
            for text, metadata, _ in results:
                if 'skills' in metadata or 'technical_skills' in metadata:
                    skills.update(metadata.get('skills', []))
                    skills.update(metadata.get('technical_skills', []))
            if skills:
                insights.append(f"**💻 Skills:** {', '.join(list(skills)[:15])}")
        
        # Section summary
        if section_data:
            insights.append(f"\n**📑 Sections Retrieved:** {', '.join(section_data.keys())}")
        
        return '\n'.join(insights) if insights else ""
    
    def _generate_response(self, question: str, context: str) -> str:
        """Generate LLM response"""
        if not self.llm:
            return ""
        
        prompt = f"""You are an expert resume analyst. Answer the question based on the resume sections provided.

RESUME SECTIONS:
{context}

QUESTION: {question}

Provide a clear, structured answer with:
1. **Direct Answer** - Start with the key information
2. **Details** - Support with specific evidence from the resume
3. **Summary** - Brief conclusion if needed

Use bullet points and clear formatting.

ANSWER:"""
        
        try:
            response = self.llm.invoke(prompt)
            if hasattr(response, 'content'):
                return response.content
            elif hasattr(response, 'text'):
                return response.text
            return str(response)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return ""
    
    def _combine_responses(self, llm_response: str, pattern_insights: str, question: str) -> str:
        """Combine LLM and pattern-based insights"""
        if not llm_response and not pattern_insights:
            return "Unable to generate response."
        
        if not llm_response:
            return f"## 📊 Analysis\n\n{pattern_insights}"
        
        if not pattern_insights:
            return f"## 🤖 Answer\n\n{llm_response}"
        
        return f"""## 🤖 Comprehensive Analysis

{llm_response}

---

## 📊 Additional Insights

{pattern_insights}"""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            **self.stats,
            'total_documents': len(self.search_engine.documents),
            'llm_provider': self.current_llm_provider
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Alias for get_stats() for backward compatibility"""
        return self.get_stats()


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_advanced_rag_engine(storage_path: str = "data/advanced_rag") -> AdvancedRAGEngine:
    """Create advanced RAG engine instance"""
    return AdvancedRAGEngine(storage_path)
