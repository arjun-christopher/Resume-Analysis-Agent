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

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import re
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    _HAS_QDRANT = True
except ImportError:
    _HAS_QDRANT = False

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
    # Fallback Document class if langchain not available
    class Document:
        """Fallback Document class for when langchain is not available"""
        def __init__(self, page_content: str = "", metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}

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
    from google import genai
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
    from sentence_transformers import CrossEncoder
    _HAS_CROSS_ENCODER = True
except ImportError:
    _HAS_CROSS_ENCODER = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# QUERY EXPANSION & ENHANCEMENT
# ============================================================================

class QueryExpander:
    """Expand queries with synonyms and related terms for better retrieval"""
    
    # Domain-specific synonyms for resume queries
    SYNONYMS = {
        'experience': ['work', 'employment', 'job', 'position', 'role', 'career'],
        'skills': ['abilities', 'competencies', 'expertise', 'proficiency', 'capabilities'],
        'education': ['academic', 'degree', 'qualification', 'university', 'college', 'study'],
        'projects': ['work', 'assignments', 'initiatives', 'portfolio'],
        'certifications': ['certificates', 'licenses', 'credentials', 'qualifications'],
        'achievements': ['accomplishments', 'awards', 'recognition', 'honors'],
        'python': ['py', 'python3', 'python2', 'python programming'],
        'javascript': ['js', 'ecmascript', 'node', 'nodejs', 'javascript programming'],
        'machine learning': ['ml', 'deep learning', 'ai', 'artificial intelligence', 'neural networks'],
        'data science': ['data analytics', 'data analysis', 'big data', 'analytics'],
        'full stack': ['fullstack', 'full-stack', 'frontend and backend'],
        'frontend': ['front-end', 'front end', 'ui', 'user interface', 'client-side'],
        'backend': ['back-end', 'back end', 'server-side', 'server'],
        'devops': ['dev ops', 'sre', 'infrastructure', 'ci/cd'],
        'cloud': ['aws', 'azure', 'gcp', 'google cloud', 'cloud computing'],
        'database': ['db', 'sql', 'nosql', 'data storage', 'rdbms'],
        'agile': ['scrum', 'kanban', 'sprint', 'iterative development'],
        'leadership': ['management', 'lead', 'manage', 'supervise', 'team lead'],
        'years': ['yrs', 'year', 'experience'],
        'bachelor': ['bachelors', 'bs', 'ba', 'undergraduate'],
        'master': ['masters', 'ms', 'ma', 'graduate', 'postgraduate'],
        'phd': ['ph.d', 'doctorate', 'doctoral'],
    }
    
    # Technical variations
    TECH_VARIATIONS = {
        'react': ['reactjs', 'react.js'],
        'angular': ['angularjs', 'angular.js'],
        'vue': ['vuejs', 'vue.js'],
        'node': ['nodejs', 'node.js'],
        'typescript': ['ts'],
        'docker': ['containerization', 'containers'],
        'kubernetes': ['k8s'],
        'tensorflow': ['tf'],
        'pytorch': ['torch'],
    }
    
    @classmethod
    def expand_query(cls, query: str, max_expansions: int = 3) -> List[str]:
        """
        Expand query with synonyms and variations
        
        Args:
            query: Original query
            max_expansions: Maximum number of expanded queries to generate
            
        Returns:
            List of query variations including original
        """
        query_lower = query.lower()
        expanded = [query]  # Always include original
        
        # Find matching terms
        matched_terms = {}
        for key, synonyms in {**cls.SYNONYMS, **cls.TECH_VARIATIONS}.items():
            if key in query_lower:
                matched_terms[key] = synonyms
        
        # Generate variations
        if matched_terms:
            # Strategy 1: Replace one term at a time
            for key, synonyms in list(matched_terms.items())[:max_expansions]:
                for synonym in synonyms[:2]:  # Use top 2 synonyms
                    # Case-insensitive replacement
                    import re
                    pattern = re.compile(re.escape(key), re.IGNORECASE)
                    expanded_query = pattern.sub(synonym, query, count=1)
                    if expanded_query != query and expanded_query not in expanded:
                        expanded.append(expanded_query)
                        if len(expanded) >= max_expansions + 1:
                            break
                if len(expanded) >= max_expansions + 1:
                    break
        
        return expanded[:max_expansions + 1]
    
    @classmethod
    def classify_query_type(cls, query: str) -> str:
        """
        Classify query type for specialized retrieval
        
        Returns:
            'exact_match' | 'semantic' | 'hybrid' | 'comparison' | 'aggregation'
        """
        query_lower = query.lower()
        
        # Exact match queries (specific years, numbers, names)
        if any(keyword in query_lower for keyword in ['how many years', 'specific', 'exactly', 'in year']):
            return 'exact_match'
        
        # Comparison queries
        if any(keyword in query_lower for keyword in ['compare', 'best', 'top', 'rank', 'versus', 'vs', 'better']):
            return 'comparison'
        
        # Aggregation queries
        if any(keyword in query_lower for keyword in ['all candidates', 'everyone', 'list all', 'total', 'count', 'how many']):
            return 'aggregation'
        
        # Check for technical terms (prefer semantic)
        technical_terms = ['python', 'java', 'react', 'machine learning', 'aws', 'cloud', 'docker']
        if any(term in query_lower for term in technical_terms):
            return 'semantic'
        
        # Default: hybrid
        return 'hybrid'


class CrossEncoderReranker:
    """Rerank retrieved documents using cross-encoder for better accuracy"""
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = None
        self.model_name = model_name
        if _HAS_CROSS_ENCODER:
            try:
                self.model = CrossEncoder(model_name)
                logger.info(f"✅ Loaded cross-encoder reranker: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder: {e}")
    
    def rerank(
        self, 
        query: str, 
        documents: List[Tuple[str, Dict[str, Any], float]], 
        top_k: int = None
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Rerank documents using cross-encoder
        
        Args:
            query: User query
            documents: List of (text, metadata, score) tuples
            top_k: Number of top documents to return (None = return all)
            
        Returns:
            Reranked list of documents with updated scores
        """
        if not self.model or not documents:
            return documents
        
        try:
            # Prepare pairs for cross-encoder
            pairs = [[query, doc[0]] for doc in documents]
            
            # Get cross-encoder scores
            scores = self.model.predict(pairs)
            
            # Combine with original documents
            reranked = []
            for i, (text, metadata, original_score) in enumerate(documents):
                # Use cross-encoder score as primary, keep original as secondary
                reranked.append((
                    text, 
                    metadata, 
                    float(scores[i])  # Cross-encoder score
                ))
            
            # Sort by cross-encoder score
            reranked.sort(key=lambda x: x[2], reverse=True)
            
            # Return top_k if specified
            if top_k:
                return reranked[:top_k]
            return reranked
            
        except Exception as e:
            logger.warning(f"Cross-encoder reranking failed: {e}")
            return documents

# ============================================================================
# Google GenAI Wrapper for LangChain Compatibility
# ============================================================================
class GoogleGenAIWrapper:
    """Wrapper to make Google GenAI SDK compatible with LangChain interface"""
    
    def __init__(self, model: str, temperature: float = 0.1, max_output_tokens: int = 2048, api_key: str = None):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        
        # Initialize Google GenAI client
        if api_key:
            os.environ['GOOGLE_API_KEY'] = api_key
        self.client = genai.Client()
    
    def invoke(self, prompt: str):
        """Invoke the model with a prompt (LangChain-compatible interface)"""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    'temperature': self.temperature,
                    'max_output_tokens': self.max_output_tokens,
                }
            )
            
            # Create a response object with 'content' attribute for compatibility
            class Response:
                def __init__(self, text):
                    self.content = text
                    self.text = text
            
            return Response(response.text)
        except Exception as e:
            logger.error(f"Google GenAI error: {e}")
            raise


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
            r'\blinguistic\s+skills\b',
            r'\blanguage\s+proficiency\b',
            r'\bspoken\s+languages?\b'
        ],
        'hobbies': [
            r'\bhobbies?\b',
            r'\binterests?\b',
            r'\bpersonal\s+interests?\b',
            r'\bhobbies?\s+(?:and|&)\s+interests?\b',
            r'\bleisure\s+activities?\b',
            r'\brecreational\s+activities?\b',
            r'\bpastimes?\b',
            r'\boutside\s+(?:interests?|work)\b'
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
    """Combines dense (semantic) and sparse (BM25) retrieval with persistent storage"""
    
    # File limits for indexing
    MAX_FILES_PER_SESSION = 5
    MAX_BATCH_SIZE = 5
    
    def __init__(self, embedding_model: str = "BAAI/bge-small-en-v1.5", storage_path: str = "data/index"):
        self.embedding_model_name = embedding_model
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.indexed_resume_ids = set()  # Track unique resumes indexed
        
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
        
        # Qdrant client for persistent storage
        self.qdrant_client = None
        self.collection_name = "resume_embeddings"
        self._init_qdrant()
        
        # Load existing index if available
        self._load_index()
    
    def _init_qdrant(self):
        """Initialize Qdrant client for persistent vector storage"""
        if not _HAS_QDRANT:
            logger.warning("Qdrant not available, falling back to FAISS (no persistence)")
            return
        
        try:
            # Use local storage mode (no server required)
            qdrant_path = str(self.storage_path / "qdrant_storage")
            self.qdrant_client = QdrantClient(path=qdrant_path)
            logger.info(f"✅ Initialized Qdrant at {qdrant_path}")
        except Exception as e:
            logger.warning(f"Failed to initialize Qdrant: {e}")
            self.qdrant_client = None
    
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
    
    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Embed texts using configured method with batch processing for efficiency
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch (default: 64)
                       Optimal range is 32-128 for better BLAS operations
        
        Returns:
            numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        if self.embed_method == "fastembed":
            # FastEmbed handles batching internally
            embeddings = list(self.embedder.embed(texts))
            return np.array(embeddings)
        elif self.embed_method == "sentence_transformers":
            # Process in batches for better performance
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.embedder.encode(
                    batch, 
                    show_progress_bar=False,
                    batch_size=batch_size,
                    convert_to_numpy=True
                )
                all_embeddings.append(batch_embeddings)
            
            # Concatenate all batches
            return np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
        else:
            # Dummy embeddings
            return np.random.random((len(texts), 384)).astype(np.float32)
    
    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]]):
        """Add documents to both dense and sparse indices with file limit enforcement"""
        if not documents:
            return
        
        # Extract unique resume IDs from metadata
        new_resume_ids = set()
        for meta in metadata:
            resume_id = meta.get('resume_id')
            if resume_id:
                new_resume_ids.add(resume_id)
        
        # Check file limit
        total_resumes_after_add = len(self.indexed_resume_ids | new_resume_ids)
        if total_resumes_after_add > self.MAX_FILES_PER_SESSION:
            allowed_new = self.MAX_FILES_PER_SESSION - len(self.indexed_resume_ids)
            raise ValueError(
                f"Cannot add documents. Session limit: {self.MAX_FILES_PER_SESSION} files maximum. "
                f"Currently indexed: {len(self.indexed_resume_ids)} files. "
                f"Attempting to add: {len(new_resume_ids)} files. "
                f"You can only add {allowed_new} more file(s)."
            )
        
        # Update resume tracking
        self.indexed_resume_ids.update(new_resume_ids)
        
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
        
        # Save to Qdrant for persistence
        self._save_to_qdrant(documents, new_embeddings, metadata)
        
        logger.info(f"Added {len(documents)} documents. Total: {len(self.documents)}")
    
    def _save_to_qdrant(self, documents: List[str], embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """Save documents and embeddings to Qdrant for persistence"""
        if not self.qdrant_client:
            return
        
        try:
            # Get embedding dimension
            dimension = embeddings.shape[1]
            
            # Create collection if it doesn't exist
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if not collection_exists:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            
            # Get current offset (number of existing points)
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            offset = collection_info.points_count
            
            # Prepare points for upload
            points = []
            for i, (doc, emb, meta) in enumerate(zip(documents, embeddings, metadata)):
                point = PointStruct(
                    id=offset + i,
                    vector=emb.tolist(),
                    payload={
                        "text": doc,
                        "metadata": meta
                    }
                )
                points.append(point)
            
            # Upload to Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Saved {len(points)} documents to Qdrant")
        except Exception as e:
            logger.warning(f"Failed to save to Qdrant: {e}")
    
    def _load_index(self):
        """Load existing index from Qdrant"""
        if not self.qdrant_client:
            logger.info("No Qdrant client available, starting with empty index")
            return
        
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if not collection_exists:
                logger.info("No existing index found, starting fresh")
                return
            
            # Get collection info
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            total_points = collection_info.points_count
            
            if total_points == 0:
                logger.info("Index exists but is empty")
                return
            
            logger.info(f"Loading {total_points} documents from Qdrant...")
            
            # Retrieve all points (in batches if necessary)
            batch_size = 1000
            all_documents = []
            all_metadata = []
            all_embeddings = []
            
            offset = 0
            while offset < total_points:
                # Scroll through points
                points, next_offset = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                for point in points:
                    all_documents.append(point.payload["text"])
                    all_metadata.append(point.payload["metadata"])
                    all_embeddings.append(point.vector)
                
                if next_offset is None:
                    break
                offset = next_offset
            
            # Restore state
            self.documents = all_documents
            self.metadata = all_metadata
            self.embeddings = np.array(all_embeddings, dtype=np.float32)
            
            # Rebuild resume ID tracking
            self.indexed_resume_ids = set()
            for meta in all_metadata:
                resume_id = meta.get('resume_id')
                if resume_id:
                    self.indexed_resume_ids.add(resume_id)
            
            # Rebuild FAISS and BM25 indices
            self._build_faiss_index()
            self._build_bm25_index()
            
            logger.info(f"✅ Loaded {len(self.documents)} documents from persistent storage ({len(self.indexed_resume_ids)} unique resumes)")
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")
    
    def has_documents(self) -> bool:
        """Check if index has any documents"""
        return len(self.documents) > 0
    
    def clear_index(self):
        """Clear all documents and reset index"""
        self.documents = []
        self.metadata = []
        self.embeddings = None
        self.faiss_index = None
        self.bm25 = None
        self.tokenized_corpus = []
        self.indexed_resume_ids = set()  # Clear resume tracking
        
        # Clear Qdrant collection
        if self.qdrant_client:
            try:
                collections = self.qdrant_client.get_collections().collections
                if any(c.name == self.collection_name for c in collections):
                    self.qdrant_client.delete_collection(self.collection_name)
                    logger.info(f"Cleared Qdrant collection: {self.collection_name}")
            except Exception as e:
                logger.warning(f"Failed to clear Qdrant collection: {e}")
    
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
        bm25_weight: float = 0.3,
        resume_id: Optional[str] = None,
        resume_name: Optional[str] = None,
        use_reranking: bool = True,
        section_filter: Optional[List[str]] = None
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Enhanced hybrid search with query expansion, reranking, and section filtering
        
        Args:
            query: Search query
            k: Number of results
            semantic_weight: Weight for semantic search (0-1)
            bm25_weight: Weight for BM25 search (0-1)
            resume_id: Optional resume ID to filter results (for per-resume queries)
            resume_name: Optional resume name to filter results (alternative to resume_id)
            use_reranking: Whether to use cross-encoder reranking (default: True)
            section_filter: Optional list of section types to prioritize (e.g., ['experience', 'skills'])
        """
        if not self.documents:
            return []
        
        # Normalize weights
        total_weight = semantic_weight + bm25_weight
        semantic_weight /= total_weight
        bm25_weight /= total_weight
        
        # ENHANCEMENT 1: Query expansion for better recall
        expanded_queries = QueryExpander.expand_query(query, max_expansions=2)
        
        # Retrieve results for each query variation
        all_semantic_results = []
        all_bm25_results = []
        
        for exp_query in expanded_queries:
            # Get semantic search results
            semantic_results = self._semantic_search(exp_query, k * 2)
            all_semantic_results.extend(semantic_results)
            
            # Get BM25 results
            bm25_results = self._bm25_search(exp_query, k * 2)
            all_bm25_results.extend(bm25_results)
        
        # Deduplicate results (keep highest score for each document)
        semantic_results = self._deduplicate_results(all_semantic_results)[:k * 2]
        bm25_results = self._deduplicate_results(all_bm25_results)[:k * 2]
        
        # Fuse results using reciprocal rank fusion
        fused_results = self._fuse_results(semantic_results, bm25_results, semantic_weight, bm25_weight)
        
        # ENHANCEMENT 2: Section-aware filtering
        if section_filter:
            # Boost results from target sections
            boosted_results = []
            for text, metadata, score in fused_results:
                section_type = metadata.get('section_type', '').lower()
                if section_type in section_filter:
                    # Boost score by 30% for matching sections
                    score *= 1.3
                boosted_results.append((text, metadata, score))
            # Re-sort after boosting
            boosted_results.sort(key=lambda x: x[2], reverse=True)
            fused_results = boosted_results
        
        # Filter by resume if specified
        if resume_id or resume_name:
            filtered_results = []
            for text, metadata, score in fused_results:
                if resume_id and metadata.get('resume_id') == resume_id:
                    filtered_results.append((text, metadata, score))
                elif resume_name and metadata.get('resume_name') == resume_name:
                    filtered_results.append((text, metadata, score))
            fused_results = filtered_results
        
        # Return top k
        results = fused_results[:k * 3]  # Get more for reranking
        
        # ENHANCEMENT 3: Cross-encoder reranking for better accuracy
        if use_reranking and _HAS_CROSS_ENCODER and len(results) > 0:
            reranker = CrossEncoderReranker()
            results = reranker.rerank(query, results, top_k=k)
        else:
            results = results[:k]
        
        return results
    
    def group_results_by_resume(
        self, 
        results: List[Tuple[str, Dict[str, Any], float]]
    ) -> Dict[str, List[Tuple[str, Dict[str, Any], float]]]:
        """
        Group search results by resume_id for multi-resume queries
        
        Args:
            results: List of search results
            
        Returns:
            Dictionary mapping resume_id to list of results for that resume
        """
        grouped = defaultdict(list)
        
        for text, metadata, score in results:
            resume_id = metadata.get('resume_id', 'unknown')
            resume_name = metadata.get('resume_name', 'Unknown Resume')
            
            # Store resume name in the group key for easy access
            group_key = f"{resume_id}|{resume_name}"
            grouped[group_key].append((text, metadata, score))
        
        return dict(grouped)
    
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
            if 0 <= idx < len(self.documents):
                results.append((self.documents[idx], self.metadata[idx], score))
        
        return results
    
    def _deduplicate_results(self, results: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """
        Deduplicate results, keeping highest score for each document index
        
        Args:
            results: List of (index, score) tuples
            
        Returns:
            Deduplicated list sorted by score
        """
        # Use dictionary to keep highest score per index
        best_scores = {}
        for idx, score in results:
            if idx not in best_scores or score > best_scores[idx]:
                best_scores[idx] = score
        
        # Convert back to list and sort by score
        deduplicated = [(idx, score) for idx, score in best_scores.items()]
        deduplicated.sort(key=lambda x: x[1], reverse=True)
        
        return deduplicated


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
        self.search_engine = HybridSearchEngine(storage_path=str(self.storage_path))
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
        
        # Prioritize Gemini first, then Ollama for fallback
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
                # If Gemini fails, try Ollama fallback immediately
                if provider == 'google':
                    logger.info("Gemini failed, trying Ollama fallback...")
        
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
            model = os.getenv('GOOGLE_MODEL', 'gemini-2.5-flash')
            if api_key and api_key != 'your_google_api_key_here':
                return GoogleGenAIWrapper(model=model, temperature=temperature, max_output_tokens=max_tokens, api_key=api_key)
        
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
        """Add documents with section-based chunking and parallel processing"""
        start_time = time.time()
        
        if metadata is None:
            metadata = [{'source': f'doc_{i}'} for i in range(len(documents))]
        
        all_chunks = []
        all_metadata = []
        total_sections = 0
        
        # Use parallel processing for batch document ingestion
        if len(documents) > 1:
            logger.info(f"Processing {len(documents)} documents in parallel...")
            
            def process_document(doc_meta_pair):
                """Process a single document (for parallel execution)"""
                doc, meta = doc_meta_pair
                chunks_data = self.chunker.chunk_document(doc, meta)
                sections = self.chunker.section_detector.detect_sections(doc)
                return chunks_data, len(sections)
            
            # Process documents in parallel with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(len(documents), os.cpu_count() or 4)) as executor:
                futures = {executor.submit(process_document, (doc, meta)): (doc, meta) 
                          for doc, meta in zip(documents, metadata)}
                
                for future in as_completed(futures):
                    try:
                        chunks_data, section_count = future.result()
                        for chunk_data in chunks_data:
                            all_chunks.append(chunk_data['text'])
                            all_metadata.append(chunk_data['metadata'])
                        total_sections += section_count
                    except Exception as e:
                        logger.error(f"Error processing document: {e}")
        else:
            # Single document - no need for parallel processing
            for doc, meta in zip(documents, metadata):
                chunks_data = self.chunker.chunk_document(doc, meta)
                
                for chunk_data in chunks_data:
                    all_chunks.append(chunk_data['text'])
                    all_metadata.append(chunk_data['metadata'])
                
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
    
    def query(self, question: str, k: int = 5, resume_id: Optional[str] = None, 
              resume_name: Optional[str] = None, group_by_resume: bool = True) -> Dict[str, Any]:
        """
        Enhanced query with classification, expansion, and reranking
        
        Args:
            question: User query
            k: Number of results to retrieve
            resume_id: Optional resume ID to filter results (for per-resume queries)
            resume_name: Optional resume name to filter results
            group_by_resume: Whether to group results by resume in the response
        """
        start_time = time.time()
        
        try:
            # ENHANCEMENT 1: Query Classification
            query_type = QueryExpander.classify_query_type(question)
            logger.info(f"Query type detected: {query_type}")
            
            # Detect if query is asking for comparison/ranking across resumes
            is_comparison_query = self._is_comparison_query(question) or query_type == 'comparison'
            
            # ENHANCEMENT 2: Adaptive search weights based on query type
            semantic_weight, bm25_weight = self._determine_search_weights_by_type(question, query_type)
            
            # ENHANCEMENT 3: Section filtering based on query content
            section_filter = self._determine_section_filter(question)
            
            # For comparison queries, get more results to ensure we cover multiple resumes
            search_k = k * 3 if is_comparison_query else k
            
            # ENHANCEMENT 4: Hybrid search with query expansion, reranking, and section filtering
            results = self.search_engine.search(
                question, 
                k=search_k, 
                semantic_weight=semantic_weight,
                bm25_weight=bm25_weight,
                resume_id=resume_id,
                resume_name=resume_name,
                use_reranking=True,  # Enable cross-encoder reranking
                section_filter=section_filter  # Prioritize relevant sections
            )
            
            if not results:
                return {
                    'answer': "No relevant documents found.",
                    'source_documents': [],
                    'processing_time': time.time() - start_time
                }
            
            # Group results by resume if requested or if comparison query
            grouped_results = None
            if group_by_resume or is_comparison_query:
                grouped_results = self.search_engine.group_results_by_resume(results)
            
            # Extract pattern insights (with grouping info if available)
            pattern_insights = self._extract_insights(question, results, grouped_results)
            
            # Generate LLM response with grouping context if available
            llm_response = ""
            if self.llm:
                context = self._build_context_for_llm(results, grouped_results, is_comparison_query)
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
            
            response = {
                'answer': final_answer,
                'source_documents': source_docs,
                'processing_time': time.time() - start_time,
                'method': 'enhanced_hybrid_rag_v2',
                'query_type': query_type,
                'search_weights': {'semantic': semantic_weight, 'bm25': bm25_weight},
                'enhancements_used': ['query_expansion', 'cross_encoder_reranking', 'section_filtering', 'adaptive_weights']
            }
            
            # Add grouping info if available
            if grouped_results:
                response['grouped_by_resume'] = True
                response['resume_count'] = len(grouped_results)
            
            return response
        
        except Exception as e:
            logger.error(f"Query error: {e}")
            return {
                'answer': f"Error: {str(e)}",
                'source_documents': [],
                'processing_time': time.time() - start_time
            }
    
    def _is_comparison_query(self, question: str) -> bool:
        """Detect if query is asking for comparison across multiple resumes"""
        question_lower = question.lower()
        comparison_keywords = [
            'compare', 'rank', 'best', 'top', 'most', 'who has', 'which candidate',
            'all candidates', 'everyone', 'across', 'between', 'versus', 'vs',
            'find candidates', 'list candidates', 'show all', 'all resumes'
        ]
        return any(keyword in question_lower for keyword in comparison_keywords)
    
    def _build_context_for_llm(
        self, 
        results: List[Tuple[str, Dict, float]], 
        grouped_results: Optional[Dict[str, List[Tuple[str, Dict, float]]]] = None,
        is_comparison: bool = False
    ) -> str:
        """Build context string for LLM with optional grouping and candidate names"""
        if grouped_results and is_comparison:
            # Group context by resume for comparison queries - WITH CANDIDATE NAMES
            context_parts = []
            for group_key, group_results in grouped_results.items():
                resume_id, resume_name = group_key.split('|', 1)
                
                # Try to get candidate name from metadata
                candidate_name = None
                if group_results:
                    candidate_name = group_results[0][1].get('candidate_name')
                
                # Use candidate name if available, otherwise use resume name
                display_name = candidate_name if candidate_name else resume_name
                
                context_parts.append(f"\n{'='*80}")
                context_parts.append(f"CANDIDATE: {display_name}")
                context_parts.append(f"Resume File: {resume_name}")
                context_parts.append(f"{'='*80}\n")
                
                for text, meta, score in group_results[:5]:  # More content per resume
                    section_type = meta.get('section_type', 'unknown')
                    context_parts.append(f"[{section_type.upper()}]")
                    context_parts.append(f"{text}\n")
                
            return '\n'.join(context_parts)
        else:
            # Standard context format - also include candidate name if available
            context_parts = []
            for text, meta, score in results:
                candidate_name = meta.get('candidate_name')
                resume_name = meta.get('resume_name', 'Unknown Resume')
                section_type = meta.get('section_type', 'unknown')
                
                if candidate_name:
                    context_parts.append(f"[Candidate: {candidate_name} | Section: {section_type}]")
                else:
                    context_parts.append(f"[File: {resume_name} | Section: {section_type}]")
                
                context_parts.append(f"{text}\n")
            
            return '\n'.join(context_parts)
    
    def get_resume_list(self) -> List[Dict[str, str]]:
        """Get list of all indexed resumes with their IDs, names, and candidate names"""
        resume_map = {}
        for metadata in self.search_engine.metadata:
            resume_id = metadata.get('resume_id')
            resume_name = metadata.get('resume_name')
            candidate_name = metadata.get('candidate_name')
            
            if resume_id and resume_id not in resume_map:
                resume_map[resume_id] = {
                    'resume_name': resume_name,
                    'candidate_name': candidate_name
                }
        
        return [
            {
                'resume_id': rid, 
                'resume_name': data['resume_name'],
                'candidate_name': data['candidate_name'] if data['candidate_name'] else data['resume_name']
            } 
            for rid, data in resume_map.items()
        ]
    
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
    
    def _determine_search_weights_by_type(self, question: str, query_type: str) -> Tuple[float, float]:
        """
        Enhanced search weight determination using query classification
        
        Args:
            question: User query
            query_type: Classified query type from QueryExpander
            
        Returns:
            Tuple of (semantic_weight, bm25_weight)
        """
        question_lower = question.lower()
        
        # Type-based weights
        if query_type == 'exact_match':
            # Favor keyword search for exact matches
            return 0.3, 0.7
        
        elif query_type == 'semantic':
            # Strong semantic focus for conceptual queries
            return 0.85, 0.15
        
        elif query_type == 'comparison':
            # Balanced for comparison queries
            return 0.7, 0.3
        
        elif query_type == 'aggregation':
            # Slightly favor BM25 for aggregation/counting
            return 0.6, 0.4
        
        # Hybrid type - refine with keyword indicators
        keyword_indicators = ['name', 'email', 'phone', 'specific', 'exactly', 'list', 'year', 'date']
        if any(indicator in question_lower for indicator in keyword_indicators):
            return 0.4, 0.6  # Favor BM25
        
        # Semantic indicators
        semantic_indicators = ['analyze', 'describe', 'explain', 'how', 'why', 'summarize']
        if any(indicator in question_lower for indicator in semantic_indicators):
            return 0.8, 0.2  # Favor semantic
        
        # Default balanced
        return 0.7, 0.3
    
    def _determine_section_filter(self, question: str) -> Optional[List[str]]:
        """
        Determine which resume sections are most relevant for the query
        
        Args:
            question: User query
            
        Returns:
            List of relevant section types to prioritize, or None for all sections
        """
        question_lower = question.lower()
        
        # Map keywords to sections
        section_keywords = {
            'experience': ['experience', 'work', 'job', 'role', 'position', 'employment', 'worked', 'company'],
            'skills': ['skill', 'technology', 'programming', 'language', 'framework', 'tool', 'expertise', 'proficiency'],
            'education': ['education', 'degree', 'university', 'college', 'study', 'academic', 'major', 'graduated'],
            'projects': ['project', 'built', 'developed', 'created', 'implemented'],
            'certifications': ['certification', 'certificate', 'license', 'certified', 'credential'],
            'achievements': ['achievement', 'award', 'recognition', 'accomplishment', 'honor'],
            'summary': ['summary', 'profile', 'objective', 'about'],
        }
        
        # Find matching sections
        matched_sections = []
        for section, keywords in section_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                matched_sections.append(section)
        
        # Return None if no specific match (search all sections)
        # Or if multiple broad sections matched (likely needs all context)
        if len(matched_sections) == 0 or len(matched_sections) > 3:
            return None
        
        return matched_sections
    
    def _extract_insights(
        self, 
        question: str, 
        results: List[Tuple[str, Dict, float]], 
        grouped_results: Optional[Dict[str, List[Tuple[str, Dict, float]]]] = None
    ) -> str:
        """Extract structured insights from results with candidate names and grouping"""
        question_lower = question.lower()
        insights = []
        
        # If grouped results available, show resume breakdown with candidate names
        if grouped_results:
            candidate_info = []
            for group_key, group_results in grouped_results.items():
                resume_id, resume_name = group_key.split('|', 1)
                # Get candidate name from first result in group
                candidate_name = None
                if group_results:
                    candidate_name = group_results[0][1].get('candidate_name')
                
                if candidate_name:
                    candidate_info.append(f"**{candidate_name}** ({resume_name})")
                else:
                    candidate_info.append(f"{resume_name}")
            
            insights.append(f"**Candidates Analyzed:** {len(grouped_results)}")
            insights.append('\n'.join(f"  {i+1}. {info}" for i, info in enumerate(candidate_info)))
            insights.append("")  # Empty line for spacing
        
        # Aggregate data by section type
        section_data = defaultdict(list)
        candidate_data = defaultdict(lambda: defaultdict(set))
        
        for text, metadata, score in results:
            section_type = metadata.get('section_type', 'unknown')
            candidate_name = metadata.get('candidate_name', 'Unknown')
            
            section_data[section_type].append((text, metadata, score))
            
            # Aggregate candidate-specific data
            if 'emails' in metadata:
                candidate_data[candidate_name]['emails'].update(metadata['emails'])
            if 'phones' in metadata:
                candidate_data[candidate_name]['phones'].update(metadata['phones'])
            if 'skills' in metadata:
                candidate_data[candidate_name]['skills'].update(metadata.get('skills', []))
            if 'technical_skills' in metadata:
                candidate_data[candidate_name]['skills'].update(metadata.get('technical_skills', []))
        
        # Extract specific data based on question
        if any(word in question_lower for word in ['email', 'contact']):
            if candidate_data:
                insights.append("**Contact Information:**")
                for candidate_name, data in candidate_data.items():
                    if data.get('emails'):
                        emails_str = ', '.join(sorted(data['emails']))
                        insights.append(f"  • **{candidate_name}:** {emails_str}")
                    if data.get('phones'):
                        phones_str = ', '.join(sorted(data['phones']))
                        insights.append(f"  • **{candidate_name}** (Phone): {phones_str}")
        
        if any(word in question_lower for word in ['skill', 'technology', 'expertise']):
            if candidate_data:
                insights.append("**Skills Summary:**")
                for candidate_name, data in candidate_data.items():
                    if data.get('skills'):
                        skills_list = list(data['skills'])[:10]  # Top 10 skills
                        skills_str = ', '.join(skills_list)
                        insights.append(f"  • **{candidate_name}:** {skills_str}")
        
        # Section summary
        if section_data and not grouped_results:  # Only show if not already showing grouped info
            insights.append(f"\n**Sections Retrieved:** {', '.join(section_data.keys())}")
        
        return '\n'.join(insights) if insights else ""
    
    def _generate_response(self, question: str, context: str) -> str:
        """Generate LLM response with improved prompting for multi-resume scenarios"""
        if not self.llm:
            return ""
        
        # Detect if this is a multi-candidate query
        is_multi_candidate = "CANDIDATE:" in context and context.count("CANDIDATE:") > 1
        
        if is_multi_candidate:
            prompt = f"""You are an expert resume analyst comparing multiple candidates. Answer the question based on the resume sections provided.

RESUME DATA (Multiple Candidates):
{context}

QUESTION: {question}

IMPORTANT INSTRUCTIONS:
1. **Start each candidate's information with their name in BOLD** (e.g., **John Smith:**)
2. Present information for EACH candidate separately
3. Use clear formatting with bullet points
4. If comparing or ranking, explain your reasoning
5. Extract specific details (skills, experience, education) for each candidate
6. Be factual and cite evidence from the resumes

Provide a comprehensive answer that clearly identifies each candidate and their relevant qualifications.

ANSWER:"""
        else:
            prompt = f"""You are an expert resume analyst. Answer the question based on the resume sections provided.

RESUME SECTIONS:
{context}

QUESTION: {question}

Provide a clear, structured answer with:
1. **Direct Answer** - Start with the key information
2. **Details** - Support with specific evidence from the resume
3. **Summary** - Brief conclusion if needed

Use bullet points and clear formatting. If a candidate name is provided, use it in your response.

ANSWER:"""
        
        try:
            response = self.llm.invoke(prompt)
            if hasattr(response, 'content'):
                return response.content
            elif hasattr(response, 'text'):
                return response.text
            return str(response)
        except Exception as e:
            logger.error(f"LLM generation failed with {self.current_llm_provider}: {e}")
            
            # Try fallback to Ollama if current provider is not Ollama
            if self.current_llm_provider != 'ollama':
                logger.info("Attempting fallback to Ollama...")
                try:
                    ollama_llm = self._init_provider_llm('ollama', 0.1, 2048, 60)
                    if ollama_llm:
                        response = ollama_llm.invoke(prompt)
                        logger.info("✅ Ollama fallback successful")
                        if hasattr(response, 'content'):
                            return response.content
                        elif hasattr(response, 'text'):
                            return response.text
                        return str(response)
                except Exception as fallback_error:
                    logger.error(f"Ollama fallback also failed: {fallback_error}")
            
            return ""
    
    def _combine_responses(self, llm_response: str, pattern_insights: str, question: str) -> str:
        """Combine LLM and pattern-based insights"""
        if not llm_response and not pattern_insights:
            return "Unable to generate response."
        
        if not llm_response:
            return f"## Analysis\n\n{pattern_insights}"
        
        if not pattern_insights:
            return f"## Answer\n\n{llm_response}"
        
        return f"""## Comprehensive Analysis

{llm_response}

---

## Additional Insights

{pattern_insights}"""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            **self.stats,
            'total_documents': len(self.search_engine.documents),
            'llm_provider': self.current_llm_provider,
            'has_index': self.has_index()
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Alias for get_stats() for backward compatibility"""
        return self.get_stats()
    
    def has_index(self) -> bool:
        """Check if the engine has indexed documents"""
        return self.search_engine.has_documents()
    
    def clear_index(self):
        """Clear all indexed documents"""
        self.search_engine.clear_index()
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'queries_processed': 0,
            'sections_detected': 0
        }
        logger.info("Index cleared")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_advanced_rag_engine(storage_path: str = "data/advanced_rag") -> AdvancedRAGEngine:
    """Create advanced RAG engine instance"""
    return AdvancedRAGEngine(storage_path)