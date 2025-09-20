# app/fast_semantic_rag.py - High-Speed Semantic RAG with Advanced EDA
"""
Ultra-fast RAG system optimized for speed and semantic understanding:
- FastEmbed for lightning-fast embeddings
- Optimized chunking with semantic sentence boundaries
- Fast vector search with FAISS
- Advanced pattern detection and semantic EDA
- Minimal computational overhead
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
import re
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd
from tqdm import tqdm

# Fast embedding models
try:
    from fastembed import TextEmbedding
    _HAS_FASTEMBED = True
except ImportError:
    _HAS_FASTEMBED = False

# Lightweight sentence transformers  
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False

# Fast vector search
try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

# Advanced NLP for pattern detection
try:
    import spacy
    try:
        _NLP = spacy.load("en_core_web_sm")  # Lightweight model for speed
    except OSError:
        _NLP = None
except ImportError:
    _NLP = None

# Fast text processing
try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

# LangChain for document management and LLM integrations
try:
    from langchain.schema import Document
    from langchain_community.llms import Ollama
    from langchain_community.chat_models import ChatOllama
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False

# API-based LLM integrations
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

# Environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    _HAS_DOTENV = True
except ImportError:
    _HAS_DOTENV = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FastRAGConfig:
    """Optimized configuration for speed"""
    # Lightweight embedding model for speed
    embedding_model: str = "BAAI/bge-small-en-v1.5"  # Fast and effective
    chunk_size: int = 256  # Smaller for faster processing
    chunk_overlap: int = 32
    max_chunks_per_doc: int = 50  # Limit for speed
    similarity_threshold: float = 0.6
    top_k: int = 5  # Reduced for speed
    enable_semantic_chunking: bool = True
    enable_fast_eda: bool = True
    enable_pattern_extraction: bool = True
    # LLM settings - using existing fallback order
    llm_model: str = "qwen2.5:7b"  # Fast model
    max_tokens: int = 2048
    temperature: float = 0.1

class FastSemanticChunker:
    """Ultra-fast semantic chunking optimized for speed"""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.embedder = None
        self._init_embedder()
    
    def _init_embedder(self):
        """Initialize the fastest available embedder with robust error handling"""
        
        # Try FastEmbed first (fastest option)
        if _HAS_FASTEMBED:
            try:
                logger.info("Attempting to initialize FastEmbed...")
                self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
                self.embed_method = "fastembed"
                logger.info("✅ Successfully initialized FastEmbed for ultra-fast embeddings")
                return
            except Exception as e:
                logger.warning(f"❌ FastEmbed initialization failed: {e}")
        
        # Try SentenceTransformers with multiple fallback models
        if _HAS_SENTENCE_TRANSFORMERS:
            models_to_try = [
                "all-MiniLM-L6-v2",  # Very fast and lightweight
                "paraphrase-MiniLM-L3-v2",  # Even smaller fallback
                "all-MiniLM-L12-v2"  # Slightly larger but more robust
            ]
            
            for model_name in models_to_try:
                try:
                    logger.info(f"Attempting to initialize SentenceTransformer with {model_name}...")
                    
                    # Set environment variables to avoid PyTorch issues
                    import os
                    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
                    
                    # Initialize with specific device settings to avoid CUDA issues
                    self.embedder = SentenceTransformer(
                        model_name, 
                        device='cpu',  # Force CPU to avoid CUDA issues
                        cache_folder=None  # Use default cache
                    )
                    self.embed_method = "sentence_transformers"
                    logger.info(f"✅ Successfully initialized SentenceTransformer with {model_name}")
                    return
                    
                except Exception as e:
                    logger.warning(f"❌ SentenceTransformer with {model_name} failed: {e}")
                    continue
        
        # Try a simple sklearn-based embedder as additional fallback
        try:
            logger.info("Attempting to initialize simple TF-IDF embedder...")
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.embedder = TfidfVectorizer(max_features=384, stop_words='english')
            self.embed_method = "tfidf"
            logger.info("✅ Successfully initialized TF-IDF embedder as fallback")
            return
        except ImportError:
            logger.warning("❌ scikit-learn not available for TF-IDF fallback")
        except Exception as e:
            logger.warning(f"❌ TF-IDF embedder failed: {e}")
        
        # Final fallback to dummy embeddings
        self.embed_method = "dummy"
        logger.warning("⚠️  Using dummy embeddings - all embedding methods failed. Install compatible versions of fastembed or sentence-transformers")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Fast text embedding with multiple method support"""
        try:
            if self.embed_method == "fastembed":
                embeddings = list(self.embedder.embed(texts))
                return np.array(embeddings)
            
            elif self.embed_method == "sentence_transformers":
                return self.embedder.encode(texts, show_progress_bar=False)
            
            elif self.embed_method == "tfidf":
                # For TF-IDF, we need to fit_transform or transform
                if not hasattr(self.embedder, 'vocabulary_'):
                    # First time - fit and transform
                    embeddings = self.embedder.fit_transform(texts)
                else:
                    # Already fitted - just transform
                    embeddings = self.embedder.transform(texts)
                return embeddings.toarray().astype(np.float32)
            
            else:
                # Dummy embeddings for testing
                return np.random.random((len(texts), 384)).astype(np.float32)
                
        except Exception as e:
            logger.error(f"Embedding failed with {self.embed_method}: {e}")
            # Fallback to dummy embeddings
            logger.warning("Falling back to dummy embeddings")
            return np.random.random((len(texts), 384)).astype(np.float32)
    
    def semantic_chunk(self, text: str, chunk_size: int = 256, similarity_threshold: float = 0.7) -> List[str]:
        """Fast semantic chunking with sentence boundaries"""
        if not text or len(text) < chunk_size:
            return [text]
        
        # Fast sentence splitting
        sentences = self._fast_sentence_split(text)
        if len(sentences) <= 1:
            return [text]
        
        # Group sentences into chunks with semantic awareness
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if current_length + sentence_length > chunk_size and current_chunk:
                # Finalize current chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + 1  # +1 for space
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _fast_sentence_split(self, text: str) -> List[str]:
        """Fast sentence splitting using regex"""
        # Simple but effective sentence splitting
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]

class FastPatternExtractor:
    """Lightning-fast pattern extraction and semantic analysis"""
    
    def __init__(self):
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b')
        self.url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
        self.linkedin_pattern = re.compile(r'linkedin\.com/in/[\w\-\.]+', re.IGNORECASE)
        self.github_pattern = re.compile(r'github\.com/[\w\-\.]+', re.IGNORECASE)
        
        # Skills patterns (common resume skills)
        self.tech_skills = {
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 'ruby', 'php',
            'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'fastapi', 'spring',
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
            'machine learning', 'deep learning', 'ai', 'nlp', 'computer vision',
            'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy'
        }
        
        # Compile skill patterns for speed
        self.skill_patterns = {
            skill: re.compile(rf'\b{re.escape(skill)}\b', re.IGNORECASE)
            for skill in self.tech_skills
        }
    
    def extract_patterns(self, text: str) -> Dict[str, Any]:
        """Fast extraction of all patterns from text"""
        result = {
            'emails': [],
            'phones': [],
            'urls': [],
            'linkedin': [],
            'github': [],
            'skills': [],
            'experience_years': [],
            'education': [],
            'certifications': []
        }
        
        # Extract basic patterns
        result['emails'] = self.email_pattern.findall(text)
        result['phones'] = self.phone_pattern.findall(text)
        result['urls'] = self.url_pattern.findall(text)
        result['linkedin'] = self.linkedin_pattern.findall(text)
        result['github'] = self.github_pattern.findall(text)
        
        # Extract skills
        text_lower = text.lower()
        for skill, pattern in self.skill_patterns.items():
            if pattern.search(text):
                result['skills'].append(skill)
        
        # Extract experience years
        exp_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
            r'(\d+)\+?\s*yrs?\s*(?:of\s*)?(?:experience|exp)',
            r'experience[:\s]*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*in',
        ]
        
        for pattern in exp_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            result['experience_years'].extend([int(m) for m in matches if m.isdigit()])
        
        # Extract education (simple patterns)
        edu_patterns = [
            r'\b(?:bachelor|master|phd|doctorate|mba|bs|ms|ba|ma)\b',
            r'\b(?:university|college|institute)\b',
            r'\b(?:computer science|engineering|mathematics|physics|business)\b'
        ]
        
        for pattern in edu_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            result['education'].extend(matches)
        
        # Extract certifications
        cert_patterns = [
            r'\b(?:aws|azure|gcp|google)\s*(?:certified|certification)\b',
            r'\b(?:cissp|cism|cisa|pmp|scrum master)\b',
            r'\bcertified\s+[\w\s]+\b'
        ]
        
        for pattern in cert_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            result['certifications'].extend(matches)
        
        return result

class FastEDAProcessor:
    """Ultra-fast Exploratory Data Analysis for resume text"""
    
    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
    
    def fast_eda(self, texts: List[str]) -> Dict[str, Any]:
        """Lightning-fast EDA analysis"""
        start_time = time.time()
        
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Tokenize
        tokens = self._fast_tokenize(combined_text)
        
        # Filter meaningful tokens
        meaningful_tokens = [
            token for token in tokens 
            if token.lower() not in self.stop_words and len(token) > 2 and token.isalpha()
        ]
        
        # Count frequencies
        token_counts = Counter(meaningful_tokens)
        
        # Generate bigrams
        bigrams = [
            f"{meaningful_tokens[i]} {meaningful_tokens[i+1]}"
            for i in range(len(meaningful_tokens) - 1)
        ]
        bigram_counts = Counter(bigrams)
        
        # Basic statistics
        stats = {
            'total_texts': len(texts),
            'total_tokens': len(tokens),
            'unique_tokens': len(token_counts),
            'avg_tokens_per_text': len(tokens) / len(texts) if texts else 0,
            'top_terms': dict(token_counts.most_common(20)),
            'top_bigrams': dict(bigram_counts.most_common(10)),
            'processing_time': time.time() - start_time
        }
        
        return stats
    
    def _fast_tokenize(self, text: str) -> List[str]:
        """Fast tokenization using regex"""
        return re.findall(r'\b\w+\b', text.lower())
    
    def semantic_summary(self, texts: List[str]) -> str:
        """Generate semantic summary from EDA"""
        eda_stats = self.fast_eda(texts)
        
        summary_parts = []
        
        # Document overview
        summary_parts.append(f"📊 **Dataset Overview:**")
        summary_parts.append(f"- {eda_stats['total_texts']} documents processed")
        summary_parts.append(f"- {eda_stats['total_tokens']} total words")
        summary_parts.append(f"- {eda_stats['unique_tokens']} unique terms")
        summary_parts.append(f"- {eda_stats['avg_tokens_per_text']:.1f} average words per document")
        
        # Top terms
        if eda_stats['top_terms']:
            summary_parts.append(f"\n🔍 **Most Frequent Terms:**")
            for term, count in list(eda_stats['top_terms'].items())[:10]:
                summary_parts.append(f"- {term}: {count} occurrences")
        
        # Top bigrams
        if eda_stats['top_bigrams']:
            summary_parts.append(f"\n🔗 **Common Phrases:**")
            for bigram, count in list(eda_stats['top_bigrams'].items())[:5]:
                summary_parts.append(f"- '{bigram}': {count} times")
        
        summary_parts.append(f"\n⚡ Processed in {eda_stats['processing_time']:.3f} seconds")
        
        return '\n'.join(summary_parts)

class FastVectorStore:
    """Ultra-fast vector storage using FAISS"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.documents = []
        self.metadata = []
        self._init_index()
    
    def _init_index(self):
        """Initialize FAISS index for speed"""
        if _HAS_FAISS:
            # Use IndexFlatIP for fastest search (inner product)
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info(f"Initialized FAISS index with dimension {self.dimension}")
        else:
            logger.warning("FAISS not available - falling back to linear search")
    
    def add_documents(self, texts: List[str], embeddings: np.ndarray, metadata: List[Dict]):
        """Add documents to vector store"""
        if self.index is not None:
            # Normalize embeddings for cosine similarity via inner product
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
        
        self.documents.extend(texts)
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(texts)} documents to vector store")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, Dict, float]]:
        """Fast similarity search"""
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(k, len(self.documents)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append((
                    self.documents[idx],
                    self.metadata[idx],
                    float(score)
                ))
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            'total_documents': len(self.documents),
            'index_dimension': self.dimension,
            'index_type': 'FAISS' if self.index else 'Linear'
        }

class FastSemanticRAG:
    """Ultra-fast semantic RAG system optimized for speed"""
    
    def __init__(self, config: FastRAGConfig = None, storage_path: str = "data/fast_rag"):
        self.config = config or FastRAGConfig()
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.chunker = FastSemanticChunker(self.config.embedding_model)
        self.pattern_extractor = FastPatternExtractor()
        self.eda_processor = FastEDAProcessor()
        self.vector_store = FastVectorStore()
        
        # Initialize LLM with existing fallback order
        self.llm = self._init_llm()
        self.current_llm_provider = None  # Track which provider is being used
        
        # Statistics
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'queries_processed': 0,
            'avg_processing_time': 0.0,
            'total_processing_time': 0.0
        }
        
        logger.info("FastSemanticRAG initialized successfully")
    
    def _init_llm(self):
        """Initialize LLM with API-first fallback order from .env configuration"""
        if not _HAS_LANGCHAIN:
            logger.warning("LangChain not available")
            return None
        
        # Get fallback order from environment
        fallback_order = os.getenv('LLM_FALLBACK_ORDER', 'openai,anthropic,google,groq,huggingface,ollama').split(',')
        
        # Get common parameters
        temperature = float(os.getenv('LLM_TEMPERATURE', self.config.temperature))
        max_tokens = int(os.getenv('LLM_MAX_TOKENS', self.config.max_tokens))
        timeout = int(os.getenv('LLM_TIMEOUT', 30))
        
        logger.info(f"Attempting LLM initialization with fallback order: {fallback_order}")
        
        for provider in fallback_order:
            provider = provider.strip().lower()
            
            # Check if provider is enabled
            if not self._is_provider_enabled(provider):
                logger.info(f"Skipping {provider} - disabled in configuration")
                continue
            
            try:
                llm = self._init_provider_llm(provider, temperature, max_tokens, timeout)
                if llm:
                    self.current_llm_provider = provider
                    logger.info(f"Successfully initialized LLM: {provider}")
                    return llm
            except Exception as e:
                logger.warning(f"Failed to initialize {provider}: {e}")
                continue
        
        logger.error("All LLM providers failed to initialize")
        return None
    
    def _is_provider_enabled(self, provider: str) -> bool:
        """Check if a provider is enabled in configuration"""
        enable_key = f'ENABLE_{provider.upper()}'
        return os.getenv(enable_key, 'true').lower() == 'true'
    
    def _init_provider_llm(self, provider: str, temperature: float, max_tokens: int, timeout: int):
        """Initialize specific LLM provider"""
        
        if provider == 'openai' and _HAS_OPENAI:
            api_key = os.getenv('OPENAI_API_KEY')
            model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
            if api_key and api_key != 'your_openai_api_key_here':
                return ChatOpenAI(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    api_key=api_key
                )
        
        elif provider == 'anthropic' and _HAS_ANTHROPIC:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            model = os.getenv('ANTHROPIC_MODEL', 'claude-3-haiku-20240307')
            if api_key and api_key != 'your_anthropic_api_key_here':
                return ChatAnthropic(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    api_key=api_key
                )
        
        elif provider == 'google' and _HAS_GOOGLE:
            api_key = os.getenv('GOOGLE_API_KEY')
            model = os.getenv('GOOGLE_MODEL', 'gemini-pro')
            if api_key and api_key != 'your_google_api_key_here':
                return ChatGoogleGenerativeAI(
                    model=model,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    timeout=timeout,
                    google_api_key=api_key
                )
        
        elif provider == 'groq' and _HAS_GROQ:
            api_key = os.getenv('GROQ_API_KEY')
            model = os.getenv('GROQ_MODEL', 'llama3-8b-8192')
            if api_key and api_key != 'your_groq_api_key_here':
                return ChatGroq(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    api_key=api_key
                )
        
        elif provider == 'huggingface' and _HAS_HUGGINGFACE:
            api_key = os.getenv('HUGGINGFACE_API_KEY')
            model = os.getenv('HUGGINGFACE_MODEL', 'meta-llama/Llama-2-7b-chat-hf')
            if api_key and api_key != 'your_huggingface_api_key_here':
                # Create HuggingFace endpoint
                endpoint = HuggingFaceEndpoint(
                    repo_id=model,
                    temperature=temperature,
                    max_new_tokens=max_tokens,
                    timeout=timeout,
                    huggingfacehub_api_token=api_key
                )
                return ChatHuggingFace(llm=endpoint)
        
        elif provider == 'ollama':
            # Ollama as fallback (local)
            model = os.getenv('OLLAMA_MODEL', self.config.llm_model)
            base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            ollama_timeout = int(os.getenv('OLLAMA_TIMEOUT', 60))
            
            return Ollama(
                model=model,
                temperature=temperature,
                num_predict=max_tokens,
                timeout=ollama_timeout,
                base_url=base_url
            )
        
        return None
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to the RAG system with fast processing"""
        start_time = time.time()
        
        if metadata is None:
            metadata = [{'source': f'doc_{i}'} for i in range(len(documents))]
        
        all_chunks = []
        all_metadata = []
        all_patterns = []
        
        # Process documents
        for i, (doc, meta) in enumerate(zip(documents, metadata)):
            # Fast semantic chunking
            if self.config.enable_semantic_chunking:
                chunks = self.chunker.semantic_chunk(doc, self.config.chunk_size)
            else:
                # Simple chunking for maximum speed
                chunks = self._simple_chunk(doc)
            
            # Limit chunks per document for speed
            chunks = chunks[:self.config.max_chunks_per_doc]
            
            # Extract patterns from the full document
            if self.config.enable_pattern_extraction:
                patterns = self.pattern_extractor.extract_patterns(doc)
            else:
                patterns = {}
            
            # Create chunk metadata
            for j, chunk in enumerate(chunks):
                chunk_meta = meta.copy()
                chunk_meta.update({
                    'doc_id': i,
                    'chunk_id': j,
                    'chunk_index': len(all_chunks),
                    **patterns  # Include extracted patterns
                })
                
                all_chunks.append(chunk)
                all_metadata.append(chunk_meta)
                all_patterns.append(patterns)
        
        # Generate embeddings
        if all_chunks:
            embeddings = self.chunker.embed_texts(all_chunks)
            
            # Add to vector store
            self.vector_store.add_documents(all_chunks, embeddings, all_metadata)
        
        # Update statistics
        processing_time = time.time() - start_time
        self.stats['documents_processed'] += len(documents)
        self.stats['chunks_created'] += len(all_chunks)
        self.stats['total_processing_time'] += processing_time
        
        logger.info(f"Processed {len(documents)} documents ({len(all_chunks)} chunks) in {processing_time:.3f}s")
        
        return {
            'documents_added': len(documents),
            'chunks_created': len(all_chunks),
            'processing_time': processing_time,
            'patterns_extracted': len([p for p in all_patterns if p])
        }
    
    def _simple_chunk(self, text: str) -> List[str]:
        """Simple word-based chunking for maximum speed"""
        words = text.split()
        chunk_size_words = self.config.chunk_size // 4  # Approximate words per chunk
        
        chunks = []
        for i in range(0, len(words), chunk_size_words):
            chunk = ' '.join(words[i:i + chunk_size_words])
            chunks.append(chunk)
        
        return chunks
    
    def query(self, question: str) -> Dict[str, Any]:
        """Fast query processing with semantic search"""
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = self.chunker.embed_texts([question])[0]
            
            # Fast vector search
            search_results = self.vector_store.search(
                query_embedding, 
                k=self.config.top_k
            )
            
            # Prepare response data
            if not search_results:
                return {
                    'answer': "No relevant documents found for your query.",
                    'source_documents': [],
                    'processing_time': time.time() - start_time,
                    'method': 'fast_semantic'
                }
            
            # Extract pattern-based insights
            pattern_insights = self._extract_pattern_insights(question, search_results)
            
            # Generate LLM response if available
            llm_response = ""
            if self.llm:
                context = '\n\n'.join([result[0] for result in search_results])
                llm_response = self._generate_llm_response(question, context)
            
            # Combine responses with better structure
            final_answer = self._combine_responses(llm_response, pattern_insights, question)
            
            # Prepare source documents
            source_docs = []
            for text, metadata, score in search_results:
                source_docs.append(Document(
                    page_content=text,
                    metadata={**metadata, 'relevance_score': score}
                ))
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.stats['queries_processed'] += 1
            self.stats['avg_processing_time'] = (
                (self.stats['avg_processing_time'] * (self.stats['queries_processed'] - 1) + processing_time)
                / self.stats['queries_processed']
            )
            
            return {
                'answer': final_answer or "I found relevant information but couldn't generate a complete response.",
                'source_documents': source_docs,
                'processing_time': processing_time,
                'method': 'fast_semantic',
                'search_results_count': len(search_results)
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'answer': f"Error processing query: {str(e)}",
                'source_documents': [],
                'processing_time': time.time() - start_time,
                'method': 'error'
            }
    
    def _extract_pattern_insights(self, question: str, search_results: List[Tuple[str, Dict, float]]) -> str:
        """Enhanced pattern-based insights with comprehensive entity extraction"""
        question_lower = question.lower()
        insights = []
        
        # Aggregate patterns from search results with enhanced categories
        aggregated_data = {
            'names': set(),
            'emails': set(),
            'skills': set(),
            'technical_skills': set(),
            'professional_skills': set(),
            'project_skills': set(),
            'linkedin': set(),
            'github': set(),
            'portfolio': set(),
            'social_links': set(),
            'organizations': set(),
            'locations': set(),
            'certifications': set(),
            'education': set(),
            'experience_years': [],
            'phones': set()
        }
        
        # Enhanced aggregation with context-aware skill categorization
        for _, metadata, _ in search_results:
            for key in aggregated_data:
                if key in metadata:
                    if key == 'experience_years':
                        aggregated_data[key].extend(metadata[key])
                    else:
                        aggregated_data[key].update(metadata.get(key, []))
            
            # Separate skills by context if available
            skills = metadata.get('skills', [])
            for skill in skills:
                if '(' in skill and ')' in skill:
                    # Contextual skill like "python (technical_skills)"
                    base_skill = skill.split('(')[0].strip()
                    context = skill.split('(')[1].split(')')[0].strip()
                    aggregated_data['skills'].add(base_skill)
                    if context in aggregated_data:
                        aggregated_data[context].add(base_skill)
                else:
                    aggregated_data['skills'].add(skill)
        
        # Generate enhanced insights based on question patterns
        if any(word in question_lower for word in ['name', 'candidate', 'person', 'who']):
            if aggregated_data['names']:
                insights.append(f"**👤 Candidate Names:**\n" + '\n'.join(f"• {name}" for name in sorted(aggregated_data['names'])))
        
        if any(word in question_lower for word in ['email', 'contact', 'reach']):
            if aggregated_data['emails']:
                insights.append(f"**📧 Email Addresses:**\n" + '\n'.join(f"• {email}" for email in sorted(aggregated_data['emails'])))
        
        if any(word in question_lower for word in ['phone', 'mobile', 'number']):
            if aggregated_data['phones']:
                insights.append(f"**📱 Phone Numbers:**\n" + '\n'.join(f"• {phone}" for phone in sorted(aggregated_data['phones'])))
        
        if any(word in question_lower for word in ['skill', 'technology', 'tech', 'programming']):
            # Provide context-aware skill insights
            if aggregated_data['technical_skills']:
                insights.append(f"**💻 Technical Skills:**\n" + '\n'.join(f"• {skill}" for skill in sorted(aggregated_data['technical_skills'])[:10]))
            elif aggregated_data['skills']:
                skill_counts = Counter([skill.lower() for skill in aggregated_data['skills']])
                insights.append(f"**🛠️ All Skills:**\n" + '\n'.join(f"• {skill}" for skill in list(skill_counts.keys())[:10]))
        
        if any(word in question_lower for word in ['linkedin', 'social', 'profile']):
            if aggregated_data['linkedin']:
                insights.append(f"**🔗 LinkedIn Profiles:**\n" + '\n'.join(f"• {link}" for link in sorted(aggregated_data['linkedin'])))
        
        if any(word in question_lower for word in ['github', 'code', 'repository', 'repo']):
            if aggregated_data['github']:
                insights.append(f"**💻 GitHub Profiles:**\n" + '\n'.join(f"• {link}" for link in sorted(aggregated_data['github'])))
        
        if any(word in question_lower for word in ['portfolio', 'website', 'personal']):
            if aggregated_data['portfolio']:
                insights.append(f"**🌐 Portfolio/Websites:**\n" + '\n'.join(f"• {link}" for link in sorted(aggregated_data['portfolio'])))
        
        if any(word in question_lower for word in ['company', 'organization', 'employer', 'work']):
            if aggregated_data['organizations']:
                org_counts = Counter(aggregated_data['organizations'])
                insights.append(f"**🏢 Organizations:**\n" + '\n'.join(f"• {org}" for org in list(org_counts.keys())[:10]))
        
        if any(word in question_lower for word in ['location', 'city', 'country', 'address']):
            if aggregated_data['locations']:
                insights.append(f"**📍 Locations:**\n" + '\n'.join(f"• {loc}" for loc in sorted(aggregated_data['locations'])))
        
        if any(word in question_lower for word in ['certification', 'certificate', 'certified']):
            if aggregated_data['certifications']:
                insights.append(f"**🏆 Certifications:**\n" + '\n'.join(f"• {cert}" for cert in sorted(aggregated_data['certifications'])))
        
        if any(word in question_lower for word in ['education', 'degree', 'university', 'college']):
            if aggregated_data['education']:
                insights.append(f"**🎓 Education:**\n" + '\n'.join(f"• {edu}" for edu in sorted(aggregated_data['education'])))
        
        if any(word in question_lower for word in ['experience', 'years', 'exp']):
            if aggregated_data['experience_years']:
                exp_years = [int(x) for x in aggregated_data['experience_years'] if isinstance(x, (int, str)) and str(x).isdigit()]
                if exp_years:
                    avg_exp = sum(exp_years) / len(exp_years)
                    max_exp = max(exp_years)
                    insights.append(f"**📈 Experience Analysis:**\n• Average: {avg_exp:.1f} years\n• Maximum: {max_exp} years\n• Experience range: {min(exp_years)}-{max_exp} years")
        
        if any(word in question_lower for word in ['eda', 'analysis', 'overview', 'summary']):
            # Generate EDA summary
            texts = [result[0] for result in search_results]
            eda_summary = self.eda_processor.semantic_summary(texts)
            insights.append(f"**📊 Semantic Analysis:**\n{eda_summary}")
        
        return '\n\n'.join(insights)
    
    def _generate_llm_response(self, question: str, context: str) -> str:
        """Generate structured LLM response with proper formatting"""
        if not self.llm:
            return ""
        
        # Create a structured prompt for better responses
        prompt = f"""You are a professional resume analyst. Based on the provided resume information, answer the question in a clear, structured format.

                    RESUME CONTEXT:
                    {context}

                    QUESTION: {question}

                    Please provide a comprehensive answer following this structure:
                    1. **Direct Answer**: Start with a clear, direct response
                    2. **Key Details**: Provide specific relevant information from the resume
                    3. **Summary**: Conclude with a brief summary if applicable

                    Use bullet points, headings, and clear formatting for readability.

                    RESPONSE:"""
        
        try:
            response = self.llm.invoke(prompt)
            
            # Handle different response types from various LLM providers
            formatted_response = self._format_llm_response(response)
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return ""
    
    def _format_llm_response(self, response) -> str:
        """Format and unwrap LLM response for better presentation"""
        try:
            # Handle different response types
            if hasattr(response, 'content'):
                # For chat-based responses (ChatOpenAI, ChatAnthropic, etc.)
                text = response.content
            elif hasattr(response, 'text'):
                # For some LLM responses
                text = response.text
            elif isinstance(response, str):
                # Direct string response
                text = response
            else:
                # Convert to string as fallback
                text = str(response)
            
            # Clean and format the response
            formatted_text = self._clean_and_structure_response(text)
            
            return formatted_text
            
        except Exception as e:
            logger.error(f"Response formatting failed: {e}")
            return str(response) if response else ""
    
    def _clean_and_structure_response(self, text: str) -> str:
        """Clean and structure the LLM response text"""
        if not text:
            return ""
        
        # Remove excessive whitespace and clean up
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Remove triple+ newlines
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)  # Remove leading/trailing spaces per line
        text = text.strip()
        
        # Enhance markdown formatting
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append('')
                continue
            
            # Enhance headings
            if line.startswith('##'):
                formatted_lines.append(f"\n{line}")
            elif line.startswith('#'):
                formatted_lines.append(f"\n{line}")
            elif line.startswith('**') and line.endswith('**') and len(line) > 4:
                # Bold headings
                formatted_lines.append(f"\n{line}")
            elif line.startswith('- ') or line.startswith('• '):
                # Bullet points
                formatted_lines.append(f"  {line}")
            elif line.startswith(('1.', '2.', '3.', '4.', '5.')):
                # Numbered lists
                formatted_lines.append(f"  {line}")
            else:
                formatted_lines.append(line)
        
        # Join and final cleanup
        formatted_text = '\n'.join(formatted_lines)
        
        # Remove excessive blank lines at start/end
        formatted_text = re.sub(r'^\n+', '', formatted_text)
        formatted_text = re.sub(r'\n+$', '', formatted_text)
        
        return formatted_text
    
    def _combine_responses(self, llm_response: str, pattern_insights: str, question: str) -> str:
        """Combine LLM response and pattern insights into a well-structured answer"""
        
        # If no LLM response, return pattern insights or fallback
        if not llm_response or llm_response.strip() == "":
            if pattern_insights:
                return f"## 📊 **Analysis Results**\n\n{pattern_insights}"
            else:
                return "I found relevant information but couldn't generate a complete response. Please try rephrasing your question."
        
        # If no pattern insights, return formatted LLM response
        if not pattern_insights or pattern_insights.strip() == "":
            return f"## 🤖 **AI Analysis**\n\n{llm_response}"
        
        # Combine both responses intelligently
        question_lower = question.lower()
        
        # For specific data queries, prioritize pattern insights
        if any(word in question_lower for word in ['email', 'contact', 'skill', 'linkedin', 'github', 'experience', 'years']):
            combined_response = f"""## 🤖 **AI Analysis**

                                {llm_response}

                                ## 📊 **Extracted Data**

                                {pattern_insights}"""
        
        # For general queries, prioritize LLM response
        else:
            combined_response = f"""## 🤖 **Comprehensive Analysis**

                                {llm_response}

---

## 📋 **Additional Insights**

{pattern_insights}"""
        
        return combined_response
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        vector_stats = self.vector_store.get_stats()
        
        return {
            **self.stats,
            **vector_stats,
            'config': {
                'embedding_model': self.config.embedding_model,
                'chunk_size': self.config.chunk_size,
                'top_k': self.config.top_k,
                'llm_model': self.config.llm_model
            },
            'components': {
                'chunker_method': self.chunker.embed_method,
                'vector_store': 'FAISS' if _HAS_FAISS else 'Fallback',
                'llm_available': self.llm is not None,
                'current_llm_provider': self.current_llm_provider
            }
        }
    
    def perform_eda(self, question: str = "Perform comprehensive EDA analysis") -> Dict[str, Any]:
        """Perform fast EDA analysis on all documents"""
        if not self.vector_store.documents:
            return {'answer': 'No documents available for EDA analysis.'}
        
        # Run EDA on all documents
        eda_results = self.eda_processor.fast_eda(self.vector_store.documents)
        eda_summary = self.eda_processor.semantic_summary(self.vector_store.documents)
        
        return {
            'answer': eda_summary,
            'eda_stats': eda_results,
            'source_documents': [],
            'processing_time': eda_results['processing_time'],
            'method': 'fast_eda'
        }

def create_fast_semantic_rag(storage_path: str = "data/fast_rag", **config_kwargs) -> FastSemanticRAG:
    """Factory function to create fast semantic RAG system"""
    config = FastRAGConfig(**config_kwargs)
    return FastSemanticRAG(config, storage_path)

# Backward compatibility with existing system
def create_advanced_rag_system(index_dir: str = "data/index", **kwargs) -> FastSemanticRAG:
    """Backward compatibility function"""
    return create_fast_semantic_rag(index_dir, **kwargs)