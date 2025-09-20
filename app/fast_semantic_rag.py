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

# LangChain for document management
try:
    from langchain.schema import Document
    from langchain_community.llms import Ollama
    from langchain_community.chat_models import ChatOllama
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False

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
        """Initialize the fastest available embedder"""
        if _HAS_FASTEMBED:
            try:
                # FastEmbed is the fastest option
                self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
                self.embed_method = "fastembed"
                logger.info("Using FastEmbed for ultra-fast embeddings")
                return
            except Exception as e:
                logger.warning(f"FastEmbed failed: {e}")
        
        if _HAS_SENTENCE_TRANSFORMERS:
            try:
                # Lightweight sentence transformers
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Very fast
                self.embed_method = "sentence_transformers"
                logger.info("Using lightweight SentenceTransformer")
                return
            except Exception as e:
                logger.warning(f"SentenceTransformer failed: {e}")
        
        # Fallback to dummy embeddings for testing
        self.embed_method = "dummy"
        logger.warning("Using dummy embeddings - install fastembed or sentence-transformers for real functionality")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Fast text embedding"""
        if self.embed_method == "fastembed":
            embeddings = list(self.embedder.embed(texts))
            return np.array(embeddings)
        elif self.embed_method == "sentence_transformers":
            return self.embedder.encode(texts, show_progress_bar=False)
        else:
            # Dummy embeddings for testing
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
        """Initialize LLM with existing fallback order"""
        if not _HAS_LANGCHAIN:
            logger.warning("LangChain not available")
            return None
        
        try:
            # Use Ollama with the configured model
            llm = Ollama(
                model=self.config.llm_model,
                temperature=self.config.temperature,
                num_predict=self.config.max_tokens
            )
            logger.info(f"Initialized LLM: {self.config.llm_model}")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
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
            
            # Combine responses
            final_answer = llm_response
            if pattern_insights:
                final_answer += f"\n\n---\n\n{pattern_insights}"
            
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
        """Extract pattern-based insights from search results"""
        question_lower = question.lower()
        insights = []
        
        # Aggregate patterns from search results
        all_emails = set()
        all_skills = set()
        all_linkedin = set()
        all_github = set()
        all_experience = []
        
        for _, metadata, _ in search_results:
            all_emails.update(metadata.get('emails', []))
            all_skills.update(metadata.get('skills', []))
            all_linkedin.update(metadata.get('linkedin', []))
            all_github.update(metadata.get('github', []))
            all_experience.extend(metadata.get('experience_years', []))
        
        # Generate insights based on question patterns
        if any(word in question_lower for word in ['email', 'contact', 'reach']):
            if all_emails:
                insights.append(f"**📧 Email Addresses Found:**\n" + '\n'.join(f"• {email}" for email in sorted(all_emails)))
        
        if any(word in question_lower for word in ['skill', 'technology', 'tech', 'programming']):
            if all_skills:
                skill_counts = Counter([skill.lower() for skill in all_skills])
                insights.append(f"**💻 Technical Skills:**\n" + '\n'.join(f"• {skill}" for skill in list(skill_counts.keys())[:10]))
        
        if any(word in question_lower for word in ['linkedin', 'profile', 'social']):
            if all_linkedin:
                insights.append(f"**🔗 LinkedIn Profiles:**\n" + '\n'.join(f"• {profile}" for profile in sorted(all_linkedin)))
        
        if any(word in question_lower for word in ['github', 'code', 'repository']):
            if all_github:
                insights.append(f"**🐙 GitHub Profiles:**\n" + '\n'.join(f"• {profile}" for profile in sorted(all_github)))
        
        if any(word in question_lower for word in ['experience', 'years', 'senior']):
            if all_experience:
                avg_exp = sum(all_experience) / len(all_experience)
                max_exp = max(all_experience)
                insights.append(f"**📈 Experience Analysis:**\n• Average: {avg_exp:.1f} years\n• Maximum: {max_exp} years\n• Experience range: {min(all_experience)}-{max_exp} years")
        
        if any(word in question_lower for word in ['eda', 'analysis', 'overview', 'summary']):
            # Generate EDA summary
            texts = [result[0] for result in search_results]
            eda_summary = self.eda_processor.semantic_summary(texts)
            insights.append(f"**📊 Semantic Analysis:**\n{eda_summary}")
        
        return '\n\n'.join(insights)
    
    def _generate_llm_response(self, question: str, context: str) -> str:
        """Generate LLM response using existing fallback order"""
        if not self.llm:
            return ""
        
        prompt = f"""Based on the following resume information, please answer the question comprehensively.

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            response = self.llm.invoke(prompt)
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return ""
    
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
                'llm_available': self.llm is not None
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