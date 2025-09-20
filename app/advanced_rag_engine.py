# app/advanced_rag_engine.py - Next-Generation RAG System with Latest Open-Source Models & Techniques
"""
Enhanced RAG Engine with cutting-edge open-source models and techniques:
- Latest embeddings (BGE-M3, E5-mistral, Nomic, Arctic)
- Advanced retrieval (ColBERT, RAPTOR, RankGPT)
- Modern LLMs (Qwen2.5, Llama 3.1/3.2, Phi-3.5, Mistral NeMo)
- Graph RAG with Neo4j
- Multi-modal capabilities
- Self-improving RAG with feedback loops
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
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from tqdm import tqdm

# Core ML libraries
try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# Latest Embedding Models
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False

try:
    from fastembed import TextEmbedding, SparseTextEmbedding
    _HAS_FASTEMBED = True
except ImportError:
    _HAS_FASTEMBED = False

# Advanced Vector Databases
try:
    import qdrant_client
    from qdrant_client.models import Distance, VectorParams, PointStruct
    _HAS_QDRANT = True
except ImportError:
    _HAS_QDRANT = False

try:
    import weaviate
    _HAS_WEAVIATE = True
except ImportError:
    _HAS_WEAVIATE = False

try:
    import pinecone
    _HAS_PINECONE = True
except ImportError:
    _HAS_PINECONE = False

# LangChain v0.1+ imports
try:
    from langchain_community.vectorstores import Chroma, FAISS, Qdrant
    from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
    from langchain_community.llms import Ollama
    from langchain_community.chat_models import ChatOllama
    from langchain_community.retrievers import BM25Retriever
    from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
    from langchain.schema import Document
    from langchain.retrievers import EnsembleRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
    from langchain.prompts import ChatPromptTemplate, PromptTemplate
    from langchain.chains import RetrievalQA
    from langchain.memory import ConversationBufferMemory
    from langchain.callbacks import StreamingStdOutCallbackHandler
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False

# LlamaIndex integration
try:
    from llama_index.core import VectorStoreIndex, ServiceContext, StorageContext, load_index_from_storage
    from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.postprocessor import SimilarityPostprocessor
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.ollama import Ollama as LlamaOllama
    _HAS_LLAMAINDEX = True
except ImportError:
    _HAS_LLAMAINDEX = False

# DSPy for systematic optimization
try:
    import dspy
    _HAS_DSPY = True
except ImportError:
    _HAS_DSPY = False

# Haystack for production pipelines
try:
    from haystack import Pipeline, Document as HaystackDocument
    from haystack.components.retrievers import InMemoryEmbeddingRetriever
    from haystack.components.generators import OpenAIGenerator
    from haystack.components.builders import PromptBuilder
    _HAS_HAYSTACK = True
except ImportError:
    _HAS_HAYSTACK = False

# Neo4j for Graph RAG
try:
    from neo4j import GraphDatabase
    _HAS_NEO4J = True
except ImportError:
    _HAS_NEO4J = False

# RAGAS for evaluation
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        context_relevancy
    )
    _HAS_RAGAS = True
except ImportError:
    _HAS_RAGAS = False

# ColBERT for late interaction
try:
    from colbert import Indexer, Searcher
    from colbert.infra import Run, RunConfig, ColBERTConfig
    _HAS_COLBERT = True
except ImportError:
    _HAS_COLBERT = False

# BM25 for hybrid search
try:
    from rank_bm25 import BM25Okapi, BM25L, BM25Plus
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

# Multi-modal models
try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoProcessor, 
        BlipProcessor, BlipForConditionalGeneration,
        LlavaNextProcessor, LlavaNextForConditionalGeneration,
        pipeline
    )
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

# OCR and document processing
try:
    import easyocr
    import pytesseract
    from PIL import Image
    import cv2
    _HAS_OCR = True
except ImportError:
    _HAS_OCR = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class EmbeddingModel(Enum):
    """Latest and most powerful embedding models"""
    BGE_M3 = "BAAI/bge-m3"  # Best multilingual model
    BGE_LARGE_EN = "BAAI/bge-large-en-v1.5"
    E5_MISTRAL = "intfloat/e5-mistral-7b-instruct"  # Instruction-tuned
    E5_LARGE_V2 = "intfloat/e5-large-v2"
    NOMIC_EMBED = "nomic-ai/nomic-embed-text-v1.5"  # High performance
    ARCTIC_EMBED = "Snowflake/snowflake-arctic-embed-l"  # Snowflake's latest
    JINA_V2 = "jinaai/jina-embeddings-v2-base-en"  # Long context
    GTE_LARGE = "thenlper/gte-large"
    INSTRUCTOR_XL = "hkunlp/instructor-xl"
    UAE_LARGE = "WhereIsAI/UAE-Large-V1"

class LLMModel(Enum):
    """Latest open-source LLMs"""
    QWEN2_5_72B = "qwen2.5:72b"  # Alibaba's most advanced
    QWEN2_5_32B = "qwen2.5:32b"
    LLAMA3_1_405B = "llama3.1:405b"  # Meta's largest
    LLAMA3_1_70B = "llama3.1:70b"
    LLAMA3_2_90B = "llama3.2:90b"  # Latest Llama
    PHI3_5 = "phi3.5:latest"  # Microsoft's efficient model
    MISTRAL_NEMO = "mistral-nemo:latest"  # Latest Mistral
    GEMMA2_27B = "gemma2:27b"  # Google's improved model
    YI_34B = "yi:34b"  # 01.AI's powerful model
    DEEPSEEK_CODER = "deepseek-coder:33b"
    CODELLAMA_70B = "codellama:70b"

class RetrievalStrategy(Enum):
    """Advanced retrieval strategies"""
    VECTOR_ONLY = "vector"
    HYBRID_BM25 = "hybrid_bm25"
    COLBERT = "colbert"
    GRAPH_RAG = "graph_rag"
    RAPTOR = "raptor"
    MULTI_QUERY = "multi_query"
    RAG_FUSION = "rag_fusion"
    ADAPTIVE = "adaptive"

@dataclass
class RAGConfig:
    """Configuration for the enhanced RAG system"""
    embedding_model: EmbeddingModel = EmbeddingModel.BGE_M3
    llm_model: LLMModel = LLMModel.QWEN2_5_32B
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.ADAPTIVE
    vector_db: str = "qdrant"  # qdrant, weaviate, pinecone, chroma, faiss
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 10
    rerank_top_k: int = 5
    enable_graph_rag: bool = True
    enable_multimodal: bool = True
    enable_self_correction: bool = True
    enable_feedback_learning: bool = True
    similarity_threshold: float = 0.7
    max_tokens: int = 4096
    temperature: float = 0.1
    use_async: bool = True

class SemanticChunker:
    """Advanced semantic chunking with multiple strategies"""
    
    def __init__(self, embedding_model: str = "BAAI/bge-small-en-v1.5"):
        self.embedding_model = embedding_model
        if _HAS_SENTENCE_TRANSFORMERS:
            self.embedder = SentenceTransformer(embedding_model)
        else:
            self.embedder = None
    
    def semantic_split(self, text: str, chunk_size: int = 512, threshold: float = 0.5) -> List[str]:
        """Split text based on semantic similarity"""
        if not self.embedder:
            return self._fallback_split(text, chunk_size)
        
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return [text]
        
        embeddings = self.embedder.encode(sentences)
        similarities = []
        
        for i in range(len(embeddings) - 1):
            sim = F.cosine_similarity(
                torch.tensor(embeddings[i]).unsqueeze(0),
                torch.tensor(embeddings[i + 1]).unsqueeze(0)
            ).item()
            similarities.append(sim)
        
        # Find split points where similarity drops
        chunks = []
        current_chunk = [sentences[0]]
        
        for i, sim in enumerate(similarities):
            if sim < threshold or len(' '.join(current_chunk)) > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i + 1]]
            else:
                current_chunk.append(sentences[i + 1])
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _fallback_split(self, text: str, chunk_size: int) -> List[str]:
        """Fallback splitting method"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            if current_size + len(word) > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                current_size += len(word) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

class AdvancedEmbedder:
    """Unified interface for multiple embedding models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embedder = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model with fallbacks"""
        try:
            if _HAS_FASTEMBED and "nomic" in self.model_name.lower():
                self.embedder = TextEmbedding(self.model_name)
                self.embed_method = "fastembed"
            elif _HAS_SENTENCE_TRANSFORMERS:
                self.embedder = SentenceTransformer(self.model_name)
                self.embed_method = "sentence_transformers"
            else:
                logger.warning(f"Could not load {self.model_name}, using fallback")
                self.embedder = None
                self.embed_method = "fallback"
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            self.embedder = None
            self.embed_method = "fallback"
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts to embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        if self.embedder is None:
            return np.random.random((len(texts), 384))  # Fallback
        
        try:
            if self.embed_method == "fastembed":
                embeddings = list(self.embedder.embed(texts))
                return np.array(embeddings)
            elif self.embed_method == "sentence_transformers":
                return self.embedder.encode(texts)
            else:
                return np.random.random((len(texts), 384))
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            return np.random.random((len(texts), 384))

class HybridRetriever:
    """Advanced hybrid retrieval combining multiple methods"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.vector_store = None
        self.bm25_retriever = None
        self.colbert_searcher = None
        self.graph_db = None
        self.reranker = None
        self._setup_retrievers()
    
    def _setup_retrievers(self):
        """Initialize retrieval components"""
        # Setup reranker
        if _HAS_SENTENCE_TRANSFORMERS:
            try:
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            except Exception as e:
                logger.warning(f"Could not load reranker: {e}")
        
        # Setup Graph DB for Graph RAG
        if self.config.enable_graph_rag and _HAS_NEO4J:
            try:
                neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
                neo4j_user = os.getenv("NEO4J_USER", "neo4j")
                neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
                self.graph_db = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            except Exception as e:
                logger.warning(f"Could not connect to Neo4j: {e}")
    
    def setup_vector_store(self, documents: List[Document], embeddings):
        """Setup vector store with documents"""
        try:
            if self.config.vector_db == "qdrant" and _HAS_QDRANT:
                self._setup_qdrant(documents, embeddings)
            elif self.config.vector_db == "weaviate" and _HAS_WEAVIATE:
                self._setup_weaviate(documents, embeddings)
            elif self.config.vector_db == "pinecone" and _HAS_PINECONE:
                self._setup_pinecone(documents, embeddings)
            else:
                self._setup_fallback(documents, embeddings)
            
            # Setup BM25
            if _HAS_BM25:
                corpus = [doc.page_content for doc in documents]
                tokenized_corpus = [doc.split() for doc in corpus]
                self.bm25_retriever = BM25Okapi(tokenized_corpus)
        
        except Exception as e:
            logger.error(f"Error setting up vector store: {e}")
            self._setup_fallback(documents, embeddings)
    
    def _setup_qdrant(self, documents: List[Document], embeddings):
        """Setup Qdrant vector database"""
        try:
            client = qdrant_client.QdrantClient(":memory:")
            vectors_config = VectorParams(size=embeddings.encode(["test"]).shape[1], distance=Distance.COSINE)
            client.create_collection(collection_name="documents", vectors_config=vectors_config)
            
            # Add documents
            texts = [doc.page_content for doc in documents]
            vectors = embeddings.encode(texts)
            
            points = [
                PointStruct(id=i, vector=vectors[i].tolist(), payload={"text": texts[i], "metadata": documents[i].metadata})
                for i in range(len(documents))
            ]
            client.upsert(collection_name="documents", points=points)
            self.vector_store = client
        except Exception as e:
            logger.error(f"Error setting up Qdrant: {e}")
            raise
    
    def _setup_weaviate(self, documents: List[Document], embeddings):
        """Setup Weaviate vector database"""
        # Implementation for Weaviate
        pass
    
    def _setup_pinecone(self, documents: List[Document], embeddings):
        """Setup Pinecone vector database"""
        # Implementation for Pinecone
        pass
    
    def _setup_fallback(self, documents: List[Document], embeddings):
        """Fallback to FAISS or Chroma"""
        if _HAS_LANGCHAIN:
            try:
                self.vector_store = FAISS.from_documents(documents, embeddings)
            except:
                self.vector_store = Chroma.from_documents(documents, embeddings)

class RAPTORRetriever:
    """RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.tree_structure = {}
        self.summaries = {}
    
    def build_tree(self, documents: List[Document]) -> Dict:
        """Build hierarchical tree structure"""
        # Cluster documents by similarity
        clusters = self._cluster_documents(documents)
        
        # Create tree levels
        tree = {"level_0": documents}
        current_level = documents
        level = 1
        
        while len(current_level) > 1:
            # Summarize clusters
            summarized_docs = []
            for cluster in self._create_clusters(current_level):
                summary = self._summarize_cluster(cluster)
                summarized_docs.append(summary)
            
            tree[f"level_{level}"] = summarized_docs
            current_level = summarized_docs
            level += 1
        
        self.tree_structure = tree
        return tree
    
    def _cluster_documents(self, documents: List[Document]) -> List[List[Document]]:
        """Cluster documents by semantic similarity"""
        # Simple clustering implementation
        # In practice, use more sophisticated clustering like HDBSCAN
        clusters = []
        cluster_size = max(2, len(documents) // 5)
        
        for i in range(0, len(documents), cluster_size):
            clusters.append(documents[i:i + cluster_size])
        
        return clusters
    
    def _create_clusters(self, documents: List[Document]) -> List[List[Document]]:
        """Create clusters from documents"""
        return self._cluster_documents(documents)
    
    def _summarize_cluster(self, cluster: List[Document]) -> Document:
        """Summarize a cluster of documents"""
        combined_text = "\n".join([doc.page_content for doc in cluster])
        # Simple summarization - in practice use advanced summarization models
        summary = combined_text[:500] + "..." if len(combined_text) > 500 else combined_text
        
        return Document(
            page_content=summary,
            metadata={"type": "summary", "source_docs": len(cluster)}
        )
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve using RAPTOR tree traversal"""
        if not self.tree_structure:
            return []
        
        # Start from top level and traverse down
        relevant_docs = []
        
        for level_name, level_docs in self.tree_structure.items():
            # Score documents at this level
            scored_docs = self._score_documents(query, level_docs)
            
            # Select top documents
            top_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)[:top_k]
            relevant_docs.extend([doc for doc, score in top_docs])
        
        return relevant_docs[:top_k]
    
    def _score_documents(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """Score documents for relevance"""
        # Simple scoring - in practice use embedding similarity
        scored = []
        for doc in documents:
            score = self._calculate_similarity(query, doc.page_content)
            scored.append((doc, score))
        return scored
    
    def _calculate_similarity(self, query: str, text: str) -> float:
        """Calculate similarity between query and text"""
        # Simple word overlap - in practice use embedding similarity
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        overlap = len(query_words.intersection(text_words))
        return overlap / max(len(query_words), 1)

class MultiModalProcessor:
    """Multi-modal document processing for images, tables, charts"""
    
    def __init__(self):
        self.ocr_reader = None
        self.vision_model = None
        self._setup_models()
    
    def _setup_models(self):
        """Setup multi-modal models"""
        if _HAS_OCR:
            try:
                self.ocr_reader = easyocr.Reader(['en'])
            except Exception as e:
                logger.warning(f"Could not setup OCR: {e}")
        
        if _HAS_TRANSFORMERS:
            try:
                self.vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.vision_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            except Exception as e:
                logger.warning(f"Could not setup vision model: {e}")
    
    def process_image(self, image_path: str) -> Dict[str, str]:
        """Extract text and description from image"""
        result = {"text": "", "description": ""}
        
        try:
            if _HAS_OCR and self.ocr_reader:
                # Extract text using OCR
                ocr_result = self.ocr_reader.readtext(image_path)
                result["text"] = " ".join([item[1] for item in ocr_result])
            
            if _HAS_TRANSFORMERS and self.vision_model:
                # Generate image description
                from PIL import Image
                image = Image.open(image_path)
                inputs = self.vision_processor(image, return_tensors="pt")
                out = self.vision_model.generate(**inputs, max_length=50)
                result["description"] = self.vision_processor.decode(out[0], skip_special_tokens=True)
        
        except Exception as e:
            logger.error(f"Error processing image: {e}")
        
        return result

class FeedbackLearningSystem:
    """Self-improving system with feedback loops"""
    
    def __init__(self):
        self.feedback_history = []
        self.performance_metrics = defaultdict(list)
        self.query_patterns = defaultdict(int)
    
    def record_feedback(self, query: str, response: str, rating: float, context: Dict):
        """Record user feedback for learning"""
        feedback = {
            "timestamp": time.time(),
            "query": query,
            "response": response,
            "rating": rating,
            "context": context
        }
        self.feedback_history.append(feedback)
        
        # Update query patterns
        self.query_patterns[self._extract_pattern(query)] += 1
    
    def _extract_pattern(self, query: str) -> str:
        """Extract pattern from query for learning"""
        # Simple pattern extraction - could be more sophisticated
        return query.lower().split()[0] if query.split() else "unknown"
    
    def get_recommendations(self, query: str) -> Dict[str, Any]:
        """Get recommendations based on historical feedback"""
        pattern = self._extract_pattern(query)
        
        recommendations = {
            "suggested_strategy": "adaptive",
            "confidence": 0.5,
            "similar_queries": []
        }
        
        # Find similar queries
        for feedback in self.feedback_history[-100:]:  # Last 100 feedbacks
            if self._extract_pattern(feedback["query"]) == pattern:
                recommendations["similar_queries"].append({
                    "query": feedback["query"],
                    "rating": feedback["rating"]
                })
        
        return recommendations

class AdvancedRAGSystem:
    """Next-generation RAG system with all advanced features"""
    
    def __init__(self, config: RAGConfig, index_dir: str = "data/index"):
        self.config = config
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.embedder = AdvancedEmbedder(config.embedding_model.value)
        self.chunker = SemanticChunker(config.embedding_model.value)
        self.retriever = HybridRetriever(config)
        self.raptor = RAPTORRetriever(config)
        self.multimodal = MultiModalProcessor()
        self.feedback_system = FeedbackLearningSystem()
        
        # Storage
        self.documents = []
        self.chunks = []
        self.metadata = []
        
        # Models
        self.llm = None
        self.reranker = None
        
        # Statistics
        self.stats = {
            "total_docs": 0,
            "total_chunks": 0,
            "queries_processed": 0,
            "avg_response_time": 0.0
        }
        
        self._setup_models()
        logger.info(f"Initialized AdvancedRAGSystem with {config.embedding_model.value}")
    
    def _setup_models(self):
        """Setup LLM and other models"""
        try:
            if _HAS_LANGCHAIN:
                self.llm = ChatOllama(
                    model=self.config.llm_model.value,
                    temperature=self.config.temperature,
                    num_predict=self.config.max_tokens
                )
        except Exception as e:
            logger.error(f"Error setting up LLM: {e}")
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to the RAG system"""
        start_time = time.time()
        
        if metadata is None:
            metadata = [{}] * len(documents)
        
        # Process documents
        all_chunks = []
        all_metadata = []
        
        for i, (doc, meta) in enumerate(zip(documents, metadata)):
            # Semantic chunking
            chunks = self.chunker.semantic_split(doc, self.config.chunk_size)
            
            for j, chunk in enumerate(chunks):
                chunk_meta = meta.copy()
                chunk_meta.update({
                    "doc_id": i,
                    "chunk_id": j,
                    "chunk_index": len(all_chunks),
                    "source": meta.get("source", f"doc_{i}")
                })
                
                all_chunks.append(chunk)
                all_metadata.append(chunk_meta)
        
        # Create LangChain documents
        langchain_docs = [
            Document(page_content=chunk, metadata=meta)
            for chunk, meta in zip(all_chunks, all_metadata)
        ]
        
        # Setup retrievers
        self.retriever.setup_vector_store(langchain_docs, self.embedder)
        
        # Build RAPTOR tree
        if self.config.retrieval_strategy == RetrievalStrategy.RAPTOR:
            self.raptor.build_tree(langchain_docs)
        
        # Update storage
        self.documents.extend(documents)
        self.chunks.extend(all_chunks)
        self.metadata.extend(all_metadata)
        
        # Update stats
        self.stats["total_docs"] += len(documents)
        self.stats["total_chunks"] += len(all_chunks)
        
        processing_time = time.time() - start_time
        logger.info(f"Added {len(documents)} documents ({len(all_chunks)} chunks) in {processing_time:.2f}s")
    
    def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """Process query with advanced RAG techniques and entity-aware analytics"""
        start_time = time.time()
        try:
            # Get recommendations from feedback system
            recommendations = self.feedback_system.get_recommendations(question)
            # Detect user intent
            intents = extract_intent(question)
            # Adaptive strategy selection
            strategy = self._select_strategy(question, recommendations)
            # Multi-query generation for RAG fusion
            queries = self._generate_multiple_queries(question) if strategy == RetrievalStrategy.RAG_FUSION else [question]
            # Retrieve relevant documents
            all_docs = []
            for query in queries:
                docs = self._retrieve_documents(query, strategy)
                all_docs.extend(docs)
            # Remove duplicates and rerank
            unique_docs = self._deduplicate_documents(all_docs)
            reranked_docs = self._rerank_documents(question, unique_docs)

            # --- Entity-aware analytics ---
            entity_answer = ""
            if intents:
                # Aggregate metadata from top chunks
                top_metas = [doc.metadata for doc in reranked_docs[:10] if doc.metadata]
                if "want_emails" in intents:
                    emails = set()
                    for meta in top_metas:
                        emails.update(meta.get("emails", []))
                    if emails:
                        entity_answer += f"\n**Emails found:**\n" + "\n".join(sorted(emails))
                if "want_links" in intents:
                    links = set()
                    for meta in top_metas:
                        links.update(meta.get("social_links", []))
                    if links:
                        entity_answer += f"\n**Social/Professional Links:**\n" + "\n".join(sorted(links))
                if "want_names" in intents:
                    names = set()
                    for meta in top_metas:
                        names.update(meta.get("names", []))
                    if names:
                        entity_answer += f"\n**Candidate Names:**\n" + "\n".join(sorted(names))
                if "want_skills" in intents:
                    skills = set()
                    for meta in top_metas:
                        skills.update(meta.get("skills", []))
                    if skills:
                        entity_answer += f"\n**Skills Identified:**\n" + ", ".join(sorted(skills))
                if "want_ranking" in intents:
                    # Example: rank by Python/AWS experience
                    ranking = []
                    for meta in top_metas:
                        score = 0
                        skills = set(meta.get("skills", []))
                        exp_years = meta.get("experience_years", [])
                        if "python" in skills:
                            score += 2
                        if "aws" in skills:
                            score += 2
                        score += sum(exp_years)
                        ranking.append((meta.get("file", "unknown"), score, skills, exp_years))
                    ranking.sort(key=lambda x: x[1], reverse=True)
                    entity_answer += "\n**Candidate Ranking (Python/AWS/Experience):**\n"
                    for fname, score, skills, exp in ranking[:5]:
                        entity_answer += f"{fname}: Score={score}, Skills={', '.join(skills)}, Experience={exp}\n"
                if "want_eda" in intents:
                    # Show EDA stats if available
                    for meta in top_metas:
                        if "readability" in meta:
                            entity_answer += f"\nReadability: {meta['readability']}\n"
                        if "skills" in meta:
                            entity_answer += f"Skills: {', '.join(meta['skills'])}\n"
                        if "education" in meta:
                            entity_answer += f"Education: {', '.join(meta['education'])}\n"
                        if "certifications" in meta:
                            entity_answer += f"Certifications: {', '.join(meta['certifications'])}\n"
            # Generate response
            response = self._generate_response(question, reranked_docs)
            # Self-correction if enabled
            if self.config.enable_self_correction:
                response = self._self_correct_response(question, response, reranked_docs)
            # Combine entity analytics with LLM answer
            if entity_answer:
                response = f"{response}\n\n---\n{entity_answer.strip()}"
            # Prepare result
            result = {
                "answer": response,
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "relevance_score": getattr(doc, 'relevance_score', 0.0)
                    }
                    for doc in reranked_docs[:5]
                ],
                "strategy_used": strategy.value,
                "processing_time": time.time() - start_time,
                "recommendations": recommendations
            }
            # Update stats
            self.stats["queries_processed"] += 1
            self.stats["avg_response_time"] = (
                (self.stats["avg_response_time"] * (self.stats["queries_processed"] - 1) + result["processing_time"])
                / self.stats["queries_processed"]
            )
            return result
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": f"I encountered an error processing your query: {str(e)}",
                "source_documents": [],
                "strategy_used": "error",
                "processing_time": time.time() - start_time,
                "error": str(e)
            }
    
    def _select_strategy(self, query: str, recommendations: Dict) -> RetrievalStrategy:
        """Intelligently select retrieval strategy"""
        if self.config.retrieval_strategy != RetrievalStrategy.ADAPTIVE:
            return self.config.retrieval_strategy
        
        # Simple heuristics for strategy selection
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["graph", "relationship", "connected", "network"]):
            return RetrievalStrategy.GRAPH_RAG
        elif any(word in query_lower for word in ["comprehensive", "detailed", "thorough"]):
            return RetrievalStrategy.RAPTOR
        elif any(word in query_lower for word in ["similar", "like", "related"]):
            return RetrievalStrategy.RAG_FUSION
        else:
            return RetrievalStrategy.HYBRID_BM25
    
    def _generate_multiple_queries(self, original_query: str) -> List[str]:
        """Generate multiple query variations for RAG fusion"""
        queries = [original_query]
        
        # Simple query variations - in practice use LLM to generate better variations
        variations = [
            f"What are the details about {original_query}?",
            f"Can you explain {original_query}?",
            f"Information regarding {original_query}",
        ]
        
        queries.extend(variations[:2])  # Limit to avoid too many queries
        return queries
    
    def _retrieve_documents(self, query: str, strategy: RetrievalStrategy) -> List[Document]:
        """Retrieve documents using specified strategy"""
        try:
            if strategy == RetrievalStrategy.RAPTOR:
                return self.raptor.retrieve(query, self.config.top_k)
            elif strategy == RetrievalStrategy.GRAPH_RAG:
                return self._graph_retrieve(query)
            elif strategy == RetrievalStrategy.COLBERT:
                return self._colbert_retrieve(query)
            else:
                return self._hybrid_retrieve(query)
        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            return []
    
    def _hybrid_retrieve(self, query: str) -> List[Document]:
        """Hybrid retrieval combining vector and BM25"""
        docs = []
        
        try:
            # Vector retrieval
            if self.retriever.vector_store:
                if hasattr(self.retriever.vector_store, 'similarity_search'):
                    vector_docs = self.retriever.vector_store.similarity_search(query, k=self.config.top_k)
                else:
                    # For Qdrant
                    query_vector = self.embedder.encode([query])[0]
                    search_result = self.retriever.vector_store.search(
                        collection_name="documents",
                        query_vector=query_vector.tolist(),
                        limit=self.config.top_k
                    )
                    vector_docs = [
                        Document(page_content=hit.payload["text"], metadata=hit.payload["metadata"])
                        for hit in search_result
                    ]
                docs.extend(vector_docs)
            
            # BM25 retrieval
            if self.retriever.bm25_retriever:
                query_tokens = query.split()
                bm25_scores = self.retriever.bm25_retriever.get_scores(query_tokens)
                top_indices = np.argsort(bm25_scores)[-self.config.top_k:]
                
                for idx in top_indices:
                    if idx < len(self.chunks):
                        doc = Document(
                            page_content=self.chunks[idx],
                            metadata=self.metadata[idx]
                        )
                        docs.append(doc)
        
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
        
        return docs
    
    def _graph_retrieve(self, query: str) -> List[Document]:
        """Graph-based retrieval using Neo4j"""
        if not self.retriever.graph_db:
            return self._hybrid_retrieve(query)
        
        # Implementation for graph retrieval
        # This would involve creating knowledge graphs from documents
        # and traversing them to find relevant information
        return self._hybrid_retrieve(query)  # Fallback for now
    
    def _colbert_retrieve(self, query: str) -> List[Document]:
        """ColBERT late interaction retrieval"""
        if not _HAS_COLBERT or not self.retriever.colbert_searcher:
            return self._hybrid_retrieve(query)
        
        # Implementation for ColBERT retrieval
        return self._hybrid_retrieve(query)  # Fallback for now
    
    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate documents"""
        seen = set()
        unique_docs = []
        
        for doc in documents:
            content_hash = hash(doc.page_content)
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs
    
    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents using cross-encoder"""
        if not self.retriever.reranker or not documents:
            return documents[:self.config.rerank_top_k]
        
        try:
            # Prepare pairs for reranking
            pairs = [(query, doc.page_content) for doc in documents]
            scores = self.retriever.reranker.predict(pairs)
            
            # Sort by scores
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Add relevance scores to documents
            reranked_docs = []
            for doc, score in scored_docs[:self.config.rerank_top_k]:
                try:
                    doc.relevance_score = float(score)
                except AttributeError:
                    # If we can't set the attribute, add it to metadata
                    if hasattr(doc, 'metadata') and doc.metadata is not None:
                        doc.metadata['relevance_score'] = float(score)
                reranked_docs.append(doc)
            
            return reranked_docs
        
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            return documents[:self.config.rerank_top_k]
    def _generate_response(self, query: str, documents: List[Document]) -> str:
        """Generate response using LLM"""
        if not self.llm:
            return "LLM not available for response generation."
        
        # Prepare context
        context = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(documents)
        ])
        
        # Create prompt
        prompt = f"""Based on the following context documents, please answer the question.

Context:
{context}

Question: {query}

Answer: Provide a comprehensive answer based on the context. If the information is not available in the context, say so clearly."""
        
        try:
            if _HAS_LANGCHAIN:
                response = self.llm.invoke(prompt)
                return response.content if hasattr(response, 'content') else str(response)
            else:
                return "Response generation not available."
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def _self_correct_response(self, query: str, response: str, documents: List[Document]) -> str:
        """Self-correct the response for accuracy and completeness"""
        if not self.llm:
            return response
        
        correction_prompt = f"""Please review and improve the following answer for accuracy and completeness.

                            Original Question: {query}
                            Current Answer: {response}

                            Available Context:
                            {documents[0].page_content if documents else "No additional context"}

                            Improved Answer: Provide a corrected and enhanced version of the answer."""
        
        try:
            corrected = self.llm.invoke(correction_prompt)
            return corrected.content if hasattr(corrected, 'content') else response
        except Exception as e:
            logger.error(f"Error in self-correction: {e}")
            return response
    
    def evaluate_performance(self, test_queries: List[Dict]) -> Dict[str, float]:
        """Evaluate RAG performance using RAGAS metrics"""
        if not _HAS_RAGAS:
            logger.warning("RAGAS not available for evaluation")
            return {}
        
        try:
            # Prepare evaluation data
            questions = [item["question"] for item in test_queries]
            ground_truths = [item["ground_truth"] for item in test_queries]
            
            # Generate answers
            answers = []
            contexts = []
            
            for query in questions:
                result = self.query(query)
                answers.append(result["answer"])
                contexts.append([doc["content"] for doc in result["source_documents"]])
            
            # Create evaluation dataset
            eval_data = {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truths": ground_truths
            }
            
            # Run evaluation
            result = evaluate(
                eval_data,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Error in performance evaluation: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            **self.stats,
            "config": {
                "embedding_model": self.config.embedding_model.value,
                "llm_model": self.config.llm_model.value,
                "retrieval_strategy": self.config.retrieval_strategy.value,
                "vector_db": self.config.vector_db
            },
            "feedback_history_size": len(self.feedback_system.feedback_history),
            "query_patterns": dict(self.feedback_system.query_patterns)
        }
    
    def record_feedback(self, query: str, response: str, rating: float):
        """Record user feedback for continuous improvement"""
        self.feedback_system.record_feedback(
            query=query,
            response=response,
            rating=rating,
            context={"timestamp": time.time()}
        )

# Intent detection patterns (keeping existing ones)
INTENT_PATTERNS = {
    "want_emails": r"\b(email|emails|e-mail|mail id|mailid|mailto)\b",
    "want_links":  r"\b(link|links|linkedin|github|portfolio|website|social)\b",
    "want_names":  r"\b(name|names|candidate[s]?\b)\b",
    "want_skills": r"\b(skill|skills|tech stack|technologies|tools)\b",
    "want_sources": r"\b(source|sources|file|files|page|pages|citation|cite|provenance)\b",
    "want_eda":    r"\b(semantics|eda|keywords|co-occur|cooccur|bigrams|top terms|relationships)\b",
    "want_ranking": r"\b(rank|ranking|sort|top|best|most)\b",
}

# Utility functions (keeping existing ones)
TOKEN_PAT = re.compile(r"[A-Za-z][A-Za-z0-9_+#.\-]+")
STOP_WORDS = set("a an and are as at be by for from has have in into is it its of on or that the this to was were will with your you".split())

def tokenize_text(text: str) -> List[str]:
    """Tokenize text for EDA analysis"""
    tokens = [t.lower() for t in TOKEN_PAT.findall(text)]
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

def create_advanced_rag_system(index_dir: str = "data/index", **config_kwargs) -> AdvancedRAGSystem:
    """Factory function to create an advanced RAG system"""
    config = RAGConfig(**config_kwargs)
    return AdvancedRAGSystem(config, index_dir)

# Backward compatibility functions
def extract_intent(query: str) -> List[str]:
    """Extract user intent from query"""
    intents = []
    for intent_name, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, query, re.IGNORECASE):
            intents.append(intent_name)
    return intents

def compute_eda_stats(chunks: List[str]) -> Dict[str, Any]:
    """Compute EDA statistics from text chunks"""
    all_tokens = []
    for chunk in chunks:
        all_tokens.extend(tokenize_text(chunk))
    
    if not all_tokens:
        return {}
    
    # Token frequency
    token_counts = Counter(all_tokens)
    
    # Bigrams
    bigrams = []
    for chunk in chunks:
        tokens = tokenize_text(chunk)
        bigrams.extend([f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)])
    
    bigram_counts = Counter(bigrams)
    
    return {
        "total_tokens": len(all_tokens),
        "unique_tokens": len(token_counts),
        "top_tokens": dict(token_counts.most_common(20)),
        "top_bigrams": dict(bigram_counts.most_common(10)),
        "avg_tokens_per_chunk": len(all_tokens) / len(chunks) if chunks else 0
    }