# app/advanced_rag_engine.py - Advanced RAG system with LangChain, LightRAG, and open-source models

from __future__ import annotations
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain.schema import Document
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.llms import Ollama, HuggingFacePipeline
from langchain.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamingStdOutCallbackHandler

# LightRAG and advanced retrieval
try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete
    from lightrag.utils import EmbeddingDim
    _HAS_LIGHTRAG = True
except ImportError:
    _HAS_LIGHTRAG = False

# Advanced embeddings
try:
    from fastembed import TextEmbedding
    _HAS_FASTEMBED = True
except ImportError:
    _HAS_FASTEMBED = False

# Reranking models
try:
    from sentence_transformers import CrossEncoder
    _HAS_CROSSENCODER = True
except ImportError:
    _HAS_CROSSENCODER = False

# BM25 for hybrid search
from rank_bm25 import BM25Okapi

# Intent detection patterns
INTENT_PATTERNS = {
    "want_emails": r"\b(email|emails|e-mail|mail id|mailid|mailto)\b",
    "want_links":  r"\b(link|links|linkedin|github|portfolio|website|social)\b",
    "want_names":  r"\b(name|names|candidate[s]?\b)\b",
    "want_skills": r"\b(skill|skills|tech stack|technologies|tools)\b",
    "want_sources": r"\b(source|sources|file|files|page|pages|citation|cite|provenance)\b",
    "want_eda":    r"\b(semantics|eda|keywords|co-occur|cooccur|bigrams|top terms|relationships)\b",
    "want_ranking": r"\b(rank|ranking|sort|top|best|most)\b",
}

# EDA utilities
TOKEN_PAT = re.compile(r"[A-Za-z][A-Za-z0-9_+#.\-]+")
STOP_WORDS = set("a an and are as at be by for from has have in into is it its of on or that the this to was were will with your you".split())

def tokenize_text(text: str) -> List[str]:
    """Tokenize text for EDA analysis"""
    tokens = [t.lower() for t in TOKEN_PAT.findall(text)]
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

def extract_intents(query: str) -> Dict[str, bool]:
    """Extract user intents from query"""
    q = query.lower()
    intents = {k: bool(re.search(p, q)) for k, p in INTENT_PATTERNS.items()}
    
    # Heuristics for implicit intents
    if not intents["want_ranking"] and re.search(r"\b(suitable|fit|match|recommend)\b", q):
        intents["want_ranking"] = True
    
    return intents

def perform_eda_analysis(chunks: List[str]) -> Dict[str, Any]:
    """Perform exploratory data analysis on text chunks"""
    all_tokens = []
    for chunk in chunks:
        all_tokens.extend(tokenize_text(chunk))
    
    token_counts = Counter(all_tokens)
    
    # Create bigrams
    bigrams = []
    for chunk in chunks:
        tokens = tokenize_text(chunk)
        bigrams.extend(list(zip(tokens, tokens[1:])))
    
    bigram_counts = Counter(bigrams)
    
    # Co-occurrence analysis
    window_size = 8
    cooccurrence = defaultdict(int)
    
    for chunk in chunks:
        tokens = tokenize_text(chunk)
        for i, token1 in enumerate(tokens):
            for j in range(i+1, min(i+1+window_size, len(tokens))):
                token2 = tokens[j]
                if token1 != token2:
                    key = tuple(sorted((token1, token2)))
                    cooccurrence[key] += 1
    
    return {
        "top_terms": [word for word, _ in token_counts.most_common(30)],
        "top_bigrams": bigram_counts.most_common(30),
        "top_cooccurrence": sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)[:30]
    }

# Evaluation framework
try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    _HAS_RAGAS = True
except ImportError:
    _HAS_RAGAS = False

class AdvancedEmbeddingManager:
    """Manages multiple embedding models for optimal performance"""
    
    def __init__(self):
        self.embeddings = {}
        self.current_model = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize various embedding models"""
        
        # Standard sentence transformers
        try:
            self.embeddings['sentence_transformers'] = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            print(f"Failed to load sentence transformers: {e}")
        
        # Instructor embeddings for better domain adaptation
        try:
            self.embeddings['instructor'] = HuggingFaceInstructEmbeddings(
                model_name="hkunlp/instructor-base",
                model_kwargs={'device': 'cpu'}
            )
        except Exception as e:
            print(f"Failed to load instructor embeddings: {e}")
        
        # FastEmbed for speed
        if _HAS_FASTEMBED:
            try:
                self.embeddings['fastembed'] = TextEmbedding(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            except Exception as e:
                print(f"Failed to load FastEmbed: {e}")
        
        # Set default
        self.current_model = self.embeddings.get('sentence_transformers') or \
                           self.embeddings.get('instructor') or \
                           list(self.embeddings.values())[0] if self.embeddings else None
    
    def get_embeddings(self, model_name: str = None):
        """Get specific embedding model"""
        if model_name and model_name in self.embeddings:
            return self.embeddings[model_name]
        return self.current_model

class AdvancedTextSplitter:
    """Advanced text splitting with multiple strategies"""
    
    def __init__(self):
        self.splitters = {
            'recursive': RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            ),
            'semantic': SentenceTransformersTokenTextSplitter(
                chunk_overlap=50,
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        }
    
    def split_documents(self, documents: List[Document], strategy: str = 'recursive') -> List[Document]:
        """Split documents using specified strategy"""
        splitter = self.splitters.get(strategy, self.splitters['recursive'])
        return splitter.split_documents(documents)

class HybridRetriever:
    """Hybrid retrieval combining dense and sparse methods"""
    
    def __init__(self, documents: List[Document], embeddings):
        self.documents = documents
        self.embeddings = embeddings
        self._setup_retrievers()
    
    def _setup_retrievers(self):
        """Setup dense and sparse retrievers"""
        
        # Dense retriever (vector similarity)
        self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
        self.dense_retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.3, "k": 10}
        )
        
        # Sparse retriever (BM25)
        texts = [doc.page_content for doc in self.documents]
        tokenized_texts = [text.split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        self.sparse_retriever = BM25Retriever.from_texts(texts)
        
        # Ensemble retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.dense_retriever, self.sparse_retriever],
            weights=[0.7, 0.3]
        )
        
        # Add reranking if available
        if _HAS_CROSSENCODER:
            try:
                compressor = CrossEncoderReranker(
                    model=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'),
                    top_k=5
                )
                self.rerank_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=self.ensemble_retriever
                )
            except Exception as e:
                print(f"Reranker setup failed: {e}")
                self.rerank_retriever = self.ensemble_retriever
        else:
            self.rerank_retriever = self.ensemble_retriever
    
    def retrieve(self, query: str, use_reranking: bool = True) -> List[Document]:
        """Retrieve relevant documents"""
        retriever = self.rerank_retriever if use_reranking else self.ensemble_retriever
        return retriever.get_relevant_documents(query)

class OpenSourceLLMManager:
    """Manages open-source language models with fallback approach"""
    
    def __init__(self):
        self.models = {}
        self.api_clients = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available models with fallback priority: OpenAI -> Gemini -> HuggingFace -> Ollama"""
        
        # Priority 1: OpenAI API
        try:
            import openai
            if os.getenv("OPENAI_API_KEY"):
                self.api_clients['openai'] = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                print("OpenAI client initialized")
        except Exception as e:
            print(f"OpenAI not available: {e}")
        
        # Priority 2: Google Gemini
        try:
            import google.generativeai as genai
            if os.getenv("GEMINI_API_KEY"):
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                self.api_clients['gemini'] = genai
                print("Gemini client initialized")
        except Exception as e:
            print(f"Gemini not available: {e}")
        
        # Priority 3: HuggingFace models
        try:
            from transformers import pipeline
            qa_pipeline = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                device_map="auto" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
            )
            self.models['huggingface'] = HuggingFacePipeline(pipeline=qa_pipeline)
            print("HuggingFace pipeline initialized")
        except Exception as e:
            print(f"HuggingFace pipeline not available: {e}")
        
        # Priority 4: Ollama models (local)
        try:
            self.models['llama2'] = ChatOllama(
                model="llama2:7b",
                temperature=0.1,
                callbacks=[StreamingStdOutCallbackHandler()]
            )
            print("Ollama Llama2 initialized")
        except Exception as e:
            print(f"Ollama Llama2 not available: {e}")
        
        try:
            self.models['mistral'] = ChatOllama(
                model="mistral:7b",
                temperature=0.1,
                callbacks=[StreamingStdOutCallbackHandler()]
            )
            print("Ollama Mistral initialized")
        except Exception as e:
            print(f"Ollama Mistral not available: {e}")
    
    def _try_openai(self, prompt: str) -> Optional[str]:
        """Try OpenAI API first"""
        if 'openai' not in self.api_clients:
            return None
        
        try:
            response = self.api_clients['openai'].chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert resume analyst providing precise, evidence-based insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI error: {e}")
            return None
    
    def _try_gemini(self, prompt: str) -> Optional[str]:
        """Try Gemini API second"""
        if 'gemini' not in self.api_clients:
            return None
        
        try:
            model = self.api_clients['gemini'].GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini error: {e}")
            return None
    
    def _try_huggingface(self, prompt: str) -> Optional[str]:
        """Try HuggingFace models third"""
        if 'huggingface' not in self.models:
            return None
        
        try:
            response = self.models['huggingface'](prompt)
            return response
        except Exception as e:
            print(f"HuggingFace error: {e}")
            return None
    
    def _try_ollama(self, prompt: str) -> Optional[str]:
        """Try Ollama models last"""
        for model_name in ['llama2', 'mistral']:
            if model_name in self.models:
                try:
                    response = self.models[model_name].invoke(prompt)
                    return response.content if hasattr(response, 'content') else str(response)
                except Exception as e:
                    print(f"Ollama {model_name} error: {e}")
                    continue
        return None
    
    def generate_response(self, prompt: str) -> Optional[str]:
        """Generate response using fallback approach"""
        # Try in priority order: OpenAI -> Gemini -> HuggingFace -> Ollama
        result = (
            self._try_openai(prompt) or
            self._try_gemini(prompt) or
            self._try_huggingface(prompt) or
            self._try_ollama(prompt)
        )
        return result
    
    def get_model(self, model_name: str = None):
        """Get specific model or best available"""
        if model_name and model_name in self.models:
            return self.models[model_name]
        
        # Return first available model
        return next(iter(self.models.values())) if self.models else None

class LightRAGIntegration:
    """Integration with LightRAG for advanced retrieval"""
    
    def __init__(self, working_dir: str):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(exist_ok=True)
        self.rag = None
        
        if _HAS_LIGHTRAG:
            self._initialize_lightrag()
    
    def _initialize_lightrag(self):
        """Initialize LightRAG system"""
        try:
            self.rag = LightRAG(
                working_dir=str(self.working_dir),
                llm_model_func=gpt_4o_mini_complete if os.getenv("OPENAI_API_KEY") else None,
                embedding_dim=EmbeddingDim.OpenAI
            )
        except Exception as e:
            print(f"LightRAG initialization failed: {e}")
    
    def insert_documents(self, texts: List[str]):
        """Insert documents into LightRAG"""
        if self.rag:
            for text in tqdm(texts, desc="Inserting into LightRAG"):
                try:
                    self.rag.insert(text)
                except Exception as e:
                    print(f"Failed to insert document: {e}")
    
    def query(self, question: str, mode: str = "hybrid") -> str:
        """Query LightRAG system"""
        if not self.rag:
            return None
        
        try:
            param = QueryParam(mode=mode)
            return self.rag.query(question, param=param)
        except Exception as e:
            print(f"LightRAG query failed: {e}")
            return None

class AdvancedRAGSystem:
    """Complete advanced RAG system with multiple frameworks"""
    
    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.embedding_manager = AdvancedEmbeddingManager()
        self.text_splitter = AdvancedTextSplitter()
        self.llm_manager = OpenSourceLLMManager()
        self.lightrag = LightRAGIntegration(self.index_dir / "lightrag")
        
        # Storage
        self.documents = []
        self.metadata = []
        self.retriever = None
        self.qa_chain = None
        
        # Memory for conversations
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Setup prompt templates"""
        
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert resume analyst and talent acquisition specialist. 
            Analyze the provided resume data and answer questions with precision and insight.
            
            Always provide:
            1. Evidence-based answers with specific examples
            2. Quantifiable insights where possible
            3. Source citations from the resumes
            4. Actionable recommendations
            
            For ranking questions: Use clear criteria and provide detailed reasoning
            For skills analysis: Categorize by proficiency and provide specific examples
            For candidate matching: Explain fit percentage with detailed reasoning"""),
            
            ("human", """Context: {context}
            
            Question: {question}
            
            Provide a comprehensive analysis:""")
        ])
        
        self.conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful resume analysis assistant. Use the context to answer questions accurately."),
            ("human", "{question}")
        ])
    
    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        """Add documents to the RAG system"""
        
        # Create LangChain documents
        documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]
        
        # Split documents
        split_docs = self.text_splitter.split_documents(documents, strategy='recursive')
        self.documents.extend(split_docs)
        self.metadata.extend(metadatas)
        
        # Setup retriever
        embeddings = self.embedding_manager.get_embeddings()
        if embeddings and self.documents:
            self.retriever = HybridRetriever(self.documents, embeddings)
            
            # Setup QA chain
            llm = self.llm_manager.get_model()
            if llm:
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=self.retriever.rerank_retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": self.qa_prompt}
                )
        
        # Add to LightRAG
        self.lightrag.insert_documents(texts)
        
        print(f"Added {len(split_docs)} document chunks to RAG system")
    
    def query(self, question: str, use_lightrag: bool = False) -> Dict[str, Any]:
        """Query the RAG system with enhanced LLM fallback"""
        start_time = time.time()
        
        # Extract user intents
        intents = extract_intents(question)
        
        # Retrieve relevant documents
        if self.retriever:
            docs = self.retriever.retrieve(question)
            context = "\n\n".join([doc.page_content for doc in docs[:5]])
            
            # Prepare enhanced prompt for LLM
            prompt = f"""You are an expert resume analyst. Analyze the provided resume data and answer the user's question with precision and insight.

Question: {question}

Resume Context:
{context}

Instructions:
1. Provide a comprehensive, evidence-based answer
2. For ranking questions: Use clear criteria and provide detailed reasoning
3. For skills analysis: Categorize by proficiency and provide specific examples
4. For candidate matching: Explain fit percentage with detailed reasoning
5. Always cite specific evidence from the resumes
6. Be precise and avoid generic statements

Answer:"""
            
            # Try LLM fallback approach: OpenAI -> Gemini -> HuggingFace -> Ollama
            llm_response = self.llm_manager.generate_response(prompt)
            
            if llm_response:
                # Add EDA analysis if requested
                if intents.get("want_eda"):
                    eda_results = perform_eda_analysis([doc.page_content for doc in docs])
                    eda_summary = f"\n\nCorpus Analysis:\n"
                    eda_summary += f"- Top terms: {', '.join(eda_results['top_terms'][:10])}\n"
                    eda_summary += f"- Common phrases: {', '.join([f'{a} {b}' for (a, b), _ in eda_results['top_bigrams'][:5]])}\n"
                    llm_response += eda_summary
                
                return {
                    "answer": llm_response,
                    "source_documents": docs,
                    "processing_time": time.time() - start_time,
                    "method": "enhanced_llm_fallback",
                    "intents_detected": intents
                }
        
        # Try LightRAG if available and requested
        if use_lightrag and self.lightrag.rag:
            response = self.lightrag.query(question, mode="hybrid")
            if response:
                return {
                    "answer": response,
                    "source_documents": [],
                    "processing_time": time.time() - start_time,
                    "method": "lightrag"
                }
        
        # Try LangChain QA chain
        if self.qa_chain:
            try:
                result = self.qa_chain({"query": question})
                return {
                    "answer": result["result"],
                    "source_documents": result.get("source_documents", []),
                    "processing_time": time.time() - start_time,
                    "method": "langchain"
                }
            except Exception as e:
                print(f"QA chain error: {e}")
        
        # Final fallback to simple retrieval
        if self.retriever:
            docs = self.retriever.retrieve(question)
            context = "\n\n".join([doc.page_content for doc in docs[:3]])
            
            return {
                "answer": f"Based on the available resumes:\n\n{context}",
                "source_documents": docs,
                "processing_time": time.time() - start_time,
                "method": "simple_retrieval"
            }
        
        return {
            "answer": "No documents indexed yet. Please upload resume files first.",
            "source_documents": [],
            "processing_time": time.time() - start_time,
            "method": "none"
        }
    
    def evaluate_performance(self, test_questions: List[str], ground_truth: List[str]) -> Dict[str, float]:
        """Evaluate RAG performance using RAGAS"""
        if not _HAS_RAGAS or not test_questions:
            return {}
        
        try:
            # Generate answers
            answers = []
            contexts = []
            
            for question in test_questions:
                result = self.query(question)
                answers.append(result["answer"])
                contexts.append([doc.page_content for doc in result["source_documents"][:3]])
            
            # Create evaluation dataset
            data = {
                "question": test_questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truths": ground_truth
            }
            
            dataset = pd.DataFrame(data)
            
            # Evaluate
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
            )
            
            return result
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return {}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "total_documents": len(self.documents),
            "total_metadata": len(self.metadata),
            "available_embeddings": list(self.embedding_manager.embeddings.keys()),
            "available_llms": list(self.llm_manager.models.keys()),
            "lightrag_available": _HAS_LIGHTRAG and self.lightrag.rag is not None,
            "retriever_ready": self.retriever is not None,
            "qa_chain_ready": self.qa_chain is not None
        }

# Factory function for easy initialization
def create_advanced_rag_system(index_dir: str) -> AdvancedRAGSystem:
    """Create and return an advanced RAG system"""
    return AdvancedRAGSystem(index_dir)
