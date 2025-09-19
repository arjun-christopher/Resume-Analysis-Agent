# config.py - Configuration for Advanced RAG Resume System

import os
from pathlib import Path
from typing import Dict, Any

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"

# Model configurations
EMBEDDING_MODELS = {
    "default": "sentence-transformers/all-MiniLM-L6-v2",
    "instructor": "hkunlp/instructor-base",
    "fastembed": "sentence-transformers/all-MiniLM-L6-v2"
}

# LLM configurations
OLLAMA_MODELS = {
    "llama2": "llama2:7b",
    "mistral": "mistral:7b",
    "codellama": "codellama:7b"
}

# RAG parameters
RAG_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "retrieval_k": 10,
    "rerank_k": 5,
    "similarity_threshold": 0.3,
    "ensemble_weights": [0.7, 0.3]  # dense, sparse
}

# LangChain settings
LANGCHAIN_CONFIG = {
    "temperature": 0.1,
    "max_tokens": 2000,
    "streaming": True
}

# LightRAG settings
LIGHTRAG_CONFIG = {
    "embedding_dim": 384,
    "working_dir": str(INDEX_DIR / "lightrag")
}

def get_config() -> Dict[str, Any]:
    """Get complete configuration"""
    return {
        "directories": {
            "base": str(BASE_DIR),
            "data": str(DATA_DIR),
            "upload": str(UPLOAD_DIR),
            "index": str(INDEX_DIR)
        },
        "models": {
            "embeddings": EMBEDDING_MODELS,
            "ollama": OLLAMA_MODELS
        },
        "rag": RAG_CONFIG,
        "langchain": LANGCHAIN_CONFIG,
        "lightrag": LIGHTRAG_CONFIG,
        "api_keys": {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY")
        }
    }

def ensure_directories():
    """Ensure all required directories exist"""
    for dir_path in [DATA_DIR, UPLOAD_DIR, INDEX_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

# Initialize directories on import
ensure_directories()
