# config.py - Advanced Configuration for Next-Generation RAG Resume System

import os
from pathlib import Path
from typing import Dict, Any
from enum import Enum

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"
CACHE_DIR = DATA_DIR / "cache"
LOGS_DIR = DATA_DIR / "logs"

# Ensure all directories exist
for dir_path in [DATA_DIR, UPLOAD_DIR, INDEX_DIR, CACHE_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Advanced Embedding Models
EMBEDDING_MODELS = {
    # State-of-the-art models
    "bge_m3": "BAAI/bge-m3",                           # Best multilingual
    "bge_large_en": "BAAI/bge-large-en-v1.5",         # Best English
    "e5_mistral": "intfloat/e5-mistral-7b-instruct",  # Instruction-tuned
    "e5_large_v2": "intfloat/e5-large-v2",            # High performance
    "nomic_embed": "nomic-ai/nomic-embed-text-v1.5",  # Fast and efficient
    "arctic_embed": "Snowflake/snowflake-arctic-embed-l", # Enterprise
    "jina_v2": "jinaai/jina-embeddings-v2-base-en",   # Long context
    "gte_large": "thenlper/gte-large",                 # General purpose
    "instructor_xl": "hkunlp/instructor-xl",           # Domain-specific
    "uae_large": "WhereIsAI/UAE-Large-V1",            # High quality
    
    # Efficient models for resource-constrained environments
    "all_minilm_l6": "sentence-transformers/all-MiniLM-L6-v2",
    "all_minilm_l12": "sentence-transformers/all-MiniLM-L12-v2",
    "default": "BAAI/bge-m3"  # Default to best overall model
}

# Latest Open-Source LLM Models
OLLAMA_MODELS = {
    # Latest advanced models
    "qwen2_5_72b": "qwen2.5:72b",          # Alibaba's most advanced
    "qwen2_5_32b": "qwen2.5:32b",          # Best balance
    "qwen2_5_14b": "qwen2.5:14b",          # Efficient option
    "llama3_1_405b": "llama3.1:405b",      # Meta's largest
    "llama3_1_70b": "llama3.1:70b",        # Production ready
    "llama3_2_90b": "llama3.2:90b",        # Latest Llama
    "phi3_5": "phi3.5:latest",             # Microsoft efficient
    "mistral_nemo": "mistral-nemo:latest", # Latest Mistral
    "gemma2_27b": "gemma2:27b",            # Google improved
    "yi_34b": "yi:34b",                    # 01.AI powerful
    "deepseek_coder": "deepseek-coder:33b", # Coding specialist
    "deepseek_coder_latest": "deepseek-coder:latest", # Latest DeepSeek Coder
    "deepseek_coder_1_3b": "deepseek-coder:1.3b", # Lightweight DeepSeek
    "codellama_70b": "codellama:70b",      # Code generation
    "mistral_latest": "mistral:latest",    # Latest Mistral
    "llama3_2_latest": "llama3.2:latest", # Latest Llama 3.2
    
    # Legacy models for compatibility
    "llama2_7b": "llama2:7b",
    "mistral_7b": "mistral:7b",
    "default": "llama2:7b"  # Default to user preference
}

# Retrieval Strategies
class RetrievalStrategy(Enum):
    VECTOR_ONLY = "vector"
    HYBRID_BM25 = "hybrid_bm25"
    COLBERT = "colbert"
    GRAPH_RAG = "graph_rag"
    RAPTOR = "raptor"
    MULTI_QUERY = "multi_query"
    RAG_FUSION = "rag_fusion"
    ADAPTIVE = "adaptive"

# Vector Database Options
class VectorDB(Enum):
    QDRANT = "qdrant"
    WEAVIATE = "weaviate"
    PINECONE = "pinecone"
    CHROMA = "chroma"
    FAISS = "faiss"

# Advanced RAG Configuration
RAG_CONFIG = {
    # Core parameters
    "chunk_size": int(os.getenv("CHUNK_SIZE", 512)),
    "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", 50)),
    "retrieval_k": int(os.getenv("TOP_K", 10)),
    "rerank_k": int(os.getenv("RERANK_TOP_K", 5)),
    "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", 0.7)),
    
    # Advanced features
    "retrieval_strategy": os.getenv("RETRIEVAL_STRATEGY", "adaptive"),
    "vector_db": os.getenv("VECTOR_DB", "qdrant"),
    "enable_graph_rag": os.getenv("ENABLE_GRAPH_RAG", "true").lower() == "true",
    "enable_multimodal": os.getenv("ENABLE_MULTIMODAL", "true").lower() == "true",
    "enable_self_correction": os.getenv("ENABLE_SELF_CORRECTION", "true").lower() == "true",
    "enable_feedback_learning": os.getenv("ENABLE_FEEDBACK_LEARNING", "true").lower() == "true",
    
    # Performance settings
    "use_async": os.getenv("USE_ASYNC", "true").lower() == "true",
    "batch_size": int(os.getenv("BATCH_SIZE", 32)),
    "max_concurrent_requests": int(os.getenv("MAX_CONCURRENT_REQUESTS", 10)),
    
    # Ensemble weights for hybrid retrieval
    "ensemble_weights": [0.7, 0.3],  # dense, sparse
    
    # RAPTOR configuration
    "raptor_cluster_size": int(os.getenv("RAPTOR_CLUSTER_SIZE", 5)),
    "raptor_max_levels": int(os.getenv("RAPTOR_MAX_LEVELS", 3)),
    
    # RAG Fusion settings
    "rag_fusion_queries": int(os.getenv("RAG_FUSION_QUERIES", 3)),
}

# LLM Generation Settings
LLM_CONFIG = {
    "temperature": float(os.getenv("TEMPERATURE", 0.1)),
    "max_tokens": int(os.getenv("MAX_TOKENS", 4096)),
    "streaming": True,
    "top_p": float(os.getenv("TOP_P", 0.9)),
    "frequency_penalty": float(os.getenv("FREQUENCY_PENALTY", 0.0)),
    "presence_penalty": float(os.getenv("PRESENCE_PENALTY", 0.0)),
}

# Model Selection and Fallback
MODEL_CONFIG = {
    "default_embedding": os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
    "instructor_model": os.getenv("INSTRUCTOR_MODEL", "hkunlp/instructor-xl"),
    "fastembed_model": os.getenv("FASTEMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5"),
    "default_llm": os.getenv("DEFAULT_LLM_MODEL", "llama2:7b"),
    "fallback_models": os.getenv("FALLBACK_MODELS", "mistral:latest,llama3.2:latest,deepseek-coder:latest,deepseek-coder:1.3b").split(","),
}

# Vector Database Configurations
VECTOR_DB_CONFIG = {
    "qdrant": {
        "url": os.getenv("QDRANT_URL", "http://localhost:6333"),
        "api_key": os.getenv("QDRANT_API_KEY"),
        "collection_name": os.getenv("QDRANT_COLLECTION_NAME", "resume_documents"),
        "distance": "cosine",
        "hnsw_ef": int(os.getenv("QDRANT_HNSW_EF", 128)),
        "hnsw_m": int(os.getenv("QDRANT_HNSW_M", 16)),
        "quantization": os.getenv("QDRANT_QUANTIZATION", "false").lower() == "true",
    },
    "weaviate": {
        "url": os.getenv("WEAVIATE_URL", "http://localhost:8080"),
        "api_key": os.getenv("WEAVIATE_API_KEY"),
        "class_name": os.getenv("WEAVIATE_CLASS_NAME", "ResumeDocument"),
        "vector_index_type": os.getenv("WEAVIATE_VECTOR_INDEX_TYPE", "hnsw"),
        "ef_construction": int(os.getenv("WEAVIATE_EF_CONSTRUCTION", 128)),
    },
    "pinecone": {
        "api_key": os.getenv("PINECONE_API_KEY"),
        "environment": os.getenv("PINECONE_ENVIRONMENT"),
        "index_name": os.getenv("PINECONE_INDEX_NAME", "resume-index"),
    },
    "chroma": {
        "persist_directory": str(INDEX_DIR / "chroma"),
        "collection_name": "resume_documents",
    },
    "faiss": {
        "index_path": str(INDEX_DIR / "faiss_index"),
        "index_type": "IVF",
    }
}

# Graph Database Configuration (Neo4j)
GRAPH_DB_CONFIG = {
    "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "user": os.getenv("NEO4J_USER", "neo4j"),
    "password": os.getenv("NEO4J_PASSWORD", "password"),
    "database": os.getenv("NEO4J_DATABASE", "neo4j"),
}

# Multi-modal Processing Configuration
MULTIMODAL_CONFIG = {
    "ocr_language": os.getenv("OCR_LANGUAGE", "en"),
    "tesseract_path": os.getenv("TESSERACT_PATH", "/usr/bin/tesseract"),
    "vision_model": os.getenv("VISION_MODEL", "Salesforce/blip-image-captioning-base"),
    "llava_model": os.getenv("LLAVA_MODEL", "llava-hf/llava-1.5-7b-hf"),
    "enable_ocr": os.getenv("ENABLE_OCR", "true").lower() == "true",
    "enable_vision": os.getenv("ENABLE_VISION", "true").lower() == "true",
}

# Performance and Optimization Settings
PERFORMANCE_CONFIG = {
    "use_gpu": os.getenv("USE_GPU", "true").lower() == "true",
    "gpu_memory_fraction": float(os.getenv("GPU_MEMORY_FRACTION", 0.8)),
    "enable_quantization": os.getenv("ENABLE_QUANTIZATION", "false").lower() == "true",
    "quantization_bits": int(os.getenv("QUANTIZATION_BITS", 8)),
    "embedding_cache_size": int(os.getenv("EMBEDDING_CACHE_SIZE", 10000)),
    "llm_cache_size": int(os.getenv("LLM_CACHE_SIZE", 1000)),
    "enable_caching": os.getenv("ENABLE_CACHING", "true").lower() == "true",
}

# Evaluation and Monitoring
EVALUATION_CONFIG = {
    "enable_evaluation": os.getenv("ENABLE_EVALUATION", "true").lower() == "true",
    "evaluation_metrics": os.getenv("EVALUATION_METRICS", "faithfulness,answer_relevancy,context_precision,context_recall").split(","),
    "log_level": os.getenv("LOG_LEVEL", "INFO"),
    "enable_detailed_logging": os.getenv("ENABLE_DETAILED_LOGGING", "false").lower() == "true",
    "metrics_collection": os.getenv("METRICS_COLLECTION", "true").lower() == "true",
}

# Security and Privacy Settings
SECURITY_CONFIG = {
    "retain_uploaded_files": os.getenv("RETAIN_UPLOADED_FILES", "true").lower() == "true",
    "auto_delete_after_days": int(os.getenv("AUTO_DELETE_AFTER_DAYS", 30)),
    "anonymize_personal_info": os.getenv("ANONYMIZE_PERSONAL_INFO", "false").lower() == "true",
    "mask_email_addresses": os.getenv("MASK_EMAIL_ADDRESSES", "false").lower() == "true",
    "mask_phone_numbers": os.getenv("MASK_PHONE_NUMBERS", "false").lower() == "true",
}

# Streamlit UI Configuration
UI_CONFIG = {
    "theme": os.getenv("STREAMLIT_THEME", "light"),
    "max_upload_size_mb": int(os.getenv("MAX_UPLOAD_SIZE_MB", 50)),
    "show_system_stats": os.getenv("SHOW_SYSTEM_STATS", "true").lower() == "true",
    "enable_feedback_ui": os.getenv("ENABLE_FEEDBACK_UI", "true").lower() == "true",
    "session_timeout_minutes": int(os.getenv("SESSION_TIMEOUT_MINUTES", 60)),
    "max_chat_history": int(os.getenv("MAX_CHAT_HISTORY", 100)),
}

# API Keys Configuration
API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "gemini": os.getenv("GEMINI_API_KEY"),
    "groq": os.getenv("GROQ_API_KEY"),
    "huggingface": os.getenv("HUGGINGFACE_API_KEY"),
}

# LangChain Configuration (for backward compatibility)
LANGCHAIN_CONFIG = {
    "temperature": LLM_CONFIG["temperature"],
    "max_tokens": LLM_CONFIG["max_tokens"],
    "streaming": LLM_CONFIG["streaming"],
    "callbacks": [],
}

# LightRAG Configuration
LIGHTRAG_CONFIG = {
    "embedding_dim": 384,  # Will be updated based on selected model
    "working_dir": str(INDEX_DIR / "lightrag"),
    "chunk_token_size": RAG_CONFIG["chunk_size"],
    "chunk_overlap_token_size": RAG_CONFIG["chunk_overlap"],
    "tiktoken_model_name": "gpt-4o-mini",
    "entity_extract_max_gleaning": 1,
    "entity_summary_to_max_tokens": 500,
    "node_group_sort_by": "degree",
    "node_top_k": 30,
    "node_max_consideration": 300,
    "edge_top_k": 30,
    "edge_max_consideration": 300,
}

def get_config() -> Dict[str, Any]:
    """Get complete configuration with all settings"""
    return {
        "directories": {
            "base": str(BASE_DIR),
            "data": str(DATA_DIR),
            "upload": str(UPLOAD_DIR),
            "index": str(INDEX_DIR),
            "cache": str(CACHE_DIR),
            "logs": str(LOGS_DIR),
        },
        "models": {
            "embeddings": EMBEDDING_MODELS,
            "ollama": OLLAMA_MODELS,
            "config": MODEL_CONFIG,
        },
        "rag": RAG_CONFIG,
        "llm": LLM_CONFIG,
        "langchain": LANGCHAIN_CONFIG,
        "lightrag": LIGHTRAG_CONFIG,
        "vector_db": VECTOR_DB_CONFIG,
        "graph_db": GRAPH_DB_CONFIG,
        "multimodal": MULTIMODAL_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "evaluation": EVALUATION_CONFIG,
        "security": SECURITY_CONFIG,
        "ui": UI_CONFIG,
        "api_keys": API_KEYS,
    }

def get_model_config(model_type: str = "embedding") -> Dict[str, Any]:
    """Get specific model configuration"""
    if model_type == "embedding":
        return {
            "default": MODEL_CONFIG["default_embedding"],
            "instructor": MODEL_CONFIG["instructor_model"],
            "fastembed": MODEL_CONFIG["fastembed_model"],
            "available": EMBEDDING_MODELS,
        }
    elif model_type == "llm":
        return {
            "default": MODEL_CONFIG["default_llm"],
            "fallback": MODEL_CONFIG["fallback_models"],
            "available": OLLAMA_MODELS,
        }
    else:
        return {}

def get_vector_db_config(db_type: str = None) -> Dict[str, Any]:
    """Get vector database configuration"""
    if db_type is None:
        db_type = RAG_CONFIG["vector_db"]
    
    return VECTOR_DB_CONFIG.get(db_type, VECTOR_DB_CONFIG["faiss"])

def update_config_from_env():
    """Update configuration from environment variables"""
    # This function can be called to refresh config from environment
    # Useful for runtime configuration updates
    pass

def ensure_directories():
    """Ensure all required directories exist"""
    for dir_path in [DATA_DIR, UPLOAD_DIR, INDEX_DIR, CACHE_DIR, LOGS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

# Initialize directories on import
ensure_directories()
