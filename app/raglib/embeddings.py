from langchain_community.embeddings import HuggingFaceEmbeddings
from .config import cfg

_embedder = None

def ensure_embedder():
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(model_name=cfg.EMBED_MODEL)
    return _embedder

def embed_model_name() -> str:
    return cfg.EMBED_MODEL
