from sentence_transformers import SentenceTransformer
from .config import settings


_model = SentenceTransformer(settings.EMBEDDING_MODEL)


def embed(texts):
    return _model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)