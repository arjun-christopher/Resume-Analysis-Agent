import os
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    PROJECT_NAME: str = "RAG-Resume (Streamlit)"
    FIREBASE_PROJECT_ID: str = Field(..., env="FIREBASE_PROJECT_ID")
    FIREBASE_SA_PATH: str = Field(..., env="FIREBASE_SA_PATH")
    FIREBASE_BUCKET: str = Field(..., env="FIREBASE_BUCKET")


    EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    VECTOR_PATH: str = Field(default=".faiss.index")
    META_PATH: str = Field(default=".faiss.meta.jsonl")


    SUPPORTED_EXT: tuple = (".pdf", ".docx", ".png", ".jpg", ".jpeg")


settings = Settings(_env_file=os.getenv("ENV_FILE", ".env"))