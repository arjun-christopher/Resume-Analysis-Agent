import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class AppConfig:
    APP_NAME: str = "RAG-Resume (Local Chat)"
    SUPPORTED_EXT = (".pdf", ".docx", ".png", ".jpg", ".jpeg")
    DATA_ROOT: Path = Path(os.environ.get("RAG_DATA_ROOT", "./data")).resolve()
    EMBED_MODEL: str = os.environ.get("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

cfg = AppConfig()

@dataclass
class SessionPaths:
    root: Path
    uploads_dir: Path
    artifacts_dir: Path
    vectors_dir: Path
    candidates_json: Path

def session_paths(session_id: str) -> SessionPaths:
    root = cfg.DATA_ROOT / "sessions" / session_id
    return SessionPaths(
        root=root,
        uploads_dir=root/"uploads",
        artifacts_dir=root/"artifacts",
        vectors_dir=root/"vectors",
        candidates_json=root/"artifacts"/"candidates.json"
    )

def ensure_session_dirs(session_id: str):
    p = session_paths(session_id)
    p.uploads_dir.mkdir(parents=True, exist_ok=True)
    p.artifacts_dir.mkdir(parents=True, exist_ok=True)
    p.vectors_dir.mkdir(parents=True, exist_ok=True)
