import os
import shutil
import tempfile
from pathlib import Path
from typing import List

def init_session_paths(force: bool = False):
    base = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "rag_resume"
    base.mkdir(parents=True, exist_ok=True)

    import uuid
    sid = uuid.uuid4().hex[:12]
    session_root = base / sid
    session_root.mkdir(parents=True, exist_ok=True)

    paths = {
        "session_root": str(session_root),
        "uploads": str(session_root / "uploads"),
        "vector": str(session_root / "vector"),
    }
    for k, v in paths.items():
        if k != "session_root":
            Path(v).mkdir(parents=True, exist_ok=True)

    import streamlit as st
    st.session_state["paths"] = paths

def max_upload_guard(total: int, max_allowed: int = 50):
    import streamlit as st
    if total > max_allowed:
        st.error(f"Maximum of {max_allowed} files per session. You tried {total}.")
        st.stop()

def human_size(n: int) -> str:
    s = float(n)
    for unit in ["B","KB","MB","GB","TB","PB"]:
        if s < 1024.0:
            return f"{s:.1f}{unit}"
        s /= 1024.0
    return f"{s:.1f}EB"

def list_supported_files(uploads_dir: str) -> List[Path]:
    p = Path(uploads_dir)
    exts = {".pdf",".docx",".png",".jpg",".jpeg"}
    return [x for x in p.iterdir() if x.is_file() and x.suffix.lower() in exts]

def clear_dir(path: str):
    p = Path(path)
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
