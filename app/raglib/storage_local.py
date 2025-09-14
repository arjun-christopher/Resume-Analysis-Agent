import os, shutil
from pathlib import Path
from typing import List
from .config import cfg, session_paths

def save_file_bytes(session_id: str, name: str, data: bytes) -> Path:
    p = session_paths(session_id)
    full = p.uploads_dir / name
    full.parent.mkdir(parents=True, exist_ok=True)
    with open(full, "wb") as f:
        f.write(data)
    return full

def list_supported_files(session_id: str) -> List[Path]:
    p = session_paths(session_id)
    out = []
    for root, _, files in os.walk(p.uploads_dir):
        for fn in files:
            ln = fn.lower()
            if ln.endswith((".pdf",".docx",".png",".jpg",".jpeg")):
                out.append(Path(root)/fn)
    return sorted(out)

def clear_session_all(session_id: str) -> int:
    p = session_paths(session_id)
    count = 0
    if p.root.exists():
        for root, _, files in os.walk(p.root):
            count += len(files)
        shutil.rmtree(p.root, ignore_errors=True)
    return count
