# app/utils.py - Minimal utilities for the new spec (PDF/DOC/DOCX only)
from __future__ import annotations
import shutil
from pathlib import Path
from typing import List, Dict, Any, Union

SUPPORTED_EXTS = {".pdf", ".doc", ".docx"}

def human_size(n: int) -> str:
    units = ["B","KB","MB","GB","TB"]
    s = float(n); i = 0
    while s >= 1024 and i < len(units)-1:
        s /= 1024.0; i += 1
    return f"{s:.2f}{units[i]}" if i else f"{int(s)}B"

def list_supported_files(dirpath: Union[str, Path]) -> List[Path]:
    p = Path(dirpath)
    if not p.exists(): return []
    return sorted([f for f in p.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS], key=lambda x: x.name.lower())

def safe_listdir(dirpath: Union[str, Path]) -> List[Path]:
    p = Path(dirpath)
    return [f for f in p.iterdir()] if p.exists() else []

def clear_dir(path: Union[str, Path]) -> None:
    p = Path(path)
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)
