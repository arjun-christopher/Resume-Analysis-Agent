from pathlib import Path
from typing import List, Dict, Any, Tuple
import hashlib
import pdfplumber
from pypdf import PdfReader
from docx import Document as DocxDocument
from PIL import Image
import pytesseract
import re

# spaCy optional (names)
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None

ALLOWED = {".pdf",".docx",".png",".jpg",".jpeg"}
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def _chunk(t: str) -> List[str]:
    t = t.replace("\r","\n")
    out = []; i=0; n=len(t)
    while i<n:
        j = min(i+CHUNK_SIZE, n)
        seg = t[i:j].strip()
        if seg: out.append(seg)
        if j>=n: break
        i = max(0, j-CHUNK_OVERLAP)
    return out

def _parse_pdf_plumber(p: Path) -> Tuple[List[str], List[int]]:
    T,P = [],[]
    try:
        with pdfplumber.open(str(p)) as pdf:
            for i,page in enumerate(pdf.pages):
                txt = page.extract_text(x_tolerance=1.5, y_tolerance=3.0) or ""
                if txt.strip():
                    for ch in _chunk(txt):
                        T.append(ch); P.append(i+1)
    except Exception:
        return [],[]
    return T,P

def _parse_pdf_fallback(p: Path) -> Tuple[List[str], List[int]]:
    try:
        rd = PdfReader(str(p))
        T,P=[],[]
        for i,pg in enumerate(rd.pages):
            txt = pg.extract_text() or ""
            if txt.strip():
                for ch in _chunk(txt):
                    T.append(ch); P.append(i+1)
        return T,P
    except Exception:
        return [],[]

def _parse_docx(p: Path) -> Tuple[List[str], List[int]]:
    try:
        d = DocxDocument(str(p))
        t = "\n".join([para.text for para in d.paragraphs])
        T = _chunk(t); P=[1]*len(T)
        return T,P
    except Exception:
        return [],[]

def _parse_image(p: Path) -> Tuple[List[str], List[int]]:
    try:
        im = Image.open(p)
        t = pytesseract.image_to_string(im, config="--psm 6")
        T = _chunk(t); P=[1]*len(T)
        return T,P
    except Exception:
        return [],[]

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(\+?\d[\d\s\-()]{7,}\d)")

def _simple_entities(text: str) -> Dict[str, List[str]]:
    emails = EMAIL.findall(text)
    phones = [m.group(0) for m in PHONE.finditer(text)]
    names = []
    if _NLP:
        try:
            doc = _NLP(text)
            names = [e.text for e in doc.ents if e.label_ == "PERSON"]
        except Exception:
            names = []
    return {"emails": list(set(emails)), "phones": list(set(phones)), "names": list(set(names))}

def _skills_from_text(text: str) -> List[str]:
    vocab = {
        "python","java","c++","c","c#","javascript","typescript","react","angular","vue",
        "node","express","django","flask","fastapi",
        "aws","azure","gcp","ec2","s3","lambda","terraform","ansible",
        "docker","kubernetes","helm","jenkins","github actions","gitlab ci","ci/cd",
        "sql","mysql","postgres","sqlite","mongodb","redis","elastic","kafka","spark","hadoop","airflow","dbt",
        "pandas","numpy","scikit-learn","xgboost","lightgbm","pytorch","tensorflow","keras","nlp","opencv",
        "microservices","rest","graphql","linux","git","tableau","power bi","snowflake","databricks",
        "android","kotlin","swift","ios"
    }
    t = text.lower()
    return sorted({s for s in vocab if s in t})

def extract_chunks_and_meta_from_paths(paths: List[Path]) -> Tuple[List[str], List[Dict[str,Any]]]:
    chunks: List[str] = []
    metas: List[Dict[str,Any]] = []

    for p in paths:
        if p.suffix.lower() not in ALLOWED: continue
        if p.suffix.lower() == ".pdf":
            T,P = _parse_pdf_plumber(p)
            if not T: T,P = _parse_pdf_fallback(p)
        elif p.suffix.lower() == ".docx":
            T,P = _parse_docx(p)
        else:
            T,P = _parse_image(p)
        if not T: continue

        fh = file_sha256(p)
        emails_all, phones_all, names_all, skills_all = set(), set(), set(), set()
        for i, t in enumerate(T):
            ent = _simple_entities(t)
            sk = _skills_from_text(t)
            emails_all |= set(ent["emails"]); phones_all |= set(ent["phones"]); names_all |= set(ent["names"]); skills_all |= set(sk)
            metas.append({
                "file": p.name,
                "path": str(p),
                "page": P[i] if i < len(P) else None,
                "file_hash": fh,
                "skills": sk
            })
            chunks.append(t)
    return chunks, metas
