# app/parsing.py - PDF & DOC/DOCX parsing with names + social links + adaptive chunking
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple
import hashlib, re
import pdfplumber
from pypdf import PdfReader
from docx import Document as DocxDocument

# Optional spaCy for more accurate PERSON extraction
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None

# ---------- Config ----------
ALLOWED = {".pdf", ".doc", ".docx"}
MAX_FILES_PER_SESSION = 50

# Social link patterns (extendable)
SOCIAL_PATTERNS = [
    r"https?://(www\.)?linkedin\.com/[A-Za-z0-9_/\-?=%.]+",
    r"https?://(www\.)?github\.com/[A-Za-z0-9_\-./]+",
    r"https?://(www\.)?(x|twitter)\.com/[A-Za-z0-9_\-./]+",
    r"https?://(www\.)?kaggle\.com/[A-Za-z0-9_\-./]+",
    r"https?://(www\.)?medium\.com/[A-Za-z0-9_\-./]+",
    r"https?://(www\.)?personal\.[A-Za-z0-9_\-./]+",
    r"https?://[A-Za-z0-9_\-./]+\.me(/[A-Za-z0-9_\-./]+)?",
    r"mailto:[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
]
SOCIAL = re.compile("|".join(SOCIAL_PATTERNS), re.IGNORECASE)

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL   = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)

# ---------- Helpers ----------
def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def _extract_person_names(text: str) -> List[str]:
    names = []
    if _NLP:
        try:
            doc = _NLP(text)
            names = [e.text for e in doc.ents if e.label_ == "PERSON"]
        except Exception:
            names = []
    else:
        # Heuristic: capitalized 2- or 3-word sequences
        pat = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")
        names = pat.findall(text)
    # De-duplicate & filter short tokens
    uniq = []
    seen = set()
    for n in names:
        k = n.strip()
        if len(k.split()) >= 2:
            key = k.lower()
            if key not in seen:
                seen.add(key); uniq.append(k)
    return uniq

def _extract_docx_hyperlinks(doc: DocxDocument) -> List[str]:
    # Collect explicit hyperlink targets from DOCX rels
    urls = []
    rels = getattr(doc.part, "rels", {})
    for r in rels.values():
        try:
            if "hyperlink" in r.reltype and r.target_ref:
                urls.append(str(r.target_ref))
        except Exception:
            continue
    return urls

def _parse_pdf(p: Path) -> Tuple[str, List[str], Dict[int, List[str]]]:
    """Return full_text, url_list, page2urls"""
    text_parts = []
    page2urls: Dict[int, List[str]] = {}
    urls = []

    # Primary: pdfplumber (layout aware)
    try:
        with pdfplumber.open(str(p)) as pdf:
            for i, page in enumerate(pdf.pages):
                t = page.extract_text() or ""
                if t.strip():
                    text_parts.append(t)
                    # Extract URLs from text
                    page_urls = URL.findall(t)
                    # Try PDF annotations (link rectangles)
                    try:
                        raw_annots = page.annotations or []
                        for a in raw_annots:
                            uri = a.get("uri") or a.get("A", {}).get("URI")
                            if uri:
                                page_urls.append(uri)
                    except Exception:
                        pass
                    if page_urls:
                        page2urls[i+1] = list(sorted(set(page_urls)))
                        urls.extend(page_urls)
    except Exception:
        # Fallback: PyPDF
        rd = PdfReader(str(p))
        for i, pg in enumerate(rd.pages):
            t = pg.extract_text() or ""
            if t.strip():
                text_parts.append(t)
                page_urls = URL.findall(t)
                if page_urls:
                    page2urls[i+1] = list(sorted(set(page_urls)))
                    urls.extend(page_urls)

    return "\n".join(text_parts), sorted(set(urls)), page2urls

def _parse_docx(p: Path) -> Tuple[str, List[str]]:
    d = DocxDocument(str(p))
    paragraphs = [para.text for para in d.paragraphs if para.text.strip()]
    tables = []
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells): tables.append(" | ".join(cells))
    full = "\n".join(paragraphs + tables)
    urls = sorted(set(URL.findall(full) + _extract_docx_hyperlinks(d)))
    return full, urls

# ---------- Adaptive EDA-informed chunking ----------
def _adaptive_params(total_bytes: int, file_count: int) -> Dict[str, int]:
    """
    Scale chunk size & overlap with corpus size.
    Small corpora => smaller chunks (finer recall)
    Larger corpora => bigger chunks (better throughput)
    """
    mb = max(1, total_bytes // (1024*1024))
    # Base values
    base = 700
    # Scale up with size but cap
    chunk = min(1600, base + int(80 * (mb**0.5)) + file_count*20)
    overlap = max(120, int(chunk * 0.18))
    return {"chunk": chunk, "overlap": overlap}

def _split_overlap(text: str, chunk: int, overlap: int) -> List[str]:
    text = text.replace("\r", "\n")
    out = []; i = 0; n = len(text)
    while i < n:
        j = min(i + chunk, n)
        seg = text[i:j].strip()
        if seg: out.append(seg)
        if j >= n: break
        i = max(0, j - overlap)
    return out

# ---------- Skills and entities ----------
EXTENDED_SKILLS = {
    # Common tech + generic skills (trimmed for brevity; extend as you wish)
    "python","java","c","c++","javascript","typescript","go","rust","sql",
    "html","css","react","angular","vue","node","express","flask","django",
    "pandas","numpy","scikit-learn","xgboost","pytorch","tensorflow","nlp","ml",
    "aws","azure","gcp","docker","kubernetes","terraform","git","linux",
    "power bi","tableau","excel","project management","leadership"
}

def _skills_from_text(text: str) -> List[str]:
    t = text.lower()
    found = []
    for s in EXTENDED_SKILLS:
        pat = r"\b" + re.escape(s).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pat, t): found.append(s)
    return sorted(set(found))

# ---------- Public API ----------
def extract_docs_to_chunks_and_records(paths: List[Path]) -> Tuple[List[str], List[Dict[str,Any]], List[Dict[str,Any]]]:
    """
    Returns:
      chunks: list of strings
      metas:  list of metadata dicts (file, page, chunk_index, names, emails, social_links, skills)
      records: list of per-doc summaries
    """
    paths = [p for p in paths if p.suffix.lower() in ALLOWED]
    if not paths:
        return [], [], []
    total_bytes = sum(p.stat().st_size for p in paths if p.exists())
    params = _adaptive_params(total_bytes, len(paths))
    CHUNK, OVER = params["chunk"], params["overlap"]

    chunks: List[str] = []
    metas: List[Dict[str,Any]] = []
    records: List[Dict[str,Any]] = []

    for p in paths:
        full_text, urls, page2urls = ("","",{})
        if p.suffix.lower() == ".pdf":
            full_text, urls, page2urls = _parse_pdf(p)
        elif p.suffix.lower() in {".doc",".docx"}:
            full_text, urls = _parse_docx(p)
            page2urls = {}

        if not full_text.strip(): continue

        # Entities
        emails = sorted(set(EMAIL.findall(full_text)))
        socials = sorted(set([u for u in urls if SOCIAL.search(u)]) or [u for u in URL.findall(full_text) if SOCIAL.search(u)])
        names  = _extract_person_names(full_text)
        skills = _skills_from_text(full_text)

        # Chunking
        doc_chunks = _split_overlap(full_text, CHUNK, OVER)

        # Meta + chunks
        for i, ch in enumerate(doc_chunks):
            # Guess page: map by simple proportion if PDF; otherwise 1
            page_guess = 1
            if page2urls:
                # crude page estimate by position in list
                total_pages = max(page2urls.keys()) if page2urls else 1
                page_guess = min(total_pages, max(1, round((i+1) / max(1, len(doc_chunks)) * total_pages)))

            metas.append({
                "file": p.name,
                "path": str(p),
                "page": page_guess,
                "chunk_index": i,
                "emails": emails[:10],
                "names": names[:10],
                "social_links": socials[:10],
                "skills": skills,
                "char_count": len(ch),
                "word_count": len(ch.split())
            })
            chunks.append(ch)

        records.append({
            "file": p.name,
            "path": str(p),
            "file_hash": file_sha256(p),
            "total_chunks": len(doc_chunks),
            "file_size": p.stat().st_size,
            "names": names,
            "emails": emails,
            "social_links": socials,
            "skills": skills,
            "chunk_params": {"chunk_size": CHUNK, "overlap": OVER}
        })

    return chunks, metas, records
   
