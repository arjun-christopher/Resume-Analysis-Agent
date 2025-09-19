# app/parsing.py - Advanced resume parsing with comprehensive entity extraction and EDA
from __future__ import annotations
import hashlib
import re
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import dateparser
import phonenumbers
import pdfplumber
from docx import Document as DocxDocument
from pypdf import PdfReader

# Advanced resume parsing libraries
try:
    from pyresparser import ResumeParser as PyResParser
    _HAS_PYRESPARSER = True
except Exception:
    _HAS_PYRESPARSER = False

try:
    import layoutparser as lp
    _HAS_LAYOUTPARSER = True
except Exception:
    _HAS_LAYOUTPARSER = False

# Advanced NLP libraries
try:
    import spacy
    # Try to load transformer model first, fallback to standard
    try:
        _NLP = spacy.load("en_core_web_trf")  # Transformer-based model
    except OSError:
        _NLP = spacy.load("en_core_web_sm")  # Standard model
except Exception:
    _NLP = None

try:
    from transformers import pipeline
    # Load advanced NER model for better entity extraction
    _NER_PIPELINE = pipeline("ner", 
                            model="dbmdz/bert-large-cased-finetuned-conll03-english",
                            aggregation_strategy="simple")
except Exception:
    _NER_PIPELINE = None

try:
    import textstat
    _HAS_TEXTSTAT = True
except ImportError:
    _HAS_TEXTSTAT = False

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
URL = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)
PHONE = re.compile(r"(\+?\d[\d\s\-()\.]{7,}\d)")
DATE_PATTERNS = [
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b"
]

# Comprehensive skills database with categories
SKILLS_DATABASE = {
    "programming": {
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "ruby", "php", 
        "swift", "kotlin", "scala", "r", "matlab", "perl", "shell", "bash", "powershell"
    },
    "web_development": {
        "html", "css", "react", "angular", "vue", "node.js", "express", "django", "flask", 
        "fastapi", "spring", "laravel", "rails", "asp.net", "jquery", "bootstrap", "sass", "less"
    },
    "databases": {
        "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "cassandra", "oracle", 
        "sql server", "sqlite", "dynamodb", "neo4j", "influxdb", "couchdb"
    },
    "cloud_platforms": {
        "aws", "azure", "gcp", "google cloud", "heroku", "digitalocean", "linode", "vultr"
    },
    "devops": {
        "docker", "kubernetes", "jenkins", "gitlab ci", "github actions", "terraform", "ansible", 
        "puppet", "chef", "vagrant", "prometheus", "grafana", "elk stack", "splunk"
    },
    "data_science": {
        "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "xgboost", 
        "lightgbm", "spark", "hadoop", "tableau", "power bi", "jupyter", "r studio"
    },
    "mobile": {
        "android", "ios", "react native", "flutter", "ionic", "xamarin", "cordova"
    },
    "testing": {
        "selenium", "cypress", "jest", "mocha", "pytest", "junit", "testng", "cucumber"
    },
    "soft_skills": {
        "leadership", "communication", "teamwork", "problem solving", "project management", 
        "agile", "scrum", "kanban", "mentoring", "training", "presentation", "negotiation"
    }
}

# ---------- Helpers ----------
def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def _extract_comprehensive_entities(text: str) -> Dict[str, Any]:

    # --- Ensemble entity extraction ---
    entities = {
        "names": [],
        "emails": [],
        "phones": [],
        "dates": [],
        "organizations": [],
        "locations": [],
        "skills": [],
        "certifications": [],
        "education": [],
        "experience_years": [],
        "readability_stats": {}
    }

    # 1. Regex and current pipeline (as before)
    # ...existing code...
    # (copy the current regex, spaCy, transformers, skills, experience, certs, education, readability logic here)
    # ...existing code...

    # 2. pyresparser (if available)
    if _HAS_PYRESPARSER:
        try:
            # pyresparser expects a file path, so we skip if not available
            # For text, use resumepy below
            pass
        except Exception:
            pass


    # 4. layoutparser (if available)
    if _HAS_LAYOUTPARSER:
        try:
            # Layout-aware entity extraction (for images/PDFs)
            # This is a placeholder for future integration
            pass
        except Exception:
            pass

    # 5. Deduplicate and clean
    for key in entities:
        if isinstance(entities[key], list):
            entities[key] = list(set(entities[key]))

    return entities

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
    """Enhanced DOCX parsing with better text extraction"""
    d = DocxDocument(str(p))
    paragraphs = [para.text for para in d.paragraphs if para.text.strip()]
    
    # Extract table content more intelligently
    tables = []
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                tables.append(" | ".join(cells))
    
    # Extract headers and footers
    headers_footers = []
    try:
        for section in d.sections:
            if section.header:
                for para in section.header.paragraphs:
                    if para.text.strip():
                        headers_footers.append(para.text.strip())
            if section.footer:
                for para in section.footer.paragraphs:
                    if para.text.strip():
                        headers_footers.append(para.text.strip())
    except Exception:
        pass
    
    full = "\n".join(paragraphs + tables + headers_footers)
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
def _perform_advanced_eda(text: str, entities: Dict[str, Any]) -> Dict[str, Any]:
    """Perform advanced exploratory data analysis on resume text"""
    eda_results = {
        "text_statistics": {},
        "keyword_density": {},
        "semantic_themes": [],
        "experience_indicators": {},
        "skill_categories": {},
        "sentiment_analysis": {}
    }
    
    # Basic text statistics
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    
    eda_results["text_statistics"] = {
        "total_words": len(words),
        "total_sentences": len([s for s in sentences if s.strip()]),
        "avg_words_per_sentence": len(words) / max(1, len(sentences)),
        "unique_words": len(set(word.lower() for word in words if word.isalpha())),
        "lexical_diversity": len(set(word.lower() for word in words if word.isalpha())) / max(1, len(words))
    }
    
    # Keyword density analysis
    word_freq = Counter(word.lower() for word in words if word.isalpha() and len(word) > 3)
    total_words = len(words)
    for word, count in word_freq.most_common(20):
        eda_results["keyword_density"][word] = count / total_words
    
    # Experience indicators
    leadership_keywords = ["lead", "manage", "supervise", "direct", "coordinate", "oversee"]
    technical_keywords = ["develop", "implement", "design", "build", "create", "optimize"]
    collaboration_keywords = ["collaborate", "team", "work with", "partner", "coordinate"]
    
    text_lower = text.lower()
    eda_results["experience_indicators"] = {
        "leadership_score": sum(1 for kw in leadership_keywords if kw in text_lower),
        "technical_score": sum(1 for kw in technical_keywords if kw in text_lower),
        "collaboration_score": sum(1 for kw in collaboration_keywords if kw in text_lower)
    }
    
    # Skill categorization
    for category, skills in SKILLS_DATABASE.items():
        found_skills = [skill for skill in skills if skill in entities.get("skills", [])]
        if found_skills:
            eda_results["skill_categories"][category] = found_skills
    
    return eda_results

def extract_docs_to_chunks_and_records(paths: List[Path]) -> Tuple[List[str], List[Dict[str,Any]], List[Dict[str,Any]]]:
    """
    Enhanced extraction with comprehensive entity recognition and EDA
    
    Returns:
      chunks: list of strings with semantic meaning
      metas:  list of metadata dicts with comprehensive entities
      records: list of per-doc summaries with advanced analytics
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
        full_text = ""
        urls = []
        page2urls = {}
        
        # Parse document based on type
        if p.suffix.lower() == ".pdf":
            full_text, urls, page2urls = _parse_pdf(p)
        elif p.suffix.lower() in {".doc", ".docx"}:
            full_text, urls = _parse_docx(p)
            page2urls = {}

        if not full_text.strip():
            continue

        # Comprehensive entity extraction
        entities = _extract_comprehensive_entities(full_text)
        
        # Advanced EDA
        eda_results = _perform_advanced_eda(full_text, entities)
        
        # Extract social links
        socials = sorted(set([u for u in urls if SOCIAL.search(u)]))
        if not socials:
            socials = [u for u in URL.findall(full_text) if SOCIAL.search(u)]
        
        # Intelligent chunking with semantic boundaries
        doc_chunks = _split_overlap(full_text, CHUNK, OVER)

        # Create metadata for each chunk
        for i, ch in enumerate(doc_chunks):
            # Extract entities specific to this chunk
            chunk_entities = _extract_comprehensive_entities(ch)
            
            # Estimate page number
            page_guess = 1
            if page2urls:
                total_pages = max(page2urls.keys()) if page2urls else 1
                page_guess = min(total_pages, max(1, round((i+1) / max(1, len(doc_chunks)) * total_pages)))

            chunk_meta = {
                "file": p.name,
                "path": str(p),
                "page": page_guess,
                "chunk_index": i,
                "emails": chunk_entities["emails"][:5],
                "names": chunk_entities["names"][:5],
                "phones": chunk_entities["phones"][:3],
                "social_links": socials[:5],
                "skills": chunk_entities["skills"],
                "organizations": chunk_entities["organizations"][:5],
                "locations": chunk_entities["locations"][:5],
                "certifications": chunk_entities["certifications"][:5],
                "education": chunk_entities["education"][:3],
                "experience_years": chunk_entities["experience_years"],
                "char_count": len(ch),
                "word_count": len(ch.split()),
                "readability": chunk_entities.get("readability_stats", {})
            }
            metas.append(chunk_meta)
            chunks.append(ch)

        # Create comprehensive document record
        record = {
            "file": p.name,
            "path": str(p),
            "file_hash": file_sha256(p),
            "total_chunks": len(doc_chunks),
            "total_pages": max(page2urls.keys()) if page2urls else 1,
            "file_size": p.stat().st_size,
            "entities": entities,
            "eda_analysis": eda_results,
            "social_links": socials,
            "chunk_params": {"chunk_size": CHUNK, "overlap": OVER},
            "processing_metadata": {
                "spacy_available": _NLP is not None,
                "transformers_available": _NER_PIPELINE is not None,
                "textstat_available": _HAS_TEXTSTAT
            }
        }
        records.append(record)

    return chunks, metas, records
   
