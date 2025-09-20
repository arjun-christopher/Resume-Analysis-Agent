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

def _extract_names_from_resume(text: str) -> List[str]:
    """Advanced name extraction using multiple strategies"""
    names = []
    lines = text.split('\n')
    
    # Strategy 1: First few lines often contain names
    for i, line in enumerate(lines[:10]):
        line = line.strip()
        if not line or len(line) < 3:
            continue
            
        # Skip lines with common resume headers
        skip_patterns = [
            r'resume|curriculum|vitae|cv|profile|summary|objective|contact|phone|email|address',
            r'skills|experience|education|work|employment|projects|achievements|awards',
            r'linkedin|github|portfolio|website|references|languages|certifications'
        ]
        
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in skip_patterns):
            continue
            
        # Look for name patterns
        name_patterns = [
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]*\.?\s*)*[A-Z][a-z]+)$',  # John Smith, John A. Smith
            r'^([A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+)$',  # John A. Smith
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})$',  # John Smith Brown
        ]
        
        for pattern in name_patterns:
            match = re.match(pattern, line)
            if match:
                candidate_name = match.group(1).strip()
                # Validate it's likely a name (not too long, reasonable words)
                words = candidate_name.split()
                if 2 <= len(words) <= 4 and all(len(w) >= 2 for w in words):
                    names.append(candidate_name)
    
    # Strategy 2: Look for "Name:" or similar patterns
    name_label_patterns = [
        r'(?:name|full\s*name|candidate):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]*\.?\s*)*[A-Z][a-z]+)',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]*\.?\s*)*[A-Z][a-z]+)\s*(?:\||\-)\s*(?:resume|cv|profile)',
    ]
    
    for pattern in name_label_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        names.extend(matches)
    
    # Strategy 3: Extract from email addresses (common pattern: firstname.lastname@domain)
    emails = EMAIL.findall(text)
    for email in emails:
        username = email.split('@')[0]
        if '.' in username:
            parts = username.split('.')
            if len(parts) == 2 and all(part.isalpha() and len(part) >= 2 for part in parts):
                name = f"{parts[0].title()} {parts[1].title()}"
                names.append(name)
    
    return list(set(names))

def _extract_social_links_comprehensive(text: str) -> Dict[str, List[str]]:
    """Comprehensive social links extraction with multiple detection methods"""
    social_data = {
        "all_links": [],
        "linkedin": [],
        "github": [],
        "portfolio": [],
        "twitter": [],
        "kaggle": [],
        "medium": []
    }
    
    # Enhanced social patterns with hyperlink detection
    enhanced_patterns = {
        "linkedin": [
            r'https?://(www\.)?linkedin\.com/in/[A-Za-z0-9_\-]+/?',
            r'linkedin\.com/in/[A-Za-z0-9_\-]+',
            r'(?:linkedin|LinkedIn):\s*([A-Za-z0-9_\-]+)',
            r'(?:linkedin|LinkedIn):\s*(https?://[^\s]+)',
        ],
        "github": [
            r'https?://(www\.)?github\.com/[A-Za-z0-9_\-]+/?',
            r'github\.com/[A-Za-z0-9_\-]+',
            r'(?:github|GitHub):\s*([A-Za-z0-9_\-]+)',
            r'(?:github|GitHub):\s*(https?://[^\s]+)',
        ],
        "portfolio": [
            r'https?://[A-Za-z0-9_\-]+\.(?:com|net|org|io|dev|me|portfolio)/?',
            r'(?:portfolio|website|personal\s*site):\s*(https?://[^\s]+)',
            r'(?:portfolio|website):\s*([A-Za-z0-9_\-]+\.(?:com|net|org|io|dev|me))',
        ],
        "twitter": [
            r'https?://(www\.)?(twitter|x)\.com/[A-Za-z0-9_\-]+/?',
            r'(?:twitter|x)\.com/[A-Za-z0-9_\-]+',
            r'(?:twitter|x):\s*@?([A-Za-z0-9_\-]+)',
        ],
        "kaggle": [
            r'https?://(www\.)?kaggle\.com/[A-Za-z0-9_\-]+/?',
            r'kaggle\.com/[A-Za-z0-9_\-]+',
        ],
        "medium": [
            r'https?://(www\.)?medium\.com/@?[A-Za-z0-9_\-]+/?',
            r'medium\.com/@?[A-Za-z0-9_\-]+',
        ]
    }
    
    # Extract using patterns
    for platform, patterns in enhanced_patterns.items():
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[-1]  # Take the last group if tuple
                if match and len(match) > 3:
                    # Normalize URLs
                    if not match.startswith('http'):
                        if platform == 'linkedin':
                            match = f"https://linkedin.com/in/{match}"
                        elif platform == 'github':
                            match = f"https://github.com/{match}"
                        elif platform == 'twitter':
                            match = f"https://twitter.com/{match}"
                        elif platform == 'kaggle':
                            match = f"https://kaggle.com/{match}"
                        elif platform == 'medium':
                            match = f"https://medium.com/@{match}"
                    
                    social_data[platform].append(match)
                    social_data["all_links"].append(match)
    
    # Look for hyperlinked text patterns (common in PDFs/Word docs)
    hyperlink_patterns = [
        r'(?:LinkedIn|LINKEDIN)\s*(?:\||\-|\:)?\s*([^\s\n]+)',
        r'(?:GitHub|GITHUB)\s*(?:\||\-|\:)?\s*([^\s\n]+)',
        r'(?:Portfolio|PORTFOLIO|Website|WEBSITE)\s*(?:\||\-|\:)?\s*([^\s\n]+)',
    ]
    
    for pattern in hyperlink_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if '.' in match or 'http' in match:
                social_data["all_links"].append(match)
    
    # Deduplicate
    for key in social_data:
        social_data[key] = list(set(social_data[key]))
    
    return social_data

def _extract_skills_with_context(text: str) -> List[str]:
    """Enhanced skills extraction with context awareness"""
    skills_found = []
    text_lower = text.lower()
    
    # Create context-aware skill patterns
    skill_contexts = {
        "technical_skills": ["technical skills", "programming languages", "technologies", "tools"],
        "professional_skills": ["professional experience", "work experience", "employment"],
        "project_skills": ["projects", "personal projects", "side projects"],
        "academic_skills": ["coursework", "education", "academic", "university"],
        "internship_skills": ["internship", "intern", "training", "apprenticeship"],
        "certification_skills": ["certified", "certification", "certificate", "licensed"]
    }
    
    # Extract skills by category to maintain context
    for category, contexts in skill_contexts.items():
        for context in contexts:
            # Find sections with this context
            context_pattern = rf'{re.escape(context)}.*?(?=\n\n|\n[A-Z][A-Z\s]*:|\Z)'
            context_matches = re.findall(context_pattern, text, re.IGNORECASE | re.DOTALL)
            
            for context_text in context_matches:
                # Extract skills from this context
                for skill_category, skills in SKILLS_DATABASE.items():
                    for skill in skills:
                        # Use word boundaries to avoid partial matches
                        skill_pattern = rf'\b{re.escape(skill)}\b'
                        if re.search(skill_pattern, context_text, re.IGNORECASE):
                            # Add context information to skill
                            contextual_skill = f"{skill} ({category})"
                            skills_found.append(contextual_skill)
                            skills_found.append(skill)  # Also add base skill
    
    # Also extract skills without context (fallback)
    for skill_category, skills in SKILLS_DATABASE.items():
        for skill in skills:
            skill_pattern = rf'\b{re.escape(skill)}\b'
            if re.search(skill_pattern, text_lower):
                skills_found.append(skill)
    
    return list(set(skills_found))

def _calculate_dynamic_chunk_size(total_file_size: int, num_files: int) -> int:
    """Calculate optimal chunk size based on total file size and number of files"""
    
    # Base chunk sizes for different file size ranges
    if total_file_size < 50 * 1024:  # < 50KB
        base_chunk = 128
    elif total_file_size < 200 * 1024:  # < 200KB
        base_chunk = 256
    elif total_file_size < 1024 * 1024:  # < 1MB
        base_chunk = 512
    elif total_file_size < 5 * 1024 * 1024:  # < 5MB
        base_chunk = 1024
    else:  # > 5MB
        base_chunk = 2048
    
    # Adjust based on number of files
    if num_files > 20:
        base_chunk = min(base_chunk, 512)  # Smaller chunks for many files
    elif num_files > 10:
        base_chunk = min(base_chunk, 1024)
    
    # Ensure reasonable bounds
    return max(128, min(base_chunk, 2048))

def _extract_comprehensive_entities(text: str) -> Dict[str, Any]:
    """Enhanced entity extraction with advanced name detection and social links"""
    
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
        "social_links": [],
        "linkedin": [],
        "github": [],
        "portfolio": [],
        "readability_stats": {}
    }

    # 1. Enhanced Name Extraction
    names = _extract_names_from_resume(text)
    entities["names"].extend(names)

    # 2. Advanced Social Links Detection
    social_data = _extract_social_links_comprehensive(text)
    entities["social_links"].extend(social_data["all_links"])
    entities["linkedin"].extend(social_data["linkedin"])
    entities["github"].extend(social_data["github"])
    entities["portfolio"].extend(social_data["portfolio"])

    # 3. Enhanced Skills with Context
    skills_data = _extract_skills_with_context(text)
    entities["skills"].extend(skills_data)

    # 4. Basic regex patterns (existing logic)
    entities["emails"].extend(EMAIL.findall(text))
    entities["phones"].extend(PHONE.findall(text))
    
    # Extract dates
    for pattern in DATE_PATTERNS:
        entities["dates"].extend(re.findall(pattern, text, re.IGNORECASE))

    # 5. Advanced NLP processing
    if _NLP:
        doc = _NLP(text[:1000000])  # Limit text size for performance
        
        # Extract person names using spaCy
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities["names"].append(ent.text.strip())
            elif ent.label_ == "ORG":
                entities["organizations"].append(ent.text.strip())
            elif ent.label_ in ["GPE", "LOC"]:
                entities["locations"].append(ent.text.strip())

    # 6. Transformer-based NER (if available)
    if _NER_PIPELINE:
        try:
            # Process in chunks to avoid memory issues
            chunk_size = 500
            text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            
            for chunk in text_chunks[:5]:  # Limit to first 5 chunks for performance
                ner_results = _NER_PIPELINE(chunk)
                for entity in ner_results:
                    if entity['entity_group'] == 'PER':
                        entities["names"].append(entity['word'].strip())
                    elif entity['entity_group'] == 'ORG':
                        entities["organizations"].append(entity['word'].strip())
        except Exception:
            pass

    # 7. Extract experience years
    experience_patterns = [
        r"(\d+)[\s]*(?:\+|\-)?[\s]*(?:years?|yrs?|year)[\s]*(?:of[\s]*)?(?:experience|exp)",
        r"(?:experience|exp)[\s]*(?:of[\s]*)?(\d+)[\s]*(?:\+|\-)?[\s]*(?:years?|yrs?|year)",
        r"(\d+)[\s]*(?:\+|\-)?[\s]*(?:years?|yrs?)[\s]*(?:in|with|of)",
    ]
    
    for pattern in experience_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities["experience_years"].extend([int(m) for m in matches if m.isdigit()])

    # 8. Extract certifications
    cert_patterns = [
        r"(?:certified|certification|certificate)[\s]+([A-Z][A-Za-z\s\-]{2,30})",
        r"([A-Z]{2,10})[\s]*(?:certified|certification|certificate)",
        r"(?:AWS|Azure|Google|Microsoft|Oracle|Cisco|CompTIA)[\s]+([A-Za-z\s\-]{2,30})",
    ]
    
    for pattern in cert_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities["certifications"].extend([m.strip() for m in matches])

    # 9. Extract education
    education_patterns = [
        r"(?:Bachelor|Master|PhD|B\.?S\.?|M\.?S\.?|B\.?A\.?|M\.?A\.?)[\s]*(?:of|in|degree)?[\s]*([A-Za-z\s]{3,50})",
        r"([A-Za-z\s]{3,50})[\s]*(?:degree|diploma|certificate)",
    ]
    
    for pattern in education_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities["education"].extend([m.strip() for m in matches])

    # 10. Readability stats (if available)
    if _HAS_TEXTSTAT:
        entities["readability_stats"] = {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            "automated_readability_index": textstat.automated_readability_index(text),
            "word_count": len(text.split()),
            "sentence_count": textstat.sentence_count(text),
        }

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
    
    # Use dynamic chunk sizing
    dynamic_chunk_size = _calculate_dynamic_chunk_size(total_bytes, len(paths))
    
    # Adaptive parameters with dynamic chunk size
    params = _adaptive_params(total_bytes, len(paths))
    CHUNK = dynamic_chunk_size  # Override with dynamic size
    OVER = min(params["overlap"], CHUNK // 4)  # Ensure overlap is reasonable

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
   
