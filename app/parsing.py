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

# Enhanced social link patterns - handles various styles and formats
SOCIAL_PATTERNS = [
    # LinkedIn patterns (various formats)
    r"https?://(www\.)?linkedin\.com/in/[A-Za-z0-9_\-/]+",
    r"https?://(www\.)?linkedin\.com/pub/[A-Za-z0-9_\-/]+",
    r"https?://(www\.)?linkedin\.com/profile/[A-Za-z0-9_\-/]+",
    r"linkedin\.com/in/[A-Za-z0-9_\-/]+",  # Without protocol
    
    # GitHub patterns (various formats)
    r"https?://(www\.)?github\.com/[A-Za-z0-9_\-./]+",
    r"github\.com/[A-Za-z0-9_\-./]+",  # Without protocol
    r"git@github\.com:[A-Za-z0-9_\-./]+",  # Git SSH format
    
    # Twitter/X patterns
    r"https?://(www\.)?(x|twitter)\.com/[A-Za-z0-9_\-./]+",
    r"(x|twitter)\.com/[A-Za-z0-9_\-./]+",  # Without protocol
    
    # Portfolio and personal websites
    r"https?://(www\.)?portfolio\.[A-Za-z0-9_\-./]+",
    r"https?://[A-Za-z0-9_\-./]+\.portfolio\.[A-Za-z0-9_\-./]+",
    r"https?://[A-Za-z0-9_\-./]+\.me(/[A-Za-z0-9_\-./]+)?",
    r"https?://(www\.)?personal\.[A-Za-z0-9_\-./]+",
    
    # Professional platforms
    r"https?://(www\.)?kaggle\.com/[A-Za-z0-9_\-./]+",
    r"https?://(www\.)?medium\.com/[A-Za-z0-9_\-./]+",
    r"https?://(www\.)?dev\.to/[A-Za-z0-9_\-./]+",
    r"https?://(www\.)?stackoverflow\.com/users/[A-Za-z0-9_\-./]+",
    r"https?://(www\.)?behance\.net/[A-Za-z0-9_\-./]+",
    r"https?://(www\.)?dribbble\.com/[A-Za-z0-9_\-./]+",
    
    # Academic profiles
    r"https?://(www\.)?researchgate\.net/profile/[A-Za-z0-9_\-./]+",
    r"https?://(www\.)?scholar\.google\.com/citations\?user=[A-Za-z0-9_\-./]+",
    r"https?://(www\.)?orcid\.org/[A-Za-z0-9_\-./]+",
    
    # Email patterns
    r"mailto:[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    
    # Personal domain patterns (common for developers/designers)
    r"https?://[A-Za-z0-9_\-]+\.(dev|io|co|tech|design|portfolio)(/[A-Za-z0-9_\-./]+)?",
]

SOCIAL = re.compile("|".join(SOCIAL_PATTERNS), re.IGNORECASE)

# Additional patterns for finding social links in different contexts
HYPERLINKED_SOCIAL_PATTERNS = [
    # Pattern for "Name (LinkedIn)" or "John Smith (linkedin.com/in/johnsmith)"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]*\.?\s+)?[A-Z][a-z]+)\s*\(([^)]*(?:linkedin|github|twitter)\.com[^)]*)\)",
    # Pattern for "LinkedIn: linkedin.com/in/profile" 
    r"(LinkedIn|GitHub|Twitter|Portfolio|Website):\s*([A-Za-z0-9._\-/:.]+)",
    # Pattern for markdown-style links [Name](url)
    r"\[([^\]]+)\]\(([^)]*(?:linkedin|github|twitter|portfolio)\.com[^)]*)\)",
]

HYPERLINKED_SOCIAL = re.compile("|".join(HYPERLINKED_SOCIAL_PATTERNS), re.IGNORECASE)

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
    """Enhanced entity extraction with advanced name detection and semantic skills analysis"""
    
    # --- Ensemble entity extraction ---
    entities = {
        "names": [],
        "emails": [],
        "phones": [],
        "dates": [],
        "organizations": [],
        "locations": [],
        "skills": [],
        "skill_contexts": {},  # Track where skills were found (general, experience, education)
        "certifications": [],
        "education": [],
        "experience_years": [],
        "readability_stats": {}
    }

    # 1. Basic regex extractions
    entities["emails"] = EMAIL.findall(text)
    entities["phones"] = PHONE.findall(text)
    
    # Extract dates
    for pattern in DATE_PATTERNS:
        entities["dates"].extend(re.findall(pattern, text, re.IGNORECASE))

    # 2. Enhanced name extraction - research-based approach
    entities["names"] = _extract_names_from_resume(text)
    
    # 3. Enhanced skills extraction with context awareness
    skill_analysis = _extract_skills_with_context(text)
    entities["skills"] = skill_analysis["skills"]
    entities["skill_contexts"] = skill_analysis["contexts"]
    
    # 4. Extract experience years with better patterns
    entities["experience_years"] = _extract_experience_years(text)
    
    # 5. Extract organizations and locations using spaCy if available
    if _NLP:
        try:
            doc = _NLP(text)
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    entities["organizations"].append(ent.text)
                elif ent.label_ in ["GPE", "LOC"]:
                    entities["locations"].append(ent.text)
                elif ent.label_ == "PERSON" and len(ent.text.split()) <= 3:
                    # Additional person names from NER
                    entities["names"].append(ent.text)
        except Exception:
            pass

    # 6. Extract certifications and education
    entities["certifications"] = _extract_certifications(text)
    entities["education"] = _extract_education(text)

    # 7. Advanced NER with transformers if available
    if _NER_PIPELINE:
        try:
            ner_results = _NER_PIPELINE(text[:512])  # Limit for speed
            for item in ner_results:
                if item["entity_group"] == "PER":
                    entities["names"].append(item["word"])
                elif item["entity_group"] == "ORG":
                    entities["organizations"].append(item["word"])
                elif item["entity_group"] == "LOC":
                    entities["locations"].append(item["word"])
        except Exception:
            pass

    # 8. pyresparser integration (if available)
    if _HAS_PYRESPARSER:
        try:
            # pyresparser expects a file path, skip for text-only processing
            pass
        except Exception:
            pass

    # 9. Deduplicate and clean all lists
    for key in entities:
        if isinstance(entities[key], list):
            # Clean and deduplicate
            cleaned = []
            for item in entities[key]:
                if isinstance(item, str):
                    cleaned_item = item.strip()
                    if cleaned_item and len(cleaned_item) > 1:
                        cleaned.append(cleaned_item)
                else:
                    cleaned.append(item)
            entities[key] = list(dict.fromkeys(cleaned))  # Preserve order while removing duplicates

    return entities


def _extract_names_from_resume(text: str) -> List[str]:
    """
    Research-based name extraction from resumes.
    Names typically appear in these locations (in order of likelihood):
    1. First few lines (header section)
    2. After 'Name:' or similar labels
    3. In contact information section
    4. Before email addresses or phone numbers
    """
    names = []
    lines = text.split('\n')
    
    # Strategy 1: Check first 5 lines for names (most common location)
    for i in range(min(5, len(lines))):
        line = lines[i].strip()
        if line and not any(x in line.lower() for x in ['email', 'phone', 'address', 'linkedin', 'github', 'http']):
            # Look for typical name patterns
            name_candidates = _find_name_patterns(line)
            names.extend(name_candidates)
    
    # Strategy 2: Look for explicit name labels
    name_labels = ['name:', 'full name:', 'candidate name:', 'applicant:', 'resume of']
    for line in lines[:10]:  # Check first 10 lines
        line_lower = line.lower()
        for label in name_labels:
            if label in line_lower:
                # Extract text after the label
                after_label = line[line_lower.find(label) + len(label):].strip()
                if after_label:
                    name_candidates = _find_name_patterns(after_label)
                    names.extend(name_candidates)
    
    # Strategy 3: Names often appear before contact info
    for i, line in enumerate(lines[:15]):
        if any(pattern in line.lower() for pattern in ['email', 'phone', '@', 'linkedin']):
            # Check the line before contact info
            if i > 0:
                prev_line = lines[i-1].strip()
                if prev_line and len(prev_line.split()) <= 4:
                    name_candidates = _find_name_patterns(prev_line)
                    names.extend(name_candidates)
            break
    
    # Strategy 4: Look for capitalized words that could be names
    for line in lines[:8]:
        words = line.split()
        if 2 <= len(words) <= 4:  # Typical name length
            if all(word[0].isupper() for word in words if word.isalpha()):
                # All words start with capital - likely a name
                full_name = ' '.join(words)
                if not any(x in full_name.lower() for x in ['email', 'phone', 'address', 'resume', 'cv']):
                    names.append(full_name)
    
    return names


def _find_name_patterns(text: str) -> List[str]:
    """Find name patterns in a given text line"""
    names = []
    
    # Skip lines that are likely not names
    skip_keywords = ['email', 'phone', 'address', 'resume', 'cv', 'objective', 'summary', 
                    'experience', 'education', 'skills', 'linkedin', 'github', 'position', 'engineer']
    
    if any(keyword in text.lower() for keyword in skip_keywords):
        return names
    
    # Pattern 1: First Last or First Middle Last (more strict)
    name_pattern = r'\b([A-Z][a-z]{2,}\s+(?:[A-Z][a-z]*\.?\s+)?[A-Z][a-z]{2,})\b'
    matches = re.findall(name_pattern, text)
    for match in matches:
        # Additional validation - should look like a real name
        words = match.split()
        if 2 <= len(words) <= 3 and all(len(word.strip('.')) >= 2 for word in words):
            names.append(match)
    
    # Pattern 2: Handle initials - J. Smith, John A. Smith (more strict)
    initial_pattern = r'\b([A-Z]\.\s+[A-Z][a-z]{2,}|[A-Z][a-z]{2,}\s+[A-Z]\.\s+[A-Z][a-z]{2,})\b'
    matches = re.findall(initial_pattern, text)
    names.extend(matches)
    
    return names


def _extract_skills_with_context(text: str) -> Dict[str, Any]:
    """
    Enhanced skills extraction that tracks context to distinguish between:
    - General skills (in skills section)
    - Experience skills (learned during jobs/internships)
    - Education skills (learned during studies)
    """
    result = {
        "skills": [],
        "contexts": {}  # skill -> [contexts where found]
    }
    
    # Split text into sections for context analysis
    sections = _identify_resume_sections(text)
    
    # Extract skills from each section with context
    for section_name, section_text in sections.items():
        section_skills = _find_skills_in_text(section_text)
        
        for skill in section_skills:
            if skill not in result["skills"]:
                result["skills"].append(skill)
            
            if skill not in result["contexts"]:
                result["contexts"][skill] = []
            
            # Determine specific context within section
            context = _determine_skill_context(skill, section_text, section_name)
            if context not in result["contexts"][skill]:
                result["contexts"][skill].append(context)
    
    return result


def _identify_resume_sections(text: str) -> Dict[str, str]:
    """Identify different sections of the resume for context-aware parsing"""
    sections = {
        "header": "",
        "objective": "",
        "skills": "",
        "experience": "",
        "education": "",
        "projects": "",
        "certifications": "",
        "other": ""
    }
    
    lines = text.split('\n')
    current_section = "header"
    section_content = []
    
    # Common section headers
    section_keywords = {
        "skills": ["skills", "technical skills", "core competencies", "technologies", "expertise"],
        "experience": ["experience", "work experience", "employment", "professional experience", "internship"],
        "education": ["education", "academic", "qualification", "degree", "university", "college"],
        "projects": ["projects", "personal projects", "work projects"],
        "certifications": ["certifications", "certificates", "awards", "achievements"],
        "objective": ["objective", "summary", "profile", "about"]
    }
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Check if line is a section header
        new_section = None
        for section, keywords in section_keywords.items():
            if any(keyword in line_lower for keyword in keywords) and len(line.strip()) < 50:
                new_section = section
                break
        
        if new_section:
            # Save previous section
            sections[current_section] = '\n'.join(section_content)
            current_section = new_section
            section_content = []
        else:
            section_content.append(line)
    
    # Save the last section
    sections[current_section] = '\n'.join(section_content)
    
    return sections


def _find_skills_in_text(text: str) -> List[str]:
    """Find skills in text using comprehensive skill database"""
    found_skills = []
    text_lower = text.lower()
    
    # Check all skill categories
    for category, skills in SKILLS_DATABASE.items():
        for skill in skills:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.append(skill)
    
    return found_skills


def _determine_skill_context(skill: str, section_text: str, section_name: str) -> str:
    """Determine the specific context where a skill was mentioned"""
    text_lower = section_text.lower()
    skill_lower = skill.lower()
    
    # Find the sentence containing the skill
    sentences = re.split(r'[.!?]+', section_text)
    skill_sentence = ""
    for sentence in sentences:
        if skill_lower in sentence.lower():
            skill_sentence = sentence.lower()
            break
    
    # Determine context based on section and surrounding text
    if section_name == "skills":
        return "general_skills"
    elif section_name == "experience":
        if any(word in skill_sentence for word in ["intern", "internship"]):
            return "internship_skills"
        elif any(word in skill_sentence for word in ["develop", "built", "created", "implemented"]):
            return "practical_experience"
        else:
            return "work_experience"
    elif section_name == "education":
        return "educational_skills"
    elif section_name == "projects":
        return "project_skills"
    else:
        return f"{section_name}_context"


def _extract_experience_years(text: str) -> List[int]:
    """Extract experience years with enhanced patterns"""
    years = []
    
    # Patterns for experience years
    patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
        r'(\d+)\+?\s*yrs?\s*(?:of\s*)?(?:experience|exp)',
        r'(?:experience|exp)\s*:?\s*(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s*in',
        r'over\s*(\d+)\s*years?',
        r'more\s*than\s*(\d+)\s*years?'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        for match in matches:
            try:
                year = int(match)
                if 0 < year <= 50:  # Reasonable range
                    years.append(year)
            except ValueError:
                continue
    
    return years


def _extract_certifications(text: str) -> List[str]:
    """Extract certifications from text"""
    certifications = []
    
    # Common certification patterns
    cert_patterns = [
        r'([A-Z]{2,}(?:\s+[A-Z]{2,})*)\s*(?:certified|certification)',
        r'(?:certified|certification)\s+([A-Z][A-Za-z\s]+)',
        r'([A-Z][A-Za-z\s]+)\s+(?:certificate|cert)',
        r'(?:aws|azure|google|microsoft|oracle|cisco|comptia)\s+([A-Za-z\s]+)',
    ]
    
    for pattern in cert_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        certifications.extend(matches)
    
    return certifications


def _extract_social_links_enhanced(text: str, urls: List[str]) -> List[str]:
    """
    Enhanced social links extraction that handles different styles:
    1. Direct URLs (standard extraction)
    2. Hyperlinked names (e.g., "John Smith (linkedin.com/in/johnsmith)")
    3. Website names with hyperlinks (e.g., "LinkedIn: linkedin.com/profile")
    4. Actual links with hyperlinks (various formats)
    """
    socials = []
    
    # Strategy 1: Extract from existing URLs list
    for url in urls:
        if SOCIAL.search(url):
            socials.append(url)
    
    # Strategy 2: Extract social links directly from text (in case URLs missed some)
    social_matches = SOCIAL.findall(text)
    for match in social_matches:
        # Handle tuple results from regex groups
        if isinstance(match, tuple):
            for part in match:
                if part and any(platform in part.lower() for platform in ['linkedin', 'github', 'twitter', 'portfolio']):
                    socials.append(part)
        else:
            socials.append(match)
    
    # Strategy 3: Extract hyperlinked social patterns (Name + link combinations)
    hyperlinked_matches = HYPERLINKED_SOCIAL.findall(text)
    for match in hyperlinked_matches:
        if isinstance(match, tuple) and len(match) == 2:
            name, url = match
            # Ensure URL is properly formatted
            if not url.startswith('http'):
                if '://' not in url:
                    url = 'https://' + url
            socials.append(url)
    
    # Strategy 4: Look for social platform mentions with nearby URLs
    social_platforms = ['linkedin', 'github', 'twitter', 'portfolio', 'website', 'blog']
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        for platform in social_platforms:
            if platform in line_lower:
                # Look for URLs in the same line or next few lines
                search_lines = lines[i:min(i+3, len(lines))]
                for search_line in search_lines:
                    urls_in_line = URL.findall(search_line)
                    for url in urls_in_line:
                        if platform in url.lower() or any(p in url.lower() for p in social_platforms):
                            socials.append(url)
    
    # Strategy 5: Extract email addresses as social links
    emails = EMAIL.findall(text)
    for email in emails:
        socials.append(f"mailto:{email}")
    
    # Clean and deduplicate
    cleaned_socials = []
    for social in socials:
        social = social.strip()
        if social and len(social) > 5:  # Minimum reasonable length
            # Normalize URLs
            if not social.startswith(('http://', 'https://', 'mailto:')):
                if '@' in social:
                    social = f"mailto:{social}"
                else:
                    social = f"https://{social}"
            cleaned_socials.append(social)
    
    return sorted(set(cleaned_socials))


def _extract_education(text: str) -> List[str]:
    """Extract education information"""
    education = []
    
    # Education degree patterns
    degree_patterns = [
        r'(Bachelor(?:\'?s)?(?:\s+of\s+[A-Za-z\s]+)?)',
        r'(Master(?:\'?s)?(?:\s+of\s+[A-Za-z\s]+)?)',
        r'(Ph\.?D\.?(?:\s+in\s+[A-Za-z\s]+)?)',
        r'(MBA)',
        r'(B\.?[A-Z]\.?(?:\s+[A-Za-z\s]+)?)',
        r'(M\.?[A-Z]\.?(?:\s+[A-Za-z\s]+)?)',
        r'(Associate(?:\s+Degree)?)',
    ]
    
    for pattern in degree_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        education.extend(matches)
    
    return education

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
    Enhanced dynamic chunk size adjustment based on total file size and content complexity.
    
    Strategy:
    - Very small files (< 100KB): Use smaller chunks for fine-grained retrieval
    - Medium files (100KB - 5MB): Use balanced chunks for optimal performance
    - Large files (> 5MB): Use larger chunks for better throughput
    - Multiple files: Adjust for processing efficiency
    """
    mb = max(1, total_bytes // (1024*1024))
    kb = total_bytes // 1024
    
    # Base chunk size calculation with multiple factors
    if kb < 100:  # Very small files
        base = 400
        scaling_factor = 30
    elif kb < 500:  # Small files
        base = 600
        scaling_factor = 50
    elif mb < 1:  # Medium-small files
        base = 800
        scaling_factor = 60
    elif mb < 5:  # Medium files
        base = 1000
        scaling_factor = 80
    else:  # Large files
        base = 1200
        scaling_factor = 100
    
    # Dynamic scaling based on file size and count
    chunk = min(2000, base + int(scaling_factor * (mb**0.5)) + file_count*15)
    
    # Ensure minimum viable chunk size
    chunk = max(256, chunk)
    
    # Calculate overlap as percentage of chunk size (15-25% range)
    overlap_percentage = 0.20 if mb > 5 else 0.18 if mb > 1 else 0.15
    overlap = max(50, int(chunk * overlap_percentage))
    
    # Ensure overlap doesn't exceed reasonable limits
    overlap = min(overlap, chunk // 3)
    
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
        
        # Extract social links with enhanced pattern matching
        socials = _extract_social_links_enhanced(full_text, urls)
        
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
   
