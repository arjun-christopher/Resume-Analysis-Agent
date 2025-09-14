# app/parsing.py - Enhanced with better ZIP handling and multi-format support
from pathlib import Path
from typing import List, Dict, Any, Tuple
import hashlib
import pdfplumber
from pypdf import PdfReader
from docx import Document as DocxDocument
from PIL import Image
import pytesseract
import re
import zipfile
import tempfile
import shutil

# spaCy optional (names)
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None

ALLOWED = {".pdf", ".docx", ".png", ".jpg", ".jpeg"}
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
MAX_FILES_PER_SESSION = 50

def file_sha256(p: Path) -> str:
    """Generate SHA256 hash for file integrity checking"""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def _chunk(t: str) -> List[str]:
    """Split text into overlapping chunks for better context preservation"""
    t = t.replace("\r", "\n")
    out = []
    i = 0
    n = len(t)
    
    while i < n:
        j = min(i + CHUNK_SIZE, n)
        seg = t[i:j].strip()
        if seg:
            out.append(seg)
        if j >= n:
            break
        i = max(0, j - CHUNK_OVERLAP)
    
    return out

def handle_zip_extraction(zip_path: Path, extract_dir: Path) -> List[Path]:
    """Extract ZIP files and return list of extracted file paths"""
    extracted_files = []
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files in ZIP
            file_list = zip_ref.namelist()
            
            # Filter for supported file types
            supported_files = [f for f in file_list 
                             if Path(f).suffix.lower() in ALLOWED and not f.startswith('__MACOSX')]
            
            if len(supported_files) > MAX_FILES_PER_SESSION:
                raise ValueError(f"ZIP contains {len(supported_files)} files. Maximum {MAX_FILES_PER_SESSION} allowed.")
            
            # Extract supported files
            for file_info in zip_ref.infolist():
                if file_info.filename in supported_files and not file_info.is_dir():
                    # Sanitize filename to prevent path traversal
                    safe_filename = Path(file_info.filename).name
                    extract_path = extract_dir / safe_filename
                    
                    with zip_ref.open(file_info) as source, open(extract_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
                    
                    extracted_files.append(extract_path)
                    
    except zipfile.BadZipFile:
        raise ValueError("Invalid or corrupted ZIP file")
    except Exception as e:
        raise ValueError(f"Error extracting ZIP file: {str(e)}")
    
    return extracted_files

def _parse_pdf_plumber(p: Path) -> Tuple[List[str], List[int]]:
    """Parse PDF using pdfplumber with enhanced text extraction"""
    T, P = [], []
    try:
        with pdfplumber.open(str(p)) as pdf:
            for i, page in enumerate(pdf.pages):
                # Enhanced text extraction with better tolerance settings
                txt = page.extract_text(
                    x_tolerance=1.5, 
                    y_tolerance=3.0,
                    layout=True,
                    x_density=7.25,
                    y_density=13
                ) or ""
                
                if txt.strip():
                    for ch in _chunk(txt):
                        T.append(ch)
                        P.append(i + 1)
    except Exception as e:
        print(f"pdfplumber failed for {p}: {e}")
        return [], []
    
    return T, P

def _parse_pdf_fallback(p: Path) -> Tuple[List[str], List[int]]:
    """Fallback PDF parser using pypdf"""
    try:
        rd = PdfReader(str(p))
        T, P = [], []
        
        for i, pg in enumerate(rd.pages):
            txt = pg.extract_text() or ""
            if txt.strip():
                for ch in _chunk(txt):
                    T.append(ch)
                    P.append(i + 1)
        
        return T, P
    except Exception as e:
        print(f"pypdf fallback failed for {p}: {e}")
        return [], []

def _parse_docx(p: Path) -> Tuple[List[str], List[int]]:
    """Parse DOCX files with enhanced content extraction"""
    try:
        d = DocxDocument(str(p))
        
        # Extract paragraphs
        paragraphs = [para.text for para in d.paragraphs if para.text.strip()]
        
        # Extract tables
        table_text = []
        for table in d.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells]
                table_text.append(" | ".join(row_text))
        
        # Combine all text
        full_text = "\n".join(paragraphs + table_text)
        
        if not full_text.strip():
            return [], []
        
        T = _chunk(full_text)
        P = [1] * len(T)  # DOCX doesn't have pages like PDF
        
        return T, P
    except Exception as e:
        print(f"DOCX parsing failed for {p}: {e}")
        return [], []

def _parse_image(p: Path) -> Tuple[List[str], List[int]]:
    """Parse images using OCR with enhanced preprocessing"""
    try:
        im = Image.open(p)
        
        # Enhance image for better OCR
        im = im.convert('L')  # Convert to grayscale
        
        # OCR with multiple PSM modes for better accuracy
        configs = [
            "--psm 6",  # Uniform block of text
            "--psm 4",  # Single column of text
            "--psm 3",  # Fully automatic page segmentation
        ]
        
        best_text = ""
        for config in configs:
            try:
                text = pytesseract.image_to_string(im, config=config)
                if len(text.strip()) > len(best_text.strip()):
                    best_text = text
            except:
                continue
        
        if not best_text.strip():
            return [], []
        
        T = _chunk(best_text)
        P = [1] * len(T)
        
        return T, P
    except Exception as e:
        print(f"Image OCR failed for {p}: {e}")
        return [], []

# Email and phone regex patterns
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(\+?\d[\d\s\-()]{7,}\d)")

# Enhanced skill vocabulary covering multiple industries
EXTENDED_SKILLS = {
    # Programming Languages
    "python", "java", "c", "c++", "c#", "go", "rust", "ruby", "perl", "scala", "php", "r", 
    "swift", "objective-c", "javascript", "typescript", "kotlin", "dart", "matlab", "shell", 
    "bash", "powershell", "haskell", "fortran", "cobol", "assembly",
    
    # Web Development
    "html", "css", "sass", "less", "jquery", "react", "angular", "vue", "svelte", "next.js", 
    "nuxt.js", "node", "express", "fastapi", "nestjs", "spring boot", "flask", "django", 
    "rails", "laravel", "asp.net", "blazor",
    
    # Mobile Development
    "android", "ios", "react native", "flutter", "ionic", "xamarin", "swiftui", 
    "kotlin multiplatform", "cordova", "phonegap",
    
    # Databases & Big Data
    "sql", "mysql", "postgres", "postgresql", "oracle", "mssql", "sqlite", "mongodb", 
    "cassandra", "couchdb", "dynamodb", "elasticsearch", "redis", "neo4j", "janusgraph", 
    "snowflake", "redshift", "bigquery", "hive", "hbase", "presto", "trino",
    
    # Data Engineering & Analytics
    "kafka", "spark", "hadoop", "beam", "flink", "airflow", "databricks", "luigi", "storm",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly", "statsmodels",
    
    # Cloud Platforms
    "aws", "azure", "gcp", "ibm cloud", "oracle cloud", "digitalocean", "heroku", 
    "cloudflare", "vercel", "netlify",
    
    # DevOps & Infrastructure
    "docker", "kubernetes", "openshift", "terraform", "ansible", "puppet", "chef", 
    "jenkins", "github actions", "gitlab ci", "circleci", "argo cd", "helm", "prometheus", 
    "grafana", "nagios", "splunk", "elastic stack", "vault", "consul",
    
    # AI/ML/Data Science
    "scikit-learn", "xgboost", "lightgbm", "catboost", "pytorch", "tensorflow", "keras", 
    "jax", "huggingface", "transformers", "spacy", "nltk", "gensim", "mlflow", "dvc", 
    "ray", "optuna", "onnx", "openvino", "accelerate", "nlp", "llm", "chatgpt", 
    "computer vision", "opencv", "stable diffusion", "deep learning", "reinforcement learning",
    
    # Business Intelligence
    "tableau", "power bi", "qlikview", "looker", "superset", "metabase", "mode", 
    "excel", "sas", "stata", "spss",
    
    # Testing & QA
    "selenium", "cypress", "pytest", "junit", "mocha", "chai", "jest", "playwright", 
    "postman", "soapui", "appium", "loadrunner", "jmeter", "rest assured", "karate",
    
    # Project Management
    "scrum", "kanban", "agile", "safe", "prince2", "pmp", "jira", "confluence", 
    "trello", "asana", "monday.com",
    
    # Design & UX
    "figma", "adobe xd", "sketch", "photoshop", "illustrator", "canva", "invision", 
    "zeplin", "wireframing", "prototyping", "usability testing",
    
    # Cybersecurity
    "penetration testing", "ethical hacking", "burpsuite", "metasploit", "wireshark", 
    "nmap", "nessus", "splunk security", "siem", "firewalls", "ids", "ips", "zero trust",
    
    # Finance & Trading
    "bloomberg", "reuters", "matlab", "r", "quantlib", "pandas", "algo trading", 
    "risk management", "derivatives", "fixed income", "equity research",
    
    # Healthcare & Biotech
    "healthcare", "clinical", "fhir", "hl7", "epic", "cerner", "allscripts", 
    "bioinformatics", "genomics", "clinical trials",
    
    # Manufacturing & Engineering
    "autocad", "solidworks", "catia", "ansys", "matlab", "simulink", "plc", "scada", 
    "lean manufacturing", "six sigma",
    
    # Sales & Marketing
    "salesforce", "hubspot", "marketo", "google analytics", "facebook ads", 
    "google ads", "seo", "sem", "crm"
}

def _simple_entities(text: str) -> Dict[str, List[str]]:
    """Extract entities: emails, phones, names using regex and spaCy"""
    emails = EMAIL.findall(text)
    phones = [m.group(0) for m in PHONE.finditer(text)]
    names = []
    
    if _NLP:
        try:
            doc = _NLP(text)
            names = [e.text for e in doc.ents if e.label_ == "PERSON"]
        except Exception:
            names = []
    
    return {
        "emails": list(set(emails)),
        "phones": list(set(phones)),
        "names": list(set(names))
    }

def _skills_from_text(text: str) -> List[str]:
    """Extract technical skills from text using enhanced vocabulary"""
    t = text.lower()
    found_skills = []
    
    for skill in EXTENDED_SKILLS:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(skill.replace(' ', r'\s+')) + r'\b'
        if re.search(pattern, t):
            found_skills.append(skill)
    
    return sorted(found_skills)

def extract_docs_to_chunks_and_records(paths: List[Path]) -> Tuple[List[str], List[Dict[str,Any]], List[Dict[str,Any]]]:
    """
    Enhanced document processing with ZIP support and comprehensive metadata extraction
    
    Returns:
        - chunks: List of text chunks
        - metas: List of metadata for each chunk
        - records: List of document-level records with aggregated info
    """
    chunks: List[str] = []
    metas: List[Dict[str,Any]] = []
    records: List[Dict[str,Any]] = []
    
    # First pass: handle ZIP files
    processed_paths = []
    for p in paths:
        if p.suffix.lower() == '.zip':
            try:
                # Create temporary extraction directory
                extract_dir = p.parent / f"{p.stem}_extracted"
                extract_dir.mkdir(exist_ok=True)
                
                # Extract and get file paths
                extracted_files = handle_zip_extraction(p, extract_dir)
                processed_paths.extend(extracted_files)
                
                # Remove the ZIP file after extraction
                p.unlink()
                
            except Exception as e:
                print(f"Failed to process ZIP file {p}: {e}")
                continue
        else:
            processed_paths.append(p)
    
    # Second pass: process all files
    for p in processed_paths:
        if p.suffix.lower() not in ALLOWED:
            continue
        
        # Parse based on file type
        if p.suffix.lower() == ".pdf":
            T, P = _parse_pdf_plumber(p)
            if not T:
                T, P = _parse_pdf_fallback(p)
        elif p.suffix.lower() == ".docx":
            T, P = _parse_docx(p)
        else:  # Image files
            T, P = _parse_image(p)
        
        if not T:
            continue
        
        # Generate file hash for integrity
        fh = file_sha256(p)
        
        # Aggregate document-level information
        all_text = " ".join(T)
        doc_entities = _simple_entities(all_text)
        doc_skills = _skills_from_text(all_text)
        
        # Create document record
        record = {
            "file": p.name,
            "path": str(p),
            "file_hash": fh,
            "total_chunks": len(T),
            "total_pages": max(P) if P else 1,
            "emails": doc_entities["emails"],
            "phones": doc_entities["phones"],
            "names": doc_entities["names"],
            "skills": doc_skills,
            "file_size": p.stat().st_size,
            "content_preview": all_text[:500] + "..." if len(all_text) > 500 else all_text
        }
        records.append(record)
        
        # Process chunks
        for i, chunk_text in enumerate(T):
            chunk_entities = _simple_entities(chunk_text)
            chunk_skills = _skills_from_text(chunk_text)
            
            chunk_meta = {
                "file": p.name,
                "path": str(p),
                "page": P[i] if i < len(P) else 1,
                "chunk_index": i,
                "file_hash": fh,
                "skills": chunk_skills,
                "emails": chunk_entities["emails"],
                "phones": chunk_entities["phones"],
                "names": chunk_entities["names"],
                "char_count": len(chunk_text),
                "word_count": len(chunk_text.split())
            }
            
            metas.append(chunk_meta)
            chunks.append(chunk_text)
    
    return chunks, metas, records

# Alias for backward compatibility
def extract_chunks_and_meta_from_paths(paths: List[Path]) -> Tuple[List[str], List[Dict[str,Any]]]:
    """Backward compatibility wrapper"""
    chunks, metas, _ = extract_docs_to_chunks_and_records(paths)
    return chunks, metas