# app/parsing_pymupdf.py - Advanced resume parsing using PyMuPDF with comprehensive information extraction
from __future__ import annotations
import hashlib
import re
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Core libraries
import fitz  # PyMuPDF
import dateparser
import phonenumbers
from docx import Document as DocxDocument

# NLP and ML libraries
try:
    import spacy
    # Try to load transformer model first, fallback to standard
    try:
        _NLP = spacy.load("en_core_web_trf")
    except OSError:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            _NLP = None
except ImportError:
    _NLP = None

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
    # Load advanced NER model (with error handling for compatibility issues)
    try:
        _NER_PIPELINE = pipeline(
            "ner", 
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            aggregation_strategy="simple",
            device=-1  # Use CPU
        )
        _HAS_TRANSFORMERS = True
    except (RuntimeError, AttributeError, ImportError) as e:
        print(f"Transformers NER pipeline disabled due to compatibility issue: {e}")
        _NER_PIPELINE = None
        _HAS_TRANSFORMERS = False
except ImportError:
    _NER_PIPELINE = None
    _HAS_TRANSFORMERS = False

try:
    from sentence_transformers import SentenceTransformer
    try:
        _SENTENCE_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        _HAS_SENTENCE_TRANSFORMERS = True
    except (RuntimeError, AttributeError, ImportError) as e:
        print(f"Sentence transformers disabled due to compatibility issue: {e}")
        _SENTENCE_MODEL = None
        _HAS_SENTENCE_TRANSFORMERS = False
except ImportError:
    _SENTENCE_MODEL = None
    _HAS_SENTENCE_TRANSFORMERS = False

try:
    from rapidfuzz import fuzz, process
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

try:
    from flashtext import KeywordProcessor
    _HAS_FLASHTEXT = True
except ImportError:
    _HAS_FLASHTEXT = False

try:
    import textstat
    _HAS_TEXTSTAT = True
except ImportError:
    _HAS_TEXTSTAT = False

# ---------- Configuration ----------
ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx"}
MAX_FILES_PER_SESSION = 50
MAX_TEXT_LENGTH = 1_000_000  # Limit text processing for performance

# Pre-compiled regex patterns for efficiency
EMAIL_PATTERN = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    re.IGNORECASE
)

PHONE_PATTERN = re.compile(
    r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\b\d{10}\b',
    re.MULTILINE
)

URL_PATTERN = re.compile(
    r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
    re.IGNORECASE
)

# Enhanced hyperlink classification patterns (ordered from most specific to most general)
from collections import OrderedDict

HYPERLINK_PATTERNS = OrderedDict([
    ('email', re.compile(r'mailto:([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})', re.IGNORECASE)),
    ('linkedin', re.compile(r'(?:https?://)?(?:www\.)?linkedin\.com/(?:in|pub|company)/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('github', re.compile(r'(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('twitter', re.compile(r'(?:https?://)?(?:www\.)?(?:twitter|x)\.com/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('youtube', re.compile(r'(?:https?://)?(?:www\.)?(?:youtube\.com/(?:c/|channel/|user/)?|youtu\.be/)([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('kaggle', re.compile(r'(?:https?://)?(?:www\.)?kaggle\.com/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('medium', re.compile(r'(?:https?://)?(?:www\.)?medium\.com/@?([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('stackoverflow', re.compile(r'(?:https?://)?(?:www\.)?stackoverflow\.com/users/(\d+/[A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('behance', re.compile(r'(?:https?://)?(?:www\.)?behance\.net/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('dribbble', re.compile(r'(?:https?://)?(?:www\.)?dribbble\.com/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('instagram', re.compile(r'(?:https?://)?(?:www\.)?instagram\.com/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('facebook', re.compile(r'(?:https?://)?(?:www\.)?facebook\.com/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    # Portfolio pattern for specific domains
    ('portfolio', re.compile(r'(?:https?://)?(?:www\.)?([A-Za-z0-9\-_.]+\.(?:portfolio|website|dev|me|io))', re.IGNORECASE)),
    # Website pattern - most general, should be last
    ('website', re.compile(r'(?:https?://)?(?:www\.)?([A-Za-z0-9\-_.]+\.[A-Za-z]{2,})', re.IGNORECASE)),
])

# Text label patterns for hyperlinks with generic display text
TEXT_LABEL_PATTERNS = {
    'linkedin': [
        r'\blinkedin\b', r'\blinked\s*in\b', r'\bprofile\b', r'\bprofessional\s*profile\b',
        r'\bconnect\b', r'\bconnect\s*with\s*me\b', r'\bview\s*profile\b'
    ],
    'github': [
        r'\bgithub\b', r'\bgit\s*hub\b', r'\bcode\b', r'\brepository\b', r'\brepo\b',
        r'\bprojects\b', r'\bmy\s*code\b', r'\bview\s*code\b', r'\bgit\b'
    ],
    'twitter': [
        r'\btwitter\b', r'\bfollow\s*me\b', r'\btweet\b', r'\bfollow\b', r'\b@\w+\b',
        r'\bx\.com\b', r'\bx\b'
    ],
    'website': [
        r'\bwebsite\b', r'\bvisit\b', r'\bhomepage\b', r'\bweb\b', r'\bsite\b',
        r'\bclick\s*here\b', r'\bmore\s*info\b', r'\blearn\s*more\b', r'\bview\b'
    ],
    'portfolio': [
        r'\bportfolio\b', r'\bwork\b', r'\bmy\s*work\b', r'\bprojects\b', 
        r'\bshowcase\b', r'\bgallery\b', r'\bexamples\b'
    ],
    'medium': [
        r'\bmedium\b', r'\bblog\b', r'\barticles\b', r'\bwriting\b', r'\bposts\b',
        r'\bread\s*more\b', r'\bmy\s*blog\b'
    ],
    'email': [
        r'\bemail\b', r'\bcontact\b', r'\bmail\b', r'\bget\s*in\s*touch\b',
        r'\breach\s*out\b', r'\bmessage\b'
    ],
    'youtube': [
        r'\byoutube\b', r'\bvideo\b', r'\bvideos\b', r'\bchannel\b', r'\bwatch\b',
        r'\bsubscribe\b', r'\bmy\s*channel\b'
    ],
    'kaggle': [
        r'\bkaggle\b', r'\bdata\s*science\b', r'\bcompetitions\b', r'\bdatasets\b'
    ],
    'stackoverflow': [
        r'\bstack\s*overflow\b', r'\bso\b', r'\bstackoverflow\b', r'\breputation\b'
    ]
}

# Compile text label patterns for efficiency
COMPILED_TEXT_PATTERNS = {}
for platform, patterns in TEXT_LABEL_PATTERNS.items():
    COMPILED_TEXT_PATTERNS[platform] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

# Legacy patterns for backward compatibility
SOCIAL_PATTERNS = {
    'linkedin': HYPERLINK_PATTERNS['linkedin'],
    'github': HYPERLINK_PATTERNS['github'],
    'twitter': HYPERLINK_PATTERNS['twitter'],
    'kaggle': HYPERLINK_PATTERNS['kaggle'],
    'medium': HYPERLINK_PATTERNS['medium'],
    'stackoverflow': HYPERLINK_PATTERNS['stackoverflow'],
}

DATE_PATTERNS = [
    re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b', re.IGNORECASE),
    re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
    re.compile(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'),
    re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', re.IGNORECASE),
]

# Name extraction patterns
NAME_PATTERNS = [
    re.compile(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]*\.?\s*)*[A-Z][a-z]+)$'),
    re.compile(r'^([A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+)$'),
    re.compile(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})$'),
]

# ---------- Comprehensive Skills Database ----------
SKILLS_DATABASE = {
    # Programming Languages
    "programming_languages": {
        "python", "java", "javascript", "typescript", "c++", "c#", "c", "go", "rust", 
        "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "perl", "shell", 
        "bash", "powershell", "sql", "nosql", "plsql", "t-sql", "vba", "objective-c",
        "dart", "elixir", "erlang", "haskell", "clojure", "f#", "lua", "groovy",
        "assembly", "cobol", "fortran", "lisp", "prolog", "scheme", "smalltalk"
    },
    
    # Web Development & Frontend
    "web_frontend": {
        "html", "html5", "css", "css3", "react", "angular", "vue", "vue.js", "svelte", 
        "jquery", "bootstrap", "tailwind", "tailwind css", "material-ui", "mui",
        "sass", "scss", "less", "stylus", "webpack", "vite", "rollup", "parcel",
        "babel", "typescript", "javascript", "es6", "json", "ajax", "dom", "bom",
        "responsive design", "pwa", "spa", "ssr", "next.js", "nuxt.js", "gatsby"
    },
    
    # Backend Development
    "web_backend": {
        "node.js", "express", "express.js", "django", "flask", "fastapi", "spring", 
        "spring boot", "laravel", "rails", "ruby on rails", "asp.net", ".net core",
        "phoenix", "gin", "fiber", "echo", "chi", "koa", "hapi", "nestjs",
        "strapi", "graphql", "rest api", "soap", "microservices", "serverless"
    },
    
    # Databases
    "databases": {
        "mysql", "postgresql", "postgres", "mongodb", "redis", "elasticsearch", 
        "cassandra", "oracle", "sql server", "sqlite", "dynamodb", "neo4j", 
        "influxdb", "couchdb", "mariadb", "firebase", "firestore", "snowflake",
        "bigquery", "redshift", "clickhouse", "timescaledb", "cockroachdb",
        "etcd", "consul", "vault", "couchbase", "riak", "datomic"
    },
    
    # Cloud Platforms & Services
    "cloud_platforms": {
        "aws", "amazon web services", "azure", "microsoft azure", "gcp", "google cloud", 
        "google cloud platform", "heroku", "digitalocean", "linode", "vultr", 
        "cloudflare", "vercel", "netlify", "firebase", "supabase", "planetscale",
        "railway", "render", "fly.io", "cyclic", "surge.sh", "github pages"
    },
    
    # DevOps & Infrastructure
    "devops_infrastructure": {
        "docker", "kubernetes", "k8s", "jenkins", "gitlab ci", "github actions", 
        "terraform", "ansible", "puppet", "chef", "vagrant", "prometheus", "grafana", 
        "elk stack", "splunk", "nginx", "apache", "traefik", "consul", "vault",
        "istio", "linkerd", "helm", "kustomize", "argo cd", "flux", "tekton",
        "packer", "pulumi", "cloudformation", "arm templates", "bicep"
    },
    
    # Data Science & Analytics
    "data_science": {
        "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", 
        "xgboost", "lightgbm", "catboost", "spark", "pyspark", "hadoop", "dask", "ray", 
        "tableau", "power bi", "looker", "qlik", "d3.js", "plotly", "seaborn", 
        "matplotlib", "ggplot2", "jupyter", "r studio", "anaconda", "spyder",
        "mlflow", "kubeflow", "airflow", "prefect", "dagster", "great expectations"
    },
    
    # Machine Learning & AI
    "machine_learning": {
        "machine learning", "deep learning", "neural networks", "cnn", "rnn", "lstm",
        "transformer", "bert", "gpt", "nlp", "computer vision", "opencv", "yolo",
        "reinforcement learning", "supervised learning", "unsupervised learning",
        "feature engineering", "model deployment", "mlops", "automl", "hyperparameter tuning"
    },
    
    # Mobile Development
    "mobile_development": {
        "android", "ios", "react native", "flutter", "ionic", "xamarin", "cordova", 
        "kotlin multiplatform", "swift ui", "jetpack compose", "android studio",
        "xcode", "unity", "unreal engine", "phonegap", "titanium", "nativescript"
    },
    
    # Testing & Quality Assurance
    "testing_qa": {
        "selenium", "cypress", "playwright", "jest", "mocha", "chai", "pytest", 
        "junit", "testng", "cucumber", "postman", "insomnia", "k6", "locust",
        "unit testing", "integration testing", "e2e testing", "performance testing",
        "load testing", "stress testing", "security testing", "automation testing",
        "tdd", "bdd", "test driven development", "behavior driven development"
    },
    
    # Soft Skills & Management
    "leadership_management": {
        "leadership", "team leadership", "project management", "people management",
        "strategic planning", "decision making", "delegation", "coaching", "mentoring",
        "performance management", "change management", "conflict resolution",
        "stakeholder management", "vendor management", "budget management"
    },
    
    "communication_interpersonal": {
        "communication", "verbal communication", "written communication", "presentation",
        "public speaking", "negotiation", "active listening", "empathy", "collaboration",
        "teamwork", "cross-functional collaboration", "client relations", "customer service",
        "networking", "relationship building", "cultural awareness", "emotional intelligence"
    },
    
    "analytical_problem_solving": {
        "problem solving", "analytical thinking", "critical thinking", "logical reasoning",
        "data analysis", "research", "troubleshooting", "root cause analysis",
        "process improvement", "optimization", "innovation", "creativity", "strategic thinking"
    },
    
    "organizational_productivity": {
        "time management", "multitasking", "prioritization", "organization", "planning",
        "attention to detail", "quality focus", "process oriented", "results oriented",
        "goal setting", "self-motivated", "proactive", "adaptability", "flexibility"
    },
    
    # Methodologies & Practices
    "methodologies": {
        "agile", "scrum", "kanban", "waterfall", "lean", "six sigma", "itil", "prince2",
        "safe", "devops", "ci/cd", "continuous integration", "continuous deployment",
        "pair programming", "code review", "version control", "git", "svn", "mercurial"
    },
    
    # Languages (Human Languages)
    "human_languages": {
        "english", "spanish", "french", "german", "italian", "portuguese", "chinese",
        "mandarin", "japanese", "korean", "arabic", "russian", "hindi", "dutch",
        "polish", "turkish", "swedish", "norwegian", "danish", "finnish", "greek",
        "hebrew", "thai", "vietnamese", "indonesian", "malay", "tagalog", "urdu"
    },
    
    # Certifications & Qualifications
    "certifications": {
        "aws certified", "azure certified", "google cloud certified", "pmp", "cissp", 
        "comptia", "cisco", "microsoft certified", "oracle certified", "kubernetes certified",
        "certified scrum master", "csm", "safe", "itil", "prince2", "six sigma",
        "cfa", "frm", "cpa", "cma", "cissp", "cism", "cisa", "crisc", "ccna", "ccnp"
    },
    
    # Industry Specific Skills
    "finance_accounting": {
        "financial analysis", "budgeting", "forecasting", "risk management", "compliance",
        "audit", "tax", "accounting", "bookkeeping", "financial modeling", "valuation",
        "investment analysis", "portfolio management", "derivatives", "fixed income"
    },
    
    "marketing_sales": {
        "digital marketing", "seo", "sem", "social media marketing", "content marketing",
        "email marketing", "marketing automation", "lead generation", "sales",
        "business development", "crm", "salesforce", "hubspot", "marketo", "pardot"
    },
    
    "design_creative": {
        "ui design", "ux design", "graphic design", "web design", "logo design",
        "branding", "typography", "color theory", "wireframing", "prototyping",
        "adobe creative suite", "photoshop", "illustrator", "indesign", "figma",
        "sketch", "invision", "principle", "framer", "zeplin", "adobe xd"
    }
}

# Flatten skills for faster lookup
ALL_SKILLS = set()
for category_skills in SKILLS_DATABASE.values():
    ALL_SKILLS.update(category_skills)

# Section detection patterns for skills extraction
SKILLS_SECTION_PATTERNS = {
    'technical_skills': [
        r'\btechnical\s+skills?\b', r'\btech\s+skills?\b', r'\btechnologies\b',
        r'\bprogramming\s+skills?\b', r'\bcoding\s+skills?\b', r'\bhard\s+skills?\b',
        r'\bcomputer\s+skills?\b', r'\bsoftware\s+skills?\b', r'\btools?\s+&?\s*technologies\b'
    ],
    'soft_skills': [
        r'\bsoft\s+skills?\b', r'\binterpersonal\s+skills?\b', r'\bpeople\s+skills?\b',
        r'\bcommunication\s+skills?\b', r'\bleadership\s+skills?\b', r'\bmanagement\s+skills?\b',
        r'\bcore\s+competencies\b', r'\bkey\s+strengths?\b', r'\bpersonal\s+skills?\b'
    ],
    'languages': [
        r'\blanguages?\b', r'\bhuman\s+languages?\b', r'\bspoken\s+languages?\b',
        r'\bforeign\s+languages?\b', r'\blinguistic\s+skills?\b', r'\bmultilingual\b'
    ],
    'certifications': [
        r'\bcertifications?\b', r'\bcertified?\b', r'\blicenses?\b', r'\bqualifications?\b',
        r'\bcredentials?\b', r'\bacreditations?\b', r'\bprofessional\s+certifications?\b'
    ],
    'general_skills': [
        r'\bskills?\b', r'\bcompetencies\b', r'\bexpertise\b', r'\bproficiencies?\b',
        r'\bcapabilities\b', r'\bstrengths?\b', r'\babilities\b', r'\bknowledge\b'
    ]
}

# Compile section patterns
COMPILED_SECTION_PATTERNS = {}
for section_type, patterns in SKILLS_SECTION_PATTERNS.items():
    COMPILED_SECTION_PATTERNS[section_type] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

# Common skill separators and delimiters
SKILL_SEPARATORS = [
    r'[,;|•·▪▫◦‣⁃]',  # Punctuation separators
    r'\s*[/\\]\s*',    # Slash separators
    r'\s+[-–—]\s+',    # Dash separators
    r'\s*\|\s*',       # Pipe separators
    r'\s*&\s*',        # Ampersand separators
    r'\n+',            # Line breaks
    r'\s{2,}',         # Multiple spaces
]

# Skill level indicators
SKILL_LEVEL_PATTERNS = {
    'beginner': r'\b(?:beginner|basic|novice|elementary|introductory|fundamental)\b',
    'intermediate': r'\b(?:intermediate|moderate|average|competent|proficient)\b',
    'advanced': r'\b(?:advanced|expert|senior|experienced|professional|master|skilled)\b',
    'expert': r'\b(?:expert|master|guru|ninja|wizard|architect|lead|principal)\b'
}

# Programming language variations and aliases
PROGRAMMING_LANGUAGE_ALIASES = {
    'javascript': ['js', 'ecmascript', 'es6', 'es2015', 'es2016', 'es2017', 'es2018', 'es2019', 'es2020'],
    'typescript': ['ts'],
    'c++': ['cpp', 'c plus plus'],
    'c#': ['csharp', 'c sharp'],
    'objective-c': ['objc', 'objective c'],
    'python': ['py'],
    'ruby': ['rb'],
    'go': ['golang'],
    'r': ['r language', 'r programming']
}

# Initialize FlashText processor if available
if _HAS_FLASHTEXT:
    _KEYWORD_PROCESSOR = KeywordProcessor(case_sensitive=False)
    for skill in ALL_SKILLS:
        _KEYWORD_PROCESSOR.add_keyword(skill)
    
    # Add programming language aliases
    for main_lang, aliases in PROGRAMMING_LANGUAGE_ALIASES.items():
        if main_lang in ALL_SKILLS:
            for alias in aliases:
                _KEYWORD_PROCESSOR.add_keyword(alias, main_lang)
else:
    _KEYWORD_PROCESSOR = None

# ---------- Skills Extraction Functions ----------
def detect_skills_sections(text: str) -> Dict[str, List[tuple]]:
    """
    Detect different skill sections in text and return their positions
    Returns: {'section_type': [(start_pos, end_pos, section_text), ...]}
    """
    sections = {}
    lines = text.split('\n')
    
    for section_type, patterns in COMPILED_SECTION_PATTERNS.items():
        sections[section_type] = []
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean:
                continue
                
            for pattern in patterns:
                if pattern.search(line_clean):
                    # Found a section header, look for content
                    start_line = i
                    end_line = start_line + 1
                    
                    # Find the end of this section (next section header or end of text)
                    for j in range(i + 1, min(len(lines), i + 20)):  # Look ahead max 20 lines
                        next_line = lines[j].strip()
                        if not next_line:
                            continue
                            
                        # Check if this is another section header
                        is_section_header = False
                        for other_patterns in COMPILED_SECTION_PATTERNS.values():
                            for other_pattern in other_patterns:
                                if other_pattern.search(next_line):
                                    is_section_header = True
                                    break
                            if is_section_header:
                                break
                        
                        if is_section_header:
                            end_line = j
                            break
                        end_line = j + 1
                    
                    # Extract section content
                    section_lines = lines[start_line:end_line]
                    section_text = '\n'.join(section_lines)
                    
                    sections[section_type].append((start_line, end_line, section_text))
                    break
    
    return sections

def extract_skills_from_text(text: str, section_type: str = 'general') -> Dict[str, Any]:
    """
    Extract skills from text with enhanced parsing for different formats
    """
    skills_data = {
        'raw_skills': [],
        'categorized_skills': {},
        'skill_levels': {},
        'confidence_scores': {},
        'extraction_method': []
    }
    
    # Method 1: FlashText keyword extraction (fastest)
    if _KEYWORD_PROCESSOR:
        found_keywords = _KEYWORD_PROCESSOR.extract_keywords(text, span_info=True)
        for keyword, start, end in found_keywords:
            # Get surrounding context for level detection
            context_start = max(0, start - 50)
            context_end = min(len(text), end + 50)
            context = text[context_start:context_end]
            
            level = detect_skill_level(context, keyword)
            confidence = calculate_skill_confidence(text, keyword, context)
            
            skills_data['raw_skills'].append(keyword)
            skills_data['skill_levels'][keyword] = level
            skills_data['confidence_scores'][keyword] = confidence
            skills_data['extraction_method'].append('flashtext')
    
    # Method 2: Regex-based extraction for missed skills
    regex_skills = extract_skills_with_regex(text)
    for skill, level, confidence in regex_skills:
        if skill not in skills_data['raw_skills']:
            skills_data['raw_skills'].append(skill)
            skills_data['skill_levels'][skill] = level
            skills_data['confidence_scores'][skill] = confidence
            skills_data['extraction_method'].append('regex')
    
    # Method 3: Pattern-based extraction for structured formats
    pattern_skills = extract_skills_from_patterns(text, section_type)
    for skill, level, confidence in pattern_skills:
        if skill not in skills_data['raw_skills']:
            skills_data['raw_skills'].append(skill)
            skills_data['skill_levels'][skill] = level
            skills_data['confidence_scores'][skill] = confidence
            skills_data['extraction_method'].append('pattern')
    
    # Categorize skills
    skills_data['categorized_skills'] = categorize_extracted_skills(skills_data['raw_skills'])
    
    return skills_data

def extract_skills_with_regex(text: str) -> List[tuple]:
    """Extract skills using regex patterns for various formats"""
    skills_found = []
    
    # Common skill listing patterns
    patterns = [
        # Bullet points: • Python, Java, C++
        r'[•·▪▫◦‣⁃]\s*([A-Za-z0-9+#\s./-]+?)(?=[•·▪▫◦‣⁃\n]|$)',
        
        # Comma separated: Python, Java, C++, JavaScript
        r'\b([A-Z][A-Za-z0-9+#\s./-]{2,20}?)(?:,\s*(?=[A-Z])|$)',
        
        # Pipe separated: Python | Java | C++
        r'\b([A-Z][A-Za-z0-9+#\s./-]{2,20}?)\s*\|\s*',
        
        # Line by line format
        r'^([A-Z][A-Za-z0-9+#\s./-]{2,30})$',
        
        # Skills with levels: Python (Advanced), Java (Intermediate)
        r'\b([A-Z][A-Za-z0-9+#\s./-]{2,20}?)\s*\(([^)]+)\)',
        
        # Skills with years: Python - 5 years, Java (3+ years)
        r'\b([A-Z][A-Za-z0-9+#\s./-]{2,20}?)\s*[-–—]\s*(\d+\+?\s*years?)',
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
        for match in matches:
            skill_name = match.group(1).strip()
            level = 'intermediate'  # default
            confidence = 0.7
            
            # Extract level information if available
            if len(match.groups()) > 1:
                level_info = match.group(2).lower()
                level = extract_level_from_text(level_info)
                confidence = 0.8
            
            # Validate if it's a real skill
            if is_valid_skill(skill_name):
                skills_found.append((skill_name.lower(), level, confidence))
    
    return skills_found

def extract_skills_from_patterns(text: str, section_type: str) -> List[tuple]:
    """Extract skills using section-specific patterns"""
    skills_found = []
    
    if section_type == 'languages':
        # Language proficiency patterns
        language_patterns = [
            r'\b([A-Z][a-z]{2,15})\s*[-–—]\s*(native|fluent|proficient|conversational|basic|beginner|intermediate|advanced)\b',
            r'\b(native|fluent|proficient|conversational|basic|beginner|intermediate|advanced)\s+([A-Z][a-z]{2,15})\b',
            r'\b([A-Z][a-z]{2,15})\s*\(([^)]+)\)',
        ]
        
        for pattern in language_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if 'native' in match.group(0).lower() or 'fluent' in match.group(0).lower():
                    lang = match.group(1) if match.group(1).lower() not in ['native', 'fluent'] else match.group(2)
                else:
                    lang = match.group(1)
                
                level = extract_level_from_text(match.group(0))
                if lang.lower() in SKILLS_DATABASE['human_languages']:
                    skills_found.append((lang.lower(), level, 0.9))
    
    elif section_type == 'technical_skills':
        # Technical skills often listed with versions
        tech_patterns = [
            r'\b([A-Za-z0-9+#.]{2,20})\s+v?(\d+(?:\.\d+)*)\b',  # Python 3.9, Java 11
            r'\b([A-Za-z0-9+#.]{2,20})\s+([\d.]+)\b',          # React 18, Angular 14
        ]
        
        for pattern in tech_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                skill = match.group(1).lower()
                if is_valid_skill(skill):
                    skills_found.append((skill, 'intermediate', 0.8))
    
    return skills_found

def detect_skill_level(context: str, skill: str) -> str:
    """Detect skill level from context"""
    context_lower = context.lower()
    
    for level, pattern in SKILL_LEVEL_PATTERNS.items():
        if re.search(pattern, context_lower):
            return level
    
    # Look for years of experience
    years_pattern = r'(\d+)\+?\s*years?'
    years_match = re.search(years_pattern, context_lower)
    if years_match:
        years = int(years_match.group(1))
        if years >= 7:
            return 'expert'
        elif years >= 4:
            return 'advanced'
        elif years >= 2:
            return 'intermediate'
        else:
            return 'beginner'
    
    return 'intermediate'  # Default

def extract_level_from_text(text: str) -> str:
    """Extract skill level from text"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['expert', 'master', 'guru', 'architect', 'lead']):
        return 'expert'
    elif any(word in text_lower for word in ['advanced', 'senior', 'experienced', 'professional']):
        return 'advanced'
    elif any(word in text_lower for word in ['intermediate', 'moderate', 'competent', 'proficient']):
        return 'intermediate'
    elif any(word in text_lower for word in ['beginner', 'basic', 'novice', 'elementary']):
        return 'beginner'
    elif any(word in text_lower for word in ['native', 'fluent']):
        return 'expert'
    elif any(word in text_lower for word in ['conversational', 'working']):
        return 'intermediate'
    
    return 'intermediate'

def calculate_skill_confidence(text: str, skill: str, context: str) -> float:
    """Calculate confidence score for skill extraction"""
    confidence = 0.5
    
    # Higher confidence if in skills section
    if re.search(r'\bskills?\b|\btechnologies\b|\bcompetencies\b', context, re.IGNORECASE):
        confidence += 0.2
    
    # Higher confidence if skill appears multiple times
    skill_count = len(re.findall(re.escape(skill), text, re.IGNORECASE))
    if skill_count > 1:
        confidence += min(0.2, skill_count * 0.05)
    
    # Higher confidence if surrounded by other known skills
    surrounding_text = context.lower()
    known_skills_count = sum(1 for known_skill in ALL_SKILLS 
                           if known_skill in surrounding_text and known_skill != skill)
    if known_skills_count >= 2:
        confidence += 0.1
    
    # Lower confidence for very common words that might not be skills
    common_words = {'work', 'experience', 'good', 'knowledge', 'understanding'}
    if skill.lower() in common_words:
        confidence -= 0.3
    
    return min(1.0, max(0.0, confidence))

def is_valid_skill(skill: str) -> bool:
    """Check if extracted text is likely a valid skill"""
    skill_lower = skill.lower().strip()
    
    # Skip empty or very short strings
    if len(skill_lower) < 2:
        return False
    
    # Skip common non-skill words
    non_skills = {
        'and', 'or', 'the', 'with', 'for', 'in', 'on', 'at', 'by', 'from', 'to', 'of',
        'experience', 'knowledge', 'understanding', 'working', 'good', 'excellent',
        'years', 'year', 'months', 'month', 'level', 'skill', 'skills', 'ability',
        'strong', 'proficient', 'familiar', 'basic', 'advanced', 'expert'
    }
    
    if skill_lower in non_skills:
        return False
    
    # Check if it's in our known skills database or looks like a valid technical term
    if (skill_lower in ALL_SKILLS or 
        any(skill_lower in alias_list for alias_list in PROGRAMMING_LANGUAGE_ALIASES.values()) or
        re.match(r'^[a-zA-Z][a-zA-Z0-9+#.\-_]{1,25}$', skill_lower)):
        return True
    
    return False

def categorize_extracted_skills(skills: List[str]) -> Dict[str, List[str]]:
    """Categorize extracted skills into predefined categories"""
    categorized = {category: [] for category in SKILLS_DATABASE.keys()}
    categorized['uncategorized'] = []
    
    for skill in skills:
        skill_lower = skill.lower()
        categorized_flag = False
        
        for category, category_skills in SKILLS_DATABASE.items():
            if skill_lower in category_skills:
                categorized[category].append(skill)
                categorized_flag = True
                break
        
        if not categorized_flag:
            # Check aliases
            for main_lang, aliases in PROGRAMMING_LANGUAGE_ALIASES.items():
                if skill_lower in aliases and main_lang in SKILLS_DATABASE['programming_languages']:
                    categorized['programming_languages'].append(main_lang)
                    categorized_flag = True
                    break
        
        if not categorized_flag:
            categorized['uncategorized'].append(skill)
    
    # Remove empty categories
    return {k: v for k, v in categorized.items() if v}

def extract_comprehensive_skills(text: str) -> Dict[str, Any]:
    """
    Main function to extract all types of skills from resume text
    """
    # Detect different skill sections
    sections = detect_skills_sections(text)
    
    comprehensive_skills = {
        'by_section': {},
        'all_skills': {
            'raw_skills': [],
            'categorized_skills': {},
            'skill_levels': {},
            'confidence_scores': {},
            'statistics': {}
        },
        'sections_found': list(sections.keys()),
        'extraction_summary': {}
    }
    
    # Extract skills from each detected section
    for section_type, section_instances in sections.items():
        comprehensive_skills['by_section'][section_type] = []
        
        for start_line, end_line, section_text in section_instances:
            section_skills = extract_skills_from_text(section_text, section_type)
            comprehensive_skills['by_section'][section_type].append({
                'location': f"lines {start_line}-{end_line}",
                'skills_data': section_skills
            })
            
            # Add to overall skills
            for skill in section_skills['raw_skills']:
                if skill not in comprehensive_skills['all_skills']['raw_skills']:
                    comprehensive_skills['all_skills']['raw_skills'].append(skill)
                    comprehensive_skills['all_skills']['skill_levels'][skill] = section_skills['skill_levels'].get(skill, 'intermediate')
                    comprehensive_skills['all_skills']['confidence_scores'][skill] = section_skills['confidence_scores'].get(skill, 0.5)
    
    # Extract from full text if no specific sections found
    if not sections or len(comprehensive_skills['all_skills']['raw_skills']) < 5:
        full_text_skills = extract_skills_from_text(text, 'general')
        
        for skill in full_text_skills['raw_skills']:
            if skill not in comprehensive_skills['all_skills']['raw_skills']:
                comprehensive_skills['all_skills']['raw_skills'].append(skill)
                comprehensive_skills['all_skills']['skill_levels'][skill] = full_text_skills['skill_levels'].get(skill, 'intermediate')
                comprehensive_skills['all_skills']['confidence_scores'][skill] = full_text_skills['confidence_scores'].get(skill, 0.5)
    
    # Final categorization
    comprehensive_skills['all_skills']['categorized_skills'] = categorize_extracted_skills(
        comprehensive_skills['all_skills']['raw_skills']
    )
    
    # Generate statistics
    comprehensive_skills['all_skills']['statistics'] = {
        'total_skills': len(comprehensive_skills['all_skills']['raw_skills']),
        'technical_skills': len([s for category in ['programming_languages', 'web_frontend', 'web_backend', 'databases', 'cloud_platforms', 'devops_infrastructure', 'data_science', 'machine_learning', 'mobile_development', 'testing_qa'] 
                                for s in comprehensive_skills['all_skills']['categorized_skills'].get(category, [])]),
        'soft_skills': len([s for category in ['leadership_management', 'communication_interpersonal', 'analytical_problem_solving', 'organizational_productivity'] 
                          for s in comprehensive_skills['all_skills']['categorized_skills'].get(category, [])]),
        'languages': len(comprehensive_skills['all_skills']['categorized_skills'].get('human_languages', [])),
        'certifications': len(comprehensive_skills['all_skills']['categorized_skills'].get('certifications', [])),
        'skill_levels': {
            level: len([s for s, l in comprehensive_skills['all_skills']['skill_levels'].items() if l == level])
            for level in ['beginner', 'intermediate', 'advanced', 'expert']
        },
        'high_confidence_skills': len([s for s, conf in comprehensive_skills['all_skills']['confidence_scores'].items() if conf >= 0.8])
    }
    
    return comprehensive_skills

# Comprehensive degree patterns covering all formats
DEGREE_PATTERNS = {
    'doctorate': [
        r'\b(?:Ph\.?D\.?|Doctor\s+of\s+Philosophy|Doctorate|D\.?Phil\.?|Ed\.?D\.?|Doctor\s+of\s+Education|'
        r'M\.?D\.?|Doctor\s+of\s+Medicine|J\.?D\.?|Juris\s+Doctor|D\.?Sc\.?|Doctor\s+of\s+Science|'
        r'D\.?B\.?A\.?|Doctor\s+of\s+Business\s+Administration|Psy\.?D\.?|Doctor\s+of\s+Psychology|'
        r'D\.?M\.?A\.?|Doctor\s+of\s+Musical\s+Arts|Pharm\.?D\.?|Doctor\s+of\s+Pharmacy|'
        r'D\.?P\.?T\.?|Doctor\s+of\s+Physical\s+Therapy|D\.?V\.?M\.?|Doctor\s+of\s+Veterinary\s+Medicine)\b'
    ],
    'masters': [
        r'\b(?:M\.?S\.?|Master\s+of\s+Science|M\.?A\.?|Master\s+of\s+Arts|M\.?B\.?A\.?|Master\s+of\s+Business\s+Administration|'
        r'M\.?Eng\.?|Master\s+of\s+Engineering|M\.?Ed\.?|Master\s+of\s+Education|M\.?F\.?A\.?|Master\s+of\s+Fine\s+Arts|'
        r'M\.?P\.?H\.?|Master\s+of\s+Public\s+Health|M\.?S\.?W\.?|Master\s+of\s+Social\s+Work|'
        r'M\.?P\.?A\.?|Master\s+of\s+Public\s+Administration|LL\.?M\.?|Master\s+of\s+Laws|'
        r'M\.?Sc\.?|M\.?Phil\.?|Master\s+of\s+Philosophy|M\.?Tech\.?|Master\s+of\s+Technology|'
        r'M\.?C\.?A\.?|Master\s+of\s+Computer\s+Applications|M\.?I\.?M\.?|Master\s+in\s+Management|'
        r'M\.?F\.?in\.?|Master\s+of\s+Finance|M\.?Acc\.?|Master\s+of\s+Accountancy|'
        r'M\.?Arch\.?|Master\s+of\s+Architecture|M\.?L\.?I\.?S\.?|Master\s+of\s+Library\s+Science|'
        r'M\.?M\.?|Master\s+of\s+Management|M\.?I\.?T\.?|Master\s+of\s+Information\s+Technology)\b'
    ],
    'bachelors': [
        r'\b(?:B\.?S\.?|Bachelor\s+of\s+Science|B\.?A\.?|Bachelor\s+of\s+Arts|B\.?Eng\.?|Bachelor\s+of\s+Engineering|'
        r'B\.?Tech\.?|Bachelor\s+of\s+Technology|B\.?B\.?A\.?|Bachelor\s+of\s+Business\s+Administration|'
        r'B\.?Ed\.?|Bachelor\s+of\s+Education|B\.?F\.?A\.?|Bachelor\s+of\s+Fine\s+Arts|'
        r'B\.?Sc\.?|B\.?Com\.?|Bachelor\s+of\s+Commerce|B\.?C\.?A\.?|Bachelor\s+of\s+Computer\s+Applications|'
        r'LL\.?B\.?|Bachelor\s+of\s+Laws|B\.?Arch\.?|Bachelor\s+of\s+Architecture|'
        r'B\.?Pharm\.?|Bachelor\s+of\s+Pharmacy|B\.?D\.?S\.?|Bachelor\s+of\s+Dental\s+Surgery|'
        r'B\.?V\.?Sc\.?|Bachelor\s+of\s+Veterinary\s+Science|B\.?N\.?|Bachelor\s+of\s+Nursing|'
        r'B\.?M\.?|Bachelor\s+of\s+Medicine|B\.?S\.?W\.?|Bachelor\s+of\s+Social\s+Work|'
        r'B\.?Mus\.?|Bachelor\s+of\s+Music|B\.?Des\.?|Bachelor\s+of\s+Design)\b'
    ],
    'associate': [
        r'\b(?:A\.?S\.?|Associate\s+of\s+Science|A\.?A\.?|Associate\s+of\s+Arts|'
        r'A\.?A\.?S\.?|Associate\s+of\s+Applied\s+Science|A\.?S\.?N\.?|Associate\s+of\s+Science\s+in\s+Nursing)\b'
    ],
    'diploma': [
        r'\b(?:Diploma|Advanced\s+Diploma|Graduate\s+Diploma|Postgraduate\s+Diploma|'
        r'Professional\s+Diploma|Higher\s+Diploma|National\s+Diploma)\b'
    ],
    'certificate': [
        r'\b(?:Certificate|Professional\s+Certificate|Graduate\s+Certificate|Postgraduate\s+Certificate|'
        r'Advanced\s+Certificate|Specialist\s+Certificate)\b'
    ],
    'high_school': [
        r'\b(?:High\s+School|Secondary\s+School|Higher\s+Secondary|Senior\s+Secondary|'
        r'12th\s+(?:Grade|Standard)|Class\s+12|Grade\s+12|HSC|SSC|CBSE|ICSE|'
        r'A[-\s]?Levels?|O[-\s]?Levels?|GCE|GCSE|Matriculation|Intermediate)\b'
    ]
}

# Field of study patterns
FIELD_OF_STUDY_PATTERNS = [
    r'\b(?:in|of|for)\s+([A-Z][A-Za-z\s&,\-]{2,60}?)(?:\s+(?:from|at|,|\.|$))',
    r'\b(?:major|majoring|specialization|specializing|concentration|focus)(?:\s+in)?\s*:?\s*([A-Z][A-Za-z\s&,\-]{2,60}?)(?:\s+(?:from|at|,|\.|$))',
]

# Institution patterns
INSTITUTION_PATTERNS = [
    r'\b([A-Z][A-Za-z\s&\'\-\.]{3,80}?)\s+(?:University|College|Institute|School|Academy)\b',
    r'\b(?:University|College|Institute|School|Academy)\s+of\s+([A-Z][A-Za-z\s&\'\-\.]{3,60})\b',
    r'\b([A-Z]{2,}(?:\s+[A-Z]{2,})*)\s*[-–—]\s*(?:University|College|Institute)\b',
]

# GPA patterns
GPA_PATTERNS = [
    r'\b(?:GPA|CGPA|Grade\s+Point\s+Average)[\s:]*(\d+\.?\d*)\s*(?:/|out\s+of)?\s*(\d+\.?\d*)?\b',
    r'\b(\d+\.?\d*)\s*/\s*(\d+\.?\d*)\s+(?:GPA|CGPA)\b',
    r'\b(?:GPA|CGPA)[\s:]*(\d+\.?\d*)\b',
]

# Date patterns for education
EDUCATION_DATE_PATTERNS = [
    r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\s*[-–—]\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|Present|Current|Ongoing)\b',
    r'\b(\d{4})\s*[-–—]\s*(\d{4}|Present|Current|Ongoing)\b',
    r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b',
    r'\b(?:Graduated|Graduation|Completed)[\s:]*([A-Z][a-z]+\.?\s+\d{4}|\d{4})\b',
    r'\b(?:Expected|Anticipated)[\s:]*([A-Z][a-z]+\.?\s+\d{4}|\d{4})\b',
]

# Education section headers
EDUCATION_SECTION_HEADERS = [
    r'\b(?:EDUCATION|ACADEMIC\s+(?:BACKGROUND|QUALIFICATIONS?|HISTORY)|EDUCATIONAL\s+(?:BACKGROUND|QUALIFICATIONS?)|'
    r'ACADEMIC\s+CREDENTIALS|QUALIFICATIONS?|DEGREES?|CERTIFICATIONS?\s+&\s+EDUCATION)\b'
]

# Honors and achievements patterns
HONORS_PATTERNS = [
    r'\b(?:Honors?|Honours?|Distinction|Cum\s+Laude|Magna\s+Cum\s+Laude|Summa\s+Cum\s+Laude|'
    r'First\s+Class|Second\s+Class|Dean\'?s\s+List|Merit|Scholarship|Award|Prize)\b'
]

# Compile all patterns
COMPILED_DEGREE_PATTERNS = {}
for degree_type, patterns in DEGREE_PATTERNS.items():
    COMPILED_DEGREE_PATTERNS[degree_type] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

COMPILED_FIELD_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in FIELD_OF_STUDY_PATTERNS]
COMPILED_INSTITUTION_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in INSTITUTION_PATTERNS]
COMPILED_GPA_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in GPA_PATTERNS]
COMPILED_EDUCATION_DATE_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in EDUCATION_DATE_PATTERNS]
COMPILED_EDUCATION_HEADERS = [re.compile(pattern, re.IGNORECASE) for pattern in EDUCATION_SECTION_HEADERS]
COMPILED_HONORS_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in HONORS_PATTERNS]


def detect_education_sections(text: str) -> List[Tuple[int, int, str]]:
    """
    Detect education sections in resume text
    Returns: List of (start_pos, end_pos, section_text) tuples
    """
    sections = []
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean:
            continue
        
        # Check if line is an education section header
        for pattern in COMPILED_EDUCATION_HEADERS:
            if pattern.search(line_clean):
                start_line = i
                end_line = start_line + 1
                
                # Find the end of this section
                for j in range(i + 1, min(len(lines), i + 50)):
                    next_line = lines[j].strip()
                    if not next_line:
                        continue
                    
                    # Check if this is another major section header
                    if re.match(r'^[A-Z\s]{3,30}$', next_line) and len(next_line) > 5:
                        common_sections = ['EXPERIENCE', 'WORK', 'EMPLOYMENT', 'SKILLS', 'PROJECTS', 
                                         'CERTIFICATIONS', 'PUBLICATIONS', 'AWARDS', 'INTERESTS']
                        if any(section in next_line.upper() for section in common_sections):
                            end_line = j
                            break
                    
                    end_line = j + 1
                
                section_lines = lines[start_line:end_line]
                section_text = '\n'.join(section_lines)
                sections.append((start_line, end_line, section_text))
                break
    
    return sections


def extract_degree_info(text: str) -> List[Dict[str, Any]]:
    """Extract degree information from text with comprehensive pattern matching"""
    degrees = []
    
    for degree_type, patterns in COMPILED_DEGREE_PATTERNS.items():
        for pattern in patterns:
            matches = pattern.finditer(text)
            for match in matches:
                degree_text = match.group(0)
                start_pos = match.start()
                end_pos = match.end()
                
                context_start = max(0, start_pos - 200)
                context_end = min(len(text), end_pos + 200)
                context = text[context_start:context_end]
                
                degree_info = {
                    'degree_type': degree_type,
                    'degree': degree_text.strip(),
                    'context': context,
                    'position': (start_pos, end_pos)
                }
                
                degrees.append(degree_info)
    
    return degrees


def extract_field_of_study(context: str, degree_text: str) -> Optional[str]:
    """Extract field of study from context around degree mention"""
    for pattern in COMPILED_FIELD_PATTERNS:
        match = pattern.search(context)
        if match:
            field = match.group(1).strip()
            field = re.sub(r'\s+', ' ', field)
            field = field.rstrip('.,;:')
            if len(field) > 3 and len(field) < 100:
                return field
    
    in_pattern = re.search(r'\b(?:in|of)\s+([A-Z][A-Za-z\s&,\-]{2,60}?)$', degree_text, re.IGNORECASE)
    if in_pattern:
        field = in_pattern.group(1).strip()
        if len(field) > 3:
            return field
    
    return None


def extract_institution(context: str) -> Optional[str]:
    """Extract institution name from context"""
    for pattern in COMPILED_INSTITUTION_PATTERNS:
        match = pattern.search(context)
        if match:
            institution = match.group(0).strip()
            institution = re.sub(r'\s+', ' ', institution)
            institution = institution.rstrip('.,;:')
            
            false_positives = ['Bachelor of Science', 'Master of Arts', 'High School']
            if not any(fp.lower() in institution.lower() for fp in false_positives):
                if len(institution) > 5 and len(institution) < 150:
                    return institution
    
    return None


def extract_gpa(context: str) -> Optional[Dict[str, float]]:
    """Extract GPA information from context"""
    for pattern in COMPILED_GPA_PATTERNS:
        match = pattern.search(context)
        if match:
            try:
                gpa_value = float(match.group(1))
                gpa_scale = float(match.group(2)) if match.lastindex >= 2 and match.group(2) else None
                
                if gpa_scale:
                    if 0 <= gpa_value <= gpa_scale:
                        return {'value': gpa_value, 'scale': gpa_scale}
                else:
                    if 0 <= gpa_value <= 4.0:
                        return {'value': gpa_value, 'scale': 4.0}
                    elif 0 <= gpa_value <= 10.0:
                        return {'value': gpa_value, 'scale': 10.0}
                    elif 0 <= gpa_value <= 100.0:
                        return {'value': gpa_value, 'scale': 100.0}
            except (ValueError, IndexError):
                continue
    
    return None


def extract_education_dates(context: str) -> Dict[str, Optional[str]]:
    """Extract start and end dates from education context"""
    dates = {'start_date': None, 'end_date': None, 'is_current': False}
    
    for pattern in COMPILED_EDUCATION_DATE_PATTERNS:
        match = pattern.search(context)
        if match:
            if match.lastindex >= 2:
                dates['start_date'] = match.group(1).strip()
                end_date = match.group(2).strip()
                
                if end_date.lower() in ['present', 'current', 'ongoing']:
                    dates['is_current'] = True
                    dates['end_date'] = 'Present'
                else:
                    dates['end_date'] = end_date
                break
            elif match.lastindex == 1:
                single_date = match.group(1).strip()
                
                if 'expected' in context.lower() or 'anticipated' in context.lower():
                    dates['end_date'] = single_date
                    dates['is_current'] = True
                else:
                    dates['end_date'] = single_date
                break
    
    return dates


def extract_honors_and_achievements(context: str) -> List[str]:
    """Extract honors and achievements from education context"""
    honors = []
    
    for pattern in COMPILED_HONORS_PATTERNS:
        matches = pattern.finditer(context)
        for match in matches:
            honor = match.group(0).strip()
            if honor not in honors:
                honors.append(honor)
    
    return honors


def extract_location_from_education(context: str) -> Optional[str]:
    """Extract location (city, state, country) from education context"""
    location_patterns = [
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z]{2})\b',
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z][a-z]+)\b',
        r'\b([A-Z][a-z]+),\s*([A-Z][a-z]+),\s*([A-Z]{2,})\b',
    ]
    
    for pattern in location_patterns:
        match = re.search(pattern, context)
        if match:
            return match.group(0).strip()
    
    return None


def merge_education_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge duplicate or overlapping education entries"""
    if not entries:
        return []
    
    entries.sort(key=lambda x: x.get('position', (0, 0))[0])
    
    merged = []
    skip_indices = set()
    
    for i, entry in enumerate(entries):
        if i in skip_indices:
            continue
        
        for j in range(i + 1, len(entries)):
            if j in skip_indices:
                continue
            
            other = entries[j]
            pos1 = entry.get('position', (0, 0))
            pos2 = other.get('position', (0, 0))
            
            if abs(pos1[0] - pos2[0]) < 300:
                if not entry.get('field_of_study') and other.get('field_of_study'):
                    entry['field_of_study'] = other['field_of_study']
                if not entry.get('institution') and other.get('institution'):
                    entry['institution'] = other['institution']
                if not entry.get('gpa') and other.get('gpa'):
                    entry['gpa'] = other['gpa']
                if not entry.get('start_date') and other.get('start_date'):
                    entry['start_date'] = other['start_date']
                if not entry.get('end_date') and other.get('end_date'):
                    entry['end_date'] = other['end_date']
                if not entry.get('location') and other.get('location'):
                    entry['location'] = other['location']
                if other.get('honors'):
                    entry['honors'] = list(set(entry.get('honors', []) + other['honors']))
                
                skip_indices.add(j)
        
        merged.append(entry)
    
    return merged


def extract_education_with_nlp(text: str) -> List[Dict[str, Any]]:
    """Extract education information using NLP and spaCy (if available)"""
    education_entries = []
    
    if not _NLP:
        return education_entries
    
    try:
        doc = _NLP(text[:100000])
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            education_keywords = ['university', 'college', 'bachelor', 'master', 'phd', 'degree', 
                                'graduated', 'studied', 'major', 'diploma', 'certificate']
            
            if any(keyword in sent_text.lower() for keyword in education_keywords):
                entry = {'raw_text': sent_text}
                
                orgs = [ent.text for ent in sent.ents if ent.label_ == 'ORG']
                if orgs:
                    entry['institution'] = orgs[0]
                
                dates = [ent.text for ent in sent.ents if ent.label_ == 'DATE']
                if dates:
                    entry['dates'] = dates
                
                locations = [ent.text for ent in sent.ents if ent.label_ in ['GPE', 'LOC']]
                if locations:
                    entry['location'] = ', '.join(locations)
                
                if len(entry) > 1:
                    education_entries.append(entry)
    
    except Exception as e:
        print(f"NLP extraction error: {e}")
    
    return education_entries


def extract_education_comprehensive(text: str) -> List[Dict[str, Any]]:
    """
    Comprehensive education information extraction using multiple NLP techniques
    
    This function covers all possible resume formats including:
    - Traditional degree formats (BS, MS, PhD, etc.)
    - International degree formats
    - Diploma and certificate programs
    - High school education
    - Online courses and certifications
    - Multiple date formats
    - GPA in various scales
    - Honors and achievements
    - Institution names and locations
    
    Args:
        text: Resume text to extract education information from
    
    Returns:
        List of dictionaries containing structured education information with fields:
        - degree: The degree name (e.g., "Bachelor of Science")
        - degree_type: Category (bachelors, masters, doctorate, etc.)
        - field_of_study: Major/specialization (e.g., "Computer Science")
        - institution: University/college name
        - location: City, state, country
        - gpa: Dictionary with 'value' and 'scale'
        - start_date: Start date of education
        - end_date: End/graduation date
        - is_current: Boolean indicating if currently enrolled
        - honors: List of honors and achievements
    """
    education_entries = []
    
    # Step 1: Detect education sections
    education_sections = detect_education_sections(text)
    
    # If no explicit section found, use entire text
    if not education_sections:
        education_sections = [(0, len(text), text)]
    
    # Step 2: Extract education information from each section
    for start, end, section_text in education_sections:
        degrees = extract_degree_info(section_text)
        
        for degree_info in degrees:
            context = degree_info['context']
            
            entry = {
                'degree': degree_info['degree'],
                'degree_type': degree_info['degree_type'],
                'field_of_study': extract_field_of_study(context, degree_info['degree']),
                'institution': extract_institution(context),
                'gpa': extract_gpa(context),
                'location': extract_location_from_education(context),
                'honors': extract_honors_and_achievements(context),
                'raw_text': context[:300],
            }
            
            date_info = extract_education_dates(context)
            entry.update(date_info)
            entry['position'] = degree_info['position']
            
            education_entries.append(entry)
    
    # Step 3: Use NLP-based extraction as supplementary
    if _NLP:
        nlp_entries = extract_education_with_nlp(text)
        
        for nlp_entry in nlp_entries:
            is_duplicate = False
            for existing in education_entries:
                if (nlp_entry.get('institution') and 
                    existing.get('institution') and 
                    nlp_entry['institution'].lower() in existing['institution'].lower()):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                nlp_entry['extraction_method'] = 'nlp'
                education_entries.append(nlp_entry)
    
    # Step 4: Merge duplicate entries
    education_entries = merge_education_entries(education_entries)
    
    # Step 5: Clean up and format final entries
    final_entries = []
    for entry in education_entries:
        entry.pop('position', None)
        cleaned_entry = {k: v for k, v in entry.items() if v}
        
        if cleaned_entry.get('degree') or cleaned_entry.get('institution'):
            final_entries.append(cleaned_entry)
    
    # Step 6: Sort by date (most recent first)
    def get_sort_key(entry):
        end_date = entry.get('end_date', '')
        if not end_date or end_date == 'Present':
            return 9999
        
        year_match = re.search(r'\d{4}', str(end_date))
        if year_match:
            return int(year_match.group(0))
        return 0
    
    final_entries.sort(key=get_sort_key, reverse=True)
    
    return final_entries


# Backward compatibility - alias for the main function
extract_education_info = extract_education_comprehensive

# ---------- Utility Functions ----------
def file_sha256(file_path: Path) -> str:
    """Calculate SHA256 hash of file"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def normalize_phone_number(phone: str) -> Optional[str]:
    """Normalize phone number using phonenumbers library"""
    try:
        parsed = phonenumbers.parse(phone, "US")  # Default to US
        if phonenumbers.is_valid_number(parsed):
            return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    except:
        pass
    return phone

def parse_date_with_dateparser(date_str: str) -> Optional[str]:
    """Parse date string using dateparser"""
    try:
        parsed_date = dateparser.parse(date_str)
        if parsed_date:
            return parsed_date.strftime("%Y-%m-%d")
    except:
        pass
    return date_str

def _is_plausible_match(url: str, platform: str) -> bool:
    """Check if a URL could plausibly belong to a platform based on domain or common patterns"""
    url_lower = url.lower()
    
    # Common domain patterns that would exclude certain platforms
    if platform == 'linkedin' and 'linkedin' not in url_lower:
        # Could still be valid if it's a redirect or short URL
        return not any(domain in url_lower for domain in ['github.com', 'twitter.com', 'instagram.com', 'facebook.com'])
    elif platform == 'github' and 'github' not in url_lower:
        return not any(domain in url_lower for domain in ['linkedin.com', 'twitter.com', 'instagram.com', 'facebook.com'])
    elif platform == 'twitter' and not any(x in url_lower for x in ['twitter', 'x.com']):
        return not any(domain in url_lower for domain in ['linkedin.com', 'github.com', 'instagram.com', 'facebook.com'])
    
    # If URL contains the platform name, it's very likely a match
    if platform.lower() in url_lower:
        return True
    
    # For generic labels like "website", "visit", accept most URLs
    if platform in ['website', 'portfolio']:
        return True
    
    # If no obvious conflicts, allow the match
    return True

def _extract_username_from_url(url: str, platform: str) -> Optional[str]:
    """Try to extract username from URL for a given platform"""
    if platform in HYPERLINK_PATTERNS:
        match = HYPERLINK_PATTERNS[platform].search(url)
        if match and match.groups():
            return match.group(1)
    
    # Fallback: try to extract username from common URL patterns
    try:
        # Remove protocol and www
        clean_url = re.sub(r'^https?://(www\.)?', '', url)
        # Split by / and take relevant parts
        parts = clean_url.split('/')
        if len(parts) >= 2:
            return parts[1]  # Usually the username part
    except:
        pass
    
    return None

def extract_and_classify_hyperlinks(doc: fitz.Document) -> Dict[str, Any]:
    """
    Extract hyperlinks from PDF using PyMuPDF and classify them by type
    
    Args:
        doc: PyMuPDF document object
        
    Returns:
        Dictionary with classified hyperlinks by type and page
    """
    
    hyperlinks_data = {
        'by_type': {
            'email': [],
            'linkedin': [],
            'github': [],
            'twitter': [],
            'kaggle': [],
            'medium': [],
            'stackoverflow': [],
            'behance': [],
            'dribbble': [],
            'portfolio': [],
            'youtube': [],
            'instagram': [],
            'facebook': [],
            'website': [],
            'other': []
        },
        'by_page': {},
        'all_links': [],
        'statistics': {
            'total_links': 0,
            'pages_with_links': 0,
            'most_common_type': None
        }
    }
    
    pages_with_links = 0
    
    try:
        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_links = []
            
            # Extract hyperlinks using the method you provided
            links = page.get_links()
            
            for link in links:
                if "uri" in link:
                    uri = link["uri"]
                    if uri and isinstance(uri, str):
                        page_links.append(uri)
                        
                        # Extract display text from the link rectangle area
                        display_text = ""
                        if "from" in link:
                            try:
                                # Get the rectangle coordinates
                                rect = link["from"]
                                # Extract text from the rectangle area
                                text_instances = page.get_text("dict", clip=rect)
                                if text_instances and "blocks" in text_instances:
                                    for block in text_instances["blocks"]:
                                        if "lines" in block:
                                            for line in block["lines"]:
                                                if "spans" in line:
                                                    for span in line["spans"]:
                                                        if "text" in span:
                                                            display_text += span["text"] + " "
                                display_text = display_text.strip()
                            except:
                                pass
                        
                        hyperlinks_data['all_links'].append({
                            'url': uri,
                            'display_text': display_text,
                            'page': page_num + 1,
                            'type': None  # Will be classified below
                        })
            
            # Also extract hyperlinks from annotations (alternative method)
            try:
                for annot in page.annots():
                    if annot.type[1] == 'Link':  # Link annotation
                        link_data = annot.info.get('content', '') or annot.info.get('title', '')
                        if link_data and 'http' in link_data:
                            page_links.append(link_data)
                            
                            # Try to get display text from annotation
                            display_text = annot.info.get('title', '') or ""
                            
                            hyperlinks_data['all_links'].append({
                                'url': link_data,
                                'display_text': display_text,
                                'page': page_num + 1,
                                'type': None
                            })
            except:
                pass  # Annotations might not be accessible
            
            if page_links:
                pages_with_links += 1
                hyperlinks_data['by_page'][page_num + 1] = page_links
        
        # Classify all extracted hyperlinks using both URL and text patterns
        for link_entry in hyperlinks_data['all_links']:
            url = link_entry['url']
            display_text = link_entry.get('display_text', '').lower().strip()
            classified = False
            username = None
            
            # First, try to classify by URL pattern (most reliable)
            for link_type, pattern in HYPERLINK_PATTERNS.items():
                match = pattern.search(url)
                if match:
                    username = match.group(1) if match.groups() else None
                    hyperlinks_data['by_type'][link_type].append({
                        'url': url,
                        'display_text': link_entry.get('display_text', ''),
                        'page': link_entry['page'],
                        'username': username,
                        'classification_method': 'url_pattern'
                    })
                    link_entry['type'] = link_type
                    classified = True
                    break
            
            # If not classified by URL, try text label patterns
            if not classified and display_text:
                for platform, text_patterns in COMPILED_TEXT_PATTERNS.items():
                    for text_pattern in text_patterns:
                        if text_pattern.search(display_text):
                            # Double-check that the URL could plausibly match this platform
                            if _is_plausible_match(url, platform):
                                hyperlinks_data['by_type'][platform].append({
                                    'url': url,
                                    'display_text': link_entry.get('display_text', ''),
                                    'page': link_entry['page'],
                                    'username': _extract_username_from_url(url, platform),
                                    'classification_method': 'text_pattern'
                                })
                                link_entry['type'] = platform
                                classified = True
                                break
                    if classified:
                        break
            
            # If still not classified, mark as other
            if not classified:
                hyperlinks_data['by_type']['other'].append({
                    'url': url,
                    'display_text': link_entry.get('display_text', ''),
                    'page': link_entry['page'],
                    'username': None,
                    'classification_method': 'unclassified'
                })
                link_entry['type'] = 'other'
        
        # Calculate statistics
        hyperlinks_data['statistics']['total_links'] = len(hyperlinks_data['all_links'])
        hyperlinks_data['statistics']['pages_with_links'] = pages_with_links
        
        # Find most common link type
        type_counts = {
            link_type: len(links) 
            for link_type, links in hyperlinks_data['by_type'].items() 
            if links
        }
        
        if type_counts:
            most_common_type = max(type_counts.items(), key=lambda x: x[1])
            hyperlinks_data['statistics']['most_common_type'] = {
                'type': most_common_type[0],
                'count': most_common_type[1]
            }
    
    except Exception as e:
        print(f"Error extracting hyperlinks: {e}")
    
    return hyperlinks_data

# ---------- PyMuPDF-based PDF Extraction ----------
def extract_pdf_with_pymupdf(file_path: Path) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    Extract text, links, and metadata from PDF using PyMuPDF (fitz) with enhanced hyperlink classification
    
    Returns:
        tuple: (full_text, urls, metadata)
    """
    full_text = []
    urls = []
    page_metadata = {}
    
    try:
        doc = fitz.open(str(file_path))
        
        # Enhanced hyperlink extraction and classification
        hyperlinks_data = extract_and_classify_hyperlinks(doc)
        
        # Document metadata
        metadata = {
            "page_count": doc.page_count,
            "metadata": doc.metadata,
            "is_encrypted": doc.is_encrypted,
            "is_pdf": doc.is_pdf,
            "hyperlinks": hyperlinks_data,
            "page_details": {}
        }
        
        # Collect all URLs for backward compatibility
        urls = [link['url'] for link in hyperlinks_data['all_links']]
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            
            # Extract text with layout information
            text = page.get_text()
            if text.strip():
                full_text.append(text)
            
            # Get page-specific hyperlinks from our classification
            page_links = hyperlinks_data['by_page'].get(page_num + 1, [])
            
            # Extract images metadata (for future OCR if needed)
            images = page.get_images()
            
            # Extract text blocks with position information
            text_blocks = page.get_text("dict")
            
            page_metadata[page_num] = {
                "links": page_links,
                "link_count": len(page_links),
                "image_count": len(images),
                "text_blocks": len(text_blocks.get("blocks", [])),
                "page_rect": list(page.rect)
            }
        
        doc.close()
        
        # Extract comprehensive skills from the full text
        full_text_str = "\n".join(full_text)
        comprehensive_skills = extract_comprehensive_skills(full_text_str)
        
        metadata["page_details"] = page_metadata
        metadata["skills"] = comprehensive_skills
        
        return full_text_str, urls, metadata
        
    except Exception as e:
        print(f"Error extracting PDF {file_path}: {e}")
        return "", [], {}

def extract_docx_content(file_path: Path) -> Tuple[str, List[str]]:
    """Extract text and hyperlinks from DOCX file"""
    try:
        doc = DocxDocument(str(file_path))
        
        # Extract paragraphs
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text.strip())
        
        # Extract table content
        tables = []
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    tables.append(" | ".join(cells))
        
        # Extract hyperlinks
        urls = []
        rels = getattr(doc.part, "rels", {})
        for rel in rels.values():
            try:
                if "hyperlink" in rel.reltype and rel.target_ref:
                    urls.append(str(rel.target_ref))
            except:
                continue
        
        full_text = "\n".join(paragraphs + tables)
        return full_text, urls
        
    except Exception as e:
        print(f"Error extracting DOCX {file_path}: {e}")
        return "", []

# ---------- Information Extraction Functions ----------
def extract_emails(text: str) -> List[str]:
    """Extract and validate email addresses"""
    emails = EMAIL_PATTERN.findall(text)
    # Additional validation
    valid_emails = []
    for email in emails:
        if "@" in email and "." in email.split("@")[-1]:
            valid_emails.append(email.lower())
    return list(set(valid_emails))

def extract_phone_numbers(text: str) -> List[str]:
    """Extract and normalize phone numbers"""
    phones = PHONE_PATTERN.findall(text)
    normalized_phones = []
    
    for phone in phones:
        # Clean the phone number
        cleaned = re.sub(r'[^\d+]', '', phone)
        if len(cleaned) >= 10:  # Valid phone number length
            normalized = normalize_phone_number(phone)
            if normalized:
                normalized_phones.append(normalized)
    
    return list(set(normalized_phones))

def extract_urls_and_social_links(text: str, hyperlinks_data: Optional[Dict] = None) -> Dict[str, List[str]]:
    """
    Extract URLs and categorize social media links from text and hyperlinks
    
    Args:
        text: Text content to search for URLs
        hyperlinks_data: Optional classified hyperlinks from PDF extraction
    """
    # Extract URLs from text using regex
    text_urls = URL_PATTERN.findall(text)
    
    social_links = {platform: [] for platform in SOCIAL_PATTERNS.keys()}
    social_links['other_urls'] = []
    social_links['email'] = []
    social_links['portfolio'] = []
    social_links['website'] = []
    
    # Process URLs found in text
    for url in text_urls:
        categorized = False
        for platform, pattern in SOCIAL_PATTERNS.items():
            match = pattern.search(url)
            if match:
                social_links[platform].append(url)
                categorized = True
                break
        
        if not categorized:
            # Check for other types
            if EMAIL_PATTERN.search(url):
                social_links['email'].append(url)
            elif any(domain in url.lower() for domain in ['.portfolio', '.me', '.dev', '.website']):
                social_links['portfolio'].append(url)
            else:
                social_links['other_urls'].append(url)
    
    # If we have classified hyperlinks data from PDF, merge it
    if hyperlinks_data and isinstance(hyperlinks_data, dict):
        hyperlink_types = hyperlinks_data.get('by_type', {})
        
        for link_type, links in hyperlink_types.items():
            if links and link_type in social_links:
                # Extract just the URLs from the link objects
                urls_to_add = [link['url'] for link in links if isinstance(link, dict) and 'url' in link]
                social_links[link_type].extend(urls_to_add)
            elif links and link_type == 'other':
                urls_to_add = [link['url'] for link in links if isinstance(link, dict) and 'url' in link]
                social_links['other_urls'].extend(urls_to_add)
    
    # Remove duplicates and clean up
    for key in social_links:
        social_links[key] = list(set(social_links[key]))
    
    return social_links

def extract_dates(text: str) -> List[Dict[str, str]]:
    """Extract and normalize dates"""
    dates = []
    
    for pattern in DATE_PATTERNS:
        matches = pattern.findall(text)
        for match in matches:
            normalized_date = parse_date_with_dateparser(match)
            dates.append({
                "original": match,
                "normalized": normalized_date
            })
    
    return dates

def extract_names_advanced(text: str) -> List[str]:
    """Advanced name extraction using multiple strategies"""
    names = set()
    lines = text.split('\n')
    
    # Strategy 1: Look in first few lines
    for i, line in enumerate(lines[:10]):
        line = line.strip()
        if len(line) < 3 or len(line) > 100:
            continue
        
        # Skip lines with common resume keywords
        skip_keywords = [
            'resume', 'cv', 'curriculum', 'vitae', 'profile', 'summary', 
            'contact', 'email', 'phone', 'address', 'skills', 'experience'
        ]
        
        if any(keyword in line.lower() for keyword in skip_keywords):
            continue
        
        # Try name patterns
        for pattern in NAME_PATTERNS:
            match = pattern.match(line)
            if match:
                candidate_name = match.group(1).strip()
                words = candidate_name.split()
                if 2 <= len(words) <= 4 and all(len(w) >= 2 for w in words):
                    names.add(candidate_name)
    
    # Strategy 2: Extract from email usernames
    emails = extract_emails(text)
    for email in emails:
        username = email.split('@')[0]
        if '.' in username:
            parts = username.split('.')[:2]  # Take first two parts
            if all(part.isalpha() and len(part) >= 2 for part in parts):
                name = ' '.join(part.title() for part in parts)
                names.add(name)
    
    # Strategy 3: Use spaCy NER if available
    if _NLP:
        doc = _NLP(text[:5000])  # Limit for performance
        for ent in doc.ents:
            if ent.label_ == "PERSON" and len(ent.text.strip()) > 3:
                names.add(ent.text.strip())
    
    return list(names)

def extract_skills_flashtext(text: str) -> Dict[str, List[str]]:
    """Extract skills using FlashText for fast keyword matching"""
    if _KEYWORD_PROCESSOR:
        found_skills = _KEYWORD_PROCESSOR.extract_keywords(text.lower())
        
        # Categorize skills
        categorized_skills = {category: [] for category in SKILLS_DATABASE.keys()}
        
        for skill in found_skills:
            for category, skills_set in SKILLS_DATABASE.items():
                if skill in skills_set:
                    categorized_skills[category].append(skill)
        
        return categorized_skills
    
    return {}

def extract_skills_fuzzy(text: str, threshold: int = 80) -> Dict[str, List[str]]:
    """Extract skills using fuzzy matching with RapidFuzz"""
    if not _HAS_RAPIDFUZZ:
        return {}
    
    words = re.findall(r'\b\w+\b', text.lower())
    text_phrases = []
    
    # Create 1-3 word phrases
    for i in range(len(words)):
        text_phrases.append(words[i])
        if i < len(words) - 1:
            text_phrases.append(f"{words[i]} {words[i+1]}")
        if i < len(words) - 2:
            text_phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
    
    found_skills = {category: [] for category in SKILLS_DATABASE.keys()}
    
    for category, skills_set in SKILLS_DATABASE.items():
        for skill in skills_set:
            # Find best match
            best_match = process.extractOne(
                skill, 
                text_phrases, 
                scorer=fuzz.partial_ratio,
                score_cutoff=threshold
            )
            
            if best_match:
                found_skills[category].append(skill)
    
    return found_skills

def extract_organizations_and_locations(text: str) -> Tuple[List[str], List[str]]:
    """Extract organizations and locations using NER"""
    organizations = []
    locations = []
    
    if _NLP:
        doc = _NLP(text[:10000])  # Limit for performance
        
        for ent in doc.ents:
            if ent.label_ == "ORG":
                organizations.append(ent.text.strip())
            elif ent.label_ in ["GPE", "LOC"]:
                locations.append(ent.text.strip())
    
    # Also use transformer-based NER if available
    if _NER_PIPELINE:
        try:
            # Process in chunks
            chunk_size = 500
            for i in range(0, min(len(text), 5000), chunk_size):
                chunk = text[i:i+chunk_size]
                entities = _NER_PIPELINE(chunk)
                
                for entity in entities:
                    if entity['entity_group'] == 'ORG':
                        organizations.append(entity['word'].strip())
                    elif entity['entity_group'] == 'LOC':
                        locations.append(entity['word'].strip())
        except Exception as e:
            print(f"Error in transformer NER: {e}")
    
    return list(set(organizations)), list(set(locations))

def calculate_readability_stats(text: str) -> Dict[str, float]:
    """Calculate readability statistics"""
    if not _HAS_TEXTSTAT:
        return {}
    
    try:
        stats = {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            "automated_readability_index": textstat.automated_readability_index(text),
            "coleman_liau_index": textstat.coleman_liau_index(text),
            "word_count": len(text.split()),
            "sentence_count": textstat.sentence_count(text),
            "avg_sentence_length": textstat.avg_sentence_length(text),
            "syllable_count": textstat.syllable_count(text),
        }
        return stats
    except Exception as e:
        print(f"Error calculating readability: {e}")
        return {}

def extract_experience_years(text: str) -> List[int]:
    """Extract years of experience mentions"""
    experience_patterns = [
        r'(\d+)[\s]*(?:\+|\-)?[\s]*(?:years?|yrs?|year)[\s]*(?:of[\s]*)?(?:experience|exp)',
        r'(?:experience|exp)[\s]*(?:of[\s]*)?(\d+)[\s]*(?:\+|\-)?[\s]*(?:years?|yrs?|year)',
        r'(\d+)[\s]*(?:\+|\-)?[\s]*(?:years?|yrs?)[\s]*(?:in|with|of)',
    ]
    
    years = []
    for pattern in experience_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if match.isdigit():
                year_val = int(match)
                if 0 <= year_val <= 50:  # Reasonable range
                    years.append(year_val)
    
    return sorted(set(years))

def extract_education_info(text: str) -> List[Dict[str, str]]:
    """Extract education information"""
    education_patterns = [
        r'(?:Bachelor|Master|PhD|B\.?S\.?|M\.?S\.?|B\.?A\.?|M\.?A\.?)\s*(?:of|in|degree)?\s*([A-Za-z\s]{3,50})',
        r'([A-Za-z\s]{3,50})\s*(?:degree|diploma|certificate)',
        r'(?:University|College|Institute)\s+of\s+([A-Za-z\s]{3,50})',
    ]
    
    education = []
    for pattern in education_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            education.append({
                "degree": match.strip(),
                "type": "inferred"
            })
    
    return education

# ---------- Main Extraction Function ----------
def extract_comprehensive_entities(text: str, hyperlinks_data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Comprehensive entity extraction using multiple techniques
    
    Args:
        text: Text content to analyze
        hyperlinks_data: Optional classified hyperlinks from PDF extraction
    """
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
    
    # Basic extractions
    emails = extract_emails(text)
    phones = extract_phone_numbers(text)
    social_data = extract_urls_and_social_links(text, hyperlinks_data)
    dates = extract_dates(text)
    names = extract_names_advanced(text)
    
    # Enhanced comprehensive skills extraction
    comprehensive_skills = extract_comprehensive_skills(text)
    
    # Legacy skills extraction for backward compatibility
    legacy_skills = extract_skills_flashtext(text)
    if not any(legacy_skills.values()) and _HAS_RAPIDFUZZ:
        legacy_skills = extract_skills_fuzzy(text)
    
    # NER extractions
    organizations, locations = extract_organizations_and_locations(text)
    
    # Additional extractions
    experience_years = extract_experience_years(text)
    education = extract_education_info(text)
    readability = calculate_readability_stats(text)
    
    # Compile results
    entities = {
        "names": names,
        "emails": emails,
        "phones": phones,
        "social_links": social_data,
        "dates": dates,
        "skills": comprehensive_skills,
        "legacy_skills": legacy_skills,
        "organizations": organizations,
        "locations": locations,
        "experience_years": experience_years,
        "education": education,
        "readability_stats": readability,
        "processing_info": {
            "text_length": len(text),
            "spacy_available": _NLP is not None,
            "transformers_available": _HAS_TRANSFORMERS,
            "flashtext_available": _HAS_FLASHTEXT,
            "rapidfuzz_available": _HAS_RAPIDFUZZ,
            "textstat_available": _HAS_TEXTSTAT,
        }
    }
    
    return entities

# ---------- Semantic Similarity (if sentence-transformers available) ----------
def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts"""
    if _HAS_SENTENCE_TRANSFORMERS and _SENTENCE_MODEL:
        try:
            embeddings = _SENTENCE_MODEL.encode([text1, text2])
            similarity = float(_SENTENCE_MODEL.similarity(embeddings[0], embeddings[1]))
            return similarity
        except Exception as e:
            print(f"Error calculating similarity: {e}")
    
    return 0.0

def extract_semantic_chunks(text: str, chunk_size: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
    """Extract semantically meaningful chunks with embeddings"""
    chunks = []
    
    # Split text into sentences first
    sentences = re.split(r'[.!?]+', text)
    
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sentence_length = len(sentence.split())
        
        if current_length + sentence_length > chunk_size and current_chunk:
            # Create chunk
            chunk_text = '. '.join(current_chunk)
            chunk_info = {
                "text": chunk_text,
                "word_count": len(chunk_text.split()),
                "sentence_count": len(current_chunk)
            }
            
            # Add embeddings if available
            if _HAS_SENTENCE_TRANSFORMERS and _SENTENCE_MODEL:
                try:
                    embedding = _SENTENCE_MODEL.encode(chunk_text)
                    chunk_info["embedding"] = embedding.tolist()
                except:
                    pass
            
            chunks.append(chunk_info)
            
            # Handle overlap
            overlap_sentences = current_chunk[-overlap//50:] if overlap > 0 else []
            current_chunk = overlap_sentences + [sentence]
            current_length = sum(len(s.split()) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    # Add final chunk
    if current_chunk:
        chunk_text = '. '.join(current_chunk)
        chunk_info = {
            "text": chunk_text,
            "word_count": len(chunk_text.split()),
            "sentence_count": len(current_chunk)
        }
        
        if _HAS_SENTENCE_TRANSFORMERS and _SENTENCE_MODEL:
            try:
                embedding = _SENTENCE_MODEL.encode(chunk_text)
                chunk_info["embedding"] = embedding.tolist()
            except:
                pass
        
        chunks.append(chunk_info)
    
    return chunks

# ---------- Main Processing Function ----------
def process_documents_with_pymupdf(file_paths: List[Path]) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process documents using PyMuPDF with comprehensive information extraction
    
    Args:
        file_paths: List of file paths to process
        
    Returns:
        tuple: (chunks, chunk_metadata, document_records)
    """
    
    file_paths = [p for p in file_paths if p.suffix.lower() in ALLOWED_EXTENSIONS]
    if not file_paths:
        return [], [], []
    
    chunks = []
    chunk_metadata = []
    document_records = []
    
    for file_path in file_paths:
        if not file_path.exists():
            continue
        
        print(f"Processing: {file_path.name}")
        
        # Extract content based on file type
        if file_path.suffix.lower() == ".pdf":
            full_text, urls, pdf_metadata = extract_pdf_with_pymupdf(file_path)
            hyperlinks_data = pdf_metadata.get('hyperlinks', {})
        elif file_path.suffix.lower() in {".doc", ".docx"}:
            full_text, urls = extract_docx_content(file_path)
            pdf_metadata = {}
            hyperlinks_data = {}
        else:
            continue
        
        if not full_text.strip():
            print(f"No text extracted from {file_path.name}")
            continue
        
        # Extract comprehensive entities with hyperlink data
        entities = extract_comprehensive_entities(full_text, hyperlinks_data)
        
        # Create semantic chunks
        semantic_chunks = extract_semantic_chunks(full_text)
        
        # Process each chunk
        for i, chunk_info in enumerate(semantic_chunks):
            chunk_text = chunk_info["text"]
            
            # Extract chunk-specific entities (pass hyperlinks for consistency)
            chunk_entities = extract_comprehensive_entities(chunk_text, hyperlinks_data)
            
            # Create chunk metadata
            chunk_meta = {
                "file": file_path.name,
                "path": str(file_path),
                "chunk_index": i,
                "total_chunks": len(semantic_chunks),
                "char_count": len(chunk_text),
                "word_count": chunk_info["word_count"],
                "sentence_count": chunk_info["sentence_count"],
                "entities": chunk_entities,
                "urls": urls,
                "has_embedding": "embedding" in chunk_info
            }
            
            if "embedding" in chunk_info:
                chunk_meta["embedding"] = chunk_info["embedding"]
            
            chunks.append(chunk_text)
            chunk_metadata.append(chunk_meta)
        
        # Create document record
        document_record = {
            "file": file_path.name,
            "path": str(file_path),
            "file_hash": file_sha256(file_path),
            "file_size": file_path.stat().st_size,
            "total_chunks": len(semantic_chunks),
            "entities": entities,
            "urls": urls,
            "pdf_metadata": pdf_metadata,
            "processing_timestamp": None,  # Can be added later if needed
        }
        
        document_records.append(document_record)
    
    print(f"Processed {len(document_records)} documents into {len(chunks)} chunks")
    return chunks, chunk_metadata, document_records

# ---------- Export Function for Compatibility ----------
def extract_docs_to_chunks_and_records(paths: List[Path]) -> Tuple[List[str], List[Dict[str,Any]], List[Dict[str,Any]]]:
    """
    Compatibility function with the original parsing interface
    """
    return process_documents_with_pymupdf(paths)