# app/extractors/skills_extractor.py - Skills extraction functionality
import re
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter, defaultdict

# Try to import optional dependencies
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