"""
Comprehensive Project Information Extractor for Resumes

This module provides advanced NLP-based project extraction covering all possible resume formats:
- Academic projects
- Professional/Work projects
- Personal/Side projects
- Open source contributions
- Freelance projects
- Research projects
- Capstone projects
- Hackathon projects
- Client projects
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Try to import spaCy for NLP-based extraction
try:
    import spacy
    try:
        _NLP = spacy.load("en_core_web_trf")
    except OSError:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            _NLP = None
except ImportError:
    _NLP = None

# Try to import FlashText for fast keyword extraction
try:
    from flashtext import KeywordProcessor
    _FLASHTEXT_AVAILABLE = True
except ImportError:
    _FLASHTEXT_AVAILABLE = False


# ---------- Project Extraction Configuration ----------

# Project section headers
PROJECT_SECTION_HEADERS = [
    r'\b(?:PROJECTS?|ACADEMIC\s+PROJECTS?|PROFESSIONAL\s+PROJECTS?|PERSONAL\s+PROJECTS?|'
    r'KEY\s+PROJECTS?|MAJOR\s+PROJECTS?|NOTABLE\s+PROJECTS?|SELECTED\s+PROJECTS?|'
    r'PROJECT\s+(?:EXPERIENCE|WORK|PORTFOLIO)|TECHNICAL\s+PROJECTS?|'
    r'OPEN\s+SOURCE\s+(?:PROJECTS?|CONTRIBUTIONS?)|SIDE\s+PROJECTS?|'
    r'FREELANCE\s+PROJECTS?|CLIENT\s+PROJECTS?|RESEARCH\s+PROJECTS?|'
    r'CAPSTONE\s+PROJECTS?|PORTFOLIO\s+PROJECTS?)\b'
]

# Project title patterns (various formats)
PROJECT_TITLE_PATTERNS = [
    # Bold/Capitalized title at start of line
    r'^([A-Z][A-Za-z0-9\s\-:&,\.]{3,80})(?:\s*[-–—|]\s*|\s*\n)',
    # Title with pipe or dash separator
    r'(?:Project|Title)[\s:]*([A-Z][A-Za-z0-9\s\-&,\.]{3,80})(?:\s*[-–—|]|\s*\n)',
    # Quoted or emphasized titles
    r'["\']([A-Z][A-Za-z0-9\s\-&,\.]{3,60})["\']',
]

# Project role patterns
PROJECT_ROLE_PATTERNS = [
    r'\b(?:Role|Position|As)[\s:]*([A-Z][A-Za-z\s]{3,40}?)(?:\s*[-–—|,\n]|$)',
    r'\b(Team\s+Lead|Project\s+Lead|Lead\s+Developer|Senior\s+Developer|Developer|'
    r'Software\s+Engineer|Full\s+Stack\s+Developer|Frontend\s+Developer|Backend\s+Developer|'
    r'Data\s+Scientist|ML\s+Engineer|DevOps\s+Engineer|Architect|Designer|'
    r'Product\s+Manager|Scrum\s+Master|Technical\s+Lead|Contributor)\b',
]

# Technology/Stack patterns
TECHNOLOGY_PATTERNS = [
    r'\b(?:Technologies?|Tech\s+Stack|Stack|Tools?|Built\s+(?:with|using)|Developed\s+(?:with|using|in))[\s:]*([A-Za-z0-9\s,|/&+#.\-]{10,200})',
    r'\b(?:Languages?|Frameworks?|Libraries?)[\s:]*([A-Za-z0-9\s,|/&+#.\-]{10,150})',
    r'\b(?:Using|With)[\s:]*([A-Z][A-Za-z0-9\s,|/&+#.\-]{5,100})',
]

# Project date patterns
PROJECT_DATE_PATTERNS = [
    # Month Year - Month Year
    r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\s*[-–—]\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|Present|Current|Ongoing)\b',
    # Year - Year
    r'\b(\d{4})\s*[-–—]\s*(\d{4}|Present|Current|Ongoing)\b',
    # Single date
    r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b',
    r'\b(\d{4})\b',
    # Month/Year format
    r'\b(\d{1,2}/\d{4})\s*[-–—]\s*(\d{1,2}/\d{4}|Present)\b',
]

# Project duration patterns
PROJECT_DURATION_PATTERNS = [
    r'\b(\d+)\s*(?:months?|mos?\.?)\b',
    r'\b(\d+)\s*(?:weeks?|wks?\.?)\b',
    r'\b(\d+)\s*(?:years?|yrs?\.?)\b',
]

# Team size patterns
TEAM_SIZE_PATTERNS = [
    r'\b(?:Team\s+(?:of|size)|Team)[\s:]*(\d+)(?:\s*(?:members?|people|developers?))?\b',
    r'\b(\d+)[-\s]member\s+team\b',
    r'\b(\d+)[-\s]person\s+team\b',
]

# Project URL/Link patterns
PROJECT_URL_PATTERNS = [
    r'(?:GitHub|Repository|Repo|Source|Code|Demo|Live|Link|URL|Website|Project\s+Link)[\s:]*(?:https?://[^\s]+)',
    r'https?://(?:github\.com|gitlab\.com|bitbucket\.org)/[^\s]+',
    r'https?://[^\s]+',
]

# Project description indicators
DESCRIPTION_INDICATORS = [
    r'\b(?:Description|Summary|Overview|About|Details?)[\s:]*',
    r'\b(?:Developed|Created|Built|Designed|Implemented|Engineered)\b',
]

# Achievement/Impact patterns
ACHIEVEMENT_PATTERNS = [
    r'\b(?:Achieved|Improved|Increased|Reduced|Decreased|Optimized|Enhanced|Delivered)\s+[^.]{10,150}',
    r'\b(?:\d+%|\d+x)\s+(?:improvement|increase|reduction|faster|better|more)',
    r'\b(?:Successfully|Effectively)\s+[^.]{10,100}',
]

# Project type indicators
PROJECT_TYPE_PATTERNS = {
    'web_application': [
        r'\b(?:Web\s+(?:Application|App|Platform|Portal|Site)|Website|Web-based)\b',
        r'\b(?:E-commerce|Dashboard|CMS|Blog|Forum)\b'
    ],
    'mobile_application': [
        r'\b(?:Mobile\s+(?:Application|App)|Android\s+App|iOS\s+App|Cross-platform\s+App)\b',
        r'\b(?:Flutter|React\s+Native|Ionic|Xamarin)\s+(?:App|Application)\b'
    ],
    'machine_learning': [
        r'\b(?:Machine\s+Learning|ML|Deep\s+Learning|AI|Artificial\s+Intelligence)\s+(?:Project|Model|System)\b',
        r'\b(?:Neural\s+Network|CNN|RNN|NLP|Computer\s+Vision|Predictive\s+Model)\b'
    ],
    'data_analysis': [
        r'\b(?:Data\s+Analysis|Data\s+Analytics|Data\s+Visualization|BI\s+Dashboard)\b',
        r'\b(?:ETL|Data\s+Pipeline|Data\s+Warehouse)\b'
    ],
    'api_backend': [
        r'\b(?:REST\s+API|RESTful\s+API|GraphQL\s+API|Backend\s+API|Microservice)\b',
        r'\b(?:API\s+Development|Backend\s+Service)\b'
    ],
    'devops_infrastructure': [
        r'\b(?:DevOps|CI/CD|Infrastructure|Deployment|Automation)\s+(?:Project|Pipeline|System)\b',
        r'\b(?:Docker|Kubernetes|Terraform|Jenkins)\s+(?:Implementation|Setup)\b'
    ],
    'game_development': [
        r'\b(?:Game|Gaming)\s+(?:Development|Project|Application)\b',
        r'\b(?:Unity|Unreal\s+Engine|Godot)\s+Game\b'
    ],
    'iot': [
        r'\b(?:IoT|Internet\s+of\s+Things|Embedded\s+System|Hardware)\s+Project\b',
        r'\b(?:Arduino|Raspberry\s+Pi|ESP32)\s+(?:Project|Based)\b'
    ],
    'blockchain': [
        r'\b(?:Blockchain|Cryptocurrency|Smart\s+Contract|DApp|Web3)\s+(?:Project|Application)\b',
        r'\b(?:Ethereum|Solidity|NFT)\s+(?:Project|Platform)\b'
    ],
    'research': [
        r'\b(?:Research\s+Project|Academic\s+Research|Thesis|Dissertation|Paper)\b',
        r'\b(?:Published|Presented)\s+(?:at|in)\b'
    ]
}

# Compile patterns
COMPILED_PROJECT_SECTION_HEADERS = [re.compile(pattern, re.IGNORECASE) for pattern in PROJECT_SECTION_HEADERS]
COMPILED_PROJECT_TITLE_PATTERNS = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in PROJECT_TITLE_PATTERNS]
COMPILED_PROJECT_ROLE_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in PROJECT_ROLE_PATTERNS]
COMPILED_TECHNOLOGY_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in TECHNOLOGY_PATTERNS]
COMPILED_PROJECT_DATE_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in PROJECT_DATE_PATTERNS]
COMPILED_PROJECT_DURATION_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in PROJECT_DURATION_PATTERNS]
COMPILED_TEAM_SIZE_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in TEAM_SIZE_PATTERNS]
COMPILED_PROJECT_URL_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in PROJECT_URL_PATTERNS]
COMPILED_DESCRIPTION_INDICATORS = [re.compile(pattern, re.IGNORECASE) for pattern in DESCRIPTION_INDICATORS]
COMPILED_ACHIEVEMENT_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in ACHIEVEMENT_PATTERNS]

COMPILED_PROJECT_TYPE_PATTERNS = {}
for project_type, patterns in PROJECT_TYPE_PATTERNS.items():
    COMPILED_PROJECT_TYPE_PATTERNS[project_type] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

# Initialize FlashText processor for O(n) technology matching
_TECHNOLOGY_PROCESSOR = None
if _FLASHTEXT_AVAILABLE:
    _TECHNOLOGY_PROCESSOR = KeywordProcessor(case_sensitive=False)
    
    # Add common technologies for fast extraction
    technologies = [
        'python', 'javascript', 'java', 'react', 'angular', 'vue', 'node.js',
        'django', 'flask', 'spring', 'docker', 'kubernetes', 'aws', 'azure',
        'postgresql', 'mongodb', 'redis', 'tensorflow', 'pytorch', 'scikit-learn'
    ]
    
    for tech in technologies:
        _TECHNOLOGY_PROCESSOR.add_keyword(tech)


def detect_project_sections(text: str) -> List[Tuple[int, int, str]]:
    """
    Detect project sections in resume text
    Returns: List of (start_line, end_line, section_text) tuples
    """
    sections = []
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean:
            continue
        
        # Check if line is a project section header
        for pattern in COMPILED_PROJECT_SECTION_HEADERS:
            if pattern.search(line_clean):
                start_line = i
                end_line = start_line + 1
                
                # Find the end of this section
                for j in range(i + 1, min(len(lines), i + 100)):  # Look ahead max 100 lines
                    next_line = lines[j].strip()
                    if not next_line:
                        continue
                    
                    # Check if this is another major section header
                    if re.match(r'^[A-Z\s]{3,30}$', next_line) and len(next_line) > 5:
                        common_sections = ['EDUCATION', 'EXPERIENCE', 'WORK', 'EMPLOYMENT', 'SKILLS', 
                                         'CERTIFICATIONS', 'PUBLICATIONS', 'AWARDS', 'INTERESTS', 'SUMMARY']
                        if any(section in next_line.upper() for section in common_sections):
                            end_line = j
                            break
                    
                    end_line = j + 1
                
                section_lines = lines[start_line:end_line]
                section_text = '\n'.join(section_lines)
                sections.append((start_line, end_line, section_text))
                break
    
    return sections


def split_into_individual_projects(text: str) -> List[str]:
    """
    Split project section text into individual project entries
    Uses various heuristics to identify project boundaries
    """
    projects = []
    lines = text.split('\n')
    
    current_project = []
    in_project = False
    
    for i, line in enumerate(lines):
        line_clean = line.strip()
        
        # Skip empty lines at the start
        if not line_clean and not in_project:
            continue
        
        # Check if this line starts a new project
        is_new_project = False
        
        # Heuristic 1: Line starts with bullet point or number
        if re.match(r'^(?:•|●|○|◦|▪|▫|–|—|\*|\d+\.|\d+\))\s+[A-Z]', line_clean):
            is_new_project = True
        
        # Heuristic 2: Line is all caps or title case and short (likely a title)
        elif re.match(r'^[A-Z][A-Za-z0-9\s\-:&,\.]{3,80}$', line_clean) and len(line_clean) < 100:
            # Check if it looks like a project title
            if any(keyword in line_clean.lower() for keyword in ['project', 'system', 'application', 'platform', 'tool', 'website', 'app', 'portal', 'dashboard']):
                is_new_project = True
            # Or if next line has project indicators
            elif i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if any(keyword in next_line.lower() for keyword in ['role:', 'technologies:', 'description:', 'developed', 'built', 'created']):
                    is_new_project = True
        
        # Heuristic 3: Line starts with "Project:" or similar
        elif re.match(r'^(?:Project|Title)[\s:]+', line_clean, re.IGNORECASE):
            is_new_project = True
        
        # If we found a new project and we have accumulated content, save it
        if is_new_project and current_project:
            project_text = '\n'.join(current_project)
            if len(project_text.strip()) > 20:  # Minimum length check
                projects.append(project_text)
            current_project = []
            in_project = True
        
        # Add line to current project
        if line_clean or in_project:  # Include empty lines if we're in a project
            current_project.append(line)
            in_project = True
    
    # Don't forget the last project
    if current_project:
        project_text = '\n'.join(current_project)
        if len(project_text.strip()) > 20:
            projects.append(project_text)
    
    return projects


def extract_project_title(text: str) -> Optional[str]:
    """Extract project title from text"""
    lines = text.split('\n')
    
    # Try first non-empty line (often the title)
    for line in lines[:3]:  # Check first 3 lines
        line_clean = line.strip()
        if line_clean:
            # Remove bullet points and numbers
            title = re.sub(r'^(?:•|●|○|◦|▪|▫|–|—|\*|\d+\.|\d+\))\s*', '', line_clean)
            # Remove "Project:" prefix if present
            title = re.sub(r'^(?:Project|Title)[\s:]+', '', title, flags=re.IGNORECASE)
            
            # Check if it's a reasonable title length
            if 3 < len(title) < 150:
                # Remove trailing separators
                title = re.sub(r'\s*[-–—|:]\s*$', '', title)
                return title.strip()
    
    # Try pattern matching
    for pattern in COMPILED_PROJECT_TITLE_PATTERNS:
        match = pattern.search(text)
        if match:
            title = match.group(1).strip()
            if 3 < len(title) < 150:
                return title
    
    return None


def extract_project_role(text: str) -> Optional[str]:
    """Extract role/position in the project"""
    for pattern in COMPILED_PROJECT_ROLE_PATTERNS:
        match = pattern.search(text)
        if match:
            role = match.group(1).strip() if match.lastindex >= 1 else match.group(0).strip()
            role = re.sub(r'\s+', ' ', role)
            if 3 < len(role) < 60:
                return role
    
    return None


def extract_technologies(text: str) -> List[str]:
    """Extract technologies/tools used in the project"""
    technologies = []
    
    for pattern in COMPILED_TECHNOLOGY_PATTERNS:
        match = pattern.search(text)
        if match:
            tech_text = match.group(1).strip()
            
            # Split by common separators
            tech_list = re.split(r'[,;|/&]|\s+and\s+', tech_text)
            
            for tech in tech_list:
                tech_clean = tech.strip()
                # Remove parentheses and extra spaces
                tech_clean = re.sub(r'[()]', '', tech_clean)
                tech_clean = re.sub(r'\s+', ' ', tech_clean)
                
                # Validate technology name
                if 2 < len(tech_clean) < 50 and not re.match(r'^\d+$', tech_clean):
                    technologies.append(tech_clean)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_technologies = []
    for tech in technologies:
        tech_lower = tech.lower()
        if tech_lower not in seen:
            seen.add(tech_lower)
            unique_technologies.append(tech)
    
    return unique_technologies[:20]  # Limit to 20 technologies


def extract_project_dates(text: str) -> Dict[str, Optional[str]]:
    """Extract start and end dates from project text"""
    dates = {'start_date': None, 'end_date': None, 'is_current': False}
    
    for pattern in COMPILED_PROJECT_DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            if match.lastindex >= 2:
                # Date range found
                dates['start_date'] = match.group(1).strip()
                end_date = match.group(2).strip()
                
                if end_date.lower() in ['present', 'current', 'ongoing']:
                    dates['is_current'] = True
                    dates['end_date'] = 'Present'
                else:
                    dates['end_date'] = end_date
                break
            elif match.lastindex == 1:
                # Single date found (assume it's completion date)
                dates['end_date'] = match.group(1).strip()
    
    return dates


def extract_project_duration(text: str) -> Optional[str]:
    """Extract project duration"""
    for pattern in COMPILED_PROJECT_DURATION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0).strip()
    
    return None


def extract_team_size(text: str) -> Optional[int]:
    """Extract team size from project description"""
    for pattern in COMPILED_TEAM_SIZE_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                team_size = int(match.group(1))
                if 1 <= team_size <= 1000:  # Reasonable range
                    return team_size
            except (ValueError, IndexError):
                continue
    
    return None


def extract_project_urls(text: str) -> List[str]:
    """Extract project URLs (GitHub, demo links, etc.)"""
    urls = []
    
    for pattern in COMPILED_PROJECT_URL_PATTERNS:
        matches = pattern.finditer(text)
        for match in matches:
            # Extract just the URL part
            url_match = re.search(r'https?://[^\s]+', match.group(0))
            if url_match:
                url = url_match.group(0).rstrip('.,;:)')
                if url not in urls:
                    urls.append(url)
    
    return urls


def extract_project_description(text: str) -> Optional[str]:
    """Extract project description/summary"""
    # Remove title (first line)
    lines = text.split('\n')
    if len(lines) > 1:
        description_lines = lines[1:]
    else:
        description_lines = lines
    
    description_text = '\n'.join(description_lines).strip()
    
    # Remove metadata lines (role, technologies, dates)
    description_text = re.sub(r'(?:Role|Technologies?|Tech\s+Stack|Duration|Team|Link|URL)[\s:][^\n]+', '', description_text, flags=re.IGNORECASE)
    
    # Clean up
    description_text = re.sub(r'\n{3,}', '\n\n', description_text)
    description_text = description_text.strip()
    
    if len(description_text) > 20:
        return description_text[:1000]  # Limit length
    
    return None


def extract_achievements(text: str) -> List[str]:
    """Extract achievements/impact from project description"""
    achievements = []
    
    for pattern in COMPILED_ACHIEVEMENT_PATTERNS:
        matches = pattern.finditer(text)
        for match in matches:
            achievement = match.group(0).strip()
            if len(achievement) > 15 and achievement not in achievements:
                achievements.append(achievement)
    
    return achievements[:10]  # Limit to 10 achievements


def detect_project_type(text: str) -> List[str]:
    """Detect project type(s) from text"""
    project_types = []
    
    for project_type, patterns in COMPILED_PROJECT_TYPE_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text):
                project_types.append(project_type)
                break
    
    return project_types


def extract_projects_with_nlp(text: str) -> List[Dict[str, Any]]:
    """Extract project information using NLP and spaCy (if available)"""
    project_entries = []
    
    if not _NLP:
        return project_entries
    
    try:
        doc = _NLP(text[:100000])
        
        # Look for sentences with project keywords
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            project_keywords = ['project', 'developed', 'built', 'created', 'designed', 
                              'implemented', 'application', 'system', 'platform']
            
            if any(keyword in sent_text.lower() for keyword in project_keywords):
                entry = {'raw_text': sent_text}
                
                # Extract organizations (could be clients or companies)
                orgs = [ent.text for ent in sent.ents if ent.label_ == 'ORG']
                if orgs:
                    entry['organization'] = orgs[0]
                
                # Extract dates
                dates = [ent.text for ent in sent.ents if ent.label_ == 'DATE']
                if dates:
                    entry['dates'] = dates
                
                # Extract products/technologies
                products = [ent.text for ent in sent.ents if ent.label_ == 'PRODUCT']
                if products:
                    entry['technologies'] = products
                
                if len(entry) > 1:
                    project_entries.append(entry)
    
    except Exception as e:
        print(f"NLP project extraction error: {e}")
    
    return project_entries


def extract_projects_comprehensive(text: str) -> List[Dict[str, Any]]:
    """
    Comprehensive project extraction using multiple techniques
    
    This function covers all possible resume formats including:
    - Academic projects
    - Professional/Work projects
    - Personal/Side projects
    - Open source contributions
    - Freelance projects
    - Research projects
    - Capstone projects
    - Hackathon projects
    - Client projects
    
    Handles various formats:
    - Bullet point lists
    - Numbered lists
    - Paragraph format
    - Table format
    - Mixed formats
    
    Args:
        text: Resume text to extract project information from
    
    Returns:
        List of dictionaries containing structured project information with fields:
        - title: Project title/name
        - role: Role/position in the project
        - description: Project description
        - technologies: List of technologies used
        - start_date: Project start date
        - end_date: Project end date
        - is_current: Boolean indicating if currently working on it
        - duration: Project duration
        - team_size: Number of team members
        - urls: List of project URLs (GitHub, demo, etc.)
        - achievements: List of achievements/impacts
        - project_types: List of project type categories
    """
    project_entries = []
    
    # Step 1: Detect project sections
    project_sections = detect_project_sections(text)
    
    # If no explicit section found, try to find projects in entire text
    if not project_sections:
        # Look for project indicators in the text
        if any(keyword in text.lower() for keyword in ['project', 'developed', 'built', 'created']):
            project_sections = [(0, len(text), text)]
    
    # Step 2: Extract projects from each section
    for start, end, section_text in project_sections:
        # Split section into individual projects
        individual_projects = split_into_individual_projects(section_text)
        
        for project_text in individual_projects:
            entry = {
                'title': extract_project_title(project_text),
                'role': extract_project_role(project_text),
                'description': extract_project_description(project_text),
                'technologies': extract_technologies(project_text),
                'team_size': extract_team_size(project_text),
                'duration': extract_project_duration(project_text),
                'urls': extract_project_urls(project_text),
                'achievements': extract_achievements(project_text),
                'project_types': detect_project_type(project_text),
                'raw_text': project_text[:500],
            }
            
            # Extract dates
            date_info = extract_project_dates(project_text)
            entry.update(date_info)
            
            # Add FlashText technology extraction for O(n) speed
            if _FLASHTEXT_AVAILABLE and _TECHNOLOGY_PROCESSOR:
                flashtext_techs = _TECHNOLOGY_PROCESSOR.extract_keywords(project_text.lower())
                if flashtext_techs:
                    existing_techs = entry.get('technologies', [])
                    # Merge with existing technologies
                    all_techs = set(existing_techs + list(flashtext_techs))
                    entry['technologies'] = list(all_techs)
            
            project_entries.append(entry)
    
    # Step 3: Use NLP-based extraction as supplementary
    if _NLP and not project_entries:
        nlp_entries = extract_projects_with_nlp(text)
        
        for nlp_entry in nlp_entries:
            nlp_entry['extraction_method'] = 'nlp'
            project_entries.append(nlp_entry)
    
    # Step 4: Clean up and format final entries
    final_entries = []
    for entry in project_entries:
        # Remove empty fields
        cleaned_entry = {k: v for k, v in entry.items() if v}
        
        # Ensure at least title or description is present
        if cleaned_entry.get('title') or cleaned_entry.get('description'):
            final_entries.append(cleaned_entry)
    
    # Step 5: Sort by date (most recent first)
    def get_sort_key(entry):
        end_date = entry.get('end_date', '')
        if not end_date or end_date == 'Present':
            return 9999  # Current projects come first
        
        year_match = re.search(r'\d{4}', str(end_date))
        if year_match:
            return int(year_match.group(0))
        return 0
    
    final_entries.sort(key=get_sort_key, reverse=True)
    
    return final_entries


# Backward compatibility - alias for the main function
extract_projects_info = extract_projects_comprehensive
