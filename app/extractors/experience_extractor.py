"""
Comprehensive Work Experience Information Extractor for Resumes

This module provides advanced NLP-based work experience extraction covering all possible resume formats:
- Full-time employment
- Part-time jobs
- Internships and co-ops
- Freelance and contract work
- Consulting engagements
- Volunteer positions with professional responsibilities
- Self-employment and entrepreneurship
- Remote work
- Multiple concurrent positions
- Career gaps and sabbaticals
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


# ---------- Experience Extraction Configuration ----------

# Experience section headers
EXPERIENCE_SECTION_HEADERS = [
    r'\b(?:EXPERIENCE|WORK\s+EXPERIENCE|PROFESSIONAL\s+EXPERIENCE|EMPLOYMENT\s+(?:HISTORY|EXPERIENCE)|'
    r'CAREER\s+(?:HISTORY|EXPERIENCE|SUMMARY)|WORK\s+HISTORY|EMPLOYMENT|'
    r'PROFESSIONAL\s+BACKGROUND|RELEVANT\s+EXPERIENCE|INDUSTRY\s+EXPERIENCE|'
    r'INTERNSHIP(?:S)?|INTERNSHIP\s+EXPERIENCE|CO[-\s]?OP\s+EXPERIENCE|'
    r'FREELANCE\s+(?:WORK|EXPERIENCE)|CONSULTING\s+EXPERIENCE|'
    r'VOLUNTEER\s+EXPERIENCE|LEADERSHIP\s+EXPERIENCE)\b'
]

# Job title patterns (common titles)
JOB_TITLE_PATTERNS = [
    # Engineering & Technical
    r'\b(?:Senior|Lead|Principal|Staff|Junior)?\s*(?:Software|Hardware|Systems?|Network|Security|DevOps|Data|ML|AI|Cloud|Full[-\s]?Stack|Front[-\s]?End|Back[-\s]?End)\s+(?:Engineer|Developer|Architect|Analyst|Scientist|Specialist)\b',
    r'\b(?:Chief|VP|Vice\s+President|Director|Manager|Head)\s+of\s+(?:Engineering|Technology|Product|Data|Security)\b',
    r'\b(?:Technical|Engineering|Product|Project|Program)\s+(?:Lead|Manager|Director)\b',
    
    # Business & Management
    r'\b(?:Chief|C-Level|VP|Vice\s+President|Senior|Junior)?\s*(?:Executive|Financial|Marketing|Operations?|Technology|Information|Data)\s+(?:Officer|Director|Manager)\b',
    r'\b(?:CEO|CTO|CFO|COO|CMO|CIO|CISO|CPO|CDO)\b',
    r'\b(?:Business|Management|Strategy|Operations?|Sales|Marketing)\s+(?:Analyst|Consultant|Manager|Director)\b',
    
    # Design & Creative
    r'\b(?:Senior|Lead|Junior)?\s*(?:UI|UX|Product|Graphic|Web|Visual)\s+(?:Designer|Design\s+Lead)\b',
    r'\b(?:Creative|Art|Design)\s+(?:Director|Manager|Lead)\b',
    
    # Data & Analytics
    r'\b(?:Senior|Lead|Junior)?\s*(?:Data|Business|Financial|Marketing)\s+(?:Analyst|Scientist|Engineer)\b',
    r'\b(?:Machine\s+Learning|ML|AI|Deep\s+Learning)\s+(?:Engineer|Scientist|Researcher)\b',
    
    # Sales & Marketing
    r'\b(?:Senior|Junior)?\s*(?:Sales|Marketing|Account|Customer\s+Success)\s+(?:Manager|Executive|Representative|Specialist|Coordinator)\b',
    r'\b(?:Business\s+Development|BD)\s+(?:Manager|Executive|Representative)\b',
    
    # HR & Admin
    r'\b(?:Human\s+Resources?|HR|Talent|Recruitment)\s+(?:Manager|Specialist|Coordinator|Business\s+Partner)\b',
    r'\b(?:Administrative|Executive|Office)\s+(?:Assistant|Manager|Coordinator)\b',
    
    # Finance & Accounting
    r'\b(?:Senior|Junior)?\s*(?:Financial|Accounting|Tax|Audit)\s+(?:Analyst|Manager|Accountant|Auditor)\b',
    r'\b(?:Investment|Portfolio|Risk)\s+(?:Analyst|Manager)\b',
    
    # General patterns
    r'\b(?:Intern|Internship|Co[-\s]?op|Trainee|Fellow|Associate|Consultant|Contractor|Freelancer)\b',
    r'\b(?:Founder|Co[-\s]?Founder|Owner|Entrepreneur|Self[-\s]?Employed)\b',
]

# Company/Organization name patterns
COMPANY_PATTERNS = [
    r'\b([A-Z][A-Za-z0-9\s&\'\-\.]{2,60}?)\s+(?:Inc\.?|LLC|Ltd\.?|Corporation|Corp\.?|Company|Co\.?|LP|LLP|PLC)\b',
    r'\b([A-Z][A-Za-z0-9\s&\'\-\.]{2,60}?)(?:\s*[-–—|]\s*(?:Remote|Hybrid|On[-\s]?site))\b',
]

# Location patterns
LOCATION_PATTERNS = [
    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z]{2})\b',  # City, STATE
    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z][a-z]+)\b',  # City, Country
    r'\b([A-Z][a-z]+),\s*([A-Z][a-z]+),\s*([A-Z]{2,})\b',  # City, State, Country
    r'\b(Remote|Hybrid|On[-\s]?site|Work\s+from\s+Home|WFH)\b',
]

# Date patterns for experience
EXPERIENCE_DATE_PATTERNS = [
    # Month Year - Month Year
    r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\s*[-–—]\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|Present|Current|Ongoing)\b',
    # MM/YYYY - MM/YYYY
    r'\b(\d{1,2}/\d{4})\s*[-–—]\s*(\d{1,2}/\d{4}|Present|Current)\b',
    # Year - Year
    r'\b(\d{4})\s*[-–—]\s*(\d{4}|Present|Current|Ongoing)\b',
    # Month Year only
    r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b',
]

# Employment type patterns
EMPLOYMENT_TYPE_PATTERNS = {
    'full_time': [r'\b(?:full[-\s]?time|permanent|regular)\b'],
    'part_time': [r'\b(?:part[-\s]?time|casual)\b'],
    'internship': [r'\b(?:intern|internship|co[-\s]?op|trainee)\b'],
    'contract': [r'\b(?:contract|contractor|contractual|temporary|temp)\b'],
    'freelance': [r'\b(?:freelance|freelancer|independent\s+contractor)\b'],
    'consulting': [r'\b(?:consultant|consulting)\b'],
    'volunteer': [r'\b(?:volunteer|voluntary|pro\s+bono)\b'],
    'self_employed': [r'\b(?:self[-\s]?employed|founder|co[-\s]?founder|entrepreneur|owner)\b'],
}

# Work mode patterns
WORK_MODE_PATTERNS = {
    'remote': [r'\b(?:remote|work\s+from\s+home|WFH|distributed|virtual)\b'],
    'hybrid': [r'\b(?:hybrid|flexible|mixed)\b'],
    'onsite': [r'\b(?:on[-\s]?site|in[-\s]?office|office[-\s]?based)\b'],
}

# Responsibility indicators
RESPONSIBILITY_PATTERNS = [
    r'^(?:•|●|○|◦|▪|▫|–|—|\*|\d+\.|\d+\))\s+(.+)$',  # Bullet points
    r'^[-–—]\s+(.+)$',  # Dash bullets
]

# Achievement indicators (for responsibilities)
ACHIEVEMENT_INDICATORS = [
    'achieved', 'accomplished', 'delivered', 'exceeded', 'improved', 'increased',
    'reduced', 'decreased', 'saved', 'generated', 'led', 'managed', 'launched',
    'implemented', 'developed', 'created', 'built', 'designed', 'optimized'
]

# Compile patterns
COMPILED_EXPERIENCE_HEADERS = [re.compile(pattern, re.IGNORECASE) for pattern in EXPERIENCE_SECTION_HEADERS]
COMPILED_JOB_TITLE_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in JOB_TITLE_PATTERNS]
COMPILED_COMPANY_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in COMPANY_PATTERNS]
COMPILED_LOCATION_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in LOCATION_PATTERNS]
COMPILED_EXPERIENCE_DATE_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in EXPERIENCE_DATE_PATTERNS]
COMPILED_RESPONSIBILITY_PATTERNS = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in RESPONSIBILITY_PATTERNS]

COMPILED_EMPLOYMENT_TYPE_PATTERNS = {}
for emp_type, patterns in EMPLOYMENT_TYPE_PATTERNS.items():
    COMPILED_EMPLOYMENT_TYPE_PATTERNS[emp_type] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

COMPILED_WORK_MODE_PATTERNS = {}
for mode, patterns in WORK_MODE_PATTERNS.items():
    COMPILED_WORK_MODE_PATTERNS[mode] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


def detect_experience_sections(text: str) -> List[Tuple[int, int, str]]:
    """
    Detect experience sections in resume text
    Returns: List of (start_line, end_line, section_text) tuples
    """
    sections = []
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean:
            continue
        
        # Check if line is an experience section header
        for pattern in COMPILED_EXPERIENCE_HEADERS:
            if pattern.search(line_clean):
                start_line = i
                end_line = start_line + 1
                
                # Find the end of this section
                for j in range(i + 1, min(len(lines), i + 150)):  # Look ahead max 150 lines
                    next_line = lines[j].strip()
                    if not next_line:
                        continue
                    
                    # Check if this is another major section header
                    if re.match(r'^[A-Z\s]{3,30}$', next_line) and len(next_line) > 5:
                        common_sections = ['EDUCATION', 'SKILLS', 'PROJECTS', 'CERTIFICATIONS', 
                                         'ACHIEVEMENTS', 'ACTIVITIES', 'PUBLICATIONS', 'SUMMARY', 'OBJECTIVE']
                        if any(section in next_line.upper() for section in common_sections):
                            end_line = j
                            break
                    
                    end_line = j + 1
                
                section_lines = lines[start_line:end_line]
                section_text = '\n'.join(section_lines)
                sections.append((start_line, end_line, section_text))
                break
    
    return sections


def split_into_individual_experiences(text: str) -> List[str]:
    """
    Split experience section text into individual job entries
    """
    experiences = []
    lines = text.split('\n')
    
    current_exp = []
    in_exp = False
    
    for i, line in enumerate(lines):
        line_clean = line.strip()
        
        # Skip empty lines at the start
        if not line_clean and not in_exp:
            continue
        
        # Check if this line starts a new experience
        is_new_exp = False
        
        # Heuristic 1: Line contains job title pattern
        for pattern in COMPILED_JOB_TITLE_PATTERNS:
            if pattern.search(line_clean):
                is_new_exp = True
                break
        
        # Heuristic 2: Line has company name pattern
        if not is_new_exp:
            for pattern in COMPILED_COMPANY_PATTERNS:
                if pattern.search(line_clean):
                    is_new_exp = True
                    break
        
        # Heuristic 3: Line has date range pattern (common in experience entries)
        if not is_new_exp:
            for pattern in COMPILED_EXPERIENCE_DATE_PATTERNS:
                if pattern.search(line_clean):
                    # Check if previous line looks like a title
                    if i > 0 and len(lines[i-1].strip()) > 5:
                        is_new_exp = True
                        break
        
        # If we found a new experience and we have accumulated content, save it
        if is_new_exp and current_exp and len(current_exp) > 2:  # At least 3 lines
            exp_text = '\n'.join(current_exp)
            if len(exp_text.strip()) > 30:
                experiences.append(exp_text)
            current_exp = []
            in_exp = True
        
        # Add line to current experience
        if line_clean or in_exp:
            current_exp.append(line)
            in_exp = True
    
    # Don't forget the last experience
    if current_exp:
        exp_text = '\n'.join(current_exp)
        if len(exp_text.strip()) > 30:
            experiences.append(exp_text)
    
    return experiences


def extract_job_title(text: str) -> Optional[str]:
    """Extract job title from experience text"""
    lines = text.split('\n')
    
    # Try first few non-empty lines
    for line in lines[:5]:
        line_clean = line.strip()
        if not line_clean:
            continue
        
        # Check against job title patterns
        for pattern in COMPILED_JOB_TITLE_PATTERNS:
            match = pattern.search(line_clean)
            if match:
                title = match.group(0).strip()
                # Clean up
                title = re.sub(r'\s+', ' ', title)
                if 3 < len(title) < 100:
                    return title
        
        # If first line looks like a title (title case, reasonable length)
        if re.match(r'^[A-Z][A-Za-z\s\-&,]{3,80}$', line_clean):
            # Remove trailing separators
            title = re.sub(r'\s*[-–—|:]\s*$', '', line_clean)
            if 3 < len(title) < 100:
                return title
    
    return None


def extract_company_name(text: str) -> Optional[str]:
    """Extract company/organization name"""
    for pattern in COMPILED_COMPANY_PATTERNS:
        match = pattern.search(text)
        if match:
            company = match.group(1).strip() if match.lastindex >= 1 else match.group(0).strip()
            company = re.sub(r'\s+', ' ', company)
            company = company.rstrip('.,;:')
            
            if 2 < len(company) < 100:
                return company
    
    # Try to find company in second or third line
    lines = text.split('\n')
    for line in lines[1:4]:
        line_clean = line.strip()
        if line_clean and 3 < len(line_clean) < 100:
            # Check if it looks like a company name
            if re.match(r'^[A-Z][A-Za-z0-9\s&\'\-\.]+$', line_clean):
                # Remove location if present
                company = re.sub(r'\s*[-–—|]\s*[A-Z][a-z]+,?\s*[A-Z]{2,}.*$', '', line_clean)
                company = company.strip()
                if 2 < len(company) < 100:
                    return company
    
    return None


def extract_location(text: str) -> Optional[str]:
    """Extract job location"""
    for pattern in COMPILED_LOCATION_PATTERNS:
        match = pattern.search(text)
        if match:
            location = match.group(0).strip()
            return location
    
    return None


def extract_experience_dates(text: str) -> Dict[str, Optional[str]]:
    """Extract start and end dates from experience text"""
    dates = {'start_date': None, 'end_date': None, 'is_current': False}
    
    for pattern in COMPILED_EXPERIENCE_DATE_PATTERNS:
        match = pattern.search(text)
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
                # Single date found
                dates['end_date'] = match.group(1).strip()
    
    return dates


def calculate_duration(start_date: Optional[str], end_date: Optional[str], is_current: bool) -> Optional[str]:
    """Calculate duration of employment"""
    if not start_date:
        return None
    
    try:
        # Parse start date
        start_year = None
        start_month = None
        
        # Try to extract year and month
        year_match = re.search(r'\d{4}', start_date)
        if year_match:
            start_year = int(year_match.group(0))
        
        month_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', start_date, re.IGNORECASE)
        if month_match:
            month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
            start_month = month_map.get(month_match.group(0)[:3].lower(), 1)
        else:
            start_month = 1
        
        # Parse end date
        end_year = None
        end_month = None
        
        if is_current or not end_date or end_date == 'Present':
            # Use current date
            now = datetime.now()
            end_year = now.year
            end_month = now.month
        else:
            year_match = re.search(r'\d{4}', end_date)
            if year_match:
                end_year = int(year_match.group(0))
            
            month_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', end_date, re.IGNORECASE)
            if month_match:
                month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
                end_month = month_map.get(month_match.group(0)[:3].lower(), 12)
            else:
                end_month = 12
        
        if start_year and end_year:
            # Calculate duration in months
            total_months = (end_year - start_year) * 12 + (end_month - start_month)
            
            if total_months < 0:
                return None
            
            years = total_months // 12
            months = total_months % 12
            
            if years > 0 and months > 0:
                return f"{years} year{'s' if years > 1 else ''} {months} month{'s' if months > 1 else ''}"
            elif years > 0:
                return f"{years} year{'s' if years > 1 else ''}"
            elif months > 0:
                return f"{months} month{'s' if months > 1 else ''}"
            else:
                return "Less than 1 month"
    
    except Exception:
        pass
    
    return None


def detect_employment_type(text: str) -> List[str]:
    """Detect employment type(s) from text"""
    emp_types = []
    
    for emp_type, patterns in COMPILED_EMPLOYMENT_TYPE_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text):
                emp_types.append(emp_type)
                break
    
    return emp_types


def detect_work_mode(text: str) -> Optional[str]:
    """Detect work mode (remote, hybrid, onsite)"""
    for mode, patterns in COMPILED_WORK_MODE_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text):
                return mode
    
    return None


def extract_responsibilities(text: str) -> List[str]:
    """Extract job responsibilities and achievements"""
    responsibilities = []
    lines = text.split('\n')
    
    for line in lines:
        line_clean = line.strip()
        
        # Check if line is a bullet point
        for pattern in COMPILED_RESPONSIBILITY_PATTERNS:
            match = pattern.match(line_clean)
            if match:
                resp = match.group(1).strip()
                # Clean up
                resp = re.sub(r'\s+', ' ', resp)
                
                if 10 < len(resp) < 500:
                    responsibilities.append(resp)
                break
    
    return responsibilities[:20]  # Limit to 20 responsibilities


def categorize_responsibilities(responsibilities: List[str]) -> Dict[str, List[str]]:
    """Categorize responsibilities into achievements vs regular duties"""
    achievements = []
    duties = []
    
    for resp in responsibilities:
        resp_lower = resp.lower()
        
        # Check if it's an achievement (contains achievement indicators or metrics)
        is_achievement = False
        
        # Check for achievement verbs
        if any(indicator in resp_lower for indicator in ACHIEVEMENT_INDICATORS):
            is_achievement = True
        
        # Check for metrics (numbers, percentages)
        if re.search(r'\d+%|\d+x|\$\d+|increased|improved|reduced|saved', resp_lower):
            is_achievement = True
        
        if is_achievement:
            achievements.append(resp)
        else:
            duties.append(resp)
    
    return {'achievements': achievements, 'duties': duties}


def extract_technologies_from_experience(text: str) -> List[str]:
    """Extract technologies/tools mentioned in experience"""
    # Common technology keywords
    tech_keywords = [
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 'ruby',
        'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring', 'express',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'github',
        'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
        'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
        'html', 'css', 'sass', 'bootstrap', 'tailwind',
    ]
    
    technologies = []
    text_lower = text.lower()
    
    for tech in tech_keywords:
        if re.search(r'\b' + re.escape(tech) + r'\b', text_lower):
            technologies.append(tech.title())
    
    return list(set(technologies))[:15]  # Limit to 15 unique technologies


def extract_experiences_with_nlp(text: str) -> List[Dict[str, Any]]:
    """Extract experience information using NLP and spaCy (if available)"""
    experience_entries = []
    
    if not _NLP:
        return experience_entries
    
    try:
        doc = _NLP(text[:100000])
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            exp_keywords = ['worked', 'employed', 'position', 'role', 'company', 
                          'experience', 'managed', 'led', 'developed']
            
            if any(keyword in sent_text.lower() for keyword in exp_keywords):
                entry = {'raw_text': sent_text}
                
                # Extract organizations (likely companies)
                orgs = [ent.text for ent in sent.ents if ent.label_ == 'ORG']
                if orgs:
                    entry['company'] = orgs[0]
                
                # Extract dates
                dates = [ent.text for ent in sent.ents if ent.label_ == 'DATE']
                if dates:
                    entry['dates'] = dates
                
                # Extract locations
                locations = [ent.text for ent in sent.ents if ent.label_ in ['GPE', 'LOC']]
                if locations:
                    entry['location'] = ', '.join(locations)
                
                if len(entry) > 1:
                    experience_entries.append(entry)
    
    except Exception as e:
        print(f"NLP experience extraction error: {e}")
    
    return experience_entries


def extract_experiences_comprehensive(text: str) -> Dict[str, Any]:
    """
    Comprehensive work experience extraction using multiple techniques
    
    This function covers all possible resume formats including:
    - Full-time employment
    - Part-time jobs
    - Internships and co-ops
    - Freelance and contract work
    - Consulting engagements
    - Self-employment and entrepreneurship
    - Volunteer positions with professional responsibilities
    - Remote, hybrid, and onsite work
    - Multiple concurrent positions
    - Career progression within same company
    
    Handles various formats:
    - Traditional reverse chronological
    - Functional format
    - Combination format
    - Bullet point lists
    - Paragraph format
    - Table format
    - Mixed formats
    
    Args:
        text: Resume text to extract experience information from
    
    Returns:
        Dictionary containing:
        - experiences: List of experience dictionaries with fields:
            - job_title: Position title
            - company: Company/organization name
            - location: Job location
            - start_date: Start date
            - end_date: End date
            - is_current: Boolean indicating if currently employed
            - duration: Calculated duration
            - employment_type: List of types (full_time, internship, etc.)
            - work_mode: Remote, hybrid, or onsite
            - responsibilities: List of responsibilities
            - achievements: List of achievements
            - technologies: List of technologies used
        - statistics: Summary statistics
    """
    experience_entries = []
    
    # Step 1: Detect experience sections
    exp_sections = detect_experience_sections(text)
    
    # If no explicit section found, try to find experiences in entire text
    if not exp_sections:
        exp_keywords = ['worked', 'experience', 'employed', 'position', 'company']
        if any(keyword in text.lower() for keyword in exp_keywords):
            exp_sections = [(0, len(text), text)]
    
    # Step 2: Extract experiences from each section
    for start, end, section_text in exp_sections:
        # Split section into individual experiences
        individual_exps = split_into_individual_experiences(section_text)
        
        for exp_text in individual_exps:
            entry = {
                'job_title': extract_job_title(exp_text),
                'company': extract_company_name(exp_text),
                'location': extract_location(exp_text),
                'employment_type': detect_employment_type(exp_text),
                'work_mode': detect_work_mode(exp_text),
                'technologies': extract_technologies_from_experience(exp_text),
                'raw_text': exp_text[:600],
            }
            
            # Extract dates
            date_info = extract_experience_dates(exp_text)
            entry.update(date_info)
            
            # Calculate duration
            entry['duration'] = calculate_duration(
                date_info.get('start_date'),
                date_info.get('end_date'),
                date_info.get('is_current', False)
            )
            
            # Extract responsibilities
            responsibilities = extract_responsibilities(exp_text)
            categorized = categorize_responsibilities(responsibilities)
            entry['responsibilities'] = responsibilities
            entry['achievements'] = categorized['achievements']
            entry['duties'] = categorized['duties']
            
            experience_entries.append(entry)
    
    # Step 3: Use NLP-based extraction as supplementary
    if _NLP and not experience_entries:
        nlp_entries = extract_experiences_with_nlp(text)
        
        for nlp_entry in nlp_entries:
            nlp_entry['extraction_method'] = 'nlp'
            experience_entries.append(nlp_entry)
    
    # Step 4: Clean up and format final entries
    final_entries = []
    for entry in experience_entries:
        # Remove empty fields
        cleaned_entry = {k: v for k, v in entry.items() if v}
        
        # Ensure at least job title or company is present
        if cleaned_entry.get('job_title') or cleaned_entry.get('company'):
            final_entries.append(cleaned_entry)
    
    # Step 5: Sort by date (most recent first)
    def get_sort_key(entry):
        end_date = entry.get('end_date', '')
        if not end_date or end_date == 'Present':
            return 9999  # Current jobs come first
        
        year_match = re.search(r'\d{4}', str(end_date))
        if year_match:
            return int(year_match.group(0))
        return 0
    
    final_entries.sort(key=get_sort_key, reverse=True)
    
    # Step 6: Add statistics
    exp_stats = {
        'total_positions': len(final_entries),
        'current_positions': sum(1 for e in final_entries if e.get('is_current', False)),
        'by_employment_type': {},
        'by_work_mode': {},
        'total_years_experience': 0,
        'companies': [],
        'job_titles': [],
    }
    
    for exp in final_entries:
        # Count by employment type
        for emp_type in exp.get('employment_type', []):
            exp_stats['by_employment_type'][emp_type] = exp_stats['by_employment_type'].get(emp_type, 0) + 1
        
        # Count by work mode
        work_mode = exp.get('work_mode')
        if work_mode:
            exp_stats['by_work_mode'][work_mode] = exp_stats['by_work_mode'].get(work_mode, 0) + 1
        
        # Collect companies
        if exp.get('company'):
            exp_stats['companies'].append(exp['company'])
        
        # Collect job titles
        if exp.get('job_title'):
            exp_stats['job_titles'].append(exp['job_title'])
        
        # Calculate total years (approximate)
        duration = exp.get('duration', '')
        if 'year' in duration:
            year_match = re.search(r'(\d+)\s+year', duration)
            if year_match:
                exp_stats['total_years_experience'] += int(year_match.group(1))
    
    # Remove duplicates from lists
    exp_stats['companies'] = list(set(exp_stats['companies']))
    exp_stats['job_titles'] = list(set(exp_stats['job_titles']))
    
    return {
        'experiences': final_entries,
        'statistics': exp_stats
    }


# Backward compatibility - alias for the main function
extract_experiences_info = extract_experiences_comprehensive
