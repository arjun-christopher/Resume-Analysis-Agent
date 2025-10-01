"""
Comprehensive Activities Information Extractor for Resumes

This module provides advanced NLP-based activities extraction covering all possible resume formats:
- Extracurricular activities
- Volunteer work and community service
- Professional associations and memberships
- Clubs and organizations
- Sports and athletics
- Cultural activities
- Student government
- Hobbies and interests
- Social causes and activism
- Event organization
- Committee participation
"""

import re
from typing import List, Dict, Any, Optional, Tuple

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


# ---------- Activities Extraction Configuration ----------

# Activity section headers
ACTIVITY_SECTION_HEADERS = [
    r'\b(?:ACTIVITIES|EXTRACURRICULAR\s+ACTIVITIES|EXTRA[-\s]?CURRICULAR|'
    r'VOLUNTEER\s+(?:WORK|EXPERIENCE|ACTIVITIES)|VOLUNTEERING|COMMUNITY\s+SERVICE|'
    r'LEADERSHIP\s+(?:ACTIVITIES|EXPERIENCE|ROLES)|STUDENT\s+ACTIVITIES|'
    r'CLUBS?\s+(?:AND|&)\s+(?:ORGANIZATIONS?|SOCIETIES)|ORGANIZATIONS?|MEMBERSHIPS?|'
    r'PROFESSIONAL\s+(?:MEMBERSHIPS?|AFFILIATIONS?|ASSOCIATIONS?)|AFFILIATIONS?|'
    r'INTERESTS?\s+(?:AND|&)\s+ACTIVITIES|HOBBIES?\s+(?:AND|&)\s+INTERESTS?|'
    r'PERSONAL\s+INTERESTS?|ADDITIONAL\s+(?:ACTIVITIES|INFORMATION)|'
    r'SOCIAL\s+ACTIVITIES|CAMPUS\s+INVOLVEMENT|COMMUNITY\s+INVOLVEMENT|'
    r'CO[-\s]?CURRICULAR\s+ACTIVITIES)\b'
]

# Activity type patterns
ACTIVITY_TYPE_PATTERNS = {
    'volunteer': [
        r'\b(?:volunteer|volunteered|volunteering)\b',
        r'\b(?:community\s+service|social\s+service|public\s+service)\b',
        r'\b(?:charity|charitable|non[-\s]?profit|NGO)\b',
        r'\b(?:helped|assisted|supported|contributed\s+to)\s+(?:community|society|cause)',
    ],
    'leadership': [
        r'\b(?:president|vice\s+president|chairman|chairperson|chair|leader|head)\b',
        r'\b(?:captain|co[-\s]?captain|team\s+leader|group\s+leader)\b',
        r'\b(?:coordinator|organizer|director|manager|secretary|treasurer)\b',
        r'\b(?:founded|co[-\s]?founded|established|started|initiated)\b',
    ],
    'club': [
        r'\b(?:club|society|association|organization|group|team)\s+member\b',
        r'\b(?:member|active\s+member|founding\s+member)\s+of\b',
        r'\b(?:joined|participated\s+in|part\s+of)\s+(?:the\s+)?(?:club|society|team)\b',
    ],
    'sports': [
        r'\b(?:sports?|athletic|athletics|team\s+sports?)\b',
        r'\b(?:basketball|football|soccer|cricket|tennis|volleyball|baseball|hockey|swimming|track|field|rugby|badminton|table\s+tennis)\b',
        r'\b(?:captain|player|athlete|competitor|team\s+member)\b',
        r'\b(?:varsity|junior\s+varsity|intramural)\b',
    ],
    'cultural': [
        r'\b(?:cultural|music|musical|dance|dancing|drama|theater|theatre|art|artistic)\b',
        r'\b(?:choir|orchestra|band|ensemble|troupe|group)\b',
        r'\b(?:performed|performance|concert|recital|exhibition|show)\b',
        r'\b(?:painting|drawing|sculpture|photography|creative)\b',
    ],
    'academic': [
        r'\b(?:academic|scholarly|research|study)\s+(?:club|society|group)\b',
        r'\b(?:debate|debating|quiz|science|math|mathematics|engineering|robotics|coding)\s+(?:club|team|society)\b',
        r'\b(?:honor\s+society|honors?\s+program|academic\s+society)\b',
    ],
    'professional': [
        r'\b(?:professional\s+(?:association|organization|society|membership))\b',
        r'\b(?:IEEE|ACM|PMI|AMA|APA|SHRM|CFA\s+Institute)\b',
        r'\b(?:industry\s+(?:association|group|network))\b',
        r'\b(?:professional\s+development|networking\s+group)\b',
    ],
    'social': [
        r'\b(?:social\s+(?:club|group|organization|cause))\b',
        r'\b(?:awareness|advocacy|activism|campaign)\b',
        r'\b(?:fundraising|fundraiser|charity\s+event)\b',
    ],
    'student_government': [
        r'\b(?:student\s+(?:government|council|senate|union|body))\b',
        r'\b(?:class\s+(?:representative|president|officer))\b',
        r'\b(?:student\s+(?:representative|delegate|ambassador))\b',
    ],
    'event': [
        r'\b(?:organized|organized|coordinated|managed|hosted)\s+(?:event|conference|workshop|seminar|competition)\b',
        r'\b(?:event\s+(?:organizer|coordinator|manager|planner))\b',
        r'\b(?:festival|fest|symposium|summit|convention)\b',
    ]
}

# Role/Position patterns
ROLE_PATTERNS = [
    r'\b(?:as|role|position)[\s:]+([A-Z][A-Za-z\s]{3,40}?)(?:\s*[-–—|,\n]|$)',
    r'\b(President|Vice\s+President|Chairman|Chairperson|Secretary|Treasurer|Captain|Co[-\s]?Captain|'
    r'Coordinator|Director|Manager|Leader|Head|Member|Volunteer|Organizer|Founder|Co[-\s]?Founder)\b',
]

# Organization/Group name patterns
ORGANIZATION_PATTERNS = [
    r'\b([A-Z][A-Za-z\s&\'\-\.]{3,60}?)\s+(?:Club|Society|Association|Organization|Team|Group|Committee)\b',
    r'\b(?:Member|President|Volunteer)\s+(?:of|at|with|for)\s+([A-Z][A-Za-z\s&\'\-\.]{3,60})\b',
]

# Date patterns for activities
ACTIVITY_DATE_PATTERNS = [
    r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\s*[-–—]\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|Present|Current|Ongoing)\b',
    r'\b(\d{4})\s*[-–—]\s*(\d{4}|Present|Current|Ongoing)\b',
    r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b',
    r'\b(\d{4})\b',
]

# Duration patterns
DURATION_PATTERNS = [
    r'\b(\d+)\s*(?:\+)?\s*(?:years?|yrs?\.?)\b',
    r'\b(\d+)\s*(?:\+)?\s*(?:months?|mos?\.?)\b',
    r'\b(?:for|duration|period\s+of)[\s:]*(\d+)\s+(?:years?|months?)\b',
]

# Impact/Achievement patterns in activities
ACTIVITY_IMPACT_PATTERNS = [
    r'\b(?:raised|collected|gathered)\s+\$?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:dollars?|USD|K|M)?\b',
    r'\b(?:organized|coordinated|managed)\s+(?:event|workshop|conference)\s+(?:with|for)\s+(\d+(?:,\d{3})*(?:\+)?)\s+(?:participants?|attendees?|people)\b',
    r'\b(?:led|managed|supervised)\s+(?:team\s+of\s+)?(\d+(?:\+)?)\s+(?:members?|volunteers?|people)\b',
    r'\b(?:reached|served|helped|impacted)\s+(\d+(?:,\d{3})*(?:\+)?)\s+(?:people|individuals|students?|families)\b',
]

# Frequency patterns
FREQUENCY_PATTERNS = [
    r'\b(?:weekly|every\s+week|per\s+week)\b',
    r'\b(?:monthly|every\s+month|per\s+month)\b',
    r'\b(?:daily|every\s+day|per\s+day)\b',
    r'\b(?:bi[-\s]?weekly|twice\s+a\s+week)\b',
    r'\b(?:quarterly|every\s+quarter)\b',
    r'\b(\d+)\s+(?:hours?|hrs?)\s+per\s+(?:week|month)\b',
]

# Compile patterns
COMPILED_ACTIVITY_HEADERS = [re.compile(pattern, re.IGNORECASE) for pattern in ACTIVITY_SECTION_HEADERS]
COMPILED_ROLE_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in ROLE_PATTERNS]
COMPILED_ORGANIZATION_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in ORGANIZATION_PATTERNS]
COMPILED_ACTIVITY_DATE_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in ACTIVITY_DATE_PATTERNS]
COMPILED_DURATION_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in DURATION_PATTERNS]
COMPILED_ACTIVITY_IMPACT_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in ACTIVITY_IMPACT_PATTERNS]
COMPILED_FREQUENCY_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in FREQUENCY_PATTERNS]

COMPILED_ACTIVITY_TYPE_PATTERNS = {}
for activity_type, patterns in ACTIVITY_TYPE_PATTERNS.items():
    COMPILED_ACTIVITY_TYPE_PATTERNS[activity_type] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


def detect_activity_sections(text: str) -> List[Tuple[int, int, str]]:
    """
    Detect activity sections in resume text
    Returns: List of (start_line, end_line, section_text) tuples
    """
    sections = []
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean:
            continue
        
        # Check if line is an activity section header
        for pattern in COMPILED_ACTIVITY_HEADERS:
            if pattern.search(line_clean):
                start_line = i
                end_line = start_line + 1
                
                # Find the end of this section
                for j in range(i + 1, min(len(lines), i + 100)):
                    next_line = lines[j].strip()
                    if not next_line:
                        continue
                    
                    # Check if this is another major section header
                    if re.match(r'^[A-Z\s]{3,30}$', next_line) and len(next_line) > 5:
                        common_sections = ['EDUCATION', 'EXPERIENCE', 'WORK', 'EMPLOYMENT', 'SKILLS', 
                                         'PROJECTS', 'CERTIFICATIONS', 'ACHIEVEMENTS', 'SUMMARY', 'OBJECTIVE']
                        if any(section in next_line.upper() for section in common_sections):
                            end_line = j
                            break
                    
                    end_line = j + 1
                
                section_lines = lines[start_line:end_line]
                section_text = '\n'.join(section_lines)
                sections.append((start_line, end_line, section_text))
                break
    
    return sections


def split_into_individual_activities(text: str) -> List[str]:
    """
    Split activity section text into individual activity entries
    """
    activities = []
    lines = text.split('\n')
    
    current_activity = []
    in_activity = False
    
    for i, line in enumerate(lines):
        line_clean = line.strip()
        
        # Skip empty lines at the start
        if not line_clean and not in_activity:
            continue
        
        # Check if this line starts a new activity
        is_new_activity = False
        
        # Heuristic 1: Line starts with bullet point or number
        if re.match(r'^(?:•|●|○|◦|▪|▫|–|—|\*|\d+\.|\d+\))\s+[A-Z]', line_clean):
            is_new_activity = True
        
        # Heuristic 2: Line looks like a title/organization name
        elif re.match(r'^[A-Z][A-Za-z0-9\s\-:&,\.]{3,80}$', line_clean) and len(line_clean) < 100:
            # Check if it contains activity keywords
            activity_keywords = ['club', 'society', 'volunteer', 'member', 'president', 'team', 
                               'organization', 'association', 'committee', 'group']
            if any(keyword in line_clean.lower() for keyword in activity_keywords):
                is_new_activity = True
        
        # If we found a new activity and we have accumulated content, save it
        if is_new_activity and current_activity:
            activity_text = '\n'.join(current_activity)
            if len(activity_text.strip()) > 15:
                activities.append(activity_text)
            current_activity = []
            in_activity = True
        
        # Add line to current activity
        if line_clean or in_activity:
            current_activity.append(line)
            in_activity = True
    
    # Don't forget the last activity
    if current_activity:
        activity_text = '\n'.join(current_activity)
        if len(activity_text.strip()) > 15:
            activities.append(activity_text)
    
    return activities


def extract_activity_title(text: str) -> Optional[str]:
    """Extract activity title/name from text"""
    lines = text.split('\n')
    
    # Try first non-empty line
    for line in lines[:3]:
        line_clean = line.strip()
        if line_clean:
            # Remove bullet points
            title = re.sub(r'^(?:•|●|○|◦|▪|▫|–|—|\*|\d+\.|\d+\))\s*', '', line_clean)
            
            # Check if it's a reasonable title length
            if 3 < len(title) < 150:
                # Remove trailing separators
                title = re.sub(r'\s*[-–—|:]\s*$', '', title)
                return title.strip()
    
    return None


def extract_activity_role(text: str) -> Optional[str]:
    """Extract role/position in the activity"""
    for pattern in COMPILED_ROLE_PATTERNS:
        match = pattern.search(text)
        if match:
            role = match.group(1).strip() if match.lastindex >= 1 else match.group(0).strip()
            role = re.sub(r'\s+', ' ', role)
            if 3 < len(role) < 60:
                return role
    
    return None


def extract_organization_name(text: str) -> Optional[str]:
    """Extract organization/group name"""
    for pattern in COMPILED_ORGANIZATION_PATTERNS:
        match = pattern.search(text)
        if match:
            if match.lastindex >= 1:
                org = match.group(1).strip()
            else:
                org = match.group(0).strip()
            
            org = re.sub(r'\s+', ' ', org)
            org = org.rstrip('.,;:')
            
            if 3 < len(org) < 100:
                return org
    
    return None


def extract_activity_dates(text: str) -> Dict[str, Optional[str]]:
    """Extract start and end dates from activity text"""
    dates = {'start_date': None, 'end_date': None, 'is_current': False}
    
    for pattern in COMPILED_ACTIVITY_DATE_PATTERNS:
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
                dates['end_date'] = match.group(1).strip()
    
    return dates


def extract_activity_duration(text: str) -> Optional[str]:
    """Extract activity duration"""
    for pattern in COMPILED_DURATION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0).strip()
    
    return None


def extract_activity_frequency(text: str) -> Optional[str]:
    """Extract activity frequency (weekly, monthly, etc.)"""
    for pattern in COMPILED_FREQUENCY_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0).strip()
    
    return None


def extract_activity_impact(text: str) -> List[str]:
    """Extract impact/achievements from activity"""
    impacts = []
    
    for pattern in COMPILED_ACTIVITY_IMPACT_PATTERNS:
        matches = pattern.finditer(text)
        for match in matches:
            impact = match.group(0).strip()
            if len(impact) > 10 and impact not in impacts:
                impacts.append(impact)
    
    return impacts[:5]  # Limit to 5 impacts


def detect_activity_types(text: str) -> List[str]:
    """Detect activity type(s) from text"""
    activity_types = []
    
    for activity_type, patterns in COMPILED_ACTIVITY_TYPE_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text):
                activity_types.append(activity_type)
                break
    
    return activity_types


def extract_activity_description(text: str) -> Optional[str]:
    """Extract activity description"""
    lines = text.split('\n')
    
    # Skip first line (usually title) and extract rest
    if len(lines) > 1:
        description_lines = lines[1:]
    else:
        description_lines = lines
    
    description_text = '\n'.join(description_lines).strip()
    
    # Remove metadata lines
    description_text = re.sub(r'(?:Role|Position|Duration|Date|Organization)[\s:][^\n]+', '', description_text, flags=re.IGNORECASE)
    
    # Clean up
    description_text = re.sub(r'\n{3,}', '\n\n', description_text)
    description_text = description_text.strip()
    
    if len(description_text) > 20:
        return description_text[:800]  # Limit length
    
    return None


def extract_activities_with_nlp(text: str) -> List[Dict[str, Any]]:
    """Extract activity information using NLP and spaCy (if available)"""
    activity_entries = []
    
    if not _NLP:
        return activity_entries
    
    try:
        doc = _NLP(text[:100000])
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            activity_keywords = ['volunteer', 'club', 'member', 'president', 'organized', 
                               'participated', 'team', 'society', 'association']
            
            if any(keyword in sent_text.lower() for keyword in activity_keywords):
                entry = {'raw_text': sent_text}
                
                # Extract organizations
                orgs = [ent.text for ent in sent.ents if ent.label_ == 'ORG']
                if orgs:
                    entry['organization'] = orgs[0]
                
                # Extract dates
                dates = [ent.text for ent in sent.ents if ent.label_ == 'DATE']
                if dates:
                    entry['dates'] = dates
                
                if len(entry) > 1:
                    activity_entries.append(entry)
    
    except Exception as e:
        print(f"NLP activity extraction error: {e}")
    
    return activity_entries


def extract_activities_comprehensive(text: str) -> List[Dict[str, Any]]:
    """
    Comprehensive activity extraction using multiple techniques
    
    This function covers all possible resume formats including:
    - Extracurricular activities
    - Volunteer work and community service
    - Professional associations and memberships
    - Clubs and organizations (academic, cultural, sports, social)
    - Student government and leadership roles
    - Sports and athletics
    - Cultural activities (music, dance, drama, art)
    - Event organization and coordination
    - Committee participation
    - Hobbies and interests
    - Social causes and activism
    
    Handles various formats:
    - Dedicated activities/interests section
    - Bullet point lists
    - Paragraph format
    - Table format
    - Mixed formats
    
    Args:
        text: Resume text to extract activity information from
    
    Returns:
        List of dictionaries containing structured activity information with fields:
        - title: Activity title/name
        - role: Role/position in the activity
        - organization: Organization/group name
        - description: Activity description
        - activity_types: List of activity type categories
        - start_date: Activity start date
        - end_date: Activity end date
        - is_current: Boolean indicating if currently active
        - duration: Activity duration
        - frequency: Activity frequency (weekly, monthly, etc.)
        - impact: List of impacts/achievements
    """
    activity_entries = []
    
    # Step 1: Detect activity sections
    activity_sections = detect_activity_sections(text)
    
    # If no explicit section found, try to find activities in entire text
    if not activity_sections:
        # Look for activity indicators
        activity_indicators = ['volunteer', 'club', 'member', 'president', 'society', 
                              'association', 'organization', 'team', 'committee']
        if any(indicator in text.lower() for indicator in activity_indicators):
            activity_sections = [(0, len(text), text)]
    
    # Step 2: Extract activities from each section
    for start, end, section_text in activity_sections:
        # Split section into individual activities
        individual_activities = split_into_individual_activities(section_text)
        
        for activity_text in individual_activities:
            entry = {
                'title': extract_activity_title(activity_text),
                'role': extract_activity_role(activity_text),
                'organization': extract_organization_name(activity_text),
                'description': extract_activity_description(activity_text),
                'activity_types': detect_activity_types(activity_text),
                'duration': extract_activity_duration(activity_text),
                'frequency': extract_activity_frequency(activity_text),
                'impact': extract_activity_impact(activity_text),
                'raw_text': activity_text[:500],
            }
            
            # Extract dates
            date_info = extract_activity_dates(activity_text)
            entry.update(date_info)
            
            activity_entries.append(entry)
    
    # Step 3: Use NLP-based extraction as supplementary
    if _NLP and not activity_entries:
        nlp_entries = extract_activities_with_nlp(text)
        
        for nlp_entry in nlp_entries:
            nlp_entry['extraction_method'] = 'nlp'
            activity_entries.append(nlp_entry)
    
    # Step 4: Clean up and format final entries
    final_entries = []
    for entry in activity_entries:
        # Remove empty fields
        cleaned_entry = {k: v for k, v in entry.items() if v}
        
        # Ensure at least title, role, or organization is present
        if cleaned_entry.get('title') or cleaned_entry.get('role') or cleaned_entry.get('organization'):
            final_entries.append(cleaned_entry)
    
    # Step 5: Sort by date (most recent first)
    def get_sort_key(entry):
        end_date = entry.get('end_date', '')
        if not end_date or end_date == 'Present':
            return 9999  # Current activities come first
        
        year_match = re.search(r'\d{4}', str(end_date))
        if year_match:
            return int(year_match.group(0))
        return 0
    
    final_entries.sort(key=get_sort_key, reverse=True)
    
    # Step 6: Add statistics
    activity_stats = {
        'total_count': len(final_entries),
        'by_type': {},
        'current_activities': sum(1 for a in final_entries if a.get('is_current', False)),
        'with_leadership': sum(1 for a in final_entries if 'leadership' in a.get('activity_types', [])),
        'volunteer_activities': sum(1 for a in final_entries if 'volunteer' in a.get('activity_types', [])),
    }
    
    for activity in final_entries:
        for activity_type in activity.get('activity_types', []):
            activity_stats['by_type'][activity_type] = activity_stats['by_type'].get(activity_type, 0) + 1
    
    return {
        'activities': final_entries,
        'statistics': activity_stats
    }


# Backward compatibility - alias for the main function
extract_activities_info = extract_activities_comprehensive
