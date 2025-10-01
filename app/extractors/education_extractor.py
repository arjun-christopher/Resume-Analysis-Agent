# app/extractors/education_extractor.py - Education extraction functionality
import re
from typing import Any, Dict, List, Tuple, Optional

# Try to import optional dependencies
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