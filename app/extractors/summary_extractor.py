"""
Comprehensive Professional Summary/Objective Extractor for Resumes

This module provides advanced NLP-based extraction of professional summaries, career objectives,
and profile sections from resumes. It extracts key information about the candidate's:
- Career goals and objectives
- Professional identity and value proposition
- Key strengths and differentiators
- Years of experience
- Core competencies mentioned in summary
- Industry focus and target roles
"""

import re
from typing import List, Dict, Any, Optional

# Try to import spaCy for NLP-based extraction
try:
    import spacy
    try:
        _NLP = spacy.load("en_core_web_sm")
    except OSError:
        _NLP = None
except ImportError:
    _NLP = None


# ---------- Summary/Objective Extraction Configuration ----------

# Summary section headers
SUMMARY_SECTION_HEADERS = [
    r'\b(?:PROFESSIONAL\s+)?SUMMARY\b',
    r'\b(?:CAREER\s+)?OBJECTIVE\b',
    r'\b(?:PROFESSIONAL\s+)?PROFILE\b',
    r'\bABOUT\s+ME\b',
    r'\bEXECUTIVE\s+SUMMARY\b',
    r'\b(?:CAREER\s+)?(?:SUMMARY|GOAL|GOALS)\b',
    r'\bPERSONAL\s+STATEMENT\b',
    r'\bPROFESSIONAL\s+SYNOPSIS\b',
    r'\bOVERVIEW\b',
    r'\bINTRODUCTION\b',
]

# Experience patterns (to extract years of experience mentioned in summary)
EXPERIENCE_PATTERNS = [
    r'\b(\d+)\+?\s*years?\s+of\s+(?:professional\s+)?experience\b',
    r'\b(?:over|more\s+than)\s+(\d+)\s*years?\s+of\s+experience\b',
    r'\b(\d+)\+?\s*year\s+(?:track\s+record|background|history)\b',
    r'\bexperience\s+spanning\s+(\d+)\+?\s*years?\b',
]

# Role/Title patterns (to extract current/target role)
ROLE_PATTERNS = [
    r'\b(Senior|Lead|Principal|Staff|Junior|Associate|Chief)\s+([A-Z][a-z]+\s+){1,3}(?:Engineer|Developer|Manager|Analyst|Designer|Scientist|Architect|Consultant|Specialist)\b',
    r'\b(?:experienced|seasoned|accomplished|results[-\s]driven)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b',
    r'\bseeking\s+(?:a\s+)?(?:position|role|opportunity)\s+as\s+(?:a|an)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b',
]

# Industry patterns
INDUSTRY_PATTERNS = [
    r'\b(?:in\s+the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+industry\b',
    r'\b(?:specializing|specialized|focusing|focused)\s+(?:in|on)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b',
    r'\b(?:background|experience)\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b',
]

# Key strength indicators
STRENGTH_INDICATORS = [
    r'\b(?:expertise|expert|skilled|proficient|specialized)\s+(?:in|at|with)\s+([A-Za-z\s,&]+?)(?:\.|,|and|with|$)',
    r'\b(?:strong|solid|proven|extensive)\s+(?:background|experience|skills?|knowledge)\s+(?:in|of|with)\s+([A-Za-z\s,&]+?)(?:\.|,|and|$)',
    r'\b(?:adept|talented|accomplished|successful)\s+(?:in|at|with)\s+([A-Za-z\s,&]+?)(?:\.|,|and|$)',
]

# Compile patterns
COMPILED_SUMMARY_HEADERS = [re.compile(pattern, re.IGNORECASE) for pattern in SUMMARY_SECTION_HEADERS]
COMPILED_EXPERIENCE_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in EXPERIENCE_PATTERNS]
COMPILED_ROLE_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in ROLE_PATTERNS]
COMPILED_INDUSTRY_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in INDUSTRY_PATTERNS]
COMPILED_STRENGTH_INDICATORS = [re.compile(pattern, re.IGNORECASE) for pattern in STRENGTH_INDICATORS]


def detect_summary_section(text: str) -> Optional[Tuple[int, int, str]]:
    """
    Detect professional summary/objective section in resume text
    Returns: (start_pos, end_pos, section_text) tuple or None
    """
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean:
            continue
        
        # Check if line is a summary section header
        for pattern in COMPILED_SUMMARY_HEADERS:
            if pattern.search(line_clean):
                start_line = i
                end_line = start_line + 1
                
                # Find the end of this section (usually 2-6 lines)
                for j in range(i + 1, min(len(lines), i + 20)):
                    next_line = lines[j].strip()
                    if not next_line:
                        continue
                    
                    # Check if this is another major section header
                    if re.match(r'^[A-Z\s]{3,30}$', next_line) and len(next_line) > 5:
                        common_sections = ['EXPERIENCE', 'EDUCATION', 'SKILLS', 'PROJECTS', 
                                         'CERTIFICATIONS', 'ACHIEVEMENTS', 'WORK', 'EMPLOYMENT']
                        if any(section in next_line.upper() for section in common_sections):
                            end_line = j
                            break
                    
                    end_line = j + 1
                
                section_lines = lines[start_line:end_line]
                section_text = '\n'.join(section_lines)
                return (start_line, end_line, section_text)
    
    # If no explicit header found, check if resume starts with a paragraph (common format)
    # First few lines without a header could be a summary
    first_content_line = 0
    for i, line in enumerate(lines[:15]):  # Check first 15 lines
        if line.strip() and len(line.strip()) > 30:  # Substantial content
            # Check if it looks like contact info
            if re.search(r'@|phone|email|linkedin|github|tel:|mobile:', line, re.IGNORECASE):
                continue
            
            # This might be start of summary
            first_content_line = i
            
            # Collect paragraph (until empty line or section header)
            summary_lines = []
            for j in range(i, min(len(lines), i + 10)):
                if not lines[j].strip():
                    break
                if re.match(r'^[A-Z\s]{3,30}$', lines[j].strip()):
                    break
                summary_lines.append(lines[j])
            
            if len(summary_lines) >= 2:  # At least 2 lines
                summary_text = '\n'.join(summary_lines)
                # Check if it looks like a summary (contains career-related keywords)
                summary_keywords = ['experience', 'professional', 'skilled', 'expertise', 
                                   'background', 'seeking', 'passionate', 'dedicated']
                if any(keyword in summary_text.lower() for keyword in summary_keywords):
                    return (first_content_line, first_content_line + len(summary_lines), summary_text)
    
    return None


def extract_years_of_experience(text: str) -> Optional[int]:
    """
    Extract years of experience mentioned in summary
    """
    for pattern in COMPILED_EXPERIENCE_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                years = int(match.group(1))
                return years
            except (ValueError, IndexError):
                continue
    
    return None


def extract_current_or_target_role(text: str) -> Optional[str]:
    """
    Extract current role or target role from summary
    """
    for pattern in COMPILED_ROLE_PATTERNS:
        match = pattern.search(text)
        if match:
            # Get the last captured group (role name)
            for i in range(len(match.groups()), 0, -1):
                role = match.group(i)
                if role and len(role) > 3:
                    return role.strip()
    
    return None


def extract_industry_focus(text: str) -> List[str]:
    """
    Extract industry focus areas from summary
    """
    industries = []
    
    for pattern in COMPILED_INDUSTRY_PATTERNS:
        matches = pattern.finditer(text)
        for match in matches:
            industry = match.group(1).strip()
            if len(industry) > 3 and industry not in industries:
                industries.append(industry)
    
    return industries[:3]  # Limit to top 3


def extract_key_strengths(text: str) -> List[str]:
    """
    Extract key strengths and competencies mentioned in summary
    """
    strengths = []
    
    for pattern in COMPILED_STRENGTH_INDICATORS:
        matches = pattern.finditer(text)
        for match in matches:
            strength = match.group(1).strip()
            # Clean up the strength text
            strength = re.sub(r'\s+', ' ', strength)
            strength = strength.rstrip('.,;:')
            
            if len(strength) > 5 and len(strength) < 100 and strength not in strengths:
                strengths.append(strength)
    
    return strengths[:5]  # Limit to top 5


def extract_summary_with_nlp(text: str) -> Dict[str, Any]:
    """
    Extract summary information using spaCy NER and dependency parsing
    """
    if not _NLP:
        return {}
    
    summary_info = {
        'organizations': [],
        'roles': [],
        'skills': [],
        'key_phrases': []
    }
    
    try:
        # Limit to first 5000 chars (summary is usually at the beginning)
        doc = _NLP(text[:5000])
        
        # Look for sentences in summary-like context
        summary_keywords = ['experience', 'professional', 'skilled', 'expertise', 
                           'background', 'seeking', 'passionate', 'dedicated', 'proven']
        
        relevant_sentences = []
        for sent in doc.sents:
            if any(keyword in sent.text.lower() for keyword in summary_keywords):
                relevant_sentences.append(sent)
                if len(relevant_sentences) >= 5:  # Limit to 5 sentences
                    break
        
        # Extract entities from relevant sentences
        for sent in relevant_sentences:
            # Extract organizations
            orgs = [ent.text for ent in sent.ents if ent.label_ == 'ORG']
            summary_info['organizations'].extend(orgs)
            
            # Extract noun chunks as potential skills/phrases
            for chunk in sent.noun_chunks:
                if len(chunk.text.split()) >= 2:  # Multi-word phrases
                    summary_info['key_phrases'].append(chunk.text)
        
        # Deduplicate
        summary_info['organizations'] = list(set(summary_info['organizations']))[:3]
        summary_info['key_phrases'] = list(set(summary_info['key_phrases']))[:5]
        
    except Exception as e:
        print(f"NLP summary extraction error: {e}")
    
    return summary_info


def calculate_summary_sentiment(text: str) -> str:
    """
    Determine the tone of the summary (confident, modest, enthusiastic)
    """
    text_lower = text.lower()
    
    # Count confident/strong words
    confident_words = ['expert', 'proven', 'accomplished', 'successful', 'leading', 
                      'extensive', 'comprehensive', 'exceptional', 'outstanding']
    confident_count = sum(1 for word in confident_words if word in text_lower)
    
    # Count enthusiastic words
    enthusiastic_words = ['passionate', 'eager', 'excited', 'motivated', 'driven', 
                         'enthusiastic', 'committed', 'dedicated']
    enthusiastic_count = sum(1 for word in enthusiastic_words if word in text_lower)
    
    # Determine tone
    if confident_count >= 2:
        return 'Confident'
    elif enthusiastic_count >= 2:
        return 'Enthusiastic'
    else:
        return 'Professional'


def extract_summary_comprehensive(text: str) -> Dict[str, Any]:
    """
    Comprehensive professional summary extraction using multiple NLP techniques
    
    This function extracts:
    - Full summary text
    - Years of experience mentioned
    - Current/target role
    - Industry focus areas
    - Key strengths and competencies
    - Tone/sentiment of summary
    - NER-extracted entities (organizations, key phrases)
    
    Uses advanced NLP techniques:
    - spaCy NER for entity extraction
    - Dependency parsing for phrase extraction
    - Pattern matching for structured information
    - Sentiment analysis for tone detection
    
    Args:
        text: Resume text to extract summary from
    
    Returns:
        Dictionary containing structured summary information with fields:
        - summary_text: Full text of the summary section
        - years_of_experience: Years of experience mentioned (if any)
        - current_role: Current or target role title
        - industry_focus: List of industry areas mentioned
        - key_strengths: List of key competencies/strengths
        - tone: Tone of the summary (Confident, Enthusiastic, Professional)
        - organizations_mentioned: Organizations mentioned in summary
        - key_phrases: Important noun phrases extracted
    """
    result = {
        'summary_text': None,
        'years_of_experience': None,
        'current_role': None,
        'industry_focus': [],
        'key_strengths': [],
        'tone': None,
        'organizations_mentioned': [],
        'key_phrases': [],
        'section_found': False
    }
    
    # Step 1: Detect summary section
    summary_section = detect_summary_section(text)
    
    if not summary_section:
        return result
    
    start, end, section_text = summary_section
    result['section_found'] = True
    
    # Remove header line, keep only content
    lines = section_text.split('\n')
    content_lines = []
    for line in lines:
        # Skip if it looks like a header
        if re.match(r'^[A-Z\s]{3,30}$', line.strip()):
            continue
        if line.strip():
            content_lines.append(line)
    
    summary_content = '\n'.join(content_lines)
    result['summary_text'] = summary_content.strip()
    
    if not result['summary_text']:
        return result
    
    # Step 2: Extract years of experience
    result['years_of_experience'] = extract_years_of_experience(summary_content)
    
    # Step 3: Extract current/target role
    result['current_role'] = extract_current_or_target_role(summary_content)
    
    # Step 4: Extract industry focus
    result['industry_focus'] = extract_industry_focus(summary_content)
    
    # Step 5: Extract key strengths
    result['key_strengths'] = extract_key_strengths(summary_content)
    
    # Step 6: Calculate tone/sentiment
    result['tone'] = calculate_summary_sentiment(summary_content)
    
    # Step 7: Use NLP for additional extraction
    if _NLP:
        nlp_info = extract_summary_with_nlp(summary_content)
        result['organizations_mentioned'] = nlp_info.get('organizations', [])
        result['key_phrases'] = nlp_info.get('key_phrases', [])
    
    # Step 8: Calculate word count and reading level
    word_count = len(summary_content.split())
    result['word_count'] = word_count
    result['is_concise'] = 50 <= word_count <= 150  # Ideal summary length
    
    return result


# Backward compatibility - alias for the main function
extract_summary_info = extract_summary_comprehensive
