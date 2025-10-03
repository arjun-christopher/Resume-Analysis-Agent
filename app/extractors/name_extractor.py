"""
Name Extraction and Formatting Module

This module provides advanced name extraction and formatting capabilities for resumes.
It uses multiple strategies including pattern matching, NER, and confidence scoring.
"""

import re
from typing import List, Tuple

# Try to import optional NLP libraries
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
    from transformers import pipeline
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

# Name patterns for regex matching
NAME_PATTERNS = [
    re.compile(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]*\.?\s*)*[A-Z][a-z]+)$'),
    re.compile(r'^([A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+)$'),
    re.compile(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})$'),
]

# Enhanced skip keywords to avoid false positives
SKIP_KEYWORDS = [
    'resume', 'cv', 'curriculum', 'vitae', 'profile', 'summary', 
    'contact', 'email', 'phone', 'address', 'skills', 'experience',
    'education', 'objective', 'portfolio', 'projects', 'work',
    'professional', 'personal', 'references', 'certification',
    'achievement', 'publication', 'awards', 'languages', 'hobbies'
]

# Words to exclude from being considered as names
EXCLUDE_WORDS = [
    'the', 'and', 'for', 'with', 'from', 'about', 'university',
    'college', 'school', 'company', 'corporation', 'inc', 'llc',
    'engineer', 'developer', 'manager', 'analyst', 'designer',
    'consultant', 'specialist', 'director', 'senior', 'junior'
]

# Common name prefixes and suffixes
COMMON_PREFIXES = ['mr', 'mrs', 'ms', 'dr', 'prof']
COMMON_SUFFIXES = ['jr', 'sr', 'ii', 'iii', 'iv', 'phd', 'md']


def format_candidate_name(name: str) -> str:
    """
    Format candidate name with proper capitalization, handling initials and special cases
    
    Examples:
        "ARJUN CHRISTOPHER" -> "Arjun Christopher"
        "arjun christopher" -> "Arjun Christopher"
        "JOHN SMITH JR" -> "John Smith Jr"
        "A.R.RAHMAN" -> "A.R. Rahman"
        "MC DONALD" -> "Mc Donald"
        "O'BRIEN" -> "O'Brien"
        "Jean-Paul SARTRE" -> "Jean Paul Sartre" (hyphen removed)
        "arjun_christopher" -> "Arjun Christopher" (underscore removed)
    """
    if not name or not isinstance(name, str):
        return name
    
    # First, replace hyphens and underscores with spaces
    name = name.replace('-', ' ').replace('_', ' ')
    
    # Clean up extra whitespace
    name = ' '.join(name.split())
    
    # Handle edge case: initials stuck to name like "A.R.RAHMAN" -> "A.R. RAHMAN"
    # But preserve "A.R." together as one unit
    # Look for pattern where multiple dots exist followed by uppercase letters
    name = re.sub(r'([A-Z]\.[A-Z]\.?)([A-Z][A-Za-z]+)', r'\1 \2', name)
    
    def format_word(word: str) -> str:
        """Format a single word with proper capitalization"""
        if not word:
            return word
        
        # Handle initials (single letter or letter with period)
        if len(word) <= 2 and word[0].isalpha():
            # Single letter initial or "A." format
            if len(word) == 1:
                return word.upper() + '.'
            elif len(word) == 2 and word[1] == '.':
                return word[0].upper() + '.'
        
        # Handle initials with dots (e.g., "A.R." or "A.R" or "a.r.rahman" or "DR.")
        if '.' in word:
            # Check if it's a title like "Dr." or suffix
            word_lower = word.lower().rstrip('.')
            title_suffix_patterns = {
                'jr': 'Jr', 'sr': 'Sr', 
                'ii': 'II', 'iii': 'III', 'iv': 'IV', 'v': 'V',
                'phd': 'PhD', 'md': 'MD', 'esq': 'Esq',
                'dds': 'DDS', 'jd': 'JD', 'dr': 'Dr', 'mr': 'Mr',
                'mrs': 'Mrs', 'ms': 'Ms', 'prof': 'Prof'
            }
            if word_lower in title_suffix_patterns:
                return title_suffix_patterns[word_lower] + '.'
            
            # Handle multi-part initials like "A.R." or "a.r.something"
            parts = word.split('.')
            formatted_parts = []
            for i, p in enumerate(parts):
                if not p:  # Empty part (from consecutive dots or trailing dot)
                    continue
                if len(p) == 1 and p.isalpha():
                    # Single letter - make it uppercase (initial)
                    formatted_parts.append(p.upper())
                elif len(p) == 2 and p.isalpha():
                    # Two letters together like "AR" in "A.R.NAME"
                    formatted_parts.append(p.upper())
                else:
                    # Regular word part
                    formatted_parts.append(p.capitalize())
            
            result = '.'.join(formatted_parts)
            # Add trailing period if original had it and result doesn't
            if word.endswith('.') and not result.endswith('.'):
                result += '.'
            return result
        
        # Handle apostrophes (e.g., "O'Brien", "D'Angelo")
        if "'" in word:
            parts = word.split("'")
            if len(parts) == 2 and len(parts[0]) <= 2:
                # Handles O'Brien, D'Angelo
                return parts[0].capitalize() + "'" + parts[1].capitalize()
            return "'".join(part.capitalize() for part in parts)
        
        # Handle Mc/Mac prefixes (e.g., "McDonald", "MacLeod")
        if word.lower().startswith('mc') and len(word) > 2:
            return 'Mc' + word[2:].capitalize()
        
        # For "MacDonald" as single word, just capitalize normally
        if word.lower().startswith('mac') and len(word) > 3:
            return word.capitalize()
        
        # Handle common suffixes (Jr, Sr, II, III, IV, PhD, MD, etc.) without dots
        suffix_patterns = {
            'jr': 'Jr', 'sr': 'Sr', 
            'ii': 'II', 'iii': 'III', 'iv': 'IV', 'v': 'V',
            'phd': 'PhD', 'md': 'MD', 'esq': 'Esq',
            'dds': 'DDS', 'jd': 'JD', 'dr': 'Dr'
        }
        if word.lower() in suffix_patterns:
            return suffix_patterns[word.lower()]
        
        # Standard capitalization for regular words
        return word.capitalize()
    
    # Split name into words and format each
    words = name.split()
    formatted_words = [format_word(word) for word in words]
    
    return ' '.join(formatted_words)


def _calculate_name_confidence(name: str, position: int, context: str) -> float:
    """Calculate confidence score for a potential name"""
    score = 0.0
    words = name.split()
    
    # Position bonus (earlier in document = higher confidence)
    if position < 3:
        score += 50
    elif position < 10:
        score += 30
    elif position < 20:
        score += 10
    
    # Length bonus (2-4 words typical for names)
    if 2 <= len(words) <= 4:
        score += 30
    elif len(words) == 1:
        score -= 20
    
    # All words capitalized
    if all(word[0].isupper() for word in words if word):
        score += 20
    
    # No numbers or special characters except hyphens and apostrophes
    if not any(char.isdigit() or char in '!@#$%^&*()_+=[]{}|;:,.<>?/' for char in name):
        score += 15
    
    # Context clues
    context_lower = context.lower()
    if any(keyword in context_lower for keyword in SKIP_KEYWORDS):
        score -= 30
    
    # Check if words are common name parts
    for word in words:
        word_lower = word.lower().rstrip('.,')
        if word_lower in COMMON_PREFIXES:
            score += 10
        elif word_lower in COMMON_SUFFIXES:
            score += 5
        elif word_lower in EXCLUDE_WORDS:
            score -= 40
    
    return score


def extract_names_advanced(text: str, extract_emails_func=None) -> List[str]:
    """
    Advanced name extraction using multiple strategies with confidence scoring
    
    Args:
        text: Resume text to extract names from
        extract_emails_func: Optional function to extract emails (for email-based name extraction)
    
    Returns:
        List of candidate names (formatted, max 3)
    """
    candidate_names = {}  # Store names with confidence scores
    lines = text.split('\n')
    
    # Strategy 1: Look in first lines with enhanced pattern matching
    for i, line in enumerate(lines[:25]):
        original_line = line
        line = line.strip()
        
        if len(line) < 3 or len(line) > 150:
            continue
        
        # Remove common prefixes
        line_clean = re.sub(r'^(Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s*', '', line, flags=re.IGNORECASE)
        
        # Try to match name patterns
        # Pattern 1: Simple capitalized words at start of line
        if i < 5 and line_clean and line_clean[0].isupper():
            # Check if line looks like a name (not a sentence)
            if not line.endswith(('.', '!', '?')) and len(line.split()) <= 5:
                words = line_clean.split()
                # Filter out words that are likely not names
                name_words = [w for w in words if w and w[0].isupper() and not any(char.isdigit() for char in w)]
                
                if 2 <= len(name_words) <= 4:
                    potential_name = ' '.join(name_words)
                    # Clean up any trailing punctuation
                    potential_name = re.sub(r'[,;:]$', '', potential_name)
                    
                    confidence = _calculate_name_confidence(potential_name, i, original_line)
                    if confidence > 20:  # Threshold
                        if potential_name not in candidate_names or candidate_names[potential_name] < confidence:
                            candidate_names[potential_name] = confidence
        
        # Pattern 2: Name patterns with regex
        for pattern in NAME_PATTERNS:
            match = pattern.match(line_clean)
            if match:
                potential_name = match.group(1).strip()
                # Remove any trailing punctuation
                potential_name = re.sub(r'[,;:]$', '', potential_name)
                words = potential_name.split()
                
                if 2 <= len(words) <= 4 and all(len(w) >= 2 for w in words):
                    confidence = _calculate_name_confidence(potential_name, i, original_line)
                    if confidence > 20:
                        if potential_name not in candidate_names or candidate_names[potential_name] < confidence:
                            candidate_names[potential_name] = confidence
    
    # Strategy 2: Extract from email usernames (lower confidence)
    if extract_emails_func:
        try:
            emails = extract_emails_func(text)
            for email in emails:
                username = email.split('@')[0]
                # Handle various email username formats
                if '.' in username:
                    parts = username.split('.')[:2]  # Take first two parts
                    if all(part.isalpha() and len(part) >= 2 for part in parts):
                        name = ' '.join(part.title() for part in parts)
                        confidence = 15  # Lower confidence from email
                        if name not in candidate_names or candidate_names[name] < confidence:
                            candidate_names[name] = confidence
                elif '_' in username:
                    parts = username.split('_')[:2]
                    if all(part.isalpha() and len(part) >= 2 for part in parts):
                        name = ' '.join(part.title() for part in parts)
                        confidence = 15
                        if name not in candidate_names or candidate_names[name] < confidence:
                            candidate_names[name] = confidence
        except Exception as e:
            print(f"Email-based name extraction error: {e}")
    
    # Strategy 3: Use spaCy NER if available (high confidence for PERSON entities in first 5000 chars)
    if _NLP:
        try:
            doc = _NLP(text[:5000])  # Limit for performance
            for ent in doc.ents:
                if ent.label_ == "PERSON" and len(ent.text.strip()) > 3:
                    name = ent.text.strip()
                    # Calculate position-based confidence
                    position_in_text = text[:5000].find(name)
                    line_number = text[:position_in_text].count('\n') if position_in_text >= 0 else 999
                    
                    confidence = 25  # Base confidence for NER
                    if line_number < 10:
                        confidence += 25
                    elif line_number < 20:
                        confidence += 15
                    
                    if name not in candidate_names or candidate_names[name] < confidence:
                        candidate_names[name] = confidence
        except Exception as e:
            print(f"SpaCy NER error: {e}")
    
    # Strategy 4: Use transformer-based NER if available
    if _NER_PIPELINE:
        try:
            # Process first 2000 characters for efficiency
            text_sample = text[:2000]
            ner_results = _NER_PIPELINE(text_sample)
            
            for entity in ner_results:
                if entity.get('entity_group') == 'PER' or 'PER' in entity.get('entity', ''):
                    name = entity.get('word', '').strip()
                    # Clean up tokenization artifacts
                    name = name.replace('##', '').replace('Ġ', ' ').strip()
                    
                    if len(name) > 3:
                        # Position-based confidence
                        start_pos = entity.get('start', 1000)
                        line_number = text_sample[:start_pos].count('\n')
                        
                        confidence = 30  # Higher confidence for transformer
                        if line_number < 5:
                            confidence += 30
                        elif line_number < 15:
                            confidence += 15
                        
                        if name not in candidate_names or candidate_names[name] < confidence:
                            candidate_names[name] = confidence
        except Exception as e:
            print(f"Transformer NER error: {e}")
    
    # Sort by confidence and return top names
    sorted_names = sorted(candidate_names.items(), key=lambda x: x[1], reverse=True)
    
    # Return names with confidence > threshold, formatted properly
    result = [format_candidate_name(name) for name, conf in sorted_names if conf > 20]
    
    # If we found names, return top 3 max; if none, return empty list
    return result[:3] if result else []
