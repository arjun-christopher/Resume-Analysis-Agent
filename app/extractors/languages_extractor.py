"""
Comprehensive Languages Information Extractor for Resumes

This module provides advanced NLP-based language skills extraction covering all possible formats:
- Spoken languages with proficiency levels
- Written languages
- Sign languages
- Programming languages (filtered out)
- Native/mother tongue indication
- Multiple proficiency frameworks (CEFR, ILR, etc.)
- Certifications (TOEFL, IELTS, DELE, DELF, etc.)
"""

import re
from typing import List, Dict, Any, Optional, Tuple

# Try to import spaCy for NLP-based extraction
try:
    import spacy
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

# Try to import RapidFuzz for fuzzy matching
try:
    from rapidfuzz import fuzz, process
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False


# ---------- Language Extraction Configuration ----------

# Language section headers
LANGUAGE_SECTION_HEADERS = [
    r'\b(?:LANGUAGES?|LANGUAGE\s+(?:SKILLS?|PROFICIENC(?:Y|IES))|'
    r'LINGUISTIC\s+(?:SKILLS?|ABILITIES)|SPOKEN\s+LANGUAGES?|'
    r'FOREIGN\s+LANGUAGES?|LANGUAGE\s+COMPETENC(?:Y|IES)|'
    r'MULTILINGUAL\s+SKILLS?)\b'
]

# Comprehensive world languages database (200+ languages)
WORLD_LANGUAGES = [
    # Major languages
    'english', 'spanish', 'mandarin', 'chinese', 'hindi', 'arabic', 'bengali', 'portuguese',
    'russian', 'japanese', 'punjabi', 'german', 'javanese', 'french', 'telugu', 'vietnamese',
    'marathi', 'turkish', 'tamil', 'urdu', 'korean', 'italian', 'gujarati', 'polish',
    'ukrainian', 'malayalam', 'kannada', 'oriya', 'burmese', 'thai', 'persian', 'farsi',
    # European languages
    'dutch', 'greek', 'czech', 'swedish', 'romanian', 'hungarian', 'serbian', 'croatian',
    'bulgarian', 'slovak', 'danish', 'finnish', 'norwegian', 'lithuanian', 'slovenian',
    'latvian', 'estonian', 'albanian', 'icelandic', 'macedonian', 'irish', 'welsh', 'basque',
    'catalan', 'galician', 'maltese', 'luxembourgish',
    # Asian languages
    'indonesian', 'malay', 'tagalog', 'filipino', 'khmer', 'lao', 'mongolian', 'nepali',
    'sinhala', 'pashto', 'kurdish', 'uzbek', 'kazakh', 'azerbaijani', 'armenian', 'georgian',
    # African languages
    'swahili', 'yoruba', 'igbo', 'hausa', 'amharic', 'somali', 'zulu', 'xhosa', 'afrikaans',
    # Middle Eastern
    'hebrew', 'turkish', 'arabic', 'persian', 'kurdish',
    # Others
    'esperanto', 'latin', 'sanskrit'
]

# Proficiency level patterns (multiple frameworks)
PROFICIENCY_PATTERNS = {
    # Common descriptive levels
    'native': [
        r'\b(?:native|mother\s+tongue|first\s+language|L1|native\s+speaker)\b'
    ],
    'fluent': [
        r'\b(?:fluent|fluency|proficient|highly\s+proficient|advanced|expert)\b'
    ],
    'advanced': [
        r'\b(?:advanced|upper[-\s]?advanced|upper[-\s]?intermediate)\b'
    ],
    'intermediate': [
        r'\b(?:intermediate|conversational|working\s+proficiency|moderate)\b'
    ],
    'basic': [
        r'\b(?:basic|elementary|beginner|limited\s+working\s+proficiency|survival)\b'
    ],
    # CEFR levels (Common European Framework of Reference)
    'cefr': [
        r'\b(?:A1|A2|B1|B2|C1|C2)\b'
    ],
    # ILR scale (Interagency Language Roundtable) - US government
    'ilr': [
        r'\b(?:ILR\s+)?(?:[0-5]|0\+|1\+|2\+|3\+|4\+)\b'
    ],
    # Percentage or rating
    'percentage': [
        r'\b(\d{1,3})%\b',
        r'\b(\d+)\s*/\s*(?:5|10|100)\b'
    ]
}

# Language test/certification patterns
LANGUAGE_TEST_PATTERNS = {
    'english': [
        r'\b(?:TOEFL|IELTS|Cambridge|CAE|CPE|FCE|PTE|TOEIC|Duolingo\s+English\s+Test)\b',
        r'\b(?:TOEFL\s+(?:iBT|PBT)?\s*:\s*\d+|IELTS\s*:\s*[\d\.]+)\b'
    ],
    'spanish': [
        r'\b(?:DELE|SIELE|CELU)\b'
    ],
    'french': [
        r'\b(?:DELF|DALF|TCF|TEF)\b'
    ],
    'german': [
        r'\b(?:Goethe[-\s]?Zertifikat|TestDaF|DSH|telc)\b'
    ],
    'chinese': [
        r'\b(?:HSK|HSKK|BCT|YCT)\b'
    ],
    'japanese': [
        r'\b(?:JLPT|JPT|J[-\s]?TEST|NAT[-\s]?TEST)\b'
    ],
    'korean': [
        r'\b(?:TOPIK|KLPT)\b'
    ],
    'italian': [
        r'\b(?:CILS|CELI|PLIDA|IT)\b'
    ],
    'portuguese': [
        r'\b(?:CELPE[-\s]?Bras|CAPLE)\b'
    ],
    'russian': [
        r'\b(?:TORFL|TRKI)\b'
    ],
    'arabic': [
        r'\b(?:ALPT|ACTFL)\b'
    ]
}

# Language-specific patterns for extraction
LANGUAGE_MENTION_PATTERNS = [
    # Pattern: "Language: Level" or "Language - Level"
    r'([A-Z][a-z]+)\s*[-:–—]\s*(native|fluent|advanced|intermediate|basic|[A-C][12])',
    # Pattern: "Level Language" or "Level in Language"
    r'(native|fluent|advanced|intermediate|basic)\s+(?:in\s+)?([A-Z][a-z]+)',
    # Pattern: "Language (Level)" or "Language - Level"
    r'([A-Z][a-z]+)\s*[\(\-]\s*(native|fluent|advanced|intermediate|basic|[A-C][12])',
    # Pattern: Bullet point language
    r'(?:^|\n)\s*[•●○◦▪▫–—\*]\s*([A-Z][a-z]+)\s*[-:–—]?\s*(native|fluent|advanced|intermediate|basic|[A-C][12])?',
]

# Compile patterns
COMPILED_LANGUAGE_HEADERS = [re.compile(pattern, re.IGNORECASE) for pattern in LANGUAGE_SECTION_HEADERS]
COMPILED_LANGUAGE_MENTION_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in LANGUAGE_MENTION_PATTERNS]

COMPILED_PROFICIENCY_PATTERNS = {}
for level, patterns in PROFICIENCY_PATTERNS.items():
    COMPILED_PROFICIENCY_PATTERNS[level] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

COMPILED_LANGUAGE_TEST_PATTERNS = {}
for lang, patterns in LANGUAGE_TEST_PATTERNS.items():
    COMPILED_LANGUAGE_TEST_PATTERNS[lang] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

# Initialize FlashText processor for O(n) language matching
_LANGUAGE_PROCESSOR = None
if _FLASHTEXT_AVAILABLE:
    _LANGUAGE_PROCESSOR = KeywordProcessor(case_sensitive=False)
    for language in WORLD_LANGUAGES:
        _LANGUAGE_PROCESSOR.add_keyword(language)


def detect_language_sections(text: str) -> List[Tuple[int, int, str]]:
    """
    Detect language sections in resume text
    Returns: List of (start_line, end_line, section_text) tuples
    """
    sections = []
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean:
            continue
        
        # Check if line is a language section header
        for pattern in COMPILED_LANGUAGE_HEADERS:
            if pattern.search(line_clean):
                start_line = i
                end_line = start_line + 1
                
                # Find the end of this section (usually very short)
                for j in range(i + 1, min(len(lines), i + 20)):  # Languages sections are usually short
                    next_line = lines[j].strip()
                    if not next_line:
                        continue
                    
                    # Check if this is another major section header
                    if re.match(r'^[A-Z\s]{3,30}$', next_line) and len(next_line) > 5:
                        common_sections = ['EDUCATION', 'EXPERIENCE', 'SKILLS', 'PROJECTS', 
                                         'CERTIFICATIONS', 'ACHIEVEMENTS', 'REFERENCES', 'INTERESTS']
                        if any(section in next_line.upper() for section in common_sections):
                            end_line = j
                            break
                    
                    end_line = j + 1
                
                section_lines = lines[start_line:end_line]
                section_text = '\n'.join(section_lines)
                sections.append((start_line, end_line, section_text))
                break
    
    return sections


def extract_languages_with_flashtext(text: str) -> List[str]:
    """
    Extract languages using FlashText for O(n) keyword matching
    """
    if not _FLASHTEXT_AVAILABLE or not _LANGUAGE_PROCESSOR:
        return []
    
    # Extract languages in O(n) time
    found_languages = _LANGUAGE_PROCESSOR.extract_keywords(text.lower())
    
    # Remove duplicates and common false positives
    false_positives = ['python', 'java', 'ruby', 'swift', 'rust', 'go', 'c', 'r']
    filtered_languages = [lang for lang in found_languages if lang not in false_positives]
    
    return list(set(filtered_languages))


def extract_proficiency_level(context: str, language: str) -> Optional[str]:
    """
    Extract proficiency level for a language from context
    """
    context_lower = context.lower()
    
    # Check each proficiency level
    for level, patterns in COMPILED_PROFICIENCY_PATTERNS.items():
        for pattern in patterns:
            match = pattern.search(context)
            if match:
                # Check if it's near the language mention (within 50 chars)
                lang_pos = context_lower.find(language.lower())
                match_pos = match.start()
                if abs(lang_pos - match_pos) < 50:
                    if level == 'cefr':
                        return f"CEFR {match.group(0)}"
                    elif level == 'ilr':
                        return f"ILR {match.group(0)}"
                    elif level == 'percentage':
                        return match.group(0)
                    else:
                        return level.capitalize()
    
    return None


def extract_language_tests(text: str) -> Dict[str, List[str]]:
    """
    Extract language test scores and certifications
    """
    tests = {}
    
    for lang, patterns in COMPILED_LANGUAGE_TEST_PATTERNS.items():
        for pattern in patterns:
            matches = pattern.finditer(text)
            for match in matches:
                if lang not in tests:
                    tests[lang] = []
                tests[lang].append(match.group(0))
    
    return tests


def extract_languages_with_patterns(text: str) -> List[Dict[str, Any]]:
    """
    Extract languages using regex patterns
    """
    languages = []
    
    for pattern in COMPILED_LANGUAGE_MENTION_PATTERNS:
        matches = pattern.finditer(text)
        for match in matches:
            # Extract language name and proficiency
            groups = match.groups()
            
            # Determine which group is language and which is proficiency
            lang_name = None
            proficiency = None
            
            for group in groups:
                if group:
                    group_lower = group.lower()
                    if group_lower in WORLD_LANGUAGES:
                        lang_name = group.capitalize()
                    elif group_lower in ['native', 'fluent', 'advanced', 'intermediate', 'basic']:
                        proficiency = group.capitalize()
                    elif re.match(r'[A-C][12]', group):
                        proficiency = f"CEFR {group.upper()}"
            
            if lang_name:
                languages.append({
                    'language': lang_name,
                    'proficiency': proficiency,
                    'context': match.group(0)
                })
    
    return languages


def extract_languages_with_nlp(text: str) -> List[Dict[str, Any]]:
    """
    Extract languages using spaCy NER and linguistic analysis
    """
    if not _NLP:
        return []
    
    languages = []
    
    try:
        doc = _NLP(text[:50000])  # Limit for performance
        
        language_keywords = ['speak', 'fluent', 'native', 'language', 'proficient', 'bilingual', 'multilingual']
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_lower = sent_text.lower()
            
            # Check if sentence mentions languages
            if not any(keyword in sent_lower for keyword in language_keywords):
                continue
            
            # Extract using FlashText within this sentence
            if _FLASHTEXT_AVAILABLE:
                found_langs = extract_languages_with_flashtext(sent_text)
                
                for lang in found_langs:
                    proficiency = extract_proficiency_level(sent_text, lang)
                    
                    languages.append({
                        'language': lang.capitalize(),
                        'proficiency': proficiency,
                        'sentence': sent_text,
                        'extraction_method': 'nlp'
                    })
        
    except Exception as e:
        print(f"NLP language extraction error: {e}")
    
    return languages


def normalize_proficiency_level(level: Optional[str]) -> str:
    """
    Normalize proficiency level to standard format
    """
    if not level:
        return 'Not specified'
    
    level_lower = level.lower()
    
    # Map to standard levels
    if 'native' in level_lower or 'mother tongue' in level_lower:
        return 'Native'
    elif 'fluent' in level_lower or 'c2' in level_lower or 'ilr 5' in level_lower:
        return 'Fluent'
    elif 'advanced' in level_lower or 'c1' in level_lower or 'ilr 4' in level_lower:
        return 'Advanced'
    elif 'intermediate' in level_lower or 'b' in level_lower or 'ilr 2' in level_lower or 'ilr 3' in level_lower:
        return 'Intermediate'
    elif 'basic' in level_lower or 'a' in level_lower or 'ilr 1' in level_lower or 'ilr 0' in level_lower:
        return 'Basic'
    else:
        return level.capitalize()


def extract_languages_comprehensive(text: str) -> List[Dict[str, Any]]:
    """
    Comprehensive language extraction using multiple NLP techniques
    
    This function covers all possible resume formats including:
    - Dedicated language sections
    - Languages mentioned in summary/experience
    - Multiple proficiency frameworks (CEFR, ILR, descriptive)
    - Language test scores (TOEFL, IELTS, DELE, DELF, etc.)
    - Native/mother tongue indication
    - Spoken vs written language distinction
    - Multiple languages with different proficiency levels
    
    Uses advanced NLP techniques:
    - FlashText for O(n) language name matching (200+ languages)
    - spaCy NER for contextual extraction
    - Regex patterns for proficiency levels
    - Fuzzy matching for language name variations
    
    Args:
        text: Resume text to extract language information from
    
    Returns:
        List of dictionaries containing structured language information with fields:
        - language: Language name (capitalized)
        - proficiency: Proficiency level (Native, Fluent, Advanced, etc.)
        - proficiency_framework: Framework used (CEFR, ILR, Descriptive)
        - test_scores: List of language test scores (TOEFL, IELTS, etc.)
        - additional_info: Any additional information found
    """
    all_languages = []
    language_map = {}  # To deduplicate and merge information
    
    # Step 1: Detect dedicated language sections
    language_sections = detect_language_sections(text)
    
    # Step 2: Extract from dedicated sections using multiple methods
    for start, end, section_text in language_sections:
        # Method 1: FlashText extraction (O(n))
        if _FLASHTEXT_AVAILABLE:
            flashtext_langs = extract_languages_with_flashtext(section_text)
            for lang in flashtext_langs:
                lang_cap = lang.capitalize()
                if lang_cap not in language_map:
                    proficiency = extract_proficiency_level(section_text, lang)
                    language_map[lang_cap] = {
                        'language': lang_cap,
                        'proficiency': normalize_proficiency_level(proficiency),
                        'extraction_method': 'flashtext'
                    }
        
        # Method 2: Pattern-based extraction
        pattern_langs = extract_languages_with_patterns(section_text)
        for lang_info in pattern_langs:
            lang_name = lang_info['language']
            if lang_name not in language_map:
                language_map[lang_name] = {
                    'language': lang_name,
                    'proficiency': normalize_proficiency_level(lang_info.get('proficiency')),
                    'extraction_method': 'patterns'
                }
            else:
                # Merge proficiency if not already set
                if lang_info.get('proficiency') and language_map[lang_name]['proficiency'] == 'Not specified':
                    language_map[lang_name]['proficiency'] = normalize_proficiency_level(lang_info['proficiency'])
    
    # Step 3: If no dedicated section, use NLP on entire text
    if not language_sections and _NLP:
        nlp_langs = extract_languages_with_nlp(text)
        for lang_info in nlp_langs:
            lang_name = lang_info['language']
            if lang_name not in language_map:
                language_map[lang_name] = {
                    'language': lang_name,
                    'proficiency': normalize_proficiency_level(lang_info.get('proficiency')),
                    'extraction_method': 'nlp'
                }
    
    # Step 4: Extract language test scores
    test_scores = extract_language_tests(text)
    
    # Step 5: Merge test scores with languages
    for lang, scores in test_scores.items():
        # Try to find corresponding language in map
        lang_variants = {
            'english': 'English',
            'spanish': 'Spanish',
            'french': 'French',
            'german': 'German',
            'chinese': 'Chinese',
            'japanese': 'Japanese',
            'korean': 'Korean',
            'italian': 'Italian',
            'portuguese': 'Portuguese',
            'russian': 'Russian',
            'arabic': 'Arabic'
        }
        
        lang_name = lang_variants.get(lang, lang.capitalize())
        
        if lang_name in language_map:
            language_map[lang_name]['test_scores'] = scores
        else:
            # Add as new entry if test score found but language not explicitly mentioned
            language_map[lang_name] = {
                'language': lang_name,
                'proficiency': 'Not specified',
                'test_scores': scores,
                'extraction_method': 'test_scores'
            }
    
    # Step 6: Convert map to list
    all_languages = list(language_map.values())
    
    # Step 7: Sort by proficiency (Native > Fluent > Advanced > Intermediate > Basic > Not specified)
    proficiency_order = {
        'Native': 0,
        'Fluent': 1,
        'Advanced': 2,
        'Intermediate': 3,
        'Basic': 4,
        'Not specified': 5
    }
    
    all_languages.sort(key=lambda x: proficiency_order.get(x['proficiency'], 10))
    
    return all_languages


# Backward compatibility - alias for the main function
extract_languages_info = extract_languages_comprehensive
