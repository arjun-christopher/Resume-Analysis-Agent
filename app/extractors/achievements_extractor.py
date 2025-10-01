"""
Comprehensive Achievements Information Extractor for Resumes

This module provides advanced NLP-based achievements extraction covering all possible resume formats:
- Quantifiable achievements with metrics
- Awards and recognitions
- Publications and patents
- Performance improvements
- Revenue/cost impacts
- Leadership achievements
- Technical achievements
- Academic achievements
- Competition wins
- Certifications earned
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


# ---------- Achievements Extraction Configuration ----------

# Achievement section headers
ACHIEVEMENT_SECTION_HEADERS = [
    r'\b(?:ACHIEVEMENTS?|ACCOMPLISHMENTS?|AWARDS?\s+(?:AND|&)\s+(?:ACHIEVEMENTS?|HONORS?|RECOGNITIONS?)|'
    r'KEY\s+ACHIEVEMENTS?|MAJOR\s+ACHIEVEMENTS?|NOTABLE\s+ACHIEVEMENTS?|'
    r'HONORS?\s+(?:AND|&)\s+AWARDS?|RECOGNITIONS?|DISTINCTIONS?|'
    r'AWARDS?\s+(?:AND|&)\s+CERTIFICATIONS?|PROFESSIONAL\s+ACHIEVEMENTS?|'
    r'ACADEMIC\s+ACHIEVEMENTS?|CAREER\s+HIGHLIGHTS?|HIGHLIGHTS?)\b'
]

# Quantifiable achievement patterns (with metrics/numbers)
QUANTIFIABLE_PATTERNS = [
    # Percentage improvements
    r'\b(?:increased|improved|enhanced|boosted|grew|raised|elevated)\s+(?:[A-Za-z\s]+?\s+)?by\s+(\d+(?:\.\d+)?%)',
    r'\b(?:reduced|decreased|lowered|cut|minimized|saved)\s+(?:[A-Za-z\s]+?\s+)?by\s+(\d+(?:\.\d+)?%)',
    r'\b(\d+(?:\.\d+)?%)\s+(?:increase|improvement|growth|reduction|decrease)',
    
    # Multiplier improvements (2x, 3x, etc.)
    r'\b(\d+x|\d+\s+times?)\s+(?:faster|better|more|higher|greater|improvement)',
    r'\b(?:increased|improved)\s+(?:[A-Za-z\s]+?\s+)?by\s+(\d+x|\d+\s+times?)',
    
    # Dollar amounts
    r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:million|M|billion|B|thousand|K)?',
    r'\b(?:saved|generated|earned|revenue\s+of|profit\s+of|worth)\s+\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:million|M|billion|B|thousand|K)?',
    
    # User/customer numbers
    r'\b(\d+(?:,\d{3})*(?:\+)?)\s+(?:users?|customers?|clients?|subscribers?|members?|downloads?|installs?)',
    r'\b(?:reached|achieved|served|acquired)\s+(\d+(?:,\d{3})*(?:\+)?)\s+(?:users?|customers?|clients?)',
    
    # Time improvements
    r'\b(?:reduced|decreased|cut)\s+(?:[A-Za-z\s]+?\s+)?(?:time|duration|latency)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|minutes?|mins?|seconds?|secs?|days?|weeks?|months?)',
    r'\b(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|minutes?|mins?|seconds?|secs?)\s+(?:faster|quicker|reduction)',
    
    # Rankings and positions
    r'\b(?:ranked|positioned|placed)\s+(?:#)?(\d+)(?:st|nd|rd|th)?\s+(?:in|among|out\s+of)',
    r'\b(?:top|first|#1)\s+(?:in|among|performer)',
    
    # Scale achievements
    r'\b(?:managed|led|supervised|oversaw)\s+(?:a\s+)?(?:team\s+of\s+)?(\d+(?:\+)?)\s+(?:people|members|employees|developers|engineers)',
    r'\b(\d+(?:\+)?)\s+(?:projects?|initiatives?|campaigns?|products?)\s+(?:delivered|completed|launched)',
]

# Action verb patterns for achievements
ACTION_VERBS = [
    'achieved', 'accomplished', 'delivered', 'exceeded', 'surpassed', 'outperformed',
    'increased', 'improved', 'enhanced', 'optimized', 'boosted', 'maximized',
    'reduced', 'decreased', 'minimized', 'eliminated', 'saved', 'cut',
    'generated', 'created', 'developed', 'built', 'established', 'launched',
    'led', 'managed', 'directed', 'coordinated', 'spearheaded', 'pioneered',
    'won', 'earned', 'received', 'awarded', 'recognized', 'honored',
    'published', 'presented', 'spoke', 'contributed', 'collaborated',
    'transformed', 'revolutionized', 'innovated', 'streamlined', 'automated',
    'implemented', 'executed', 'delivered', 'completed', 'finished'
]

# Award and recognition patterns
AWARD_PATTERNS = [
    r'\b(?:won|received|awarded|earned|achieved)\s+(?:the\s+)?([A-Z][A-Za-z\s\-&]{3,60}?)\s+(?:award|prize|recognition|honor|medal|trophy|scholarship)',
    r'\b([A-Z][A-Za-z\s\-&]{3,60}?)\s+(?:award|prize|recognition|honor|medal|trophy|scholarship)\s+(?:winner|recipient|laureate)',
    r'\b(?:awarded|recognized\s+as|named)\s+([A-Z][A-Za-z\s\-&]{3,60}?)(?:\s+(?:of\s+the\s+)?(?:year|month|quarter))?',
    r'\b(?:employee|performer|contributor|developer|engineer)\s+of\s+the\s+(?:year|month|quarter)',
]

# Publication patterns
PUBLICATION_PATTERNS = [
    r'\b(?:published|authored|co-authored)\s+(?:a\s+)?(?:paper|article|book|chapter|thesis)\s+(?:in|at|on)',
    r'\b(?:publication|paper|article)[\s:]+["\']([^"\']{10,150})["\']',
    r'\b(\d+)\s+(?:publications?|papers?|articles?)\s+(?:published|in)',
]

# Patent patterns
PATENT_PATTERNS = [
    r'\b(?:patent|patents?)\s+(?:filed|granted|pending|awarded)',
    r'\b(?:filed|granted|received)\s+(\d+)\s+patents?',
    r'\b(?:inventor|co-inventor)\s+(?:of|on)\s+(\d+)\s+patents?',
]

# Competition/Hackathon patterns
COMPETITION_PATTERNS = [
    r'\b(?:won|winner|1st\s+place|first\s+place|champion)\s+(?:at|in|of)\s+([A-Z][A-Za-z\s\-&]{3,60}?)(?:\s+(?:hackathon|competition|contest|challenge))',
    r'\b([A-Z][A-Za-z\s\-&]{3,60}?)\s+(?:hackathon|competition|contest|challenge)\s+(?:winner|champion|finalist)',
    r'\b(?:finalist|runner-up|top\s+\d+)\s+(?:at|in)\s+([A-Z][A-Za-z\s\-&]{3,60}?)(?:\s+(?:hackathon|competition))?',
]

# Performance rating patterns
PERFORMANCE_PATTERNS = [
    r'\b(?:rated|scored|achieved)\s+(\d+(?:\.\d+)?)\s*(?:/|out\s+of)\s*(\d+(?:\.\d+)?)',
    r'\b(?:performance\s+rating|rating|score)[\s:]*(\d+(?:\.\d+)?)\s*(?:/|out\s+of)\s*(\d+(?:\.\d+)?)',
    r'\b(?:exceeded|surpassed)\s+(?:performance\s+)?(?:targets?|goals?|expectations?|KPIs?)',
]

# Leadership achievement patterns
LEADERSHIP_PATTERNS = [
    r'\b(?:led|managed|directed|supervised)\s+(?:a\s+)?team\s+of\s+(\d+)\s+(?:members?|people|developers?|engineers?)',
    r'\b(?:mentored|coached|trained)\s+(\d+)\s+(?:members?|people|developers?|engineers?|interns?)',
    r'\b(?:promoted|advanced)\s+to\s+([A-Z][A-Za-z\s]{3,40})',
]

# Academic achievement patterns
ACADEMIC_PATTERNS = [
    r'\b(?:GPA|CGPA)[\s:]*(\d+\.\d+)\s*(?:/|out\s+of)\s*(\d+\.\d+)',
    r'\b(?:graduated|degree)\s+(?:with\s+)?(?:summa\s+cum\s+laude|magna\s+cum\s+laude|cum\s+laude|honors?|distinction|first\s+class)',
    r'\b(?:valedictorian|salutatorian|top\s+of\s+(?:the\s+)?class)',
    r'\b(?:dean\'?s\s+list|honor\s+roll|merit\s+list)',
    r'\b(?:scholarship|fellowship)\s+(?:recipient|winner|awarded)',
]

# Speaking/Presentation patterns
SPEAKING_PATTERNS = [
    r'\b(?:presented|spoke|keynote)\s+at\s+([A-Z][A-Za-z\s\-&]{3,60}?)(?:\s+(?:conference|summit|event|meetup))?',
    r'\b(?:speaker|presenter)\s+at\s+([A-Z][A-Za-z\s\-&]{3,60}?)',
    r'\b(?:delivered|gave)\s+(?:a\s+)?(?:talk|presentation|keynote|speech)\s+(?:at|on)',
]

# Media/Press patterns
MEDIA_PATTERNS = [
    r'\b(?:featured|mentioned|interviewed|quoted)\s+in\s+([A-Z][A-Za-z\s\-&]{3,60}?)(?:\s+(?:magazine|newspaper|blog|podcast))?',
    r'\b(?:press\s+coverage|media\s+coverage)\s+in',
]

# Compile patterns
COMPILED_ACHIEVEMENT_HEADERS = [re.compile(pattern, re.IGNORECASE) for pattern in ACHIEVEMENT_SECTION_HEADERS]
COMPILED_QUANTIFIABLE_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in QUANTIFIABLE_PATTERNS]
COMPILED_AWARD_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in AWARD_PATTERNS]
COMPILED_PUBLICATION_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in PUBLICATION_PATTERNS]
COMPILED_PATENT_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in PATENT_PATTERNS]
COMPILED_COMPETITION_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in COMPETITION_PATTERNS]
COMPILED_PERFORMANCE_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in PERFORMANCE_PATTERNS]
COMPILED_LEADERSHIP_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in LEADERSHIP_PATTERNS]
COMPILED_ACADEMIC_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in ACADEMIC_PATTERNS]
COMPILED_SPEAKING_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in SPEAKING_PATTERNS]
COMPILED_MEDIA_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in MEDIA_PATTERNS]

# Create action verb pattern
ACTION_VERB_PATTERN = re.compile(r'\b(' + '|'.join(ACTION_VERBS) + r')\b', re.IGNORECASE)


def detect_achievement_sections(text: str) -> List[Tuple[int, int, str]]:
    """
    Detect achievement sections in resume text
    Returns: List of (start_line, end_line, section_text) tuples
    """
    sections = []
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean:
            continue
        
        # Check if line is an achievement section header
        for pattern in COMPILED_ACHIEVEMENT_HEADERS:
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
                        common_sections = ['EDUCATION', 'EXPERIENCE', 'WORK', 'EMPLOYMENT', 'SKILLS', 
                                         'PROJECTS', 'CERTIFICATIONS', 'SUMMARY', 'OBJECTIVE']
                        if any(section in next_line.upper() for section in common_sections):
                            end_line = j
                            break
                    
                    end_line = j + 1
                
                section_lines = lines[start_line:end_line]
                section_text = '\n'.join(section_lines)
                sections.append((start_line, end_line, section_text))
                break
    
    return sections


def extract_quantifiable_achievements(text: str) -> List[Dict[str, Any]]:
    """Extract achievements with quantifiable metrics"""
    achievements = []
    
    for pattern in COMPILED_QUANTIFIABLE_PATTERNS:
        matches = pattern.finditer(text)
        for match in matches:
            # Get the full sentence containing the achievement
            start_pos = max(0, match.start() - 100)
            end_pos = min(len(text), match.end() + 100)
            context = text[start_pos:end_pos]
            
            # Find sentence boundaries
            sentences = re.split(r'[.!?\n]', context)
            achievement_sentence = None
            for sent in sentences:
                if match.group(0) in sent:
                    achievement_sentence = sent.strip()
                    break
            
            if achievement_sentence and len(achievement_sentence) > 15:
                achievement = {
                    'type': 'quantifiable',
                    'description': achievement_sentence,
                    'metric': match.group(0),
                    'category': categorize_achievement(achievement_sentence)
                }
                achievements.append(achievement)
    
    return achievements


def extract_awards(text: str) -> List[Dict[str, Any]]:
    """Extract awards and recognitions"""
    awards = []
    
    for pattern in COMPILED_AWARD_PATTERNS:
        matches = pattern.finditer(text)
        for match in matches:
            award_text = match.group(0).strip()
            
            # Try to extract award name
            award_name = None
            if match.lastindex and match.lastindex >= 1:
                award_name = match.group(1).strip()
            
            achievement = {
                'type': 'award',
                'description': award_text,
                'award_name': award_name,
                'category': 'recognition'
            }
            awards.append(achievement)
    
    return awards


def extract_publications(text: str) -> List[Dict[str, Any]]:
    """Extract publications and papers"""
    publications = []
    
    for pattern in COMPILED_PUBLICATION_PATTERNS:
        matches = pattern.finditer(text)
        for match in matches:
            pub_text = match.group(0).strip()
            
            # Try to extract publication title
            pub_title = None
            if match.lastindex and match.lastindex >= 1:
                pub_title = match.group(1).strip()
            
            achievement = {
                'type': 'publication',
                'description': pub_text,
                'title': pub_title,
                'category': 'academic'
            }
            publications.append(achievement)
    
    return publications


def extract_patents(text: str) -> List[Dict[str, Any]]:
    """Extract patent information"""
    patents = []
    
    for pattern in COMPILED_PATENT_PATTERNS:
        matches = pattern.finditer(text)
        for match in matches:
            patent_text = match.group(0).strip()
            
            # Try to extract patent count
            patent_count = None
            if match.lastindex and match.lastindex >= 1:
                try:
                    patent_count = int(match.group(1))
                except (ValueError, IndexError):
                    pass
            
            achievement = {
                'type': 'patent',
                'description': patent_text,
                'count': patent_count,
                'category': 'innovation'
            }
            patents.append(achievement)
    
    return patents


def extract_competitions(text: str) -> List[Dict[str, Any]]:
    """Extract competition wins and placements"""
    competitions = []
    
    for pattern in COMPILED_COMPETITION_PATTERNS:
        matches = pattern.finditer(text)
        for match in matches:
            comp_text = match.group(0).strip()
            
            # Try to extract competition name
            comp_name = None
            if match.lastindex and match.lastindex >= 1:
                comp_name = match.group(1).strip()
            
            achievement = {
                'type': 'competition',
                'description': comp_text,
                'competition_name': comp_name,
                'category': 'competition'
            }
            competitions.append(achievement)
    
    return competitions


def extract_performance_ratings(text: str) -> List[Dict[str, Any]]:
    """Extract performance ratings and reviews"""
    ratings = []
    
    for pattern in COMPILED_PERFORMANCE_PATTERNS:
        matches = pattern.finditer(text)
        for match in matches:
            rating_text = match.group(0).strip()
            
            # Try to extract rating value
            rating_value = None
            rating_scale = None
            if match.lastindex >= 2:
                try:
                    rating_value = float(match.group(1))
                    rating_scale = float(match.group(2))
                except (ValueError, IndexError):
                    pass
            
            achievement = {
                'type': 'performance',
                'description': rating_text,
                'rating_value': rating_value,
                'rating_scale': rating_scale,
                'category': 'performance'
            }
            ratings.append(achievement)
    
    return ratings


def extract_leadership_achievements(text: str) -> List[Dict[str, Any]]:
    """Extract leadership-related achievements"""
    leadership = []
    
    for pattern in COMPILED_LEADERSHIP_PATTERNS:
        matches = pattern.finditer(text)
        for match in matches:
            lead_text = match.group(0).strip()
            
            # Try to extract team size or position
            detail = None
            if match.lastindex and match.lastindex >= 1:
                detail = match.group(1).strip()
            
            achievement = {
                'type': 'leadership',
                'description': lead_text,
                'detail': detail,
                'category': 'leadership'
            }
            leadership.append(achievement)
    
    return leadership


def extract_academic_achievements(text: str) -> List[Dict[str, Any]]:
    """Extract academic achievements"""
    academic = []
    
    for pattern in COMPILED_ACADEMIC_PATTERNS:
        matches = pattern.finditer(text)
        for match in matches:
            acad_text = match.group(0).strip()
            
            achievement = {
                'type': 'academic',
                'description': acad_text,
                'category': 'academic'
            }
            academic.append(achievement)
    
    return academic


def extract_speaking_engagements(text: str) -> List[Dict[str, Any]]:
    """Extract speaking engagements and presentations"""
    speaking = []
    
    for pattern in COMPILED_SPEAKING_PATTERNS:
        matches = pattern.finditer(text)
        for match in matches:
            speak_text = match.group(0).strip()
            
            # Try to extract event name
            event_name = None
            if match.lastindex and match.lastindex >= 1:
                event_name = match.group(1).strip()
            
            achievement = {
                'type': 'speaking',
                'description': speak_text,
                'event_name': event_name,
                'category': 'professional'
            }
            speaking.append(achievement)
    
    return speaking


def extract_media_mentions(text: str) -> List[Dict[str, Any]]:
    """Extract media mentions and press coverage"""
    media = []
    
    for pattern in COMPILED_MEDIA_PATTERNS:
        matches = pattern.finditer(text)
        for match in matches:
            media_text = match.group(0).strip()
            
            # Try to extract media outlet
            outlet = None
            if match.lastindex and match.lastindex >= 1:
                outlet = match.group(1).strip()
            
            achievement = {
                'type': 'media',
                'description': media_text,
                'outlet': outlet,
                'category': 'recognition'
            }
            media.append(achievement)
    
    return media


def extract_bullet_achievements(text: str) -> List[Dict[str, Any]]:
    """Extract achievements from bullet points with action verbs"""
    achievements = []
    lines = text.split('\n')
    
    for line in lines:
        line_clean = line.strip()
        
        # Check if line starts with bullet point
        if re.match(r'^(?:•|●|○|◦|▪|▫|–|—|\*|\d+\.|\d+\))\s+', line_clean):
            # Remove bullet point
            content = re.sub(r'^(?:•|●|○|◦|▪|▫|–|—|\*|\d+\.|\d+\))\s+', '', line_clean)
            
            # Check if starts with action verb
            if ACTION_VERB_PATTERN.match(content):
                # Check if it contains numbers or metrics
                has_metric = bool(re.search(r'\d+', content))
                
                if len(content) > 20 and len(content) < 500:
                    achievement = {
                        'type': 'bullet_point',
                        'description': content,
                        'has_metric': has_metric,
                        'category': categorize_achievement(content)
                    }
                    achievements.append(achievement)
    
    return achievements


def categorize_achievement(text: str) -> str:
    """Categorize achievement based on content"""
    text_lower = text.lower()
    
    # Category keywords
    categories = {
        'revenue': ['revenue', 'sales', 'profit', 'income', 'earnings', '$'],
        'performance': ['performance', 'efficiency', 'productivity', 'speed', 'faster', 'optimization'],
        'cost_savings': ['saved', 'reduced cost', 'cost reduction', 'savings', 'cut costs'],
        'user_growth': ['users', 'customers', 'subscribers', 'downloads', 'installs', 'engagement'],
        'quality': ['quality', 'accuracy', 'reliability', 'uptime', 'defects', 'bugs'],
        'team': ['team', 'led', 'managed', 'mentored', 'coached', 'supervised'],
        'innovation': ['innovative', 'pioneered', 'first', 'new', 'novel', 'patent'],
        'recognition': ['award', 'recognized', 'honored', 'winner', 'achievement'],
        'technical': ['developed', 'built', 'implemented', 'architected', 'engineered'],
        'leadership': ['led', 'directed', 'managed', 'spearheaded', 'coordinated'],
    }
    
    for category, keywords in categories.items():
        if any(keyword in text_lower for keyword in keywords):
            return category
    
    return 'general'


def deduplicate_achievements(achievements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate achievements"""
    seen = set()
    unique_achievements = []
    
    for achievement in achievements:
        desc = achievement.get('description', '').lower().strip()
        if desc and desc not in seen and len(desc) > 15:
            seen.add(desc)
            unique_achievements.append(achievement)
    
    return unique_achievements


def extract_achievements_comprehensive(text: str) -> List[Dict[str, Any]]:
    """
    Comprehensive achievement extraction using multiple techniques
    
    This function covers all possible resume formats including:
    - Quantifiable achievements with metrics (%, $, numbers)
    - Awards and recognitions
    - Publications and research papers
    - Patents and innovations
    - Competition wins and placements
    - Performance ratings and reviews
    - Leadership achievements
    - Academic achievements (GPA, honors, scholarships)
    - Speaking engagements and presentations
    - Media mentions and press coverage
    - Bullet point achievements with action verbs
    
    Handles various formats:
    - Dedicated achievements section
    - Embedded in experience/education sections
    - Bullet point lists
    - Paragraph format
    - Mixed formats
    
    Args:
        text: Resume text to extract achievement information from
    
    Returns:
        List of dictionaries containing structured achievement information with fields:
        - type: Achievement type (quantifiable, award, publication, etc.)
        - description: Full achievement description
        - category: Achievement category (revenue, performance, recognition, etc.)
        - metric: Extracted metric (for quantifiable achievements)
        - Additional type-specific fields
    """
    all_achievements = []
    
    # Step 1: Detect dedicated achievement sections
    achievement_sections = detect_achievement_sections(text)
    
    # Step 2: Extract from dedicated sections
    for start, end, section_text in achievement_sections:
        # Extract different types of achievements
        all_achievements.extend(extract_quantifiable_achievements(section_text))
        all_achievements.extend(extract_awards(section_text))
        all_achievements.extend(extract_publications(section_text))
        all_achievements.extend(extract_patents(section_text))
        all_achievements.extend(extract_competitions(section_text))
        all_achievements.extend(extract_performance_ratings(section_text))
        all_achievements.extend(extract_leadership_achievements(section_text))
        all_achievements.extend(extract_academic_achievements(section_text))
        all_achievements.extend(extract_speaking_engagements(section_text))
        all_achievements.extend(extract_media_mentions(section_text))
        all_achievements.extend(extract_bullet_achievements(section_text))
    
    # Step 3: If no dedicated section, extract from entire text
    if not achievement_sections:
        all_achievements.extend(extract_quantifiable_achievements(text))
        all_achievements.extend(extract_awards(text))
        all_achievements.extend(extract_publications(text))
        all_achievements.extend(extract_patents(text))
        all_achievements.extend(extract_competitions(text))
        all_achievements.extend(extract_performance_ratings(text))
        all_achievements.extend(extract_leadership_achievements(text))
        all_achievements.extend(extract_academic_achievements(text))
        all_achievements.extend(extract_speaking_engagements(text))
        all_achievements.extend(extract_media_mentions(text))
        all_achievements.extend(extract_bullet_achievements(text))
    
    # Step 4: Deduplicate
    all_achievements = deduplicate_achievements(all_achievements)
    
    # Step 5: Sort by type priority and add metadata
    type_priority = {
        'quantifiable': 1,
        'award': 2,
        'competition': 3,
        'patent': 4,
        'publication': 5,
        'speaking': 6,
        'performance': 7,
        'leadership': 8,
        'academic': 9,
        'media': 10,
        'bullet_point': 11
    }
    
    for achievement in all_achievements:
        achievement['priority'] = type_priority.get(achievement.get('type'), 99)
    
    all_achievements.sort(key=lambda x: x.get('priority', 99))
    
    # Remove priority field from output
    for achievement in all_achievements:
        achievement.pop('priority', None)
    
    # Step 6: Add statistics
    achievement_stats = {
        'total_count': len(all_achievements),
        'by_type': {},
        'by_category': {},
        'has_metrics': sum(1 for a in all_achievements if a.get('has_metric') or a.get('metric')),
    }
    
    for achievement in all_achievements:
        # Count by type
        ach_type = achievement.get('type', 'unknown')
        achievement_stats['by_type'][ach_type] = achievement_stats['by_type'].get(ach_type, 0) + 1
        
        # Count by category
        ach_category = achievement.get('category', 'general')
        achievement_stats['by_category'][ach_category] = achievement_stats['by_category'].get(ach_category, 0) + 1
    
    return {
        'achievements': all_achievements,
        'statistics': achievement_stats
    }


# Backward compatibility - alias for the main function
extract_achievements_info = extract_achievements_comprehensive
