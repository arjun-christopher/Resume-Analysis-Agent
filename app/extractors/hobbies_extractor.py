"""
Comprehensive Hobbies and Interests Extractor for Resumes

This module provides advanced NLP-based extraction for hobbies and personal interests,
covering all possible resume formats and hobby categories with high computational speed.

Categories covered:
- Sports and Fitness (individual and team sports)
- Arts and Creativity (music, visual arts, writing, performing arts)
- Technology and Gaming (programming, gaming, gadgets)
- Outdoor Activities (hiking, camping, adventure sports)
- Intellectual Pursuits (reading, learning, puzzles)
- Social and Community (volunteering, socializing, clubs)
- Collection and Hobbies (collecting, crafts, DIY)
- Travel and Cultural (exploring, languages, cultural activities)
- Entertainment (movies, TV, podcasts, streaming)
- Wellness and Mindfulness (yoga, meditation, wellness)

Performance optimizations:
- FlashText O(n) keyword extraction (10-100x faster than regex)
- Text size limiting (50K chars for NLP processing)
- Pre-compiled regex patterns
- Graceful fallbacks without optional dependencies
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict

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


# ---------- Hobbies/Interests Extraction Configuration ----------

# Hobbies section headers
HOBBIES_SECTION_HEADERS = [
    r'\b(?:HOBBIES?|INTERESTS?|PERSONAL\s+INTERESTS?|HOBBIES?\s+(?:AND|&)\s+INTERESTS?|'
    r'LEISURE\s+ACTIVITIES?|RECREATIONAL\s+ACTIVITIES?|PASTIMES?|'
    r'OUTSIDE\s+(?:INTERESTS?|WORK)|EXTRACURRICULAR\s+INTERESTS?|'
    r'PERSONAL\s+ACTIVITIES?|OTHER\s+INTERESTS?|ADDITIONAL\s+INTERESTS?)\b'
]

# Comprehensive hobbies database organized by category
HOBBIES_DATABASE = {
    'sports': [
        # Team sports
        'soccer', 'football', 'basketball', 'baseball', 'volleyball', 'hockey',
        'rugby', 'cricket', 'lacrosse', 'softball', 'handball', 'water polo',
        
        # Individual sports
        'tennis', 'golf', 'swimming', 'running', 'cycling', 'marathon', 'triathlon',
        'track and field', 'athletics', 'jogging', 'sprinting', 'long distance running',
        
        # Fitness
        'gym', 'fitness', 'weightlifting', 'powerlifting', 'bodybuilding', 'crossfit',
        'cardio', 'aerobics', 'pilates', 'strength training', 'workout', 'exercise',
        
        # Martial arts
        'martial arts', 'karate', 'judo', 'taekwondo', 'boxing', 'kickboxing',
        'jiu-jitsu', 'muay thai', 'kung fu', 'aikido', 'krav maga', 'mma',
        
        # Racket sports
        'badminton', 'squash', 'table tennis', 'ping pong', 'racquetball',
        
        # Water sports
        'surfing', 'kayaking', 'canoeing', 'rowing', 'sailing', 'windsurfing',
        'kitesurfing', 'paddleboarding', 'scuba diving', 'snorkeling', 'diving',
        
        # Winter sports
        'skiing', 'snowboarding', 'ice skating', 'figure skating', 'ice hockey',
        'sledding', 'curling',
        
        # Extreme sports
        'skateboarding', 'bmx', 'parkour', 'rock climbing', 'bouldering',
        'mountain biking', 'skydiving', 'paragliding', 'bungee jumping',
    ],
    
    'arts': [
        # Visual arts
        'painting', 'drawing', 'sketching', 'illustration', 'calligraphy',
        'photography', 'digital art', 'graphic design', 'sculpture', 'pottery',
        'ceramics', 'printmaking', 'watercolor', 'oil painting', 'acrylic painting',
        
        # Music
        'music', 'playing guitar', 'playing piano', 'playing drums', 'playing violin',
        'playing bass', 'singing', 'composing', 'songwriting', 'music production',
        'djing', 'beatmaking', 'playing saxophone', 'playing flute', 'playing cello',
        'playing ukulele', 'playing harmonica', 'playing keyboard', 'band',
        
        # Performing arts
        'acting', 'theater', 'drama', 'dancing', 'ballet', 'contemporary dance',
        'hip hop dance', 'salsa', 'ballroom dancing', 'choreography', 'stand-up comedy',
        'improvisation', 'performance art',
        
        # Writing
        'writing', 'creative writing', 'poetry', 'blogging', 'journaling',
        'screenwriting', 'storytelling', 'content creation',
        
        # Crafts
        'knitting', 'crocheting', 'sewing', 'embroidery', 'quilting', 'needlework',
        'origami', 'paper crafts', 'woodworking', 'carpentry', 'metalworking',
        'jewelry making', 'beading', 'candle making', 'soap making',
    ],
    
    'technology': [
        # Programming & Tech
        'coding', 'programming', 'software development', 'web development',
        'app development', 'open source', 'github', 'hackathons', 'competitive programming',
        'algorithm design', 'machine learning', 'ai', 'data science', 'robotics',
        'electronics', 'arduino', 'raspberry pi', '3d printing', 'iot',
        
        # Gaming
        'gaming', 'video games', 'pc gaming', 'console gaming', 'esports',
        'game development', 'game design', 'streaming', 'twitch', 'speedrunning',
        'board games', 'card games', 'chess', 'strategy games', 'rpg',
        
        # Tech hobbies
        'building computers', 'hardware', 'drones', 'virtual reality', 'vr',
        'augmented reality', 'cryptocurrency', 'blockchain', 'cybersecurity',
    ],
    
    'outdoor': [
        # Hiking & Nature
        'hiking', 'trekking', 'backpacking', 'camping', 'mountaineering',
        'nature walks', 'birdwatching', 'wildlife photography', 'stargazing',
        'astronomy', 'geocaching', 'orienteering', 'trail running',
        
        # Adventure
        'adventure sports', 'rock climbing', 'caving', 'spelunking', 'rafting',
        'white water rafting', 'hunting', 'fishing', 'fly fishing', 'archery',
        
        # Gardening
        'gardening', 'horticulture', 'landscaping', 'organic gardening',
        'urban gardening', 'permaculture', 'plant care', 'bonsai',
    ],
    
    'intellectual': [
        # Reading & Learning
        'reading', 'book clubs', 'literature', 'fiction', 'non-fiction',
        'poetry reading', 'audiobooks', 'learning', 'self-improvement',
        'online courses', 'moocs', 'language learning', 'studying',
        
        # Puzzles & Games
        'puzzles', 'crosswords', 'sudoku', 'brain teasers', 'logic puzzles',
        'rubiks cube', 'chess', 'go', 'scrabble', 'trivia', 'quiz',
        
        # Research & Analysis
        'research', 'writing', 'philosophy', 'history', 'science',
        'documentaries', 'podcasts', 'ted talks', 'debate', 'public speaking',
    ],
    
    'social': [
        # Community & Volunteering
        'volunteering', 'community service', 'charity work', 'fundraising',
        'mentoring', 'coaching', 'teaching', 'tutoring', 'youth programs',
        'animal rescue', 'environmental activism', 'social work',
        
        # Social activities
        'socializing', 'networking', 'meeting new people', 'attending events',
        'parties', 'concerts', 'festivals', 'social clubs', 'meetups',
        'wine tasting', 'beer tasting', 'food tasting', 'dining out',
        'cooking classes', 'book clubs', 'discussion groups',
    ],
    
    'collection': [
        # Collecting
        'collecting', 'stamp collecting', 'coin collecting', 'antiques',
        'vintage items', 'memorabilia', 'art collecting', 'book collecting',
        'vinyl records', 'action figures', 'model building', 'scale models',
        'lego', 'miniatures', 'trading cards',
        
        # DIY & Making
        'diy projects', 'home improvement', 'restoration', 'upcycling',
        'repurposing', 'furniture restoration', 'car restoration', 'fixing things',
    ],
    
    'travel': [
        # Travel & Exploration
        'traveling', 'travel', 'backpacking', 'exploring', 'road trips',
        'adventure travel', 'cultural tourism', 'ecotourism', 'solo travel',
        'travel photography', 'travel blogging', 'exploring new cultures',
        'learning languages', 'cultural exchange', 'international cuisine',
        'food tourism', 'visiting museums', 'historical sites', 'world heritage sites',
    ],
    
    'entertainment': [
        # Media consumption
        'movies', 'cinema', 'film', 'tv shows', 'series', 'netflix',
        'streaming', 'anime', 'manga', 'comics', 'graphic novels',
        'podcasts', 'radio', 'music festivals', 'concerts', 'live music',
        'theater', 'broadway', 'opera', 'stand-up comedy',
    ],
    
    'wellness': [
        # Wellness & Mindfulness
        'yoga', 'meditation', 'mindfulness', 'tai chi', 'qigong',
        'breathing exercises', 'wellness', 'health', 'nutrition',
        'healthy eating', 'meal prep', 'cooking', 'baking', 'culinary arts',
        'wine appreciation', 'coffee appreciation', 'tea ceremony',
        'spa', 'massage', 'aromatherapy', 'holistic health', 'self-care',
    ],
}

# Initialize FlashText processor with all hobbies
_HOBBIES_PROCESSOR = None
if _FLASHTEXT_AVAILABLE:
    _HOBBIES_PROCESSOR = KeywordProcessor(case_sensitive=False)
    
    # Add all hobbies from database with their categories
    for category, hobbies in HOBBIES_DATABASE.items():
        for hobby in hobbies:
            # Add hobby with category as clean name
            _HOBBIES_PROCESSOR.add_keyword(hobby, (hobby, category))

# Interest-related verbs and patterns
INTEREST_VERBS = [
    'enjoy', 'love', 'like', 'passion', 'passionate about', 'enthusiastic about',
    'interested in', 'fascinated by', 'fond of', 'keen on', 'into', 'hobby',
    'spare time', 'free time', 'leisure time', 'pastime', 'recreational',
]

# Compile patterns for performance
COMPILED_HOBBIES_HEADERS = [re.compile(pattern, re.IGNORECASE) for pattern in HOBBIES_SECTION_HEADERS]

# Interest patterns for extraction
INTEREST_PATTERNS = [
    r'\b(?:enjoy|love|like|interested\s+in|passionate\s+about|enthusiastic\s+about)\s+([a-z\s,]+?)(?:\.|,|;|\n|$)',
    r'\b(?:hobbies?|interests?|pastimes?)\s*:?\s*([^\n\.]+)',
    r'\b(?:in\s+my\s+)?(?:spare|free|leisure)\s+time[,\s]+(?:I\s+)?([^\n\.]+)',
]

COMPILED_INTEREST_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in INTEREST_PATTERNS]


def detect_hobbies_section(text: str) -> List[Tuple[int, int, str]]:
    """
    Detect hobbies/interests sections in resume text
    
    Returns:
        List of (start_line, end_line, section_text) tuples
    """
    sections = []
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean:
            continue
        
        # Check if line is a hobbies section header
        for pattern in COMPILED_HOBBIES_HEADERS:
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
                        common_sections = [
                            'EDUCATION', 'EXPERIENCE', 'WORK', 'EMPLOYMENT', 'SKILLS',
                            'PROJECTS', 'CERTIFICATIONS', 'ACHIEVEMENTS', 'ACTIVITIES',
                            'PUBLICATIONS', 'REFERENCES', 'LANGUAGES'
                        ]
                        if any(section in next_line.upper() for section in common_sections):
                            end_line = j
                            break
                    
                    end_line = j + 1
                
                section_lines = lines[start_line:end_line]
                section_text = '\n'.join(section_lines)
                sections.append((start_line, end_line, section_text))
                break
    
    return sections


def extract_hobbies_with_flashtext(text: str) -> Dict[str, Any]:
    """
    Fast hobby keyword extraction using FlashText (O(n) complexity)
    
    This provides 10-100x speedup compared to regex for large keyword sets.
    
    Args:
        text: Text to extract hobbies from
    
    Returns:
        Dictionary with detected hobbies, categories, and frequencies
    """
    if not _FLASHTEXT_AVAILABLE or not _HOBBIES_PROCESSOR:
        return {}
    
    try:
        # Limit text size for performance (50K chars)
        text_sample = text[:50000].lower()
        
        # Extract all hobby keywords with their categories
        keywords_found = _HOBBIES_PROCESSOR.extract_keywords(text_sample, span_info=False)
        
        # Organize by category
        hobbies_by_category = defaultdict(list)
        hobby_frequencies = defaultdict(int)
        all_hobbies = []
        
        for hobby, category in keywords_found:
            hobbies_by_category[category].append(hobby)
            hobby_frequencies[hobby] += 1
            if hobby not in all_hobbies:
                all_hobbies.append(hobby)
        
        # Convert defaultdict to regular dict for JSON serialization
        hobbies_by_category = dict(hobbies_by_category)
        hobby_frequencies = dict(hobby_frequencies)
        
        # Get category statistics
        category_counts = {cat: len(hobbies) for cat, hobbies in hobbies_by_category.items()}
        
        return {
            'hobbies': all_hobbies,
            'hobbies_by_category': hobbies_by_category,
            'hobby_frequencies': hobby_frequencies,
            'category_counts': category_counts,
            'total_hobbies': len(all_hobbies),
            'total_categories': len(hobbies_by_category)
        }
    
    except Exception as e:
        print(f"FlashText hobby extraction error: {e}")
        return {}


def extract_hobbies_with_patterns(text: str) -> List[str]:
    """
    Extract hobbies using regex patterns
    
    Fallback method when FlashText is not available.
    
    Args:
        text: Text to extract hobbies from
    
    Returns:
        List of extracted hobby strings
    """
    hobbies = []
    
    try:
        text_sample = text[:50000]
        
        for pattern in COMPILED_INTEREST_PATTERNS:
            matches = pattern.findall(text_sample)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match else ''
                
                # Clean and split the match
                hobby_text = match.strip()
                if len(hobby_text) > 5 and len(hobby_text) < 200:
                    # Split by common separators
                    items = re.split(r'[,;]|\band\b', hobby_text)
                    for item in items:
                        item_clean = item.strip()
                        if len(item_clean) > 2:
                            hobbies.append(item_clean)
    
    except Exception as e:
        print(f"Pattern-based hobby extraction error: {e}")
    
    return list(set(hobbies))  # Remove duplicates


def extract_hobbies_with_dependency_parsing(text: str) -> List[Dict[str, Any]]:
    """
    Extract hobbies using spaCy dependency parsing
    
    Identifies hobby-related verbs and their objects to extract structured information.
    
    Args:
        text: Text to extract hobbies from
    
    Returns:
        List of hobby dictionaries with detailed information
    """
    hobby_entries = []
    
    if not _NLP:
        return hobby_entries
    
    try:
        # Limit text for performance (100K chars for NLP)
        text_sample = text[:100000]
        doc = _NLP(text_sample)
        
        # Interest verbs to look for
        interest_verb_lemmas = ['enjoy', 'love', 'like', 'play', 'practice', 'do', 'participate']
        
        for token in doc:
            # Check if token is an interest-related verb
            if token.lemma_ in interest_verb_lemmas or token.text.lower() in INTEREST_VERBS:
                entry = {
                    'verb': token.text,
                    'raw_text': token.sent.text.strip()[:200]  # Limit length
                }
                
                # Find direct objects (what they enjoy/love/play)
                hobby_candidates = []
                for child in token.children:
                    if child.dep_ in ['dobj', 'pobj', 'attr', 'xcomp']:
                        # Get the full noun phrase or verb phrase
                        for np in child.sent.noun_chunks:
                            if child in np:
                                hobby_candidates.append(np.text.strip())
                        
                        # Also check for gerunds (playing, reading, etc.)
                        if child.pos_ == 'VERB' and child.tag_ == 'VBG':
                            hobby_candidates.append(child.text + 'ing' if not child.text.endswith('ing') else child.text)
                
                if hobby_candidates:
                    entry['hobby_candidates'] = hobby_candidates
                
                # Extract frequency/intensity if mentioned
                intensity_words = ['really', 'very', 'extremely', 'passionate', 'avid', 'keen']
                for intensity in intensity_words:
                    if intensity in token.sent.text.lower():
                        entry['intensity'] = intensity
                        break
                
                # Only add if we found hobby candidates
                if len(entry) > 2:
                    hobby_entries.append(entry)
    
    except Exception as e:
        print(f"Dependency parsing hobby extraction error: {e}")
    
    return hobby_entries


def extract_hobbies_with_nlp(text: str) -> Dict[str, Any]:
    """
    Extract hobby information using NLP and named entity recognition
    
    Args:
        text: Text to extract hobbies from
    
    Returns:
        Dictionary with NLP-extracted information
    """
    nlp_data = {
        'hobby_phrases': [],
        'organizations': [],  # Sports teams, clubs, etc.
        'activities': [],     # Activity-related entities
    }
    
    if not _NLP:
        return nlp_data
    
    try:
        text_sample = text[:100000]
        doc = _NLP(text_sample)
        
        # Extract noun phrases that might be hobbies
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            
            # Check if noun phrase relates to hobbies
            hobby_indicators = ['play', 'enjoy', 'love', 'passion', 'interest', 'hobby']
            
            # Look at the context around the noun phrase
            sent_text = chunk.sent.text.lower()
            if any(indicator in sent_text for indicator in hobby_indicators):
                if 3 < len(chunk_text) < 50:
                    nlp_data['hobby_phrases'].append(chunk.text.strip())
        
        # Extract organizations (could be sports teams, clubs)
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                nlp_data['organizations'].append(ent.text)
            elif ent.label_ in ['EVENT', 'FAC']:  # Events or facilities related to hobbies
                nlp_data['activities'].append(ent.text)
    
    except Exception as e:
        print(f"NLP hobby extraction error: {e}")
    
    return nlp_data


def categorize_hobby(hobby_text: str) -> Optional[str]:
    """
    Categorize a hobby into one of the predefined categories
    
    Args:
        hobby_text: Hobby description
    
    Returns:
        Category name or None if not categorized
    """
    hobby_lower = hobby_text.lower()
    
    for category, hobbies in HOBBIES_DATABASE.items():
        for known_hobby in hobbies:
            if known_hobby in hobby_lower or hobby_lower in known_hobby:
                return category
    
    return None


def calculate_diversity_score(hobbies_by_category: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Calculate diversity score based on hobby categories
    
    More diverse hobbies (across categories) indicate well-rounded personality.
    
    Args:
        hobbies_by_category: Dictionary of hobbies organized by category
    
    Returns:
        Dictionary with diversity metrics
    """
    num_categories = len(hobbies_by_category)
    total_categories = len(HOBBIES_DATABASE)
    
    diversity_percentage = (num_categories / total_categories) * 100
    
    # Categorize diversity level
    if diversity_percentage >= 60:
        diversity_level = "Highly Diverse"
    elif diversity_percentage >= 40:
        diversity_level = "Moderately Diverse"
    elif diversity_percentage >= 20:
        diversity_level = "Somewhat Diverse"
    else:
        diversity_level = "Focused"
    
    return {
        'categories_represented': num_categories,
        'total_categories': total_categories,
        'diversity_percentage': round(diversity_percentage, 1),
        'diversity_level': diversity_level,
        'strongest_category': max(hobbies_by_category, key=lambda k: len(hobbies_by_category[k])) if hobbies_by_category else None
    }


def infer_personality_traits(hobbies_by_category: Dict[str, List[str]]) -> List[str]:
    """
    Infer personality traits based on hobby categories
    
    This provides insights into candidate's personality based on their interests.
    
    Args:
        hobbies_by_category: Dictionary of hobbies by category
    
    Returns:
        List of inferred personality traits
    """
    traits = []
    
    # Map categories to personality traits
    category_traits = {
        'sports': ['Team Player', 'Competitive', 'Goal-Oriented', 'Active'],
        'arts': ['Creative', 'Expressive', 'Detail-Oriented', 'Artistic'],
        'technology': ['Analytical', 'Innovative', 'Problem Solver', 'Tech-Savvy'],
        'outdoor': ['Adventurous', 'Nature Lover', 'Risk-Taker', 'Explorer'],
        'intellectual': ['Curious', 'Knowledge Seeker', 'Analytical', 'Thoughtful'],
        'social': ['Outgoing', 'Community-Minded', 'Empathetic', 'Collaborative'],
        'collection': ['Organized', 'Patient', 'Detail-Oriented', 'Dedicated'],
        'travel': ['Adventurous', 'Open-Minded', 'Cultural', 'Explorer'],
        'entertainment': ['Culture-Appreciative', 'Well-Rounded', 'Relaxed'],
        'wellness': ['Balanced', 'Health-Conscious', 'Mindful', 'Self-Aware'],
    }
    
    seen_traits = set()
    for category, hobbies in hobbies_by_category.items():
        if category in category_traits and hobbies:
            for trait in category_traits[category]:
                if trait not in seen_traits:
                    traits.append(trait)
                    seen_traits.add(trait)
                    # Limit to top 8 traits
                    if len(traits) >= 8:
                        return traits
    
    return traits


def extract_hobbies_comprehensive(text: str) -> Dict[str, Any]:
    """
    Comprehensive hobby and interest extraction using multiple advanced NLP techniques
    
    This function provides complete hobby extraction with:
    - FlashText O(n) keyword extraction (10-100x faster than regex)
    - spaCy dependency parsing for context understanding
    - Pattern-based extraction for various formats
    - NLP entity recognition
    - Category classification
    - Personality trait inference
    - Diversity scoring
    
    Performance optimizations:
    - Text size limiting (50K for FlashText, 100K for spaCy)
    - Pre-compiled regex patterns
    - Graceful fallbacks without optional dependencies
    
    Args:
        text: Resume text to extract hobbies from
    
    Returns:
        Dictionary containing:
        - hobbies: List of hobby dictionaries with details
        - by_category: Hobbies organized by category
        - statistics: Summary statistics
        - personality_traits: Inferred personality traits
        - diversity_score: Hobby diversity metrics
    """
    # Step 1: Detect hobbies sections
    hobby_sections = detect_hobbies_section(text)
    
    # If explicit section found, use it; otherwise search entire text
    search_text = text
    if hobby_sections:
        # Combine all hobby sections
        search_text = '\n'.join([section_text for _, _, section_text in hobby_sections])
    
    # Step 2: Use FlashText for fast keyword extraction (PRIMARY METHOD)
    flashtext_results = {}
    if _FLASHTEXT_AVAILABLE and _HOBBIES_PROCESSOR:
        flashtext_results = extract_hobbies_with_flashtext(search_text)
    
    # Step 3: Use dependency parsing for structured extraction (SECONDARY METHOD)
    dependency_results = []
    if _NLP:
        dependency_results = extract_hobbies_with_dependency_parsing(search_text)
    
    # Step 4: Use pattern-based extraction as fallback (TERTIARY METHOD)
    pattern_hobbies = extract_hobbies_with_patterns(search_text)
    
    # Step 5: Use NLP for additional context (SUPPLEMENTARY)
    nlp_data = {}
    if _NLP:
        nlp_data = extract_hobbies_with_nlp(search_text)
    
    # Step 6: Merge and structure results
    final_hobbies = []
    hobbies_by_category = {}
    all_hobby_names = set()
    
    # Primary: Use FlashText results if available
    if flashtext_results and flashtext_results.get('hobbies'):
        hobbies_by_category = flashtext_results.get('hobbies_by_category', {})
        
        for hobby in flashtext_results['hobbies']:
            if hobby not in all_hobby_names:
                category = None
                for cat, hobbies_list in hobbies_by_category.items():
                    if hobby in hobbies_list:
                        category = cat
                        break
                
                final_hobbies.append({
                    'hobby': hobby,
                    'category': category,
                    'extraction_method': 'flashtext',
                    'confidence': 'high'
                })
                all_hobby_names.add(hobby)
    
    # Secondary: Add dependency parsing results
    for dep_entry in dependency_results:
        hobby_candidates = dep_entry.get('hobby_candidates', [])
        for candidate in hobby_candidates:
            if candidate.lower() not in all_hobby_names:
                category = categorize_hobby(candidate)
                
                hobby_entry = {
                    'hobby': candidate,
                    'category': category,
                    'extraction_method': 'dependency_parsing',
                    'confidence': 'medium',
                    'context': dep_entry.get('raw_text', '')[:100]
                }
                
                if 'intensity' in dep_entry:
                    hobby_entry['intensity'] = dep_entry['intensity']
                
                final_hobbies.append(hobby_entry)
                all_hobby_names.add(candidate.lower())
                
                # Add to category dict
                if category:
                    if category not in hobbies_by_category:
                        hobbies_by_category[category] = []
                    hobbies_by_category[category].append(candidate)
    
    # Tertiary: Add pattern-based results
    for pattern_hobby in pattern_hobbies:
        if pattern_hobby.lower() not in all_hobby_names:
            category = categorize_hobby(pattern_hobby)
            
            final_hobbies.append({
                'hobby': pattern_hobby,
                'category': category,
                'extraction_method': 'pattern',
                'confidence': 'low'
            })
            all_hobby_names.add(pattern_hobby.lower())
            
            if category:
                if category not in hobbies_by_category:
                    hobbies_by_category[category] = []
                hobbies_by_category[category].append(pattern_hobby)
    
    # Step 7: Calculate statistics
    statistics = {
        'total_count': len(final_hobbies),
        'by_category': {cat: len(hobbies) for cat, hobbies in hobbies_by_category.items()},
        'extraction_methods': {
            'flashtext': sum(1 for h in final_hobbies if h.get('extraction_method') == 'flashtext'),
            'dependency_parsing': sum(1 for h in final_hobbies if h.get('extraction_method') == 'dependency_parsing'),
            'pattern': sum(1 for h in final_hobbies if h.get('extraction_method') == 'pattern'),
        },
        'has_explicit_section': len(hobby_sections) > 0,
        'categories_represented': list(hobbies_by_category.keys())
    }
    
    # Step 8: Calculate diversity score
    diversity_score = {}
    if hobbies_by_category:
        diversity_score = calculate_diversity_score(hobbies_by_category)
    
    # Step 9: Infer personality traits
    personality_traits = []
    if hobbies_by_category:
        personality_traits = infer_personality_traits(hobbies_by_category)
    
    # Step 10: Return comprehensive results
    return {
        'hobbies': final_hobbies,
        'by_category': hobbies_by_category,
        'statistics': statistics,
        'diversity_score': diversity_score,
        'personality_traits': personality_traits,
        'nlp_insights': nlp_data if nlp_data else None,
        'flashtext_analysis': flashtext_results if flashtext_results else None
    }


# Backward compatibility - alias for the main function
extract_hobbies_info = extract_hobbies_comprehensive
