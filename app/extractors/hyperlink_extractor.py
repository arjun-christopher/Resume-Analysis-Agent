"""
Hyperlink Extraction and Classification Module

This module provides comprehensive hyperlink extraction and classification for resumes.
Supports 50+ platforms including professional networks, code repositories, social media,
and more. Includes advanced name-embedded hyperlink detection.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from collections import OrderedDict

try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except ImportError:
    _HAS_PYMUPDF = False
    fitz = None

# Enhanced hyperlink classification patterns (ordered from most specific to most general)
HYPERLINK_PATTERNS = OrderedDict([
    ('email', re.compile(r'mailto:([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})', re.IGNORECASE)),
    # Professional Networks
    ('linkedin', re.compile(r'(?:https?://)?(?:www\.)?linkedin\.com/(?:in|pub|company)/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    # Code Repositories
    ('github', re.compile(r'(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('gitlab', re.compile(r'(?:https?://)?(?:www\.)?gitlab\.com/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('bitbucket', re.compile(r'(?:https?://)?(?:www\.)?bitbucket\.org/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    # Social Media
    ('twitter', re.compile(r'(?:https?://)?(?:www\.)?(?:twitter|x)\.com/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('instagram', re.compile(r'(?:https?://)?(?:www\.)?instagram\.com/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('facebook', re.compile(r'(?:https?://)?(?:www\.)?facebook\.com/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('reddit', re.compile(r'(?:https?://)?(?:www\.)?reddit\.com/(?:u|user)/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('tiktok', re.compile(r'(?:https?://)?(?:www\.)?tiktok\.com/@?([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('pinterest', re.compile(r'(?:https?://)?(?:www\.)?pinterest\.com/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('snapchat', re.compile(r'(?:https?://)?(?:www\.)?snapchat\.com/add/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    # Messaging Apps
    ('telegram', re.compile(r'(?:https?://)?(?:t\.me|telegram\.me)/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('whatsapp', re.compile(r'(?:https?://)?(?:wa\.me|api\.whatsapp\.com)/([0-9]+)', re.IGNORECASE)),
    ('discord', re.compile(r'(?:https?://)?(?:www\.)?discord\.(?:gg|com/invite)/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    # Content Platforms
    ('youtube', re.compile(r'(?:https?://)?(?:www\.)?(?:youtube\.com/(?:c/|channel/|user/|@)?|youtu\.be/)([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('twitch', re.compile(r'(?:https?://)?(?:www\.)?twitch\.tv/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('medium', re.compile(r'(?:https?://)?(?:www\.)?medium\.com/@?([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('substack', re.compile(r'(?:https?://)?([A-Za-z0-9\-_.]+)\.substack\.com', re.IGNORECASE)),
    ('devto', re.compile(r'(?:https?://)?(?:www\.)?dev\.to/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('hashnode', re.compile(r'(?:https?://)?(?:www\.)?hashnode\.com/@?([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    # Professional Portfolios
    ('behance', re.compile(r'(?:https?://)?(?:www\.)?behance\.net/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('dribbble', re.compile(r'(?:https?://)?(?:www\.)?dribbble\.com/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('artstation', re.compile(r'(?:https?://)?(?:www\.)?artstation\.com/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('codepen', re.compile(r'(?:https?://)?(?:www\.)?codepen\.io/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    # Technical Communities
    ('stackoverflow', re.compile(r'(?:https?://)?(?:www\.)?stackoverflow\.com/users/(\d+/[A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('stackexchange', re.compile(r'(?:https?://)?(?:www\.)?stackexchange\.com/users/(\d+/[A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('kaggle', re.compile(r'(?:https?://)?(?:www\.)?kaggle\.com/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('hackerrank', re.compile(r'(?:https?://)?(?:www\.)?hackerrank\.com/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('leetcode', re.compile(r'(?:https?://)?(?:www\.)?leetcode\.com/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('codeforces', re.compile(r'(?:https?://)?(?:www\.)?codeforces\.com/profile/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    # Academic & Research
    ('researchgate', re.compile(r'(?:https?://)?(?:www\.)?researchgate\.net/profile/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('orcid', re.compile(r'(?:https?://)?(?:www\.)?orcid\.org/([0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{3}[0-9X])', re.IGNORECASE)),
    ('scholar', re.compile(r'(?:https?://)?(?:www\.)?scholar\.google\.com/citations\?user=([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('academia', re.compile(r'(?:https?://)?(?:www\.)?(?:[A-Za-z]+\.)?academia\.edu/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('mendeley', re.compile(r'(?:https?://)?(?:www\.)?mendeley\.com/profiles/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    # Knowledge Sharing
    ('quora', re.compile(r'(?:https?://)?(?:www\.)?quora\.com/profile/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('goodreads', re.compile(r'(?:https?://)?(?:www\.)?goodreads\.com/(?:user/show|author/show)/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    # Music & Audio
    ('spotify', re.compile(r'(?:https?://)?(?:www\.)?open\.spotify\.com/user/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('soundcloud', re.compile(r'(?:https?://)?(?:www\.)?soundcloud\.com/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    # Personal Domains
    ('portfolio', re.compile(r'(?:https?://)?(?:www\.)?([A-Za-z0-9\-_.]+\.(?:portfolio|website|dev|me|io|tech|app|vercel\.app|netlify\.app|github\.io))', re.IGNORECASE)),
    # Website pattern - most general, should be last
    ('website', re.compile(r'(?:https?://)?(?:www\.)?([A-Za-z0-9\-_.]+\.[A-Za-z]{2,})', re.IGNORECASE)),
])

# Text label patterns for hyperlinks with generic display text
TEXT_LABEL_PATTERNS = {
    # Professional
    'linkedin': [
        r'\blinkedin\b', r'\blinked\s*in\b', r'\bprofile\b', r'\bprofessional\s*profile\b',
        r'\bconnect\b', r'\bconnect\s*with\s*me\b', r'\bview\s*profile\b'
    ],
    # Code Repositories
    'github': [
        r'\bgithub\b', r'\bgit\s*hub\b', r'\bcode\b', r'\brepository\b', r'\brepo\b',
        r'\bprojects\b', r'\bmy\s*code\b', r'\bview\s*code\b', r'\bgit\b'
    ],
    'gitlab': [r'\bgitlab\b', r'\bgit\s*lab\b'],
    'bitbucket': [r'\bbitbucket\b', r'\bbit\s*bucket\b'],
    # Social Media
    'twitter': [
        r'\btwitter\b', r'\bfollow\s*me\b', r'\btweet\b', r'\bfollow\b', r'\b@\w+\b',
        r'\bx\.com\b', r'\bx\b'
    ],
    'instagram': [r'\binstagram\b', r'\binsta\b', r'\big\b'],
    'facebook': [r'\bfacebook\b', r'\bfb\b'],
    'reddit': [r'\breddit\b'],
    'tiktok': [r'\btiktok\b', r'\btik\s*tok\b'],
    'pinterest': [r'\bpinterest\b'],
    'snapchat': [r'\bsnapchat\b', r'\bsnap\b'],
    # Messaging
    'telegram': [r'\btelegram\b', r'\bt\.me\b'],
    'whatsapp': [r'\bwhatsapp\b', r'\bwhats\s*app\b', r'\bwa\.me\b'],
    'discord': [r'\bdiscord\b'],
    # Content
    'youtube': [
        r'\byoutube\b', r'\bvideo\b', r'\bvideos\b', r'\bchannel\b', r'\bwatch\b',
        r'\bsubscribe\b', r'\bmy\s*channel\b'
    ],
    'twitch': [r'\btwitch\b', r'\bstream\b', r'\bstreaming\b'],
    'medium': [
        r'\bmedium\b', r'\bblog\b', r'\barticles\b', r'\bwriting\b', r'\bposts\b',
        r'\bread\s*more\b', r'\bmy\s*blog\b'
    ],
    'substack': [r'\bsubstack\b', r'\bnewsletter\b'],
    'devto': [r'\bdev\.to\b', r'\bdev\s*to\b'],
    # Portfolios
    'behance': [r'\bbehance\b'],
    'dribbble': [r'\bdribbble\b'],
    'artstation': [r'\bartstation\b', r'\bart\s*station\b'],
    'codepen': [r'\bcodepen\b', r'\bcode\s*pen\b'],
    # Technical
    'stackoverflow': [
        r'\bstack\s*overflow\b', r'\bso\b', r'\bstackoverflow\b', r'\breputation\b'
    ],
    'kaggle': [
        r'\bkaggle\b', r'\bdata\s*science\b', r'\bcompetitions\b', r'\bdatasets\b'
    ],
    'hackerrank': [r'\bhackerrank\b', r'\bhacker\s*rank\b'],
    'leetcode': [r'\bleetcode\b', r'\bleet\s*code\b'],
    'codeforces': [r'\bcodeforces\b', r'\bcode\s*forces\b'],
    # Academic
    'researchgate': [r'\bresearchgate\b', r'\bresearch\s*gate\b'],
    'orcid': [r'\borcid\b'],
    'scholar': [r'\bgoogle\s*scholar\b', r'\bscholar\b'],
    'academia': [r'\bacademia\.edu\b', r'\bacademia\b'],
    'quora': [r'\bquora\b'],
    # General
    'website': [
        r'\bwebsite\b', r'\bvisit\b', r'\bhomepage\b', r'\bweb\b', r'\bsite\b',
        r'\bclick\s*here\b', r'\bmore\s*info\b', r'\blearn\s*more\b', r'\bview\b'
    ],
    'portfolio': [
        r'\bportfolio\b', r'\bwork\b', r'\bmy\s*work\b', r'\bprojects\b', 
        r'\bshowcase\b', r'\bgallery\b', r'\bexamples\b'
    ],
    'email': [
        r'\bemail\b', r'\bcontact\b', r'\bmail\b', r'\bget\s*in\s*touch\b',
        r'\breach\s*out\b', r'\bmessage\b'
    ]
}

# Compile text label patterns for efficiency
COMPILED_TEXT_PATTERNS = {}
for platform, patterns in TEXT_LABEL_PATTERNS.items():
    COMPILED_TEXT_PATTERNS[platform] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


def _is_plausible_match(url: str, platform: str) -> bool:
    """Check if a URL could plausibly belong to a platform based on domain or common patterns"""
    url_lower = url.lower()
    
    # Common domain patterns that would exclude certain platforms
    if platform == 'linkedin' and 'linkedin' not in url_lower:
        # Could still be valid if it's a redirect or short URL
        return not any(domain in url_lower for domain in ['github.com', 'twitter.com', 'instagram.com', 'facebook.com'])
    elif platform == 'github' and 'github' not in url_lower:
        return not any(domain in url_lower for domain in ['linkedin.com', 'twitter.com', 'instagram.com', 'facebook.com'])
    elif platform == 'twitter' and not any(x in url_lower for x in ['twitter', 'x.com']):
        return not any(domain in url_lower for domain in ['linkedin.com', 'github.com', 'instagram.com', 'facebook.com'])
    
    # If URL contains the platform name, it's very likely a match
    if platform.lower() in url_lower:
        return True
    
    # For generic labels like "website", "visit", accept most URLs
    if platform in ['website', 'portfolio']:
        return True
    
    # If no obvious conflicts, allow the match
    return True


def _extract_username_from_url(url: str, platform: str) -> Optional[str]:
    """Try to extract username from URL for a given platform"""
    if platform in HYPERLINK_PATTERNS:
        match = HYPERLINK_PATTERNS[platform].search(url)
        if match and match.groups():
            return match.group(1)
    
    # Fallback: try to extract username from common URL patterns
    try:
        # Remove protocol and www
        clean_url = re.sub(r'^https?://(www\.)?', '', url)
        # Split by / and take relevant parts
        parts = clean_url.split('/')
        if len(parts) >= 2:
            return parts[1]  # Usually the username part
    except:
        pass
    
    return None


def _matches_candidate_name(display_text: str, candidate_names: List[str], url: str = None) -> Tuple[bool, Optional[str]]:
    """
    Check if display text matches any candidate name
    
    Args:
        display_text: Text displayed for the hyperlink
        candidate_names: List of candidate names extracted from resume
        url: Optional URL to validate username match
    
    Returns:
        Tuple of (is_match, matched_name)
    """
    if not display_text or not candidate_names:
        return False, None
    
    display_normalized = display_text.lower().strip()
    
    for name in candidate_names:
        name_normalized = name.lower().strip()
        
        # Exact match (case-insensitive)
        if display_normalized == name_normalized:
            return True, name
        
        # Match with common variations
        name_variations = [
            name_normalized,
            name_normalized.replace(' ', ''),  # "JohnDoe"
            name_normalized.replace(' ', '-'),  # "john-doe"
            name_normalized.replace(' ', '_'),  # "john_doe"
            name_normalized.replace(' ', '.'),  # "john.doe"
        ]
        
        if display_normalized in name_variations:
            return True, name
        
        # Check if display text is contained in name or vice versa
        if display_normalized in name_normalized or name_normalized in display_normalized:
            # Require at least 50% overlap for partial matches
            overlap_ratio = len(display_normalized) / len(name_normalized)
            if overlap_ratio >= 0.5:
                return True, name
    
    # If URL provided, check if username in URL matches name
    if url:
        for platform in ['linkedin', 'github', 'twitter', 'medium', 'instagram']:
            username = _extract_username_from_url(url, platform)
            if username:
                username_normalized = username.lower().replace('-', '').replace('_', '').replace('.', '')
                for name in candidate_names:
                    name_parts = name.lower().split()
                    # Check if username matches any combination of name parts
                    name_combined = ''.join(name_parts)
                    name_first = name_parts[0] if name_parts else ''
                    name_last = name_parts[-1] if len(name_parts) > 1 else ''
                    
                    if username_normalized in [name_combined, name_first, name_last, name_first + name_last]:
                        return True, name
    
    return False, None


def _detect_profile_context(text: str, position: int, window: int = 100) -> Optional[str]:
    """
    Detect if text around position indicates a profile/social media context
    
    Args:
        text: Full text content
        position: Position of the hyperlink
        window: Characters to check before and after position
    
    Returns:
        Platform name if detected, None otherwise
    """
    start = max(0, position - window)
    end = min(len(text), position + window)
    context = text[start:end].lower()
    
    # Platform keywords that indicate context
    platform_keywords = {
        'linkedin': ['linkedin', 'professional profile', 'connect with me'],
        'github': ['github', 'repository', 'code', 'projects', 'open source'],
        'twitter': ['twitter', 'follow me', 'tweet', '@'],
        'portfolio': ['portfolio', 'work', 'projects', 'website'],
        'email': ['email', 'contact', 'reach me', 'get in touch'],
    }
    
    for platform, keywords in platform_keywords.items():
        if any(keyword in context for keyword in keywords):
            return platform
    
    return None


def extract_and_classify_hyperlinks(doc, extract_names_func) -> Dict[str, Any]:
    """
    Extract hyperlinks from PDF using PyMuPDF and classify them by type
    
    Args:
        doc: PyMuPDF document object
        extract_names_func: Function to extract candidate names from text
        
    Returns:
        Dictionary with classified hyperlinks by type and page
    """
    if not _HAS_PYMUPDF:
        raise ImportError("PyMuPDF (fitz) is required for hyperlink extraction")
    
    hyperlinks_data = {
        'by_type': {
            'email': [],
            'linkedin': [],
            'github': [],
            'gitlab': [],
            'bitbucket': [],
            'twitter': [],
            'instagram': [],
            'facebook': [],
            'reddit': [],
            'tiktok': [],
            'pinterest': [],
            'snapchat': [],
            'telegram': [],
            'whatsapp': [],
            'discord': [],
            'youtube': [],
            'twitch': [],
            'medium': [],
            'substack': [],
            'devto': [],
            'hashnode': [],
            'behance': [],
            'dribbble': [],
            'artstation': [],
            'codepen': [],
            'stackoverflow': [],
            'stackexchange': [],
            'kaggle': [],
            'hackerrank': [],
            'leetcode': [],
            'codeforces': [],
            'researchgate': [],
            'orcid': [],
            'scholar': [],
            'academia': [],
            'mendeley': [],
            'quora': [],
            'goodreads': [],
            'spotify': [],
            'soundcloud': [],
            'portfolio': [],
            'website': [],
            'other': []
        },
        'by_page': {},
        'all_links': [],
        'statistics': {
            'total_links': 0,
            'pages_with_links': 0,
            'most_common_type': None,
            'platforms_found': []
        }
    }
    
    pages_with_links = 0
    
    try:
        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_links = []
            
            # Extract hyperlinks using the method you provided
            links = page.get_links()
            
            for link in links:
                if "uri" in link:
                    uri = link["uri"]
                    if uri and isinstance(uri, str):
                        page_links.append(uri)
                        
                        # Extract display text from the link rectangle area
                        display_text = ""
                        if "from" in link:
                            try:
                                # Get the rectangle coordinates
                                rect = link["from"]
                                # Extract text from the rectangle area
                                text_instances = page.get_text("dict", clip=rect)
                                if text_instances and "blocks" in text_instances:
                                    for block in text_instances["blocks"]:
                                        if "lines" in block:
                                            for line in block["lines"]:
                                                if "spans" in line:
                                                    for span in line["spans"]:
                                                        if "text" in span:
                                                            display_text += span["text"] + " "
                                display_text = display_text.strip()
                            except:
                                pass
                        
                        hyperlinks_data['all_links'].append({
                            'url': uri,
                            'display_text': display_text,
                            'page': page_num + 1,
                            'type': None  # Will be classified below
                        })
            
            # Also extract hyperlinks from annotations (alternative method)
            try:
                for annot in page.annots():
                    if annot.type[1] == 'Link':  # Link annotation
                        link_data = annot.info.get('content', '') or annot.info.get('title', '')
                        if link_data and 'http' in link_data:
                            page_links.append(link_data)
                            
                            # Try to get display text from annotation
                            display_text = annot.info.get('title', '') or ""
                            
                            hyperlinks_data['all_links'].append({
                                'url': link_data,
                                'display_text': display_text,
                                'page': page_num + 1,
                                'type': None
                            })
            except:
                pass  # Annotations might not be accessible
            
            if page_links:
                pages_with_links += 1
                hyperlinks_data['by_page'][page_num + 1] = page_links
        
        # Classify all extracted hyperlinks using both URL and text patterns
        for link_entry in hyperlinks_data['all_links']:
            url = link_entry['url']
            display_text = link_entry.get('display_text', '').lower().strip()
            classified = False
            username = None
            
            # First, try to classify by URL pattern (most reliable)
            for link_type, pattern in HYPERLINK_PATTERNS.items():
                match = pattern.search(url)
                if match:
                    username = match.group(1) if match.groups() else None
                    hyperlinks_data['by_type'][link_type].append({
                        'url': url,
                        'display_text': link_entry.get('display_text', ''),
                        'page': link_entry['page'],
                        'username': username,
                        'classification_method': 'url_pattern'
                    })
                    link_entry['type'] = link_type
                    classified = True
                    break
            
            # If not classified by URL, try text label patterns
            if not classified and display_text:
                for platform, text_patterns in COMPILED_TEXT_PATTERNS.items():
                    for text_pattern in text_patterns:
                        if text_pattern.search(display_text):
                            # Double-check that the URL could plausibly match this platform
                            if _is_plausible_match(url, platform):
                                hyperlinks_data['by_type'][platform].append({
                                    'url': url,
                                    'display_text': link_entry.get('display_text', ''),
                                    'page': link_entry['page'],
                                    'username': _extract_username_from_url(url, platform),
                                    'classification_method': 'text_pattern'
                                })
                                link_entry['type'] = platform
                                classified = True
                                break
                    if classified:
                        break
            
            # If still not classified, mark as other
            if not classified:
                hyperlinks_data['by_type']['other'].append({
                    'url': url,
                    'display_text': link_entry.get('display_text', ''),
                    'page': link_entry['page'],
                    'username': None,
                    'classification_method': 'unclassified'
                })
                link_entry['type'] = 'other'
        
        # Extract candidate names from the document for name-embedded detection
        full_text = ""
        for page_num in range(doc.page_count):
            full_text += doc[page_num].get_text()
        
        candidate_names = extract_names_func(full_text)
        
        # Detect name-embedded hyperlinks and add to candidate_profile_links
        hyperlinks_data['candidate_profile_links'] = {}
        
        for link_entry in hyperlinks_data['all_links']:
            display_text = link_entry.get('display_text', '')
            url = link_entry['url']
            platform = link_entry.get('type', 'other')
            
            # Check if this link's display text matches a candidate name
            is_name_match, matched_name = _matches_candidate_name(display_text, candidate_names, url)
            
            if is_name_match and platform != 'other':
                # This is a name-embedded hyperlink - likely a primary profile link
                if platform not in hyperlinks_data['candidate_profile_links']:
                    hyperlinks_data['candidate_profile_links'][platform] = {
                        'url': url,
                        'display_text': display_text,
                        'matched_name': matched_name,
                        'page': link_entry.get('page'),
                        'username': _extract_username_from_url(url, platform),
                        'confidence': 'high' if display_text.lower().strip() == matched_name.lower().strip() else 'medium'
                    }
                
                # Mark the link entry as name-embedded
                link_entry['is_name_embedded'] = True
                link_entry['matched_candidate_name'] = matched_name
        
        # Calculate statistics
        hyperlinks_data['statistics']['total_links'] = len(hyperlinks_data['all_links'])
        hyperlinks_data['statistics']['pages_with_links'] = pages_with_links
        
        # Find most common link type and platforms found
        type_counts = {
            link_type: len(links) 
            for link_type, links in hyperlinks_data['by_type'].items() 
            if links
        }
        
        if type_counts:
            most_common_type = max(type_counts.items(), key=lambda x: x[1])
            hyperlinks_data['statistics']['most_common_type'] = {
                'type': most_common_type[0],
                'count': most_common_type[1]
            }
            hyperlinks_data['statistics']['platforms_found'] = list(type_counts.keys())
    
    except Exception as e:
        print(f"Error extracting hyperlinks: {e}")
    
    return hyperlinks_data
