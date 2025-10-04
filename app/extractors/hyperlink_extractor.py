"""
Hyperlink Extraction and Classification Module

This module provides comprehensive hyperlink extraction and classification for resumes.
Supports 60+ platforms including professional networks, code repositories, social media,
portfolio sites, and more. Includes advanced name-embedded hyperlink detection,
URL normalization, and text-based link extraction.

Features:
- Extract hyperlinks from PDF annotations and embedded links
- Extract plain text URLs and email addresses
- Classify links by platform (60+ supported)
- Detect social media handles without full URLs
- Normalize and validate URLs
- Match profile links to candidate names
- Context-aware classification
"""

import re
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import OrderedDict
from urllib.parse import urlparse, unquote

try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except ImportError:
    _HAS_PYMUPDF = False
    fitz = None

# Enhanced hyperlink classification patterns (ordered from most specific to most general)
HYPERLINK_PATTERNS = OrderedDict([
    # Email
    ('email', re.compile(r'(?:mailto:)?([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})', re.IGNORECASE)),
    
    # Professional Networks
    ('linkedin', re.compile(r'(?:https?://)?(?:www\.)?linkedin\.com/(?:in|pub|company|profile)/([A-Za-z0-9\-_.%]+)(?:/[A-Za-z0-9\-_.%]*)?', re.IGNORECASE)),
    ('xing', re.compile(r'(?:https?://)?(?:www\.)?xing\.com/profile/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('angel', re.compile(r'(?:https?://)?(?:www\.)?angel\.co/(?:u/)?([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('wellfound', re.compile(r'(?:https?://)?(?:www\.)?wellfound\.com/u/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    
    # Code Repositories & Developer Platforms
    ('github', re.compile(r'(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9\-_.]+)(?:/[A-Za-z0-9\-_.]*)?', re.IGNORECASE)),
    ('gitlab', re.compile(r'(?:https?://)?(?:www\.)?gitlab\.com/([A-Za-z0-9\-_.]+)(?:/[A-Za-z0-9\-_.]*)?', re.IGNORECASE)),
    ('bitbucket', re.compile(r'(?:https?://)?(?:www\.)?bitbucket\.org/([A-Za-z0-9\-_.]+)(?:/[A-Za-z0-9\-_.]*)?', re.IGNORECASE)),
    ('sourceforge', re.compile(r'(?:https?://)?(?:www\.)?sourceforge\.net/u/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('launchpad', re.compile(r'(?:https?://)?(?:www\.)?launchpad\.net/~([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    
    # Social Media (Major Platforms)
    ('twitter', re.compile(r'(?:https?://)?(?:www\.)?(?:twitter|x)\.com/(?:#!/)?@?([A-Za-z0-9\-_.]+)(?:/[A-Za-z0-9\-_.]*)?', re.IGNORECASE)),
    ('instagram', re.compile(r'(?:https?://)?(?:www\.)?instagram\.com/(?:p/)?([A-Za-z0-9\-_.]+)(?:/[A-Za-z0-9\-_.]*)?', re.IGNORECASE)),
    ('facebook', re.compile(r'(?:https?://)?(?:www\.)?(?:facebook|fb)\.com/(?:profile\.php\?id=\d+|people/[^/]+/\d+|)?([A-Za-z0-9\-_.]+)(?:/[A-Za-z0-9\-_.]*)?', re.IGNORECASE)),
    ('reddit', re.compile(r'(?:https?://)?(?:www\.)?reddit\.com/(?:u|user)/([A-Za-z0-9\-_.]+)(?:/[A-Za-z0-9\-_.]*)?', re.IGNORECASE)),
    ('tiktok', re.compile(r'(?:https?://)?(?:www\.)?tiktok\.com/@([A-Za-z0-9\-_.]+)(?:/[A-Za-z0-9\-_.]*)?', re.IGNORECASE)),
    ('pinterest', re.compile(r'(?:https?://)?(?:www\.)?pinterest\.com/([A-Za-z0-9\-_.]+)(?:/[A-Za-z0-9\-_.]*)?', re.IGNORECASE)),
    ('snapchat', re.compile(r'(?:https?://)?(?:www\.)?snapchat\.com/add/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('threads', re.compile(r'(?:https?://)?(?:www\.)?threads\.net/@([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('bluesky', re.compile(r'(?:https?://)?(?:www\.)?bsky\.app/profile/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('mastodon', re.compile(r'(?:https?://)?([A-Za-z0-9\-_.]+)/(@[A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    
    # Messaging Apps
    ('telegram', re.compile(r'(?:https?://)?(?:t\.me|telegram\.me)/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('whatsapp', re.compile(r'(?:https?://)?(?:wa\.me|api\.whatsapp\.com|whatsapp\.com)/(\+?[0-9]+)', re.IGNORECASE)),
    ('discord', re.compile(r'(?:https?://)?(?:www\.)?discord\.(?:gg|com/invite|com/users?)/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('slack', re.compile(r'(?:https?://)?([A-Za-z0-9\-_.]+)\.slack\.com(?:/team/([A-Za-z0-9\-_.]+))?', re.IGNORECASE)),
    ('signal', re.compile(r'(?:https?://)?(?:www\.)?signal\.(?:group|me)/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    
    # Content Platforms
    ('youtube', re.compile(r'(?:https?://)?(?:www\.)?(?:youtube\.com/(?:c/|channel/|user/|@)|youtu\.be/)([A-Za-z0-9\-_.]+)(?:/[A-Za-z0-9\-_.]*)?', re.IGNORECASE)),
    ('twitch', re.compile(r'(?:https?://)?(?:www\.)?twitch\.tv/([A-Za-z0-9\-_.]+)(?:/[A-Za-z0-9\-_.]*)?', re.IGNORECASE)),
    ('vimeo', re.compile(r'(?:https?://)?(?:www\.)?vimeo\.com/(?:user)?([0-9]+|[A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('dailymotion', re.compile(r'(?:https?://)?(?:www\.)?dailymotion\.com/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    
    # Blogging & Writing
    ('medium', re.compile(r'(?:https?://)?(?:www\.)?medium\.com/@([A-Za-z0-9\-_.]+)(?:/[A-Za-z0-9\-_.]*)?', re.IGNORECASE)),
    ('substack', re.compile(r'(?:https?://)?([A-Za-z0-9\-_.]+)\.substack\.com', re.IGNORECASE)),
    ('devto', re.compile(r'(?:https?://)?(?:www\.)?dev\.to/([A-Za-z0-9\-_.]+)(?:/[A-Za-z0-9\-_.]*)?', re.IGNORECASE)),
    ('hashnode', re.compile(r'(?:https?://)?(?:www\.)?hashnode\.com/@([A-Za-z0-9\-_.]+)(?:/[A-Za-z0-9\-_.]*)?', re.IGNORECASE)),
    ('wordpress', re.compile(r'(?:https?://)?([A-Za-z0-9\-_.]+)\.wordpress\.com', re.IGNORECASE)),
    ('blogger', re.compile(r'(?:https?://)?([A-Za-z0-9\-_.]+)\.blogspot\.com', re.IGNORECASE)),
    ('ghost', re.compile(r'(?:https?://)?([A-Za-z0-9\-_.]+)\.ghost\.io', re.IGNORECASE)),
    
    # Professional Portfolios & Design
    ('behance', re.compile(r'(?:https?://)?(?:www\.)?behance\.net/([A-Za-z0-9\-_.]+)(?:/[A-Za-z0-9\-_.]*)?', re.IGNORECASE)),
    ('dribbble', re.compile(r'(?:https?://)?(?:www\.)?dribbble\.com/([A-Za-z0-9\-_.]+)(?:/[A-Za-z0-9\-_.]*)?', re.IGNORECASE)),
    ('artstation', re.compile(r'(?:https?://)?(?:www\.)?artstation\.com/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('deviantart', re.compile(r'(?:https?://)?(?:www\.)?deviantart\.com/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('flickr', re.compile(r'(?:https?://)?(?:www\.)?flickr\.com/(?:people|photos)/([A-Za-z0-9\-_.@]+)', re.IGNORECASE)),
    ('500px', re.compile(r'(?:https?://)?(?:www\.)?500px\.com/p/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('unsplash', re.compile(r'(?:https?://)?(?:www\.)?unsplash\.com/@([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    
    # Code & Technical
    ('codepen', re.compile(r'(?:https?://)?(?:www\.)?codepen\.io/([A-Za-z0-9\-_.]+)(?:/[A-Za-z0-9\-_.]*)?', re.IGNORECASE)),
    ('jsfiddle', re.compile(r'(?:https?://)?(?:www\.)?jsfiddle\.net/(?:user/)?([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('replit', re.compile(r'(?:https?://)?(?:www\.)?replit\.com/@([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('codesandbox', re.compile(r'(?:https?://)?(?:www\.)?codesandbox\.io/u/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('glitch', re.compile(r'(?:https?://)?(?:www\.)?glitch\.com/@([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    
    # Technical Communities & Competitive Programming
    ('stackoverflow', re.compile(r'(?:https?://)?(?:www\.)?stackoverflow\.com/users/(\d+(?:/[A-Za-z0-9\-_.]+)?)', re.IGNORECASE)),
    ('stackexchange', re.compile(r'(?:https?://)?(?:www\.)?stackexchange\.com/users/(\d+(?:/[A-Za-z0-9\-_.]+)?)', re.IGNORECASE)),
    ('kaggle', re.compile(r'(?:https?://)?(?:www\.)?kaggle\.com/([A-Za-z0-9\-_.]+)(?:/[A-Za-z0-9\-_.]*)?', re.IGNORECASE)),
    ('hackerrank', re.compile(r'(?:https?://)?(?:www\.)?hackerrank\.com/(?:profile/)?([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('leetcode', re.compile(r'(?:https?://)?(?:www\.)?leetcode\.com/([A-Za-z0-9\-_.]+)(?:/[A-Za-z0-9\-_.]*)?', re.IGNORECASE)),
    ('codeforces', re.compile(r'(?:https?://)?(?:www\.)?codeforces\.com/profile/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('topcoder', re.compile(r'(?:https?://)?(?:www\.)?topcoder\.com/members/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('codechef', re.compile(r'(?:https?://)?(?:www\.)?codechef\.com/users/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('hackerearth', re.compile(r'(?:https?://)?(?:www\.)?hackerearth\.com/@([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    
    # Academic & Research
    ('researchgate', re.compile(r'(?:https?://)?(?:www\.)?researchgate\.net/profile/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('orcid', re.compile(r'(?:https?://)?(?:www\.)?orcid\.org/([0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{3}[0-9X])', re.IGNORECASE)),
    ('scholar', re.compile(r'(?:https?://)?(?:www\.)?scholar\.google\.com/citations\?user=([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('academia', re.compile(r'(?:https?://)?(?:www\.)?(?:[A-Za-z]+\.)?academia\.edu/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('mendeley', re.compile(r'(?:https?://)?(?:www\.)?mendeley\.com/profiles/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('publons', re.compile(r'(?:https?://)?(?:www\.)?publons\.com/researcher/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('arxiv', re.compile(r'(?:https?://)?(?:www\.)?arxiv\.org/a/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    
    # Knowledge Sharing
    ('quora', re.compile(r'(?:https?://)?(?:www\.)?quora\.com/profile/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('goodreads', re.compile(r'(?:https?://)?(?:www\.)?goodreads\.com/(?:user/show|author/show)/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    
    # Music & Audio
    ('spotify', re.compile(r'(?:https?://)?(?:www\.)?open\.spotify\.com/(?:user|artist)/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    ('soundcloud', re.compile(r'(?:https?://)?(?:www\.)?soundcloud\.com/([A-Za-z0-9\-_.]+)(?:/[A-Za-z0-9\-_.]*)?', re.IGNORECASE)),
    ('bandcamp', re.compile(r'(?:https?://)?([A-Za-z0-9\-_.]+)\.bandcamp\.com', re.IGNORECASE)),
    ('mixcloud', re.compile(r'(?:https?://)?(?:www\.)?mixcloud\.com/([A-Za-z0-9\-_.]+)', re.IGNORECASE)),
    
    # Personal Domains & Portfolios
    ('portfolio', re.compile(r'(?:https?://)?(?:www\.)?([A-Za-z0-9\-_.]+\.(?:portfolio|website|dev|me|io|tech|app|co|vercel\.app|netlify\.app|github\.io|gitlab\.io|pages\.dev|web\.app))', re.IGNORECASE)),
    
    # Website pattern - most general, should be last
    ('website', re.compile(r'(?:https?://)?(?:www\.)?([A-Za-z0-9\-_.]+\.[A-Za-z]{2,}(?:/[A-Za-z0-9\-_./?=&]*)?)', re.IGNORECASE)),
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


# Social media handle patterns (for detecting @username mentions without full URLs)
SOCIAL_HANDLE_PATTERNS = {
    'twitter': re.compile(r'(?:^|[^@\w])@([A-Za-z0-9_]{1,15})(?:\s|$)', re.MULTILINE),
    'instagram': re.compile(r'(?:^|[^\w])@([A-Za-z0-9_.]{1,30})(?:\s|$)', re.MULTILINE),
    'telegram': re.compile(r'(?:^|[^\w])@([A-Za-z0-9_]{5,32})(?:\s|$)', re.MULTILINE),
}

# Plain text URL detection patterns
PLAIN_TEXT_URL_PATTERN = re.compile(
    r'\b(?:https?://|www\.)[A-Za-z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]+',
    re.IGNORECASE
)

# Email pattern for plain text extraction
PLAIN_TEXT_EMAIL_PATTERN = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
)


def normalize_url(url: str) -> str:
    """
    Normalize URL for consistent comparison and storage
    
    Args:
        url: Raw URL string
        
    Returns:
        Normalized URL string
    """
    if not url:
        return ""
    
    # Remove whitespace
    url = url.strip()
    
    # Add protocol if missing
    if not url.startswith(('http://', 'https://', 'mailto:')):
        # Check if it's an email
        if '@' in url and '.' in url.split('@')[-1]:
            url = f"mailto:{url}"
        else:
            url = f"https://{url}"
    
    # Decode URL-encoded characters
    url = unquote(url)
    
    # Remove trailing slashes
    if url.endswith('/') and url.count('/') > 2:
        url = url.rstrip('/')
    
    # Normalize common variations
    url = url.replace('http://', 'https://')  # Prefer HTTPS
    url = url.replace('www.', '')  # Remove www for consistency
    
    # Handle URL fragments and tracking parameters
    if '?' in url or '#' in url:
        parsed = urlparse(url)
        # Keep only essential query parameters (exclude tracking)
        tracking_params = {'utm_source', 'utm_medium', 'utm_campaign', 'utm_content', 'utm_term', 'ref', 'src'}
        if parsed.query:
            query_parts = [p for p in parsed.query.split('&') if p.split('=')[0] not in tracking_params]
            clean_query = '&'.join(query_parts) if query_parts else ''
            url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if clean_query:
                url += f"?{clean_query}"
    
    return url


def extract_plain_text_links(text: str) -> List[Dict[str, Any]]:
    """
    Extract URLs and emails from plain text content
    
    Args:
        text: Plain text content
        
    Returns:
        List of dictionaries containing extracted links
    """
    links = []
    seen_urls = set()
    
    # Extract URLs
    for match in PLAIN_TEXT_URL_PATTERN.finditer(text):
        url = match.group(0)
        normalized_url = normalize_url(url)
        
        if normalized_url and normalized_url not in seen_urls:
            seen_urls.add(normalized_url)
            links.append({
                'url': normalized_url,
                'original': url,
                'type': 'url',
                'position': match.start()
            })
    
    # Extract emails
    for match in PLAIN_TEXT_EMAIL_PATTERN.finditer(text):
        email = match.group(0)
        normalized_email = f"mailto:{email}"
        
        if normalized_email not in seen_urls:
            seen_urls.add(normalized_email)
            links.append({
                'url': normalized_email,
                'original': email,
                'type': 'email',
                'position': match.start()
            })
    
    return links


def extract_social_handles(text: str) -> Dict[str, List[str]]:
    """
    Extract social media handles from text (e.g., @username)
    
    Args:
        text: Plain text content
        
    Returns:
        Dictionary mapping platform to list of handles
    """
    handles = {}
    
    for platform, pattern in SOCIAL_HANDLE_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            # Remove duplicates while preserving order
            unique_handles = list(dict.fromkeys(matches))
            handles[platform] = unique_handles
    
    return handles


def validate_url(url: str) -> Tuple[bool, Optional[str]]:
    """
    Validate if a URL is well-formed and accessible
    
    Args:
        url: URL to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url:
        return False, "Empty URL"
    
    try:
        parsed = urlparse(url)
        
        # Check if scheme is valid
        if parsed.scheme not in ('http', 'https', 'mailto'):
            return False, f"Invalid scheme: {parsed.scheme}"
        
        # Check if domain exists
        if parsed.scheme != 'mailto':
            if not parsed.netloc:
                return False, "Missing domain"
            
            # Check domain format
            if '.' not in parsed.netloc:
                return False, "Invalid domain format"
        
        return True, None
        
    except Exception as e:
        return False, str(e)


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


def _generate_name_variations(name: str) -> List[str]:
    """
    Generate all possible username/profile variations from a candidate name
    
    Examples:
        "Arjun Christopher" -> ["arjunchristopher", "arjun-christopher", "arjun_christopher",
                                "arjun.christopher", "arjun", "christopher", "achristopher",
                                "arjunc", "christopher-arjun", etc.]
    
    Args:
        name: Full candidate name
        
    Returns:
        List of possible username variations
    """
    variations = set()
    name_lower = name.lower().strip()
    
    # Remove common prefixes/suffixes (Dr., Jr., etc.)
    for prefix in ['dr.', 'mr.', 'mrs.', 'ms.', 'prof.']:
        if name_lower.startswith(prefix):
            name_lower = name_lower[len(prefix):].strip()
    for suffix in ['jr.', 'jr', 'sr.', 'sr', 'ii', 'iii', 'iv', 'phd', 'md']:
        if name_lower.endswith(suffix):
            name_lower = name_lower[:-(len(suffix))].strip()
    
    # Split into parts
    parts = name_lower.split()
    
    if not parts:
        return list(variations)
    
    # Add the full name with various separators
    variations.add(name_lower)  # "arjun christopher"
    variations.add(''.join(parts))  # "arjunchristopher"
    variations.add('-'.join(parts))  # "arjun-christopher"
    variations.add('_'.join(parts))  # "arjun_christopher"
    variations.add('.'.join(parts))  # "arjun.christopher"
    
    # Add individual name parts
    for part in parts:
        variations.add(part)  # "arjun", "christopher"
    
    # First + Last combinations (most common for usernames)
    if len(parts) >= 2:
        first, last = parts[0], parts[-1]
        
        # Standard combinations
        variations.add(first + last)  # "arjunchristopher"
        variations.add(first + '-' + last)  # "arjun-christopher"
        variations.add(first + '_' + last)  # "arjun_christopher"
        variations.add(first + '.' + last)  # "arjun.christopher"
        
        # Reversed combinations
        variations.add(last + first)  # "christopherarjun"
        variations.add(last + '-' + first)  # "christopher-arjun"
        variations.add(last + '_' + first)  # "christopher_arjun"
        variations.add(last + '.' + first)  # "christopher.arjun"
        
        # Initial + Last name
        variations.add(first[0] + last)  # "achristopher"
        variations.add(first[0] + '-' + last)  # "a-christopher"
        variations.add(first[0] + '_' + last)  # "a_christopher"
        variations.add(first[0] + '.' + last)  # "a.christopher"
        
        # First + Initial of last
        variations.add(first + last[0])  # "arjunc"
        variations.add(first + '-' + last[0])  # "arjun-c"
        variations.add(first + '_' + last[0])  # "arjun_c"
        variations.add(first + '.' + last[0])  # "arjun.c"
        
        # Both initials
        variations.add(first[0] + last[0])  # "ac"
        variations.add(first[0] + '.' + last[0])  # "a.c"
    
    # For three-part names (e.g., "John Michael Smith")
    if len(parts) == 3:
        first, middle, last = parts[0], parts[1], parts[2]
        variations.add(first + middle + last)  # "johnmichaelsmith"
        variations.add(first + '-' + last)  # "john-smith" (skip middle)
        variations.add(first + last)  # "johnsmith"
        variations.add(first + middle[0] + last)  # "johnmsmith"
        variations.add(first[0] + middle[0] + last)  # "jmsmith"
    
    # Handle names with multiple parts (more than 3)
    if len(parts) > 3:
        # Use first and last only
        variations.add(parts[0] + parts[-1])
        variations.add(parts[0] + '-' + parts[-1])
    
    return list(variations)


def _matches_candidate_name(display_text: str, candidate_names: List[str], url: str = None) -> Tuple[bool, Optional[str]]:
    """
    Enhanced matching: Check if display text or URL username matches any candidate name variation
    
    This function handles scenarios like:
    - Display text: "arjun-christopher" matching name "Arjun Christopher"
    - Display text: "Arjun's GitHub" matching name "Arjun Christopher"
    - URL: github.com/arjun-christopher matching name "Arjun Christopher"
    - URL: linkedin.com/in/arjunchristopher matching name "Arjun Christopher"
    
    Args:
        display_text: Text displayed for the hyperlink (anchor text)
        candidate_names: List of candidate names extracted from resume
        url: Optional URL to validate username match
    
    Returns:
        Tuple of (is_match, matched_name) with confidence level
    """
    if not display_text and not url:
        return False, None
    
    if not candidate_names:
        return False, None
    
    display_normalized = display_text.lower().strip() if display_text else ""
    
    # Clean display text: remove possessives, common words
    display_clean = display_normalized
    for remove in ["'s", "'", 's ', 'profile', 'page', 'account', 'my ', 'visit ', 'find me on']:
        display_clean = display_clean.replace(remove, '')
    display_clean = display_clean.strip()
    
    # Try matching against each candidate name
    for name in candidate_names:
        name_normalized = name.lower().strip()
        
        # Exact match (case-insensitive)
        if display_clean == name_normalized:
            return True, name
        
        # Generate all possible variations of the name
        name_variations = _generate_name_variations(name)
        
        # Check if display text matches any variation
        for variation in name_variations:
            # Exact variation match
            if display_clean == variation:
                return True, name
            
            # Check if variation is contained in display text (for "Arjun's Profile" -> "arjun")
            if variation in display_clean and len(variation) >= 3:
                # Ensure it's a substantial match (at least 3 characters)
                return True, name
            
            # Check if display text is contained in variation (for partial matches)
            if display_clean in variation and len(display_clean) >= 3:
                return True, name
    
    # If URL provided, extract and check username/handle
    if url:
        # Try to extract username from URL for various platforms
        url_lower = url.lower()
        
        # Extract username from common URL patterns
        username = None
        
        # LinkedIn: /in/username or /company/username
        linkedin_match = re.search(r'linkedin\.com/in/([^/?&#]+)', url_lower)
        if linkedin_match:
            username = linkedin_match.group(1)
        
        # GitHub, Twitter, Instagram, etc.: platform.com/username
        generic_match = re.search(r'(?:github|twitter|instagram|medium|dev\.to|dribbble|behance)\.com/(?:@)?([^/?&#]+)', url_lower)
        if generic_match:
            username = generic_match.group(1)
        
        # If no specific pattern, try generic extraction
        if not username:
            username = _extract_username_from_url(url, 'generic')
        
        if username:
            username_clean = username.lower().strip()
            
            # Check if username matches any name variation
            for name in candidate_names:
                name_variations = _generate_name_variations(name)
                
                for variation in name_variations:
                    # Direct match
                    if username_clean == variation:
                        return True, name
                    
                    # Fuzzy match: remove separators and compare
                    username_no_sep = username_clean.replace('-', '').replace('_', '').replace('.', '')
                    variation_no_sep = variation.replace('-', '').replace('_', '').replace('.', '')
                    
                    if username_no_sep == variation_no_sep:
                        return True, name
                    
                    # Partial match for longer usernames
                    if len(username_clean) >= 5 and len(variation) >= 5:
                        if username_no_sep in variation_no_sep or variation_no_sep in username_no_sep:
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
    Extract and classify hyperlinks from PDF with enhanced social media detection
    
    Features:
    - Extract embedded hyperlinks from PDF
    - Extract plain text URLs and emails
    - Detect social media handles (@username)
    - Normalize and validate URLs
    - Classify by platform (60+ supported)
    - Match profile links to candidate names
    
    Args:
        doc: PyMuPDF document object
        extract_names_func: Function to extract candidate names from text
        
    Returns:
        Dictionary with classified hyperlinks, social handles, and statistics
    """
    if not _HAS_PYMUPDF:
        raise ImportError("PyMuPDF (fitz) is required for hyperlink extraction")
    
    # Initialize enhanced data structure
    hyperlinks_data = {
        'by_type': {},
        'by_page': {},
        'all_links': [],
        'social_handles': {},
        'plain_text_links': [],
        'candidate_profile_links': {},
        'statistics': {
            'total_links': 0,
            'embedded_links': 0,
            'plain_text_links': 0,
            'social_handles': 0,
            'pages_with_links': 0,
            'most_common_type': None,
            'platforms_found': []
        }
    }
    
    # Initialize platform categories
    for platform in HYPERLINK_PATTERNS.keys():
        hyperlinks_data['by_type'][platform] = []
    hyperlinks_data['by_type']['other'] = []
    
    seen_urls: Set[str] = set()
    pages_with_links = 0
    full_text = ""
    
    try:
        # ===== Phase 1: Extract full text and embedded hyperlinks =====
        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_text = page.get_text()
            full_text += page_text + "\n"
            page_links = []
            
            # Extract embedded hyperlinks from PDF links
            links = page.get_links()
            
            for link in links:
                if "uri" in link:
                    uri = link["uri"]
                    if uri and isinstance(uri, str):
                        # Normalize URL
                        normalized_url = normalize_url(uri)
                        
                        # Validate URL
                        is_valid, error = validate_url(normalized_url)
                        if not is_valid:
                            continue
                        
                        if normalized_url in seen_urls:
                            continue
                        
                        seen_urls.add(normalized_url)
                        page_links.append(normalized_url)
                        
                        # Extract display text from the link rectangle area
                        display_text = ""
                        if "from" in link:
                            try:
                                rect = link["from"]
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
                            'url': normalized_url,
                            'display_text': display_text,
                            'page': page_num + 1,
                            'type': None,
                            'source': 'embedded',
                            'is_valid': True
                        })
                        hyperlinks_data['statistics']['embedded_links'] += 1
            
            # Extract hyperlinks from annotations
            try:
                for annot in page.annots():
                    if annot.type[1] == 'Link':
                        link_data = annot.info.get('content', '') or annot.info.get('title', '')
                        if link_data and ('http' in link_data or '@' in link_data):
                            normalized_url = normalize_url(link_data)
                            is_valid, _ = validate_url(normalized_url)
                            
                            if is_valid and normalized_url not in seen_urls:
                                seen_urls.add(normalized_url)
                                page_links.append(normalized_url)
                                
                                display_text = annot.info.get('title', '') or ""
                                
                                hyperlinks_data['all_links'].append({
                                    'url': normalized_url,
                                    'display_text': display_text,
                                    'page': page_num + 1,
                                    'type': None,
                                    'source': 'annotation',
                                    'is_valid': True
                                })
                                hyperlinks_data['statistics']['embedded_links'] += 1
            except:
                pass
            
            if page_links:
                pages_with_links += 1
                hyperlinks_data['by_page'][page_num + 1] = page_links
        
        # ===== Phase 2: Extract plain text URLs and emails =====
        plain_text_links = extract_plain_text_links(full_text)
        
        for link_data in plain_text_links:
            url = link_data['url']
            if url not in seen_urls:
                seen_urls.add(url)
                hyperlinks_data['plain_text_links'].append(link_data)
                hyperlinks_data['all_links'].append({
                    'url': url,
                    'display_text': link_data.get('original', ''),
                    'page': None,  # Can't determine page from full text
                    'type': None,
                    'source': 'plain_text',
                    'is_valid': True
                })
                hyperlinks_data['statistics']['plain_text_links'] += 1
        
        # ===== Phase 3: Extract social media handles =====
        social_handles = extract_social_handles(full_text)
        hyperlinks_data['social_handles'] = social_handles
        hyperlinks_data['statistics']['social_handles'] = sum(len(handles) for handles in social_handles.values())
        
        # ===== Phase 4: Classify all links by platform =====
        for link_entry in hyperlinks_data['all_links']:
            url = link_entry['url']
            display_text = link_entry.get('display_text', '').lower().strip()
            classified = False
            username = None
            
            # Try to classify by URL pattern (most reliable)
            for link_type, pattern in HYPERLINK_PATTERNS.items():
                match = pattern.search(url)
                if match:
                    username = match.group(1) if match.groups() else None
                    hyperlinks_data['by_type'][link_type].append({
                        'url': url,
                        'display_text': link_entry.get('display_text', ''),
                        'page': link_entry.get('page'),
                        'username': username,
                        'source': link_entry.get('source'),
                        'classification_method': 'url_pattern'
                    })
                    link_entry['type'] = link_type
                    link_entry['username'] = username
                    classified = True
                    break
            
            # If not classified by URL, try text label patterns
            if not classified and display_text:
                for platform, text_patterns in COMPILED_TEXT_PATTERNS.items():
                    for text_pattern in text_patterns:
                        if text_pattern.search(display_text):
                            if _is_plausible_match(url, platform):
                                username = _extract_username_from_url(url, platform)
                                hyperlinks_data['by_type'][platform].append({
                                    'url': url,
                                    'display_text': link_entry.get('display_text', ''),
                                    'page': link_entry.get('page'),
                                    'username': username,
                                    'source': link_entry.get('source'),
                                    'classification_method': 'text_pattern'
                                })
                                link_entry['type'] = platform
                                link_entry['username'] = username
                                classified = True
                                break
                    if classified:
                        break
            
            # If still not classified, mark as other
            if not classified:
                hyperlinks_data['by_type']['other'].append({
                    'url': url,
                    'display_text': link_entry.get('display_text', ''),
                    'page': link_entry.get('page'),
                    'username': None,
                    'source': link_entry.get('source'),
                    'classification_method': 'unclassified'
                })
                link_entry['type'] = 'other'
        
        # ===== Phase 5: Extract candidate names and match profile links =====
        candidate_names = extract_names_func(full_text)
        
        # Match name-embedded hyperlinks (likely primary profile links)
        # This handles cases like:
        # 1. Display text "arjun-christopher" with any URL
        # 2. Display text "Arjun's GitHub" with github URL
        # 3. Display text "GitHub" but URL is github.com/arjun-christopher
        # 4. URL contains username matching name variations
        
        for link_entry in hyperlinks_data['all_links']:
            display_text = link_entry.get('display_text', '')
            url = link_entry['url']
            platform = link_entry.get('type', 'other')
            
            # Strategy 1: Check if display text OR URL matches candidate name
            is_name_match, matched_name = _matches_candidate_name(display_text, candidate_names, url)
            
            # Strategy 2: If no match yet, check if URL username matches name variations
            # This covers: "GitHub" -> github.com/arjun-christopher
            if not is_name_match and platform != 'other':
                username = link_entry.get('username')
                if username:
                    # Check if username matches any name variation
                    for name in candidate_names:
                        name_variations = _generate_name_variations(name)
                        username_clean = username.lower().replace('-', '').replace('_', '').replace('.', '')
                        
                        for variation in name_variations:
                            variation_clean = variation.replace('-', '').replace('_', '').replace('.', '')
                            if username_clean == variation_clean or (len(username_clean) >= 5 and username_clean in variation_clean):
                                is_name_match = True
                                matched_name = name
                                break
                        
                        if is_name_match:
                            break
            
            # If match found, add to candidate profile links
            if is_name_match and platform != 'other':
                # Primary profile link detected
                if platform not in hyperlinks_data['candidate_profile_links']:
                    # Determine confidence level
                    confidence = 'low'
                    if display_text and matched_name:
                        display_clean = display_text.lower().strip().replace("'s", "").replace("'", "")
                        name_clean = matched_name.lower().strip()
                        if display_clean == name_clean:
                            confidence = 'high'
                        elif any(part in display_clean for part in name_clean.split()):
                            confidence = 'medium'
                    
                    # If matched by URL username only, confidence is medium
                    if not display_text or display_text.lower() in ['github', 'linkedin', 'twitter', 'profile', 'portfolio']:
                        confidence = 'medium'
                    
                    hyperlinks_data['candidate_profile_links'][platform] = {
                        'url': url,
                        'display_text': display_text,
                        'matched_name': matched_name,
                        'page': link_entry.get('page'),
                        'username': link_entry.get('username'),
                        'source': link_entry.get('source'),
                        'confidence': confidence
                    }
                
                link_entry['is_name_embedded'] = True
                link_entry['matched_candidate_name'] = matched_name
        
        # ===== Phase 6: Enhance with social handles =====
        # Try to construct profile URLs from @handles
        for platform, handles in social_handles.items():
            for handle in handles:
                # Check if we already have a profile link for this platform
                if platform not in hyperlinks_data['candidate_profile_links']:
                    # Construct likely profile URL
                    if platform == 'twitter':
                        constructed_url = f"https://twitter.com/{handle}"
                    elif platform == 'instagram':
                        constructed_url = f"https://instagram.com/{handle}"
                    elif platform == 'telegram':
                        constructed_url = f"https://t.me/{handle}"
                    else:
                        constructed_url = None
                    
                    if constructed_url and constructed_url not in seen_urls:
                        hyperlinks_data['candidate_profile_links'][platform] = {
                            'url': constructed_url,
                            'display_text': f"@{handle}",
                            'matched_name': None,
                            'page': None,
                            'username': handle,
                            'source': 'social_handle',
                            'confidence': 'low'
                        }
        
        # ===== Phase 7: Calculate statistics =====
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
        
        # Add unique usernames count
        all_usernames = set()
        for link_entry in hyperlinks_data['all_links']:
            username = link_entry.get('username')
            if username:
                all_usernames.add(username)
        hyperlinks_data['statistics']['unique_usernames'] = len(all_usernames)
    
    except Exception as e:
        print(f"Error extracting hyperlinks: {e}")
        import traceback
        traceback.print_exc()
    
    return hyperlinks_data
