"""
Comprehensive Publications Information Extractor for Resumes

This module provides advanced NLP-based publications extraction covering all possible resume formats:
- Research papers and journal articles
- Conference papers and proceedings
- Books and book chapters
- Patents and technical reports
- Theses and dissertations
- Preprints and arXiv papers
- Blog posts and technical articles
- White papers and case studies
- Posters and presentations
- Workshop papers
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


# ---------- Publications Extraction Configuration ----------

# Publication section headers
PUBLICATION_SECTION_HEADERS = [
    r'\b(?:PUBLICATIONS?|RESEARCH\s+PUBLICATIONS?|ACADEMIC\s+PUBLICATIONS?|'
    r'PAPERS?|RESEARCH\s+PAPERS?|PUBLISHED\s+(?:WORKS?|PAPERS?)|'
    r'JOURNAL\s+(?:ARTICLES?|PAPERS?|PUBLICATIONS?)|CONFERENCE\s+(?:PAPERS?|PUBLICATIONS?)|'
    r'BOOKS?\s+(?:AND|&)\s+PUBLICATIONS?|SELECTED\s+PUBLICATIONS?|'
    r'PATENTS?|PATENTS?\s+(?:AND|&)\s+PUBLICATIONS?|'
    r'TECHNICAL\s+(?:PUBLICATIONS?|PAPERS?|REPORTS?)|'
    r'SCHOLARLY\s+(?:WORKS?|PUBLICATIONS?|ARTICLES?)|'
    r'PEER[-\s]?REVIEWED\s+PUBLICATIONS?|RESEARCH\s+OUTPUT)\b'
]

# Publication type patterns
PUBLICATION_TYPE_PATTERNS = {
    'journal': [
        r'\b(?:journal|journal\s+article|journal\s+paper)\b',
        r'\b(?:published\s+in|appeared\s+in)\s+[A-Z][A-Za-z\s&]+(?:Journal|Review|Letters?|Transactions?)\b',
        r'\b(?:IEEE|ACM|Nature|Science|PLOS|Springer|Elsevier)\s+(?:Journal|Transactions?)\b',
    ],
    'conference': [
        r'\b(?:conference|conference\s+paper|conference\s+proceedings?)\b',
        r'\b(?:presented\s+at|accepted\s+at|appeared\s+in)\s+[A-Z][A-Za-z\s&]+(?:Conference|Symposium|Workshop)\b',
        r'\b(?:IEEE|ACM|AAAI|NeurIPS|ICML|CVPR|ICLR|EMNLP|ACL)\b',
    ],
    'book': [
        r'\b(?:book|textbook|monograph|handbook)\b',
        r'\b(?:authored|co[-\s]?authored|edited|co[-\s]?edited)\s+(?:book|textbook)\b',
        r'\b(?:ISBN|Publisher)[\s:]\b',
    ],
    'book_chapter': [
        r'\b(?:book\s+chapter|chapter\s+in)\b',
        r'\b(?:contributed|wrote)\s+chapter\b',
    ],
    'patent': [
        r'\b(?:patent|patents?|patent\s+application)\b',
        r'\b(?:US\s+Patent|Patent\s+No\.?|Patent\s+Number)\b',
        r'\b(?:filed|granted|pending|issued)\s+patent\b',
    ],
    'thesis': [
        r'\b(?:thesis|dissertation|doctoral\s+thesis|master\'?s\s+thesis|PhD\s+thesis)\b',
        r'\b(?:defended|submitted)\s+(?:thesis|dissertation)\b',
    ],
    'preprint': [
        r'\b(?:preprint|pre[-\s]?print|arXiv|bioRxiv|medRxiv)\b',
        r'\barXiv:\d{4}\.\d{4,5}\b',
    ],
    'technical_report': [
        r'\b(?:technical\s+report|tech\s+report|research\s+report|white\s+paper)\b',
        r'\b(?:TR|Report\s+No\.?)\b',
    ],
    'poster': [
        r'\b(?:poster|poster\s+presentation)\b',
        r'\b(?:presented\s+poster)\b',
    ],
    'workshop': [
        r'\b(?:workshop|workshop\s+paper)\b',
        r'\b(?:presented\s+at|accepted\s+at)\s+[A-Z][A-Za-z\s&]+Workshop\b',
    ],
    'blog': [
        r'\b(?:blog\s+post|article|technical\s+article|medium\s+article)\b',
        r'\b(?:published\s+on|wrote\s+for)\s+(?:Medium|Dev\.to|Towards\s+Data\s+Science)\b',
    ]
}

# Citation format patterns (various styles)
CITATION_PATTERNS = [
    # APA style: Author, A. (Year). Title. Journal, Volume(Issue), pages.
    r'([A-Z][A-Za-z\s,\.&-]+)\s+\((\d{4})\)\.\s+"?([^"\.]+)"?\.?\s+([A-Z][A-Za-z\s&:]+),?\s*(\d+)?\(?(\d+)?\)?,?\s*(?:pp?\.)?\s*(\d+-\d+)?',
    
    # IEEE style: A. Author, "Title," Journal, vol. X, no. Y, pp. Z, Year.
    r'([A-Z][A-Za-z\s,\.&-]+),\s+"([^"]+)",\s+([A-Z][A-Za-z\s&:]+),\s+vol\.\s*(\d+),\s+no\.\s*(\d+),\s+pp\.\s*(\d+-\d+),\s+(\d{4})',
    
    # Simple format: Title, Author, Year
    r'"?([A-Z][A-Za-z\s:,\-]{10,150})"?,?\s+([A-Z][A-Za-z\s,\.&-]+),?\s+(\d{4})',
]

# Author patterns
AUTHOR_PATTERNS = [
    r'\b([A-Z][a-z]+(?:\s+[A-Z]\.?)+(?:\s+[A-Z][a-z]+)?)\b',  # First Last or First M. Last
    r'\b([A-Z][a-z]+,\s+[A-Z]\.(?:\s+[A-Z]\.)?)\b',  # Last, F. M.
    r'\b(?:by|authored\s+by|written\s+by|with)[\s:]+([A-Z][A-Za-z\s,\.&-]+)\b',
]

# Title patterns (quoted or emphasized)
TITLE_PATTERNS = [
    r'"([^"]{10,200})"',  # Quoted title
    r'\'([^\']{10,200})\'',  # Single quoted
    r'\*([^\*]{10,200})\*',  # Emphasized with asterisks
    r'_([^_]{10,200})_',  # Emphasized with underscores
]

# Venue patterns (journal, conference names)
VENUE_PATTERNS = [
    r'\b(?:in|published\s+in|appeared\s+in|presented\s+at)[\s:]+([A-Z][A-Za-z\s&:,\-]{5,100}?)(?:\s+(?:\d{4}|vol\.|,|$))',
    r'\b([A-Z][A-Za-z\s&]+(?:Journal|Conference|Symposium|Workshop|Review|Letters?|Transactions?))\b',
]

# Year patterns
YEAR_PATTERNS = [
    r'\b(19\d{2}|20\d{2})\b',
    r'\((\d{4})\)',
]

# Volume, Issue, Pages patterns
VOLUME_ISSUE_PATTERNS = [
    r'\bvol\.?\s*(\d+)\b',
    r'\bvolume\s+(\d+)\b',
    r'\bno\.?\s*(\d+)\b',
    r'\bissue\s+(\d+)\b',
    r'\bpp?\.?\s*(\d+-\d+)\b',
    r'\bpages?\s+(\d+-\d+)\b',
]

# DOI patterns
DOI_PATTERNS = [
    r'\b(?:DOI|doi)[\s:]*(?:https?://(?:dx\.)?doi\.org/)?(10\.\d{4,}/[^\s]+)',
    r'\b(10\.\d{4,}/[^\s]+)\b',
]

# arXiv patterns
ARXIV_PATTERNS = [
    r'\barXiv[\s:]*(\d{4}\.\d{4,5}(?:v\d+)?)\b',
]

# ISBN patterns
ISBN_PATTERNS = [
    r'\bISBN[\s:]*(\d{3}-?\d{1,5}-?\d{1,7}-?\d{1,7}-?\d{1})\b',
    r'\bISBN[\s:]*(\d{13}|\d{10})\b',
]

# Patent number patterns
PATENT_NUMBER_PATTERNS = [
    r'\b(?:US\s+Patent|Patent\s+No\.?|Patent\s+Number)[\s:]*([A-Z]{0,2}\d{6,10}[A-Z]?\d?)\b',
    r'\b(US\d{7,10}[A-Z]?\d?)\b',
]

# Impact metrics patterns
IMPACT_PATTERNS = [
    r'\b(?:cited|citations?)[\s:]*(\d+(?:,\d{3})*)\s+times?\b',
    r'\b(\d+(?:,\d{3})*)\s+citations?\b',
    r'\b(?:impact\s+factor|IF)[\s:]*(\d+\.\d+)\b',
    r'\bh[-\s]?index[\s:]*(\d+)\b',
]

# Status patterns
STATUS_PATTERNS = [
    r'\b(published|accepted|submitted|under\s+review|in\s+press|forthcoming|in\s+preparation)\b',
]

# Compile patterns
COMPILED_PUBLICATION_HEADERS = [re.compile(pattern, re.IGNORECASE) for pattern in PUBLICATION_SECTION_HEADERS]
COMPILED_CITATION_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in CITATION_PATTERNS]
COMPILED_AUTHOR_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in AUTHOR_PATTERNS]
COMPILED_TITLE_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in TITLE_PATTERNS]
COMPILED_VENUE_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in VENUE_PATTERNS]
COMPILED_YEAR_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in YEAR_PATTERNS]
COMPILED_VOLUME_ISSUE_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in VOLUME_ISSUE_PATTERNS]
COMPILED_DOI_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in DOI_PATTERNS]
COMPILED_ARXIV_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in ARXIV_PATTERNS]
COMPILED_ISBN_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in ISBN_PATTERNS]
COMPILED_PATENT_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in PATENT_NUMBER_PATTERNS]
COMPILED_IMPACT_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in IMPACT_PATTERNS]
COMPILED_STATUS_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in STATUS_PATTERNS]

COMPILED_PUBLICATION_TYPE_PATTERNS = {}
for pub_type, patterns in PUBLICATION_TYPE_PATTERNS.items():
    COMPILED_PUBLICATION_TYPE_PATTERNS[pub_type] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


def detect_publication_sections(text: str) -> List[Tuple[int, int, str]]:
    """
    Detect publication sections in resume text
    Returns: List of (start_line, end_line, section_text) tuples
    """
    sections = []
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean:
            continue
        
        # Check if line is a publication section header
        for pattern in COMPILED_PUBLICATION_HEADERS:
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
                                         'PROJECTS', 'CERTIFICATIONS', 'ACHIEVEMENTS', 'ACTIVITIES']
                        if any(section in next_line.upper() for section in common_sections):
                            end_line = j
                            break
                    
                    end_line = j + 1
                
                section_lines = lines[start_line:end_line]
                section_text = '\n'.join(section_lines)
                sections.append((start_line, end_line, section_text))
                break
    
    return sections


def split_into_individual_publications(text: str) -> List[str]:
    """
    Split publication section text into individual publication entries
    """
    publications = []
    lines = text.split('\n')
    
    current_pub = []
    in_pub = False
    
    for i, line in enumerate(lines):
        line_clean = line.strip()
        
        # Skip empty lines at the start
        if not line_clean and not in_pub:
            continue
        
        # Check if this line starts a new publication
        is_new_pub = False
        
        # Heuristic 1: Line starts with bullet point or number
        if re.match(r'^(?:•|●|○|◦|▪|▫|–|—|\*|\d+\.|\d+\)|\[\d+\])\s+', line_clean):
            is_new_pub = True
        
        # Heuristic 2: Line starts with author names (Last, F. format)
        elif re.match(r'^[A-Z][a-z]+,\s+[A-Z]\.', line_clean):
            is_new_pub = True
        
        # Heuristic 3: Line starts with quoted title
        elif re.match(r'^"[A-Z]', line_clean):
            is_new_pub = True
        
        # If we found a new publication and we have accumulated content, save it
        if is_new_pub and current_pub:
            pub_text = '\n'.join(current_pub)
            if len(pub_text.strip()) > 20:
                publications.append(pub_text)
            current_pub = []
            in_pub = True
        
        # Add line to current publication
        if line_clean or in_pub:
            current_pub.append(line)
            in_pub = True
    
    # Don't forget the last publication
    if current_pub:
        pub_text = '\n'.join(current_pub)
        if len(pub_text.strip()) > 20:
            publications.append(pub_text)
    
    return publications


def extract_publication_title(text: str) -> Optional[str]:
    """Extract publication title from text"""
    # Try quoted patterns first
    for pattern in COMPILED_TITLE_PATTERNS:
        match = pattern.search(text)
        if match:
            title = match.group(1).strip()
            if 10 < len(title) < 300:
                return title
    
    # Try to find title in first line or after authors
    lines = text.split('\n')
    for line in lines[:3]:
        line_clean = line.strip()
        # Remove bullet points
        line_clean = re.sub(r'^(?:•|●|○|◦|▪|▫|–|—|\*|\d+\.|\d+\)|\[\d+\])\s*', '', line_clean)
        
        # Check if it looks like a title (not too short, not all caps)
        if 10 < len(line_clean) < 300 and not line_clean.isupper():
            # Remove quotes if present
            line_clean = line_clean.strip('"\'')
            return line_clean
    
    return None


def extract_authors(text: str) -> List[str]:
    """Extract author names from publication text"""
    authors = []
    
    for pattern in COMPILED_AUTHOR_PATTERNS:
        matches = pattern.finditer(text)
        for match in matches:
            author = match.group(1).strip()
            # Clean up author name
            author = re.sub(r'\s+', ' ', author)
            if 3 < len(author) < 100 and author not in authors:
                authors.append(author)
    
    return authors[:20]  # Limit to 20 authors


def extract_venue(text: str) -> Optional[str]:
    """Extract publication venue (journal, conference name)"""
    for pattern in COMPILED_VENUE_PATTERNS:
        match = pattern.search(text)
        if match:
            venue = match.group(1).strip() if match.lastindex >= 1 else match.group(0).strip()
            venue = re.sub(r'\s+', ' ', venue)
            venue = venue.rstrip('.,;:')
            
            if 5 < len(venue) < 150:
                return venue
    
    return None


def extract_year(text: str) -> Optional[str]:
    """Extract publication year"""
    for pattern in COMPILED_YEAR_PATTERNS:
        match = pattern.search(text)
        if match:
            year = match.group(1).strip()
            # Validate year range
            try:
                year_int = int(year)
                if 1950 <= year_int <= 2030:
                    return year
            except ValueError:
                continue
    
    return None


def extract_volume_issue_pages(text: str) -> Dict[str, Optional[str]]:
    """Extract volume, issue, and page numbers"""
    info = {'volume': None, 'issue': None, 'pages': None}
    
    for pattern in COMPILED_VOLUME_ISSUE_PATTERNS:
        match = pattern.search(text)
        if match:
            matched_text = match.group(0).lower()
            value = match.group(1).strip()
            
            if 'vol' in matched_text:
                info['volume'] = value
            elif 'no' in matched_text or 'issue' in matched_text:
                info['issue'] = value
            elif 'pp' in matched_text or 'page' in matched_text:
                info['pages'] = value
    
    return info


def extract_doi(text: str) -> Optional[str]:
    """Extract DOI (Digital Object Identifier)"""
    for pattern in COMPILED_DOI_PATTERNS:
        match = pattern.search(text)
        if match:
            doi = match.group(1).strip()
            return doi
    
    return None


def extract_arxiv_id(text: str) -> Optional[str]:
    """Extract arXiv identifier"""
    for pattern in COMPILED_ARXIV_PATTERNS:
        match = pattern.search(text)
        if match:
            arxiv_id = match.group(1).strip()
            return arxiv_id
    
    return None


def extract_isbn(text: str) -> Optional[str]:
    """Extract ISBN for books"""
    for pattern in COMPILED_ISBN_PATTERNS:
        match = pattern.search(text)
        if match:
            isbn = match.group(1).strip()
            return isbn
    
    return None


def extract_patent_number(text: str) -> Optional[str]:
    """Extract patent number"""
    for pattern in COMPILED_PATENT_PATTERNS:
        match = pattern.search(text)
        if match:
            patent_num = match.group(1).strip()
            return patent_num
    
    return None


def extract_impact_metrics(text: str) -> Dict[str, Optional[str]]:
    """Extract citation count, impact factor, h-index"""
    metrics = {'citations': None, 'impact_factor': None, 'h_index': None}
    
    for pattern in COMPILED_IMPACT_PATTERNS:
        match = pattern.search(text)
        if match:
            matched_text = match.group(0).lower()
            value = match.group(1).strip()
            
            if 'citation' in matched_text or 'cited' in matched_text:
                metrics['citations'] = value
            elif 'impact' in matched_text or 'if' in matched_text:
                metrics['impact_factor'] = value
            elif 'h-index' in matched_text or 'h index' in matched_text:
                metrics['h_index'] = value
    
    return metrics


def extract_status(text: str) -> Optional[str]:
    """Extract publication status"""
    for pattern in COMPILED_STATUS_PATTERNS:
        match = pattern.search(text)
        if match:
            status = match.group(1).strip()
            return status.lower()
    
    return None


def detect_publication_type(text: str) -> List[str]:
    """Detect publication type(s) from text"""
    pub_types = []
    
    for pub_type, patterns in COMPILED_PUBLICATION_TYPE_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text):
                pub_types.append(pub_type)
                break
    
    return pub_types


def extract_publications_with_nlp(text: str) -> List[Dict[str, Any]]:
    """Extract publication information using NLP and spaCy (if available)"""
    publication_entries = []
    
    if not _NLP:
        return publication_entries
    
    try:
        doc = _NLP(text[:100000])
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            pub_keywords = ['published', 'paper', 'journal', 'conference', 'article', 
                          'proceedings', 'book', 'patent', 'thesis']
            
            if any(keyword in sent_text.lower() for keyword in pub_keywords):
                entry = {'raw_text': sent_text}
                
                # Extract dates (likely publication year)
                dates = [ent.text for ent in sent.ents if ent.label_ == 'DATE']
                if dates:
                    entry['year'] = dates[0]
                
                # Extract organizations (could be publishers or venues)
                orgs = [ent.text for ent in sent.ents if ent.label_ == 'ORG']
                if orgs:
                    entry['venue'] = orgs[0]
                
                if len(entry) > 1:
                    publication_entries.append(entry)
    
    except Exception as e:
        print(f"NLP publication extraction error: {e}")
    
    return publication_entries


def extract_publications_comprehensive(text: str) -> Dict[str, Any]:
    """
    Comprehensive publication extraction using multiple techniques
    
    This function covers all possible resume formats including:
    - Research papers (journal and conference)
    - Books and book chapters
    - Patents and technical reports
    - Theses and dissertations
    - Preprints (arXiv, bioRxiv, etc.)
    - Blog posts and technical articles
    - White papers and case studies
    - Posters and presentations
    - Workshop papers
    
    Handles various citation formats:
    - APA style
    - IEEE style
    - MLA style
    - Chicago style
    - Custom formats
    - Numbered lists
    - Bullet points
    
    Args:
        text: Resume text to extract publication information from
    
    Returns:
        Dictionary containing:
        - publications: List of publication dictionaries with fields:
            - title: Publication title
            - authors: List of author names
            - venue: Journal/conference name
            - year: Publication year
            - publication_types: List of types (journal, conference, etc.)
            - volume, issue, pages: Citation details
            - doi: Digital Object Identifier
            - arxiv_id: arXiv identifier (if applicable)
            - isbn: ISBN (for books)
            - patent_number: Patent number (if applicable)
            - citations: Citation count
            - status: Publication status (published, accepted, etc.)
        - statistics: Summary statistics
    """
    publication_entries = []
    
    # Step 1: Detect publication sections
    pub_sections = detect_publication_sections(text)
    
    # If no explicit section found, try to find publications in entire text
    if not pub_sections:
        pub_keywords = ['published', 'paper', 'journal', 'conference', 'patent', 'thesis']
        if any(keyword in text.lower() for keyword in pub_keywords):
            pub_sections = [(0, len(text), text)]
    
    # Step 2: Extract publications from each section
    for start, end, section_text in pub_sections:
        # Split section into individual publications
        individual_pubs = split_into_individual_publications(section_text)
        
        for pub_text in individual_pubs:
            entry = {
                'title': extract_publication_title(pub_text),
                'authors': extract_authors(pub_text),
                'venue': extract_venue(pub_text),
                'year': extract_year(pub_text),
                'publication_types': detect_publication_type(pub_text),
                'doi': extract_doi(pub_text),
                'arxiv_id': extract_arxiv_id(pub_text),
                'isbn': extract_isbn(pub_text),
                'patent_number': extract_patent_number(pub_text),
                'status': extract_status(pub_text),
                'raw_text': pub_text[:500],
            }
            
            # Extract volume, issue, pages
            vol_info = extract_volume_issue_pages(pub_text)
            entry.update(vol_info)
            
            # Extract impact metrics
            impact_info = extract_impact_metrics(pub_text)
            entry.update(impact_info)
            
            publication_entries.append(entry)
    
    # Step 3: Use NLP-based extraction as supplementary
    if _NLP and not publication_entries:
        nlp_entries = extract_publications_with_nlp(text)
        
        for nlp_entry in nlp_entries:
            nlp_entry['extraction_method'] = 'nlp'
            publication_entries.append(nlp_entry)
    
    # Step 4: Clean up and format final entries
    final_entries = []
    for entry in publication_entries:
        # Remove empty fields
        cleaned_entry = {k: v for k, v in entry.items() if v}
        
        # Ensure at least title or venue is present
        if cleaned_entry.get('title') or cleaned_entry.get('venue'):
            final_entries.append(cleaned_entry)
    
    # Step 5: Sort by year (most recent first)
    def get_sort_key(entry):
        year = entry.get('year', '')
        if year:
            try:
                return int(year)
            except (ValueError, TypeError):
                pass
        return 0
    
    final_entries.sort(key=get_sort_key, reverse=True)
    
    # Step 6: Add statistics
    pub_stats = {
        'total_count': len(final_entries),
        'by_type': {},
        'with_doi': sum(1 for p in final_entries if p.get('doi')),
        'with_citations': sum(1 for p in final_entries if p.get('citations')),
        'total_citations': 0,
        'years': []
    }
    
    for pub in final_entries:
        # Count by type
        for pub_type in pub.get('publication_types', []):
            pub_stats['by_type'][pub_type] = pub_stats['by_type'].get(pub_type, 0) + 1
        
        # Sum citations
        if pub.get('citations'):
            try:
                citations = int(pub['citations'].replace(',', ''))
                pub_stats['total_citations'] += citations
            except (ValueError, AttributeError):
                pass
        
        # Collect years
        if pub.get('year'):
            pub_stats['years'].append(pub['year'])
    
    # Get unique years
    pub_stats['years'] = sorted(list(set(pub_stats['years'])), reverse=True)
    
    return {
        'publications': final_entries,
        'statistics': pub_stats
    }


# Backward compatibility - alias for the main function
extract_publications_info = extract_publications_comprehensive
