"""
Comprehensive Certification Information Extractor for Resumes

This module provides advanced NLP-based certification extraction covering all possible resume formats:
- Cloud certifications (AWS, Azure, GCP, Kubernetes)
- IT & Security (CISSP, CompTIA, Cisco, CEH, CISM, CISA)
- Project Management (PMP, PRINCE2, Scrum, SAFe)
- Data & Analytics (Tableau, Power BI, Databricks, Snowflake)
- Programming (Oracle, Microsoft, Red Hat)
- Finance (CPA, CFA, FRM, CMA)
- Quality (Six Sigma, ITIL)
- Marketing (HubSpot, Google Analytics, Google Ads)
- Design (Adobe)
- And any other certification format
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

# Try to import RapidFuzz for fuzzy matching
try:
    from rapidfuzz import fuzz, process
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False

# Try to import FlashText for fast keyword extraction
try:
    from flashtext import KeywordProcessor
    _FLASHTEXT_AVAILABLE = True
except ImportError:
    _FLASHTEXT_AVAILABLE = False


# ---------- Certification Extraction Configuration ----------

# Comprehensive certification patterns covering all formats
CERTIFICATION_PATTERNS = {
    # Cloud & DevOps Certifications
    'aws': [
        r'\b(?:AWS\s+)?Certified\s+(?:Solutions?\s+Architect|Developer|SysOps\s+Administrator|DevOps\s+Engineer|Security|Machine\s+Learning|Data\s+Analytics|Database|Advanced\s+Networking)\b',
        r'\bAWS\s+Certified\s+[A-Za-z\s\-]+(?:Associate|Professional|Specialty)\b',
        r'\bAmazon\s+Web\s+Services\s+Certified\b',
        r'\bAWS\s+Solutions?\s+Architect\b',
        r'\bAWS\s+Developer\b',
        r'\bAWS\s+SysOps\b'
    ],
    'azure': [
        r'\b(?:Microsoft\s+)?Azure\s+(?:Administrator|Developer|Solutions?\s+Architect|DevOps\s+Engineer|Security\s+Engineer|Data\s+Engineer|AI\s+Engineer|Fundamentals)\b',
        r'\bMicrosoft\s+Certified:\s+Azure\s+[A-Za-z\s\-]+\b',
        r'\bAZ-\d{3}\b',
        r'\bAzure\s+Fundamentals\b'
    ],
    'gcp': [
        r'\b(?:Google\s+Cloud\s+)?(?:Professional\s+)?(?:Cloud\s+Architect|Cloud\s+Developer|Cloud\s+Engineer|Data\s+Engineer|Cloud\s+DevOps\s+Engineer|Cloud\s+Security\s+Engineer)\b',
        r'\bGoogle\s+Cloud\s+Certified\s+[A-Za-z\s\-]+\b',
        r'\bGCP\s+Certified\b',
        r'\bGoogle\s+Cloud\s+Professional\b'
    ],
    'kubernetes': [
        r'\b(?:Certified\s+Kubernetes\s+(?:Administrator|Application\s+Developer|Security\s+Specialist)|CKA|CKAD|CKS)\b',
        r'\bKubernetes\s+Certification\b',
        r'\bCertified\s+Kubernetes\b'
    ],
    'docker': [
        r'\bDocker\s+Certified\s+Associate\b',
        r'\bDCA\s+Certification\b'
    ],
    
    # IT & Security Certifications
    'cissp': [
        r'\bCISSP\b',
        r'\bCertified\s+Information\s+Systems\s+Security\s+Professional\b'
    ],
    'comptia': [
        r'\bCompTIA\s+(?:A\+|Network\+|Security\+|Linux\+|Cloud\+|CySA\+|PenTest\+|CASP\+|Server\+)\b',
        r'\bCompTIA\s+[A-Za-z\+]+\b'
    ],
    'cisco': [
        r'\b(?:CCNA|CCNP|CCIE|CCENT|CCDA|CCDP|CCNP\s+Security|CCNA\s+Security)\b',
        r'\bCisco\s+Certified\s+(?:Network\s+Associate|Network\s+Professional|Internetwork\s+Expert|Entry\s+Networking\s+Technician)\b'
    ],
    'ceh': [
        r'\b(?:CEH|Certified\s+Ethical\s+Hacker)\b',
        r'\bEC-Council\s+Certified\b'
    ],
    'cism': [
        r'\b(?:CISM|Certified\s+Information\s+Security\s+Manager)\b'
    ],
    'cisa': [
        r'\b(?:CISA|Certified\s+Information\s+Systems\s+Auditor)\b'
    ],
    'crisc': [
        r'\b(?:CRISC|Certified\s+in\s+Risk\s+and\s+Information\s+Systems\s+Control)\b'
    ],
    'oscp': [
        r'\b(?:OSCP|Offensive\s+Security\s+Certified\s+Professional)\b'
    ],
    
    # Project Management
    'pmp': [
        r'\b(?:PMP|Project\s+Management\s+Professional)\b',
        r'\bPMI\s+Certified\b',
        r'\bPMI-PMP\b'
    ],
    'capm': [
        r'\b(?:CAPM|Certified\s+Associate\s+in\s+Project\s+Management)\b'
    ],
    'prince2': [
        r'\bPRINCE2\s+(?:Foundation|Practitioner|Agile)\b',
        r'\bPRINCE2\s+Certified\b'
    ],
    'scrum': [
        r'\b(?:Certified\s+)?(?:Scrum\s+Master|Scrum\s+Product\s+Owner|Scrum\s+Developer)\b',
        r'\b(?:CSM|CSPO|CSD|PSM|PSPO|PSM\s+[I|II|III])\b',
        r'\bProfessional\s+Scrum\s+Master\b',
        r'\bScrum\s+Alliance\s+Certified\b'
    ],
    'safe': [
        r'\b(?:SAFe|Scaled\s+Agile\s+Framework)\s+(?:Agilist|Practitioner|Scrum\s+Master|Product\s+Owner|Release\s+Train\s+Engineer)\b',
        r'\bCertified\s+SAFe\b',
        r'\bSA\s+Certification\b'
    ],
    
    # Data & Analytics
    'tableau': [
        r'\bTableau\s+(?:Desktop\s+)?(?:Specialist|Certified\s+Associate|Certified\s+Professional|Server\s+Certified)\b',
        r'\bTableau\s+Certification\b'
    ],
    'powerbi': [
        r'\bMicrosoft\s+(?:Power\s+BI|PowerBI)\s+(?:Data\s+Analyst|Certification)\b',
        r'\bPower\s+BI\s+Certified\b',
        r'\bPL-300\b'
    ],
    'databricks': [
        r'\bDatabricks\s+Certified\s+(?:Associate\s+Developer|Professional\s+Data\s+Engineer|Professional\s+Data\s+Scientist)\b',
        r'\bDatabricks\s+Certification\b'
    ],
    'snowflake': [
        r'\bSnowflake\s+(?:SnowPro\s+)?(?:Core|Advanced|Data\s+Engineer|Data\s+Analyst|Architect)\b',
        r'\bSnowPro\s+Certification\b'
    ],
    'cloudera': [
        r'\bCloudera\s+Certified\s+(?:Administrator|Developer|Professional)\b',
        r'\bCCA\s+(?:Administrator|Developer|Data\s+Analyst)\b'
    ],
    
    # Programming & Development
    'oracle': [
        r'\bOracle\s+Certified\s+(?:Associate|Professional|Master|Expert)\b',
        r'\b(?:OCP|OCA)\b',
        r'\bOracle\s+Java\s+(?:SE|EE)\s+\d+\s+(?:Programmer|Developer)\b',
        r'\bOracle\s+Database\s+\d+g?\s+(?:Administrator|SQL)\b'
    ],
    'microsoft': [
        r'\bMicrosoft\s+Certified:\s+[A-Za-z\s\-]+\b',
        r'\b(?:MCSA|MCSE|MCSD|MTA)\b',
        r'\bMicrosoft\s+Technology\s+Associate\b',
        r'\bMicrosoft\s+Certified\s+Solutions\s+(?:Associate|Expert|Developer)\b'
    ],
    'redhat': [
        r'\b(?:RHCSA|RHCE|RHCA|RHCVA)\b',
        r'\bRed\s+Hat\s+Certified\s+(?:System\s+Administrator|Engineer|Architect|Specialist)\b'
    ],
    'salesforce': [
        r'\bSalesforce\s+Certified\s+(?:Administrator|Developer|Consultant|Architect)\b',
        r'\bSalesforce\s+[A-Za-z\s]+Certification\b'
    ],
    
    # Finance & Accounting
    'cpa': [
        r'\b(?:CPA|Certified\s+Public\s+Accountant)\b'
    ],
    'cfa': [
        r'\b(?:CFA|Chartered\s+Financial\s+Analyst)\b',
        r'\bCFA\s+Level\s+[I|II|III]\b'
    ],
    'frm': [
        r'\b(?:FRM|Financial\s+Risk\s+Manager)\b'
    ],
    'cma': [
        r'\b(?:CMA|Certified\s+Management\s+Accountant)\b'
    ],
    'cia': [
        r'\b(?:CIA|Certified\s+Internal\s+Auditor)\b'
    ],
    
    # Six Sigma & Quality
    'six_sigma': [
        r'\bSix\s+Sigma\s+(?:White\s+Belt|Yellow\s+Belt|Green\s+Belt|Black\s+Belt|Master\s+Black\s+Belt)\b',
        r'\bLean\s+Six\s+Sigma\s+(?:Green\s+Belt|Black\s+Belt)\b',
        r'\bCertified\s+Six\s+Sigma\b'
    ],
    
    # ITIL
    'itil': [
        r'\bITIL\s+(?:Foundation|Practitioner|Intermediate|Expert|Master)\b',
        r'\bITIL\s+v?[34]\s+(?:Foundation|Certified)\b',
        r'\bITIL\s+4\s+Foundation\b'
    ],
    
    # Sales & Marketing
    'hubspot': [
        r'\bHubSpot\s+(?:Inbound|Content\s+Marketing|Email\s+Marketing|Social\s+Media|Sales\s+Software)\s+Certification\b',
        r'\bHubSpot\s+Certified\b'
    ],
    'google_analytics': [
        r'\bGoogle\s+Analytics\s+(?:Individual\s+Qualification|Certified|IQ)\b',
        r'\bGAIQ\b',
        r'\bGoogle\s+Analytics\s+Certification\b'
    ],
    'google_ads': [
        r'\bGoogle\s+Ads\s+Certification\b',
        r'\bGoogle\s+AdWords\s+Certified\b',
        r'\bGoogle\s+Ads\s+(?:Search|Display|Video|Shopping)\s+Certification\b'
    ],
    'facebook_blueprint': [
        r'\bFacebook\s+Blueprint\s+Certification\b',
        r'\bFacebook\s+Certified\s+(?:Marketing\s+Science\s+Professional|Planning\s+Professional)\b'
    ],
    
    # Design
    'adobe': [
        r'\bAdobe\s+Certified\s+(?:Associate|Expert|Professional)\b',
        r'\bAdobe\s+(?:Photoshop|Illustrator|InDesign|XD|Premiere\s+Pro|After\s+Effects)\s+Certified\b',
        r'\bACA\s+Certification\b'
    ],
    
    # HR & Training
    'shrm': [
        r'\b(?:SHRM-CP|SHRM-SCP)\b',
        r'\bSHRM\s+Certified\s+Professional\b'
    ],
    'phr': [
        r'\b(?:PHR|SPHR|GPHR)\b',
        r'\bProfessional\s+in\s+Human\s+Resources\b'
    ],
    
    # Healthcare
    'pmi_acp': [
        r'\b(?:PMI-ACP|PMI\s+Agile\s+Certified\s+Practitioner)\b'
    ],
    
    # General patterns for any certification
    'general': [
        r'\b(?:Certified|Certification)\s+[A-Z][A-Za-z\s\-]{3,50}(?:Professional|Associate|Expert|Specialist|Administrator|Developer|Engineer)\b',
        r'\b[A-Z][A-Za-z\s\-]{3,50}\s+(?:Certified|Certification)\b',
        r'\b[A-Z]{2,10}\s+Certification\b',
        r'\bCertificate\s+in\s+[A-Z][A-Za-z\s\-]{3,50}\b'
    ]
}

# Certification issuing organizations
CERTIFICATION_ISSUERS = [
    r'\b(?:issued\s+by|from|by|provider)\s+([A-Z][A-Za-z\s&,\.]{3,60}?)(?:\s+(?:in|on|,|\.|$))',
    r'\b([A-Z][A-Za-z\s&,\.]{3,60}?)\s+(?:Certified|Certification)\b',
]

# Certification ID patterns
CERTIFICATION_ID_PATTERNS = [
    r'\b(?:Certificate\s+(?:ID|Number|No\.?)|Certification\s+ID|Credential\s+ID|License\s+(?:Number|No\.?))[\s:]*([A-Z0-9\-]{5,30})\b',
    r'\b(?:ID|#)[\s:]*([A-Z0-9\-]{8,30})\b',
    r'\bCredential\s+ID[\s:]*([A-Z0-9\-]{5,30})\b',
]

# Certification date patterns
CERTIFICATION_DATE_PATTERNS = [
    r'\b(?:Issued|Obtained|Earned|Completed|Achieved|Awarded)[\s:]*([A-Z][a-z]+\.?\s+\d{4}|\d{4}|\d{1,2}/\d{4})\b',
    r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\s*[-–—]\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|Present|Current|No\s+Expiration)\b',
    r'\b(\d{4})\s*[-–—]\s*(\d{4}|Present|Current|No\s+Expiration)\b',
    r'\b(\d{1,2}/\d{4})\s*[-–—]\s*(\d{1,2}/\d{4}|Present)\b',
]

# Expiration patterns
CERTIFICATION_EXPIRY_PATTERNS = [
    r'\b(?:Expires?|Expiration|Valid\s+(?:until|through|till))[\s:]*([A-Z][a-z]+\.?\s+\d{4}|\d{4}|\d{1,2}/\d{4})\b',
    r'\b(?:Valid|Active)\s+(?:until|through|till)[\s:]*([A-Z][a-z]+\.?\s+\d{4}|\d{4})\b',
    r'\b(?:No\s+Expiration|Does\s+not\s+expire|Lifetime\s+Valid|Never\s+Expires)\b',
]

# Certification section headers
CERTIFICATION_SECTION_HEADERS = [
    r'\b(?:CERTIFICATIONS?|CERTIFICATES?|PROFESSIONAL\s+CERTIFICATIONS?|LICENSES?\s+(?:AND|&)\s+CERTIFICATIONS?|'
    r'CREDENTIALS?|PROFESSIONAL\s+CREDENTIALS?|ACCREDITATIONS?|QUALIFICATIONS?|PROFESSIONAL\s+QUALIFICATIONS?)\b'
]

# Verification URL patterns
VERIFICATION_URL_PATTERNS = [
    r'(?:Verify|Verification|View|Link|URL)[\s:]*(?:https?://[^\s]+)',
    r'https?://(?:www\.)?(?:credly\.com|acclaim\.com|certmetrics\.com|verify\.[^\s]+|badgr\.com|youracclaim\.com)',
]

# Certification levels/tiers for ranking
CERTIFICATION_LEVELS = {
    'entry': ['foundation', 'fundamentals', 'essentials', 'entry', 'associate', 'certified associate'],
    'intermediate': ['intermediate', 'professional', 'practitioner', 'specialist', 'developer', 'administrator'],
    'advanced': ['advanced', 'expert', 'master', 'architect', 'senior', 'lead'],
    'specialty': ['specialty', 'specialized', 'security', 'devops', 'data engineer', 'ml engineer']
}

# Certification acronym standardization
CERTIFICATION_ACRONYMS = {
    'cka': 'Certified Kubernetes Administrator',
    'ckad': 'Certified Kubernetes Application Developer',
    'cks': 'Certified Kubernetes Security Specialist',
    'aws saa': 'AWS Solutions Architect Associate',
    'aws sap': 'AWS Solutions Architect Professional',
    'az-104': 'Microsoft Azure Administrator',
    'az-204': 'Microsoft Azure Developer',
    'az-303': 'Microsoft Azure Architect Technologies',
    'az-304': 'Microsoft Azure Architect Design',
    'pmp': 'Project Management Professional',
    'csm': 'Certified Scrum Master',
    'psm': 'Professional Scrum Master',
    'cissp': 'Certified Information Systems Security Professional',
    'cism': 'Certified Information Security Manager',
    'cisa': 'Certified Information Systems Auditor',
    'ceh': 'Certified Ethical Hacker',
    'oscp': 'Offensive Security Certified Professional',
}

# Compile certification patterns
COMPILED_CERT_PATTERNS = {}
for cert_type, patterns in CERTIFICATION_PATTERNS.items():
    COMPILED_CERT_PATTERNS[cert_type] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

COMPILED_CERT_ISSUER_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in CERTIFICATION_ISSUERS]
COMPILED_CERT_ID_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in CERTIFICATION_ID_PATTERNS]
COMPILED_CERT_DATE_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in CERTIFICATION_DATE_PATTERNS]
COMPILED_CERT_EXPIRY_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in CERTIFICATION_EXPIRY_PATTERNS]
COMPILED_CERT_SECTION_HEADERS = [re.compile(pattern, re.IGNORECASE) for pattern in CERTIFICATION_SECTION_HEADERS]
COMPILED_VERIFICATION_URL_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in VERIFICATION_URL_PATTERNS]

# Initialize FlashText processor for O(n) certification matching
_CERTIFICATION_PROCESSOR = None
if _FLASHTEXT_AVAILABLE:
    _CERTIFICATION_PROCESSOR = KeywordProcessor(case_sensitive=False)
    
    # Add all certification names as keywords for fast extraction
    common_certifications = [
        # Cloud
        'aws certified', 'azure certified', 'google cloud certified', 'cka', 'ckad', 'cks',
        # IT Security
        'cissp', 'comptia security+', 'comptia network+', 'ccna', 'ccnp', 'ccie', 
        'ceh', 'cism', 'cisa', 'crisc', 'oscp',
        # Project Management
        'pmp', 'capm', 'prince2', 'csm', 'psm', 'safe agilist',
        # Data
        'tableau certified', 'power bi certified', 'databricks certified', 'snowpro',
        # Programming
        'oracle certified', 'microsoft certified', 'rhcsa', 'rhce', 'salesforce certified',
        # Finance
        'cpa', 'cfa', 'frm', 'cma', 'cia',
        # Quality
        'six sigma', 'lean six sigma', 'itil foundation',
        # Marketing
        'hubspot certified', 'google analytics', 'gaiq', 'google ads certified',
        # HR
        'shrm-cp', 'shrm-scp', 'phr', 'sphr'
    ]
    
    for cert in common_certifications:
        _CERTIFICATION_PROCESSOR.add_keyword(cert)


def detect_certification_sections(text: str) -> List[Tuple[int, int, str]]:
    """
    Detect certification sections in resume text
    Returns: List of (start_line, end_line, section_text) tuples
    """
    sections = []
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line_clean = line.strip()
        if not line_clean:
            continue
        
        # Check if line is a certification section header
        for pattern in COMPILED_CERT_SECTION_HEADERS:
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
                                         'PROJECTS', 'PUBLICATIONS', 'AWARDS', 'INTERESTS', 'SUMMARY']
                        if any(section in next_line.upper() for section in common_sections):
                            end_line = j
                            break
                    
                    end_line = j + 1
                
                section_lines = lines[start_line:end_line]
                section_text = '\n'.join(section_lines)
                sections.append((start_line, end_line, section_text))
                break
    
    return sections


def extract_certification_info(text: str) -> List[Dict[str, Any]]:
    """Extract certification information from text with comprehensive pattern matching"""
    certifications = []
    
    # IMPROVEMENT: Define false positive patterns to filter out
    false_positive_patterns = [
        r'\b(?:Bachelor|Master|PhD|B\.?S\.?|M\.?S\.?|M\.?B\.?A\.?|B\.?A\.?|Associate Degree)s?\b',  # Degrees
        r'\b(?:University|College|Institute|School)\b',  # Educational institutions
        r'\b(?:Senior|Junior|Lead|Principal|Staff|Chief)\s+(?:Engineer|Developer|Analyst|Manager|Architect|Consultant|Administrator|Specialist)\b',  # Job titles with seniority
        r'\b(?:Full[- ]?Stack|Front[- ]?End|Back[- ]?End)\s+(?:Developer|Engineer)\b',  # Role descriptions
        r'\bCertified\s+(?:Java|Python|Ruby|JavaScript|C\+\+|C#|PHP|Golang|Rust|Swift|Kotlin)\s+(?:Developer|Programmer|Engineer)\b',  # Generic "Certified X Developer"
        r'\b(?:Certified|Professional)\s+in\s+[A-Za-z\s]{1,30}$',  # Incomplete certification mentions
    ]
    compiled_false_positive = [re.compile(p, re.IGNORECASE) for p in false_positive_patterns]
    
    for cert_type, patterns in COMPILED_CERT_PATTERNS.items():
        for pattern in patterns:
            matches = pattern.finditer(text)
            for match in matches:
                cert_text = match.group(0)
                
                # IMPROVEMENT: Skip if matches false positive patterns
                is_false_positive = False
                for fp_pattern in compiled_false_positive:
                    if fp_pattern.search(cert_text):
                        is_false_positive = True
                        break
                
                if is_false_positive:
                    continue
                
                # IMPROVEMENT: Validate minimum certification name length
                if len(cert_text.strip()) < 3:
                    continue
                
                start_pos = match.start()
                end_pos = match.end()
                
                # IMPROVEMENT: Extract multi-line context (look for complete lines)
                # Find line boundaries for better context
                lines_before_start = text[:start_pos].rfind('\n', max(0, start_pos - 500))
                lines_after_end = text[end_pos:].find('\n\n', 0, min(500, len(text) - end_pos))
                
                context_start = max(0, lines_before_start if lines_before_start != -1 else start_pos - 400)
                context_end = min(len(text), end_pos + lines_after_end if lines_after_end != -1 else end_pos + 400)
                context = text[context_start:context_end].strip()
                
                # Extract certification level
                cert_level = extract_certification_level(cert_text)
                
                # Standardize acronyms
                standardized_name = standardize_certification_name(cert_text)
                
                # IMPROVEMENT: Calculate confidence based on context quality
                confidence = 0.85  # Base confidence for regex matches
                
                # Boost confidence if certification appears in a certification section
                if re.search(r'\b(?:certification|certificate|license|credential)s?\b', context[:100], re.IGNORECASE):
                    confidence = min(1.0, confidence + 0.1)
                
                # Reduce confidence if in work experience section (might be requirement, not actual cert)
                if re.search(r'\b(?:experience|work history|employment|position|role)\b', context[:100], re.IGNORECASE):
                    confidence = max(0.5, confidence - 0.15)
                
                cert_info = {
                    'certification_type': cert_type,
                    'certification_name': standardized_name,
                    'original_name': cert_text.strip(),
                    'certification_level': cert_level,
                    'context': context,
                    'position': (start_pos, end_pos),
                    'confidence': confidence
                }
                
                certifications.append(cert_info)
    
    return certifications


def extract_certification_level(cert_name: str) -> str:
    """Extract certification level/tier from name"""
    cert_lower = cert_name.lower()
    
    for level, keywords in CERTIFICATION_LEVELS.items():
        for keyword in keywords:
            if keyword in cert_lower:
                return level
    
    return 'intermediate'  # Default level


def standardize_certification_name(cert_name: str) -> str:
    """Standardize certification names from acronyms"""
    cert_lower = cert_name.lower().strip()
    
    # Check if it's a known acronym
    for acronym, full_name in CERTIFICATION_ACRONYMS.items():
        if cert_lower == acronym or cert_lower.startswith(acronym + ' '):
            return full_name
    
    # Clean up whitespace and return original
    return re.sub(r'\s+', ' ', cert_name).strip()


def extract_certification_issuer(context: str) -> Optional[str]:
    """Extract issuing organization from context"""
    for pattern in COMPILED_CERT_ISSUER_PATTERNS:
        match = pattern.search(context)
        if match:
            if match.lastindex and match.lastindex >= 1:
                issuer = match.group(1).strip()
            else:
                issuer = match.group(0).strip()
            
            issuer = re.sub(r'\s+', ' ', issuer)
            issuer = issuer.rstrip('.,;:')
            
            # Filter out common false positives
            false_positives = ['Certification', 'Certified', 'Certificate', 'License', 'Credential']
            if not any(fp.lower() == issuer.lower() for fp in false_positives):
                if len(issuer) > 2 and len(issuer) < 100:
                    return issuer
    
    return None


def extract_certification_id(context: str) -> Optional[str]:
    """Extract certification ID or credential number from context"""
    for pattern in COMPILED_CERT_ID_PATTERNS:
        match = pattern.search(context)
        if match:
            if match.lastindex and match.lastindex >= 1:
                cert_id = match.group(1).strip()
                if len(cert_id) >= 5:
                    return cert_id
    
    return None


def extract_certification_dates(context: str) -> Dict[str, Optional[str]]:
    """Extract issue and expiration dates from certification context with validation"""
    dates = {
        'issue_date': None,
        'expiry_date': None,
        'is_active': True,
        'no_expiration': False,
        'validity_years': None
    }
    
    # Check for issue date
    for pattern in COMPILED_CERT_DATE_PATTERNS:
        match = pattern.search(context)
        if match:
            if match.lastindex >= 2:
                # Date range found
                issue_date_raw = match.group(1).strip()
                expiry_raw = match.group(2).strip()
                
                # Validate and standardize issue date
                dates['issue_date'] = standardize_date(issue_date_raw)
                
                if expiry_raw.lower() in ['present', 'current', 'no expiration']:
                    dates['no_expiration'] = True
                else:
                    dates['expiry_date'] = standardize_date(expiry_raw)
                break
            elif match.lastindex == 1:
                # Single date found
                dates['issue_date'] = standardize_date(match.group(1).strip())
    
    # Check for expiration date separately
    for pattern in COMPILED_CERT_EXPIRY_PATTERNS:
        match = pattern.search(context)
        if match:
            if 'no expiration' in match.group(0).lower() or 'does not expire' in match.group(0).lower() or 'never expires' in match.group(0).lower() or 'lifetime' in match.group(0).lower():
                dates['no_expiration'] = True
                dates['expiry_date'] = 'No Expiration'
            elif match.lastindex >= 1:
                dates['expiry_date'] = standardize_date(match.group(1).strip())
            break
    
    # IMPROVEMENT: Calculate validity years and determine if certification is active
    try:
        import datetime
        current_year = datetime.datetime.now().year
        current_month = datetime.datetime.now().month
        
        # Extract years from dates
        issue_year = None
        issue_month = None
        if dates['issue_date']:
            year_match = re.search(r'\d{4}', dates['issue_date'])
            if year_match:
                issue_year = int(year_match.group(0))
                # Try to extract month
                month_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', dates['issue_date'], re.IGNORECASE)
                if month_match:
                    month_names = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                                 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
                    issue_month = month_names.get(month_match.group(1)[:3].lower(), None)
        
        if dates['expiry_date'] and dates['expiry_date'] != 'No Expiration':
            expiry_year = None
            expiry_month = None
            year_match = re.search(r'\d{4}', dates['expiry_date'])
            if year_match:
                expiry_year = int(year_match.group(0))
                # Try to extract month
                month_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', dates['expiry_date'], re.IGNORECASE)
                if month_match:
                    month_names = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                                 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
                    expiry_month = month_names.get(month_match.group(1)[:3].lower(), None)
            
            # Determine if active based on year and month
            if expiry_year:
                if expiry_year > current_year:
                    dates['is_active'] = True
                elif expiry_year == current_year and expiry_month and expiry_month >= current_month:
                    dates['is_active'] = True
                elif expiry_year == current_year and not expiry_month:
                    dates['is_active'] = True  # Assume still valid if only year matches
                else:
                    dates['is_active'] = False
            
            # Calculate validity period
            if issue_year and expiry_year:
                dates['validity_years'] = expiry_year - issue_year
    except Exception as e:
        # If date parsing fails, assume active
        pass
    
    return dates


def standardize_date(date_str: str) -> str:
    """Standardize date format for consistency"""
    if not date_str:
        return date_str
    
    # Try to parse and reformat common date patterns
    # MM/YYYY -> Mon YYYY
    match = re.match(r'(\d{1,2})/(\d{4})', date_str)
    if match:
        month_num = int(match.group(1))
        year = match.group(2)
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        if 1 <= month_num <= 12:
            return f"{month_names[month_num-1]} {year}"
    
    # Already in good format, just clean up
    return re.sub(r'\s+', ' ', date_str).strip()


def extract_verification_url(context: str) -> Optional[str]:
    """Extract verification URL from certification context"""
    for pattern in COMPILED_VERIFICATION_URL_PATTERNS:
        match = pattern.search(context)
        if match:
            # Extract just the URL part
            url_match = re.search(r'https?://[^\s]+', match.group(0))
            if url_match:
                return url_match.group(0).rstrip('.,;:)')
    
    return None


def calculate_cert_similarity(cert1: Dict[str, Any], cert2: Dict[str, Any]) -> float:
    """Calculate similarity between two certification entries with improved accuracy"""
    score = 0.0
    
    # Name similarity (most important)
    name1 = cert1.get('certification_name', '').lower()
    name2 = cert2.get('certification_name', '').lower()
    
    if name1 and name2:
        # IMPROVEMENT: Check for exact or substring match first (high confidence)
        if name1 == name2:
            score += 0.7
        elif name1 in name2 or name2 in name1:
            # Only if the shorter name is at least 60% of the longer
            shorter_len = min(len(name1), len(name2))
            longer_len = max(len(name1), len(name2))
            if shorter_len / longer_len >= 0.6:
                score += 0.6
        else:
            # Token-based similarity for different variations
            tokens1 = set(name1.split())
            tokens2 = set(name2.split())
            if tokens1 and tokens2:
                intersection = len(tokens1 & tokens2)
                union = len(tokens1 | tokens2)
                token_sim = intersection / union if union > 0 else 0
                # IMPROVEMENT: Require higher token similarity (70% instead of allowing any overlap)
                if token_sim >= 0.7:
                    score += token_sim * 0.5
    
    # Position proximity (less weight)
    pos1 = cert1.get('position', (0, 0))
    pos2 = cert2.get('position', (0, 0))
    position_distance = abs(pos1[0] - pos2[0])
    if position_distance < 50:  # Very close (same line area)
        score += 0.2
    elif position_distance < 200:  # Nearby (same paragraph)
        score += 0.1
    
    # Same type adds confidence
    if cert1.get('certification_type') == cert2.get('certification_type'):
        score += 0.1
    
    # IMPROVEMENT: Penalize if confidence scores differ significantly
    conf1 = cert1.get('confidence', 0.5)
    conf2 = cert2.get('confidence', 0.5)
    if abs(conf1 - conf2) > 0.3:
        score *= 0.8  # Reduce overall score if confidence mismatch
    
    return score


def merge_certification_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge duplicate or overlapping certification entries using improved similarity scoring"""
    if not entries:
        return []
    
    entries.sort(key=lambda x: x.get('position', (0, 0))[0])
    
    merged = []
    skip_indices = set()
    
    for i, entry in enumerate(entries):
        if i in skip_indices:
            continue
        
        # Find all similar entries
        similar_entries = [entry]
        
        for j in range(i + 1, len(entries)):
            if j in skip_indices:
                continue
            
            other = entries[j]
            
            # IMPROVEMENT: Use stricter similarity threshold to reduce false merges
            similarity = calculate_cert_similarity(entry, other)
            
            if similarity >= 0.65:  # Increased from 0.5 to 0.65 for stricter matching
                similar_entries.append(other)
                skip_indices.add(j)
        
        # Merge all similar entries, keeping most complete information
        if len(similar_entries) > 1:
            # IMPROVEMENT: Start with entry that has highest confidence
            similar_entries.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            merged_entry = similar_entries[0].copy()
            
            for other_entry in similar_entries[1:]:
                # Merge fields, preferring non-null values with higher confidence
                for key in ['issuer', 'certification_id', 'issue_date', 'expiry_date', 
                           'verification_url', 'certification_level', 'validity_years']:
                    if not merged_entry.get(key) and other_entry.get(key):
                        merged_entry[key] = other_entry[key]
                    elif merged_entry.get(key) and other_entry.get(key):
                        # Both have value, prefer one with higher confidence
                        if other_entry.get('confidence', 0) > merged_entry.get('confidence', 0):
                            merged_entry[key] = other_entry[key]
                
                # IMPROVEMENT: More conservative confidence boost for merged entries
                merged_entry['confidence'] = min(1.0, merged_entry.get('confidence', 0.5) + 0.10)
            
            merged.append(merged_entry)
        else:
            merged.append(entry)
    
    return merged


def extract_certifications_with_flashtext(text: str) -> List[Dict[str, Any]]:
    """
    Extract certifications using FlashText for O(n) keyword matching
    Fast extraction of known certification names
    """
    if not _FLASHTEXT_AVAILABLE or not _CERTIFICATION_PROCESSOR:
        return []
    
    certifications = []
    
    # Extract keywords in O(n) time - much faster than regex for many keywords
    found_keywords = _CERTIFICATION_PROCESSOR.extract_keywords(text, span_info=True)
    
    for keyword, start, end in found_keywords:
        # Get context around the certification
        context_start = max(0, start - 200)
        context_end = min(len(text), end + 200)
        context = text[context_start:context_end]
        
        cert_entry = {
            'name': keyword,
            'position': (start, end),
            'context': context,
            'extraction_method': 'flashtext',
            'confidence': 0.95  # High confidence for exact keyword matches
        }
        
        # Try to extract additional information from context
        cert_id = extract_certification_id(context)
        if cert_id:
            cert_entry['certification_id'] = cert_id
        
        dates = extract_certification_dates(context)
        cert_entry.update(dates)
        
        issuer = extract_certification_issuer(context)
        if issuer:
            cert_entry['issuer'] = issuer
        
        certifications.append(cert_entry)
    
    return certifications


def extract_certifications_with_fuzzy_matching(text: str, min_score: int = 88) -> List[Dict[str, Any]]:
    """
    Extract certifications using fuzzy matching to handle typos and variations
    Catches misspellings like 'Kubernates' -> 'Kubernetes'
    """
    if not _RAPIDFUZZ_AVAILABLE:
        return []
    
    certifications = []
    
    # Common certification names to fuzzy match against
    common_cert_names = [
        'AWS Certified Solutions Architect', 'AWS Certified Developer',
        'Microsoft Azure Administrator', 'Azure Solutions Architect',
        'Google Cloud Professional', 'Certified Kubernetes Administrator',
        'CISSP', 'CompTIA Security+', 'CompTIA Network+',
        'CCNA', 'CCNP', 'CCIE',
        'PMP', 'PRINCE2', 'Certified Scrum Master',
        'Tableau Certified', 'Power BI Certified',
        'Oracle Certified Professional', 'Red Hat Certified',
        'CPA', 'CFA', 'Six Sigma Black Belt',
        'ITIL Foundation'
    ]
    
    # Extract potential certification mentions from lines
    lines = text.split('\n')
    for i, line in enumerate(lines):
        line_clean = line.strip()
        if len(line_clean) < 5:
            continue
        
        # Check if line likely contains a certification
        cert_indicators = ['certified', 'certification', 'certificate', 'credential']
        if not any(indicator in line_clean.lower() for indicator in cert_indicators):
            continue
        
        # Fuzzy match against known certifications
        matches = process.extract(
            line_clean,
            common_cert_names,
            scorer=fuzz.token_set_ratio,
            limit=3
        )
        
        for matched_cert, score, _ in matches:
            if score >= min_score:
                certifications.append({
                    'name': matched_cert,
                    'original_text': line_clean,
                    'confidence': score / 100.0,
                    'line_number': i,
                    'extraction_method': 'fuzzy_matching'
                })
                break
    
    return certifications


def extract_certifications_with_nlp(text: str) -> List[Dict[str, Any]]:
    """Extract certification information using NLP and spaCy (if available)"""
    certification_entries = []
    
    if not _NLP:
        return certification_entries
    
    try:
        doc = _NLP(text[:100000])
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            cert_keywords = ['certified', 'certification', 'certificate', 'credential', 
                           'license', 'accreditation', 'qualified', 'professional']
            
            if any(keyword in sent_text.lower() for keyword in cert_keywords):
                entry = {'raw_text': sent_text}
                
                # Extract organizations (likely issuers)
                orgs = [ent.text for ent in sent.ents if ent.label_ == 'ORG']
                if orgs:
                    entry['issuer'] = orgs[0]
                
                # Extract dates
                dates = [ent.text for ent in sent.ents if ent.label_ == 'DATE']
                if dates:
                    entry['dates'] = dates
                
                if len(entry) > 1:
                    certification_entries.append(entry)
    
    except Exception as e:
        print(f"NLP certification extraction error: {e}")
    
    return certification_entries


def extract_certifications_comprehensive(text: str) -> List[Dict[str, Any]]:
    """
    Comprehensive certification extraction using multiple NLP techniques
    
    This function covers all possible resume formats including:
    - Cloud certifications (AWS, Azure, GCP, Kubernetes, Docker)
    - IT & Security (CISSP, CompTIA, Cisco, CEH, CISM, CISA, CRISC, OSCP)
    - Project Management (PMP, CAPM, PRINCE2, Scrum, SAFe)
    - Data & Analytics (Tableau, Power BI, Databricks, Snowflake, Cloudera)
    - Programming (Oracle, Microsoft, Red Hat, Salesforce)
    - Finance (CPA, CFA, FRM, CMA, CIA)
    - Quality (Six Sigma, ITIL)
    - Marketing (HubSpot, Google Analytics, Google Ads, Facebook Blueprint)
    - Design (Adobe)
    - HR (SHRM, PHR)
    - And any other certification format
    
    Args:
        text: Resume text to extract certification information from
    
    Returns:
        List of dictionaries containing structured certification information with fields:
        - certification_name: The certification name
        - certification_type: Category (aws, azure, pmp, etc.)
        - issuer: Issuing organization
        - certification_id: Credential/License ID
        - issue_date: Date issued
        - expiry_date: Expiration date
        - is_active: Boolean indicating if currently valid
        - no_expiration: Boolean for lifetime certifications
        - verification_url: URL to verify the certification
    """
    certification_entries = []
    
    # Step 1: Detect certification sections
    cert_sections = detect_certification_sections(text)
    
    # If no explicit section found, use entire text
    if not cert_sections:
        cert_sections = [(0, len(text), text)]
    
    # Step 2: Extract certification information from each section
    for start, end, section_text in cert_sections:
        certifications = extract_certification_info(section_text)
        
        for cert_info in certifications:
            context = cert_info['context']
            
            entry = {
                'certification_name': cert_info['certification_name'],
                'certification_type': cert_info['certification_type'],
                'issuer': extract_certification_issuer(context),
                'certification_id': extract_certification_id(context),
                'verification_url': extract_verification_url(context),
                'raw_text': context[:300],
            }
            
            # Extract dates
            date_info = extract_certification_dates(context)
            entry.update(date_info)
            entry['position'] = cert_info['position']
            
            certification_entries.append(entry)
    
    # Step 3: Use NLP-based extraction as supplementary
    if _NLP:
        nlp_entries = extract_certifications_with_nlp(text)
        
        for nlp_entry in nlp_entries:
            is_duplicate = False
            for existing in certification_entries:
                if (nlp_entry.get('issuer') and 
                    existing.get('issuer') and 
                    nlp_entry['issuer'].lower() in existing['issuer'].lower()):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                certification_entries.append(nlp_entry)
    
    # Step 3.5: Add FlashText extraction for fast O(n) keyword matching
    if _FLASHTEXT_AVAILABLE:
        flashtext_entries = extract_certifications_with_flashtext(text)
        
        for flashtext_entry in flashtext_entries:
            # IMPROVEMENT: Use similarity check instead of simple substring
            is_duplicate = False
            best_match_score = 0
            
            for existing in certification_entries:
                # Calculate similarity
                if flashtext_entry.get('name') and existing.get('certification_name'):
                    name1 = flashtext_entry['name'].lower()
                    name2 = existing['certification_name'].lower()
                    
                    # Check overlap
                    if name1 in name2 or name2 in name1:
                        is_duplicate = True
                        break
                    
                    # Token-based similarity
                    tokens1 = set(name1.split())
                    tokens2 = set(name2.split())
                    if tokens1 and tokens2:
                        overlap = len(tokens1 & tokens2) / len(tokens1 | tokens2)
                        if overlap > 0.6:  # 60% token overlap
                            is_duplicate = True
                            # Enrich existing entry with flashtext data
                            if not existing.get('certification_id') and flashtext_entry.get('certification_id'):
                                existing['certification_id'] = flashtext_entry['certification_id']
                            break
            
            if not is_duplicate:
                # Rename 'name' to 'certification_name' for consistency
                flashtext_entry['certification_name'] = flashtext_entry.pop('name', 'Unknown')
                certification_entries.append(flashtext_entry)
    
    # Step 3.75: Add fuzzy matching for typo handling
    if _RAPIDFUZZ_AVAILABLE:
        fuzzy_entries = extract_certifications_with_fuzzy_matching(text, min_score=85)  # Slightly lower threshold
        
        for fuzzy_entry in fuzzy_entries:
            # Check for duplicates with similarity scoring
            is_duplicate = False
            
            for existing in certification_entries:
                if fuzzy_entry.get('name') and existing.get('certification_name'):
                    name1 = fuzzy_entry['name'].lower()
                    name2 = existing['certification_name'].lower()
                    
                    # Check if very similar
                    if name1 in name2 or name2 in name1:
                        is_duplicate = True
                        # Boost confidence of existing if fuzzy match confirms
                        if fuzzy_entry.get('confidence', 0) >= 0.85:
                            existing['confidence'] = min(1.0, existing.get('confidence', 0.5) + 0.1)
                        break
            
            if not is_duplicate:
                # Rename 'name' to 'certification_name' for consistency
                fuzzy_entry['certification_name'] = fuzzy_entry.pop('name', 'Unknown')
                certification_entries.append(fuzzy_entry)
    
    # Step 4: Merge duplicate entries
    certification_entries = merge_certification_entries(certification_entries)
    
    # Step 5: Clean up and format final entries
    final_entries = []
    for entry in certification_entries:
        entry.pop('position', None)
        
        # IMPROVEMENT: Filter out low-confidence entries (likely false positives)
        if entry.get('confidence', 0) < 0.6:
            continue
        
        cleaned_entry = {k: v for k, v in entry.items() if v}
        
        if cleaned_entry.get('certification_name') or cleaned_entry.get('issuer'):
            final_entries.append(cleaned_entry)
    
    # Step 6: Sort by issue date (most recent first)
    def get_sort_key(entry):
        issue_date = entry.get('issue_date', '')
        if not issue_date:
            return 0
        
        year_match = re.search(r'\d{4}', str(issue_date))
        if year_match:
            return int(year_match.group(0))
        return 0
    
    final_entries.sort(key=get_sort_key, reverse=True)
    
    return final_entries


# Backward compatibility - alias for the main function
extract_certifications_info = extract_certifications_comprehensive
