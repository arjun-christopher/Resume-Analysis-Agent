import re
from typing import Dict, List

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None

SKILL_VOCAB = {
    "python","java","c++","c","c#","javascript","typescript","react","angular","vue",
    "node","express","django","flask","fastapi",
    "aws","azure","gcp","ec2","s3","lambda","cloudformation","terraform","ansible",
    "docker","kubernetes","helm","jenkins","github actions","gitlab ci","ci/cd",
    "sql","mysql","postgres","postgresql","sqlite","mongodb","nosql","redis","elastic",
    "kafka","spark","hadoop","airflow","dbt",
    "pandas","numpy","scikit-learn","xgboost","lightgbm",
    "pytorch","tensorflow","keras","nlp","computer vision","opencv",
    "microservices","rest","graphql",
    "linux","git","tableau","power bi","snowflake","databricks"
}

DEGREE_PAT = re.compile(r"\b(B\.?E\.?|B\.?Tech|BSc|MSc|M\.?Tech|MBA|Ph\.?D|Bachelor|Master|Doctorate)\b", re.I)
EMAIL_PAT = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PAT = re.compile(r"(\+?\d[\d\s\-()]{7,}\d)")

SECTION_KEYWORDS = {
    "education": ["education","academics","qualifications"],
    "experience": ["experience","employment","work history","professional experience"],
    "projects": ["projects","project experience"],
    "skills": ["skills","technical skills","tech stack"],
    "certifications": ["certifications","licenses"],
    "summary": ["summary","objective","profile"],
    "achievements": ["achievements","awards","honors"],
}

def guess_section(text: str) -> str:
    t = text.lower()
    for sec, keys in SECTION_KEYWORDS.items():
        for k in keys:
            if k in t[:120]:
                return sec
    return "unknown"

def _find_skills(text: str) -> List[str]:
    t = text.lower()
    return sorted({s for s in SKILL_VOCAB if s in t})

def _find_degrees(text: str) -> List[str]:
    return sorted(set(m.group(0) for m in DEGREE_PAT.finditer(text)))

def _find_emails(text: str) -> List[str]:
    return EMAIL_PAT.findall(text)

def _find_phones(text: str) -> List[str]:
    return [m.group(0) for m in PHONE_PAT.finditer(text)]

def _find_names_spacy(text: str) -> List[str]:
    if _NLP is None: return []
    doc = _NLP(text)
    return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

def extract_resume_entities(text: str) -> Dict[str, List[str]]:
    return {
        "skills": _find_skills(text),
        "degrees": _find_degrees(text),
        "emails": _find_emails(text),
        "phones": _find_phones(text),
        "names": _find_names_spacy(text),
    }
