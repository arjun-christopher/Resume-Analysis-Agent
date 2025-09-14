import io
import re
import os
from pathlib import Path
import fitz               # PyMuPDF
from docx import Document # python-docx
from PIL import Image
import pytesseract

EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+")
PHONE_RE = re.compile(r"(\+?\d[\d \-]{8,}\d)")

# Windows: allow overriding tesseract path via env var
if os.environ.get("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]

def extract_pdf_bytes(data: bytes) -> str:
    doc = fitz.open(stream=data, filetype="pdf")
    return "\n".join(p.get_text("text") for p in doc)

def extract_docx_bytes(data: bytes) -> str:
    d = Document(io.BytesIO(data))
    return "\n".join(p.text for p in d.paragraphs)

def extract_image_bytes(data: bytes) -> str:
    img = Image.open(io.BytesIO(data))
    return pytesseract.image_to_string(img)

def extract_any(path: str | Path) -> str:
    p = str(path).lower()
    with open(path, "rb") as f:
        data = f.read()
    if p.endswith(".pdf"):  return extract_pdf_bytes(data)
    if p.endswith(".docx"): return extract_docx_bytes(data)
    if p.endswith((".png",".jpg",".jpeg")): return extract_image_bytes(data)
    raise ValueError("Unsupported file: "+str(path))

def normalize_text(t: str) -> str:
    t = t.replace("\u00ad","")          # soft hyphen
    t = re.sub(r"\n{3,}", "\n\n", t)    # collapse blank lines
    return t.strip()

def quick_facts(t: str) -> dict:
    e = EMAIL_RE.search(t)
    p = PHONE_RE.search(t)
    return {
        "email": e.group(0) if e else None,
        "phone": p.group(0) if p else None
    }
