import io
import re
import fitz
from docx import Document
from PIL import Image
import pytesseract


EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+")
PHONE_RE = re.compile(r"(\+?\d[\d \-]{8,}\d)")


def extract_pdf(b: bytes) -> str:
    doc = fitz.open(stream=b, filetype='pdf')
    return "\n".join(p.get_text('text') for p in doc)


def extract_docx(b: bytes) -> str:
    d = Document(io.BytesIO(b))
    return "\n".join(p.text for p in d.paragraphs)


def extract_image(b: bytes) -> str:
    img = Image.open(io.BytesIO(b))
    return pytesseract.image_to_string(img)


def normalize(t: str) -> str:
    t = t.replace("\u00ad", "")
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def facts(t: str) -> dict:
    e = EMAIL_RE.search(t)
    p = PHONE_RE.search(t)
    return {"email": e.group(0) if e else None, "phone": p.group(0) if p else None}