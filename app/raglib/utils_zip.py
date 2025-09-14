import io
import zipfile
from typing import List, Tuple

SUPPORTED = (".pdf",".docx",".png",".jpg",".jpeg")

def is_supported_name(name: str) -> bool:
    ln = name.lower()
    return any(ln.endswith(ext) for ext in SUPPORTED)

def iter_zip_supported(zip_bytes: bytes) -> List[Tuple[str, bytes]]:
    items = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for n in zf.namelist():
            if n.endswith("/"):
                continue
            if is_supported_name(n):
                items.append((n, zf.read(n)))
    return items
