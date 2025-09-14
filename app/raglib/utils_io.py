import io
import zipfile
import mimetypes
from typing import List, Tuple
from .config import settings


SUPPORTED = settings.SUPPORTED_EXT


def is_supported(filename: str) -> bool:
    fn = filename.lower()
    return any(fn.endswith(ext) for ext in SUPPORTED)


def iter_zip_supported(zip_bytes: bytes) -> List[Tuple[str, bytes]]:
    out = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for n in zf.namelist():
            if n.endswith('/'):
                continue
            if is_supported(n):
                out.append((n, zf.read(n)))
    return out


def guess_mime(filename: str) -> str:
    m, _ = mimetypes.guess_type(filename)
    return m or 'application/octet-stream'