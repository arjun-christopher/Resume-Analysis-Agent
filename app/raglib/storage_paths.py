import hashlib


def batch_prefix(batch_id: str) -> str:
    return f"batches/{batch_id}"


def file_key(batch_id: str, filename: str) -> str:
    return f"{batch_prefix(batch_id)}/uploads/{filename}"


def artifact_text_key(batch_id: str, file_id: str) -> str:
    return f"{batch_prefix(batch_id)}/artifacts/{file_id}/normalized.txt"


def artifact_json_key(batch_id: str, file_id: str) -> str:
    return f"{batch_prefix(batch_id)}/artifacts/{file_id}/parsed.json"


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()