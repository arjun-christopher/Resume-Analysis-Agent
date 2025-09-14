from typing import Any, Dict, Optional
from google.cloud import storage, firestore
import firebase_admin
from firebase_admin import credentials, initialize_app
from .config import settings


cred = credentials.Certificate(settings.FIREBASE_SA_PATH)
_app = initialize_app(cred, {
    'projectId': settings.FIREBASE_PROJECT_ID,
    'storageBucket': settings.FIREBASE_BUCKET,
})
_db = firestore.Client(project=settings.FIREBASE_PROJECT_ID)
_bucket = storage.Client().bucket(settings.FIREBASE_BUCKET.replace("gs://", ""))


COL_BATCH = "batches"
COL_FILEITEM = "file_items"
COL_CANDIDATE = "candidates"
COL_CHUNK = "chunks"
COL_JD = "job_descriptions"
COL_MATCH = "match_results"


class FB:
    @staticmethod
    def db():
        return _db
    @staticmethod
    def bucket():
        return _bucket
    @staticmethod
    def create(coll: str, data: Dict[str, Any]) -> str:
        ref = _db.collection(coll).document()
        ref.set(data)
        return ref.id
    @staticmethod
    def update(coll: str, doc_id: str, data: Dict[str, Any]):
        _db.collection(coll).document(doc_id).update(data)
    @staticmethod
    def get(coll: str, doc_id: str) -> Optional[Dict[str, Any]]:
        d = _db.collection(coll).document(doc_id).get()
        return d.to_dict() if d.exists else None
    @staticmethod
    def upload_bytes(dest_path: str, b: bytes, content_type: Optional[str] = None) -> str:
        blob = _bucket.blob(dest_path)
        blob.upload_from_string(b, content_type=content_type)
        return f"gs://{_bucket.name}/{dest_path}"