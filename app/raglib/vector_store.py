import json
import os
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from .config import settings


class FaissStore:
    def __init__(self, dim=384, idx_path=None, meta_path=None):
        self.dim=dim; self.idx_path=idx_path or settings.VECTOR_PATH; self.meta_path=meta_path or settings.META_PATH
        self.index = faiss.IndexFlatIP(dim)
        self.meta: List[Dict[str,Any]] = []
        self._load()
    def add(self, vecs: np.ndarray, metas: List[Dict[str,Any]]):
        if vecs.dtype != np.float32:
            vecs = vecs.astype(np.float32)
        self.index.add(vecs); self.meta.extend(metas); self._save()
    def search(self, qvec: np.ndarray, k=5) -> List[Tuple[float, Dict[str,Any]]]:
        if qvec.ndim==1:
            qvec=qvec[None,:]
        D,I = self.index.search(qvec.astype(np.float32), k)
        out=[]
        for s,idx in zip(D[0], I[0]):
            if idx==-1:
                continue
            out.append((float(s), self.meta[idx]))
        return out
    def _save(self):
        faiss.write_index(self.index, self.idx_path)
        with open(self.meta_path,'w',encoding='utf-8') as f:
            for m in self.meta:
                f.write(json.dumps(m)+"\n")
    def _load(self):
        if os.path.exists(self.idx_path):
            self.index = faiss.read_index(self.idx_path)
        if os.path.exists(self.meta_path):
            with open(self.meta_path,'r',encoding='utf-8') as f:
                self.meta=[json.loads(line) for line in f]


VS = FaissStore()