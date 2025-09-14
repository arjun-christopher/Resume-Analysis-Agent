from typing import Dict, Any
from .embeddings import embed
from .vector_store import VS


THRESH=0.65


def _query(req: Dict[str,Any]) -> str:
    return " ".join([req["name"]] + req.get("aliases",[]))


def score_candidate(candidate_id: str, jd: Dict[str,Any]) -> Dict[str,Any]:
    details=[]
    miss=[]
    agg=0.0
    tw=0.0
    for req in jd["requirements"]: 
        q = _query(req)
        qv = embed([q])[0]
        hits = VS.search(qv, k=8)
        matched=False
        ev=[]
        for s,m in hits:
            if m.get("candidate_id")!=candidate_id:
                continue
            if s>=THRESH:
                matched=True
                ev.append({"chunk_id":m["chunk_id"],"quote":m["preview"]})
        r = 5 if matched else 0
        details.append({"requirement":req["name"],"score":r,"evidence":ev,"gaps":([] if matched else [f"Missing {req['name']}"])})
        w=req.get("weight",1.0)
        agg+=r*w
        tw+=w
        if req.get("must_have") and not matched:
            miss.append(req["name"])
    final = round((agg/(tw*5))*100,2) if tw>0 else 0.0
    return {"score":final, "details":details, "missing_must_haves":miss}