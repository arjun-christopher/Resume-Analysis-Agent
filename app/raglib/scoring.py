from typing import List, Dict, Any
from .embeddings import ensure_embedder
from .vector_store import search_topk

THRESH = 0.65  # similarity threshold (approx, since we invert distance)

def prompt_to_requirements(prompt: str) -> List[Dict[str,Any]]:
    txt = (prompt or "").strip()
    if not txt:
        return []
    # comma or newline-separated preferred; else split by space
    if "," in txt:
        parts = [p.strip() for p in txt.split(",") if p.strip()]
    elif "\n" in txt:
        parts = [p.strip() for p in txt.splitlines() if p.strip()]
    else:
        parts = [p for p in txt.split() if len(p) > 2]
    seen=set(); reqs=[]
    for p in parts:
        low=p.lower()
        if low in seen: continue
        seen.add(low)
        reqs.append({"name": p, "aliases": [], "weight": 1.0, "must_have": False})
    return reqs[:12]

def _score_one_candidate(candidate_id: str, reqs: List[Dict[str,Any]], store, embedder) -> Dict[str,Any]:
    details=[]; miss=[]; agg=0.0; tw=0.0
    for req in reqs:
        q = " ".join([req["name"]] + req.get("aliases",[]))
        hits = search_topk(store, q, k=8, embedder=embedder)
        matched=False; ev=[]
        for h in hits:
            if h["metadata"].get("candidate_id") != candidate_id:
                continue
            s = h["score"]
            if s >= THRESH:
                matched=True
                ev.append({"chunk_id":h["metadata"]["chunk_id"], "quote":h["preview"]})
        r = 5 if matched else 0
        details.append({"requirement": req["name"], "score": r, "evidence": ev, "gaps": ([] if matched else [f"Missing {req['name']}"])})
        w = req.get("weight",1.0)
        agg += r*w; tw += w
        if req.get("must_have") and not matched:
            miss.append(req["name"])
    final = round((agg/(tw*5))*100,2) if tw>0 else 0.0
    return {"score": final, "details": details, "missing_must_haves": miss}

def rank_all_candidates(candidates: List[Dict[str,Any]], reqs: List[Dict[str,Any]], store, embedder=None):
    embedder = embedder or ensure_embedder()
    rows=[]
    for c in candidates:
        cid = c["candidate_id"]
        scored = _score_one_candidate(cid, reqs, store, embedder)
        rows.append({"candidate_id": cid, **scored})
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows
