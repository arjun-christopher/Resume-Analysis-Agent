from collections import Counter
from typing import List, Dict, Any

def simple_analytics(candidates: List[Dict[str,Any]]):
    n = len(candidates)
    emails = sum(1 for c in candidates if c.get("facts",{}).get("email"))
    phones = sum(1 for c in candidates if c.get("facts",{}).get("phone"))
    # dummy top domains (from email)
    ctr = Counter()
    for c in candidates:
        em = c.get("facts",{}).get("email")
        if em and "@" in em:
            ctr.update([em.split("@")[-1].lower()])
    top = dict(ctr.most_common(15))
    return {
        "num_candidates": n,
        "emails_found": emails,
        "phones_found": phones,
        "top_domains": top
    }
