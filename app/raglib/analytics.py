from collections import Counter
from typing import List, Dict


def skills_tally(all_skills: List[List[str]]) -> Dict[str,int]:
    c=Counter()
    for arr in all_skills:
        c.update([s.lower() for s in arr])
    return dict(c.most_common(50))