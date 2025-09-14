from typing import List, Dict


SECTION_HEADS = ["experience","work","skills","education","projects","summary"]


def section_blocks(text: str) -> List[Dict]:
    lines = text.splitlines()
    blocks=[]
    cur=[]
    sec="other"
    for line in lines:
        low=line.strip().lower()
        if any(h in low for h in SECTION_HEADS) and len(line) < 80:
            if cur: 
                blocks.append({"section":sec, "text":"\n".join(cur)})
                cur=[]
            sec=low
        cur.append(line)
    if cur: 
        blocks.append({"section":sec, "text":"\n".join(cur)})
    return blocks


def chunk(blocks: List[Dict], max_chars=1200) -> List[Dict]:
    out=[]
    i=0
    for b in blocks:
        t=b['text']
        for k in range(0,len(t),max_chars):
            out.append({"id":f"c{i}","section":b['section'],"text":t[k:k+max_chars]})
            i+=1
    return out