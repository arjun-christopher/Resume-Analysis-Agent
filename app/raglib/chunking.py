from typing import List, Dict

SECTION_HEADS = ["experience","work","skills","education","projects","summary","responsibilities"]

def section_blocks(text: str) -> List[Dict]:
    lines = text.splitlines()
    blocks, cur, sec = [], [], "other"
    for line in lines:
        low = line.strip().lower()
        if any(h in low for h in SECTION_HEADS) and len(line) < 80:
            if cur:
                blocks.append({"section": sec, "text": "\n".join(cur)})
                cur = []
            sec = low
        cur.append(line)
    if cur:
        blocks.append({"section": sec, "text": "\n".join(cur)})
    return blocks

def chunk_blocks(blocks: List[Dict], max_chars: int = 1200) -> List[Dict]:
    out, cid = [], 0
    for b in blocks:
        txt = b["text"]
        for i in range(0, len(txt), max_chars):
            out.append({"id": f"c{cid}", "section": b["section"], "text": txt[i:i+max_chars]})
            cid += 1
    return out
