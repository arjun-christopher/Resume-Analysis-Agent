def parse_command(text: str):
    t = (text or "").strip()
    low = t.lower()

    # normalize
    if low in ("help","/help","?"):
        return {"name":"help"}

    if low.startswith("clear") or low in ("reset","/clear"):
        return {"name":"clear"}

    if low.startswith("ingest") or low in ("process","/ingest"):
        return {"name":"ingest"}

    if low.startswith("search:") or low.startswith("find:"):
        q = t.split(":",1)[-1]
        return {"name":"search", "query": q}

    if low.startswith("rank:") or low.startswith("shortlist:"):
        args = t.split(":",1)[-1]
        return {"name":"rank", "args": args}

    if low.startswith("analytics") or low == "/analytics":
        return {"name":"analytics"}

    return {"name":"unknown"}
