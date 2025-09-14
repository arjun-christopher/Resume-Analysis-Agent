import os, uuid, json, time
from typing import List, Dict, Any
import streamlit as st

from raglib.config import cfg, ensure_session_dirs, session_paths
from raglib.storage_local import save_file_bytes, list_supported_files, clear_session_all
from raglib.utils_zip import iter_zip_supported, is_supported_name
from raglib.parsing import extract_any, normalize_text, quick_facts
from raglib.chunking import section_blocks, chunk_blocks
from raglib.embeddings import embed_model_name, ensure_embedder
from raglib.vector_store import load_or_create_store, add_chunks_to_store, search_topk
from raglib.scoring import prompt_to_requirements, rank_all_candidates
from raglib.analytics import simple_analytics
from raglib.commands import parse_command

st.set_page_config(page_title="RAG-Resume (Local Chat)", page_icon="🤖", layout="wide")

# --- Session bootstrap ---
if "sid" not in st.session_state:
    st.session_state.sid = str(uuid.uuid4())  # unique session id
SID = st.session_state.sid

paths = session_paths(SID)
ensure_session_dirs(SID)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"assistant","content":
         "Hi! Upload PDFs/DOCX/images or a ZIP on the left, then type commands like:\n"
         "- `ingest` (parse + index your uploads)\n"
         "- `rank: Python, SQL, AWS` (shortlist by skills)\n"
         "- `search: kubernetes production`\n"
         "- `analytics`\n"
         "- `clear` (delete files + vectors)\n"}
    ]

# Sidebar (uploads + session controls)
with st.sidebar:
    st.header("📎 Uploads")
    files = st.file_uploader(
        "Upload PDFs/DOCX/Images or a ZIP", type=["pdf","docx","png","jpg","jpeg","zip"],
        accept_multiple_files=True
    )
    if files:
        uploaded = 0
        for f in files:
            data = f.read()
            if f.name.lower().endswith(".zip"):
                for name, b in iter_zip_supported(data):
                    save_file_bytes(SID, name, b)
                    uploaded += 1
            else:
                save_file_bytes(SID, f.name, data)
                uploaded += 1
        st.success(f"Uploaded {uploaded} file(s).")

    st.divider()
    if st.button("🗑️ Clear session now", type="primary"):
        n_files = clear_session_all(SID)
        # Reset chat & new SID for a neat restart
        st.session_state.messages = [{"role":"assistant","content":f"Session cleared ({n_files} files removed). Start fresh!"}]
        st.session_state.sid = str(uuid.uuid4())
        st.rerun()

# Main title
st.markdown(f"## 🤖 RAG-Resume — Local Chat (Session: `{SID[:8]}`)")

# Show a quick status strip
num_files = len(list_supported_files(SID))
st.caption(f"📂 Files ready: **{num_files}** | 🔤 Embedding model: **{embed_model_name()}**")

# Chat history UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Helper: assistant reply
def say(text: str):
    st.session_state.messages.append({"role":"assistant","content":text})
    with st.chat_message("assistant"):
        st.markdown(text)

# Chat input
user_input = st.chat_input("Type a command (e.g., 'ingest', 'rank: Python, SQL', 'search: ...', 'analytics', 'clear')")
if user_input:
    st.session_state.messages.append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    cmd = parse_command(user_input)

    # Lazy-load embedder and store
    embedder = ensure_embedder()
    store = load_or_create_store(SID, embedder)

    if cmd["name"] == "help":
        say("**Commands:**\n"
            "- `ingest` — parse & index uploaded files\n"
            "- `rank: skill1, skill2, ...` — shortlist by requirements\n"
            "- `search: your query` — semantic search across all resumes\n"
            "- `analytics` — quick charts/tables\n"
            "- `clear` — delete all session files and index\n")

    elif cmd["name"] == "clear":
        n_files = clear_session_all(SID)
        say(f"🗑️ Cleared session data ({n_files} files removed). You can upload again.")
        # Reset (keep same SID to keep UI stable)
        store = load_or_create_store(SID, embedder)  # this recreates an empty store

    elif cmd["name"] == "ingest":
        files = list_supported_files(SID)
        if not files:
            say("No files found. Upload PDFs/DOCX/Images or ZIPs first (sidebar).")
        else:
            # Parse → normalize → section → chunks → index
            docs_added, candidates = 0, {}
            for path in files:
                raw = extract_any(path)
                norm = normalize_text(raw)
                facts = quick_facts(norm)
                # Save raw text artifact per file
                with open(paths.artifacts_dir / (os.path.basename(path) + ".txt"), "w", encoding="utf-8") as f:
                    f.write(norm)
                # Build candidate record (one per file for simplicity)
                cand_id = os.path.splitext(os.path.basename(path))[0]
                candidates[cand_id] = {"candidate_id": cand_id, "facts": facts}

                # chunking
                blocks = section_blocks(norm)
                chunks = chunk_blocks(blocks, max_chars=1200)
                # add to store
                n = add_chunks_to_store(store, cand_id, chunks)
                docs_added += n

            # persist candidates.json
            with open(paths.candidates_json, "w", encoding="utf-8") as f:
                json.dump(list(candidates.values()), f, ensure_ascii=False, indent=2)

            say(f"✅ Ingestion complete. Indexed **{docs_added}** chunks from **{len(files)}** files.")

    elif cmd["name"] == "search":
        q = cmd["query"].strip()
        if not q:
            say("Provide a query, e.g., `search: kubernetes production`")
        else:
            topk = search_topk(store, q, k=5, embedder=embedder)
            if not topk:
                say(f"No matches for: `{q}`")
            else:
                lines = [f"### Top results for: `{q}`"]
                for i, hit in enumerate(topk, 1):
                    score = f"{hit['score']:.3f}"
                    cand = hit["metadata"].get("candidate_id","?")
                    sec = hit["metadata"].get("section","?")
                    prev = hit["preview"].replace("\n"," ")
                    lines.append(f"**{i}. [score {score}] — candidate `{cand}` — {sec}**\n\n> {prev[:300]}…")
                say("\n\n".join(lines))

    elif cmd["name"] == "rank":
        # requirements from prompt
        reqs = prompt_to_requirements(cmd["args"])
        # load candidates
        if not os.path.exists(paths.candidates_json):
            say("No candidates found. Run `ingest` after uploading files.")
        else:
            with open(paths.candidates_json, "r", encoding="utf-8") as f:
                candidates = json.load(f)

            rows = rank_all_candidates(candidates, reqs, store, embedder)
            if not rows:
                say("No candidates scored. Have you ingested files?")
            else:
                # Render small markdown table
                header = "| Rank | Candidate | Score | Missing must-haves |\n|---:|---|---:|---|\n"
                body = ""
                for i, r in enumerate(rows, 1):
                    miss = ", ".join(r["missing_must_haves"]) if r["missing_must_haves"] else "-"
                    body += f"| {i} | `{r['candidate_id']}` | {r['score']} | {miss} |\n"
                say("### Shortlist\n" + header + body)

                # Also show an expandable with evidence per candidate
                with st.expander("See evidence per candidate"):
                    for r in rows[:20]:
                        st.markdown(f"**{r['candidate_id']}** — score {r['score']}")
                        for d in r["details"]:
                            evtxt = "\n".join([f"- {e['quote'][:200]}…" for e in d["evidence"][:2]]) or "- (no evidence)"
                            st.markdown(f"- *{d['requirement']}*: {d['score']}/5\n{evtxt}")

    elif cmd["name"] == "analytics":
        if not os.path.exists(paths.candidates_json):
            say("No candidates found. Run `ingest` after uploading files.")
        else:
            with open(paths.candidates_json, "r", encoding="utf-8") as f:
                candidates = json.load(f)
            stats = simple_analytics(candidates)
            say("### Analytics\n"
                f"- Candidates: **{stats['num_candidates']}**\n"
                f"- Emails found: **{stats['emails_found']}**\n"
                f"- Phones found: **{stats['phones_found']}**\n")
            # Simple charts
            if stats["top_domains"]:
                st.bar_chart(stats["top_domains"])

    else:
        # Unknown command → quick help
        say("I didn’t recognize that. Try: `ingest`, `rank: Python, SQL`, `search: …`, `analytics`, or `clear`.")
