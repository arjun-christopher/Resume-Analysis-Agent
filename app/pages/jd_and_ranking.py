import streamlit as st
from raglib.firebase_client import FB, COL_JD, COL_MATCH, COL_CHUNK, COL_CANDIDATE
from raglib.scoring import score_candidate


st.set_page_config(page_title="JD & Ranking", page_icon="🧮", layout="wide")
st.title("🧮 JD & Ranking")


with st.form("jd_form"):
    title = st.text_input("Job Title", value="Data Engineer")
    location = st.text_input("Location (optional)")
    st.subheader("Requirements")
    req_cols = st.columns(4)
    with req_cols[0]:
        r1 = st.text_input("Req 1 (skill)", value="Python")
    with req_cols[1]:
        r2 = st.text_input("Req 2 (skill)", value="SQL")
    with req_cols[2]:
        r3 = st.text_input("Req 3 (skill)", value="AWS")
    with req_cols[3]:
        r4 = st.text_input("Req 4 (skill)", value="Spark")
    musth = st.multiselect("Mark must-haves", options=[r1,r2,r3,r4], default=[r1,r2])
    submitted = st.form_submit_button("Save JD & Rank")


if submitted:
    reqs=[]
    for n in [r1,r2,r3,r4]:
        if n:
            reqs.append({"name":n, "aliases":[], "weight":1.0, "must_have": n in musth})
    jd_id = FB.create(COL_JD, {"title":title, "location":location, "requirements":reqs, "weights":{"skills":0.6,"experience":0.25,"domain":0.1,"location":0.05}})


    # collect candidate ids from chunks
    cids=set()
    for snap in FB.db().collection(COL_CHUNK).stream():
        d=snap.to_dict(); cid=d.get("candidate_id");
        if cid:
            cids.add(cid)


    rows=[]
    for cid in cids:
        jd = FB.get(COL_JD, jd_id)
        s = score_candidate(cid, jd)
        row = {"candidate_id":cid, "score":s["score"], "missing":", ".join(s["missing_must_haves"]) }
        FB.create(COL_MATCH, {"jd_id":jd_id, "candidate_id":cid, **s})
        rows.append(row)
    rows = sorted(rows, key=lambda x: x['score'], reverse=True)


    st.success(f"JD saved: {jd_id}. Ranked {len(rows)} candidates.")
    st.dataframe(rows, use_container_width=True)