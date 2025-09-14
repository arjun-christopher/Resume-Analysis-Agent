import streamlit as st
from raglib.firebase_client import FB, COL_CANDIDATE, COL_MATCH


st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")
st.title("📊 Analytics")


# Score distribution
scores=[]
for snap in FB.db().collection(COL_MATCH).stream():
    d=snap.to_dict()
    scores.append(d.get('score',0))


if scores:
    st.subheader("Score Distribution")
    st.bar_chart(scores)


    st.subheader("Top Candidates (latest JD runs)")
    top = sorted([d.to_dict() for d in FB.db().collection(COL_MATCH).stream()], key=lambda x: x.get('score',0), reverse=True)[:50]
    st.dataframe([{ 'candidate_id': t['candidate_id'], 'score': t['score'], 'missing': ", ".join(t.get('missing_must_haves',[])) } for t in top], use_container_width=True)
else:
    st.info("Run a JD ranking first.")