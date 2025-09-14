import streamlit as st

# Import from raglib modules
from raglib.firebase_client import FB, COL_BATCH, COL_FILEITEM, COL_CANDIDATE, COL_CHUNK
from raglib.parsing import extract_pdf, extract_docx, extract_image, normalize, facts
from raglib.chunking import section_blocks, chunk
from raglib.vector_store import VS
from raglib.storage_paths import file_key, artifact_text_key, artifact_json_key, sha256
from raglib.embeddings import embed
from raglib.utils_io import is_supported, guess_mime, iter_zip_supported

# Initialize Streamlit page
st.set_page_config(page_title="Upload & Process", page_icon="📤", layout="wide")
st.title("📤 Upload & Process")

files = st.file_uploader("Upload PDFs/DOCX/Images or ZIPs", type=["pdf","docx","png","jpg","jpeg","zip"], accept_multiple_files=True)


if st.button("Start Ingestion", disabled=not files):
    batch_id = FB.create(COL_BATCH, {"status":"processing"})
    queued=0
    skipped=0
    progress = st.progress(0)
    for i, f in enumerate(files):
        b = f.read()
        if f.name.lower().endswith('.zip'):
            for name, data in iter_zip_supported(b):
                if not is_supported(name):
                    skipped+=1
                    continue
                file_id = FB.create(COL_FILEITEM, {"batch_id":batch_id,"filename":name,"hash":sha256(data),"status":"queued"})
                FB.bucket().blob(file_key(batch_id, name)).upload_from_string(data, content_type=guess_mime(name))
                # parse
                ext = name.lower().split('.')[-1]
                if ext=='pdf':
                    raw = extract_pdf(data)
                elif ext=='docx':
                    raw = extract_docx(data)
                else:
                    raw = extract_image(data)
                norm = normalize(raw)
                FB.upload_bytes(artifact_text_key(batch_id,file_id), norm.encode('utf-8'), content_type='text/plain')
                FB.upload_bytes(artifact_json_key(batch_id,file_id), str(facts(norm)).encode('utf-8'), content_type='application/json')
                cand_id = FB.create(COL_CANDIDATE, {"batch_id":batch_id,"file_id":file_id,"filename":name,"facts":facts(norm)})
                blocks = section_blocks(norm); chunks = chunk(blocks)
                vecs = embed([c['text'] for c in chunks])
                metas=[]
                for c in chunks:
                    ch_id = FB.create(COL_CHUNK, {"candidate_id":cand_id,"section":c['section'],"text":c['text']})
                    metas.append({"candidate_id":cand_id,"chunk_id":ch_id,"preview":c['text'][:200]})
                VS.add(vecs, metas)
                FB.update(COL_FILEITEM, file_id, {"status":"success","candidate_id":cand_id})
                queued+=1
        else:
            if not is_supported(f.name):
                skipped+=1
                continue
            file_id = FB.create(COL_FILEITEM, {"batch_id":batch_id,"filename":f.name,"hash":sha256(b),"status":"queued"})
            FB.bucket().blob(file_key(batch_id, f.name)).upload_from_string(b, content_type=guess_mime(f.name))
            ext = f.name.lower().split('.')[-1]
            if ext=='pdf':
                raw = extract_pdf(b)
            elif ext=='docx':
                raw = extract_docx(b)
            else:
                raw = extract_image(b)
            norm = normalize(raw)
            FB.upload_bytes(artifact_text_key(batch_id,file_id), norm.encode('utf-8'), content_type='text/plain')
            FB.upload_bytes(artifact_json_key(batch_id,file_id), str(facts(norm)).encode('utf-8'), content_type='application/json')
            cand_id = FB.create(COL_CANDIDATE, {"batch_id":batch_id,"file_id":file_id,"filename":f.name,"facts":facts(norm)})
            blocks = section_blocks(norm); chunks = chunk(blocks)
            vecs = embed([c['text'] for c in chunks])
            metas=[]
            for c in chunks:
                ch_id = FB.create(COL_CHUNK, {"candidate_id":cand_id,"section":c['section'],"text":c['text']})
                metas.append({"candidate_id":cand_id,"chunk_id":ch_id,"preview":c['text'][:200]})
            VS.add(vecs, metas)
            FB.update(COL_FILEITEM, file_id, {"status":"success","candidate_id":cand_id})
            queued+=1
        progress.progress(int(((i+1)/len(files))*100))
    FB.update(COL_BATCH, batch_id, {"status":"done","queued":queued,"skipped":skipped})
    st.success(f"Batch {batch_id} complete. Ingested: {queued}, Skipped: {skipped}")