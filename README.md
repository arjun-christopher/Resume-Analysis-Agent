# RAG-Resume – AI-Powered Resume Filtering & Analytics


An AI-driven resume intelligence platform using Retrieval-Augmented Generation (RAG) for bulk resume filtering, structured analysis, and candidate insights.


**Tech**: FastAPI • Firebase (Firestore + Cloud Storage) • FAISS • Sentence-Transformers • PyMuPDF • python-docx • Tesseract OCR


### Quick Start
1) Create a Firebase project, Service Account JSON, and a Storage bucket.
2) Put `service-account.json` at repo root and set `.env` or docker-compose envs.
3) `docker compose up --build` (or run with Python).
4) Open docs at `http://localhost:8000/docs`.


### Features
- Upload multiple files & ZIPs; auto-unzip; ignore unsupported types.
- Parse PDF/DOCX, OCR images, normalize text, chunk & embed.
- Store metadata & artifacts in Firebase; embeddings in FAISS.
- Rank candidates for a JD with explainable evidence; export results.


> Note: This MVP uses in-memory FAISS. For production, plug Pinecone/pgvector in `vector_store.py`.