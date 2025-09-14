# RAG-Resume – AI-Powered Resume Filtering & Analytics

An AI-driven resume intelligence platform using Retrieval-Augmented Generation (RAG) for bulk resume filtering, structured analysis, and candidate insights.

**Tech Stack**:
- **Backend**: Python, Streamlit
- **Storage**: Firebase (Firestore + Cloud Storage)
- **Vector Search**: FAISS
- **NLP**: Sentence-Transformers
- **Document Processing**: PyMuPDF, python-docx, Tesseract OCR
- **Deployment**: Docker

## Quick Start

1. **Prerequisites**:
   - Docker and Docker Compose
   - Firebase project with Firestore and Cloud Storage
   - Firebase Service Account JSON file

2. **Setup**:
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd RAG-Resume
   
   # Copy and configure environment variables
   cp .env.example .env
   # Edit .env with your Firebase credentials
   
   # Place your Firebase service account JSON
   cp path/to/your/firebase-credentials.json ./firebase-service-account.json
   ```

3. **Environment Variables**:
   Create a `.env` file with:
   ```
   FIREBASE_PROJECT_ID=your-project-id
   FIREBASE_BUCKET=your-bucket-name
   ```

4. **Run with Docker**:
   ```bash
   docker-compose up --build
   ```

5. **Access the Application**:
   - Open `http://localhost:8501` in your browser
   - Use the sidebar to navigate between different sections

## Features

- **Multi-file Upload**:
  - Supports PDF, DOCX, and image formats (PNG, JPG, JPEG)
  - Automatic ZIP file extraction
  - File type validation and filtering

- **Document Processing**:
  - Text extraction from PDFs and Word documents
  - OCR for images
  - Text normalization and cleaning
  - Smart chunking of documents

- **AI-Powered Analysis**:
  - Vector embeddings using sentence-transformers
  - FAISS for efficient similarity search
  - Structured data extraction

- **Candidate Management**:
  - Store and organize candidate profiles
  - Job description parsing and requirements matching
  - Candidate ranking based on skills and experience

- **Analytics Dashboard**:
  - Score distribution visualization
  - Top candidate ranking
  - Missing skills analysis

## Project Structure

```
app/
  pages/
    upload_and_process.py    # File upload and processing
    jd_and_ranking.py       # Job description and candidate ranking
    analytics.py            # Data visualization and insights
  raglib/
    firebase_client.py      # Firebase operations
    parsing.py             # Document parsing utilities
    chunking.py            # Text chunking logic
    vector_store.py         # FAISS vector store implementation
    embeddings.py          # Text embedding utilities
    scoring.py             # Candidate scoring algorithms
    storage_paths.py       # Storage path utilities
    utils_io.py            # I/O helper functions
    config.py              # Application configuration
```

## Development

1. **Setup Python Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run Locally**:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

> Note: This MVP uses in-memory FAISS. For production, plug Pinecone/pgvector in `vector_store.py`.

---
Last Modified: September 14, 2025