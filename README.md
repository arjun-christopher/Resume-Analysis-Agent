# RAG-Resume – AI-Powered Resume Analysis

A lightweight, resume analysis tool that uses RAG (Retrieval-Augmented Generation) to help you process and analyze resumes without external services.

**Tech Stack**:
- **Frontend**: Streamlit
- **Vector Store**: FAISS (via LangChain)
- **Embeddings**: Sentence-Transformers
- **Document Processing**: 
  - PyMuPDF (PDFs)
  - python-docx (Word docs)
  - Tesseract OCR (images)
- **NLP**: LangChain, HuggingFace Embeddings

## Features

- **Local-First**: No external services required (except for optional Tesseract OCR)
- **Multi-Format Support**: Process PDFs, Word docs, and images
- **Interactive Chat Interface**: Natural language commands for all operations
- **Smart Chunking**: Intelligent section-based text chunking
- **Semantic Search**: Find relevant candidates using natural language
- **Lightweight**: Runs on CPU with minimal setup

## Quick Start

1. **Prerequisites**:
   - Python 3.8+
   - Tesseract OCR (for image processing)
     - On Windows: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
     - On macOS: `brew install tesseract`
     - On Ubuntu/Debian: `sudo apt install tesseract-ocr`

2. **Setup**:
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd RAG-Resume
   
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   streamlit run app/streamlit_app.py
   ```
   The application will be available at `http://localhost:8501`

## Usage

1. **Upload Resumes**:
   - Use the sidebar to upload PDFs, Word docs, or images
   - Supports bulk upload via ZIP files

2. **Process Resumes**:
   - Type `ingest` to process all uploaded files
   - The system will extract text, normalize it, and create searchable chunks

3. **Search and Rank**:
   - `search: your query` - Find relevant resume sections
   - `rank: skill1, skill2` - Rank candidates by skills
   - `analytics` - View statistics about processed resumes

4. **Session Management**:
   - All data is stored locally in the `./data` directory
   - Use the "Clear session" button to start fresh

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
## Project Structure

```
app/
  streamlit_app.py          # Main Streamlit application
  raglib/
    config.py              # Application configuration and paths
    commands.py            # CLI command parsing
    parsing.py             # Document parsing (PDF, DOCX, images)
    chunking.py            # Text chunking and section detection
    embeddings.py          # Text embedding utilities
    vector_store.py        # FAISS vector store implementation (via LangChain)
    scoring.py             # Candidate ranking and scoring
    storage_local.py       # Local file system operations
    utils_zip.py           # ZIP file handling
    analytics.py           # Data analysis and visualization
```

## Development

1. **Environment Setup**:
   ```bash
   # Install development dependencies
   pip install -r requirements-dev.txt
   
   # Install pre-commit hooks
   pre-commit install
   ```

2. **Running Tests**:
   ```bash
   # Run unit tests
   pytest tests/
   
   # Run with coverage
   pytest --cov=raglib tests/
   ```

3. **Code Style**:
   ```bash
   # Auto-format code
   black .
   
   # Check code style
   flake8 .
   ```

## Configuration

The application can be configured using environment variables:

- `RAG_DATA_ROOT`: Root directory for storing application data (default: `./data`)
- `RAG_EMBED_MODEL`: HuggingFace model for embeddings (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `TESSERACT_CMD`: Path to Tesseract OCR executable (if not in PATH)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Last Modified: September 14, 2025*