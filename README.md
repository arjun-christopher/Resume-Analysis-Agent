# RAG-Resume - AI-Powered Resume Analysis

A privacy-focused resume analysis tool that uses Retrieval-Augmented Generation (RAG) to process and analyze resumes locally. The application provides a clean, efficient interface for managing and querying resume collections using natural language.

## Key Features

- **Local Processing**: All data processing occurs on your machine
- **Document Support**:
  - PDF documents
  - Word documents (DOCX)
  - Images (PNG, JPG, JPEG) with OCR
  - Batch processing via ZIP files
- **Hybrid Search**:
  - FAISS for vector similarity
  - BM25 for keyword search
  - Cross-encoder reranking
- **Document Processing**:
  - Text extraction and normalization
  - Intelligent document chunking
  - Metadata management
- **Query Capabilities**:
  - Natural language search
  - Skill-based filtering
  - Contextual responses

## Tech Stack

- **Frontend**: Streamlit
- **Vector Storage**: FAISS via LangChain
- **Embeddings**: 
  - Default: `sentence-transformers/all-MiniLM-L6-v2`
  - Configurable via environment
- **Document Processing**:
  - PyMuPDF (PDFs)
  - python-docx (Word)
  - Tesseract OCR (images)
- **NLP Components**:
  - LangChain pipelines
  - HuggingFace Transformers
  - Cross-encoder reranking
- **Optional Features**:
  - RAG-Anything integration
  - Multiple LLM providers
  - Local LLM support via Ollama

## System Architecture

The application follows a modular architecture with these key components:

1. **Document Ingestion Layer**:
   - Handles multiple file formats
   - Performs OCR when needed
   - Extracts and normalizes text

2. **Processing Pipeline**:
   - Text cleaning and normalization
   - Intelligent chunking of documents
   - Metadata extraction

3. **Hybrid Retrieval System**:
   - Dense retrieval using FAISS
   - Sparse retrieval using BM25
   - Cross-encoder reranking

4. **Query Interface**:
   - Natural language processing
   - Result fusion and ranking
   - Response generation

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR (required for image processing)
  - **Windows**: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
  - **macOS**: `brew install tesseract`
  - **Ubuntu/Debian**: `sudo apt install tesseract-ocr`
  - **Arch Linux**: `sudo pacman -S tesseract tesseract-data-eng`

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/RAG-Resume.git
   cd RAG-Resume
   ```

2. **Set up a virtual environment**:
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Unix or MacOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Optional**: Configure environment variables (create a `.env` file):
   ```
   # For advanced features
   OPENAI_API_KEY=your_openai_key
   GEMINI_API_KEY=your_gemini_key
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2  # Default
   
   # For local LLM with Ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3
   ```

### Running the Application

Start the Streamlit application:

```bash
streamlit run app/streamlit_app.py
```

The application will be available at [http://localhost:8501](http://localhost:8501)

### Docker Support

You can also run the application using Docker:

```bash
docker-compose up --build
```

This will set up the application with all dependencies in an isolated container.

## Usage Examples

### Uploading Resumes

1. **Single File Upload**:
   - Click "Browse files" in the sidebar
   - Select one or more PDF, DOCX, or image files
   - Files will be automatically processed

2. **Bulk Upload**:
   - Create a ZIP file containing multiple resumes
   - Upload the ZIP file through the interface
   - Files will be extracted and processed automatically

### Processing Documents

- Documents are processed automatically upon upload
- Processing includes:
  - Text extraction
  - Normalization
  - Chunking
  - Indexing
- Progress is shown in the interface

### Searching and Querying

1. **Natural Language Search**:
   - Type your query in the search box
   - Example: "Find candidates with Python and machine learning experience"

2. **Structured Queries**:
   - `search: python machine learning` - Basic search
   - `rank: python, tensorflow, pytorch` - Rank by skills
   - `analytics` - Show document statistics

3. **Advanced Queries**:
   - Use quotation marks for exact phrases
   - Use `AND`, `OR`, `NOT` for boolean queries
   - Filter by document type: `type:pdf`

### Session Management

- **Data Storage**:
  - All data is stored locally in the `./data` directory
  - Vector indexes are persisted between sessions

- **Clearing Data**:
  - Use the "Clear session" button in the sidebar
  - This removes all uploaded files and indexes
  - Starts a fresh session

## Advanced Features

### Document Processing Pipeline

1. **Text Extraction**:
   - PDF parsing with PyMuPDF
   - DOCX processing with python-docx
   - Image OCR with Tesseract
   - Automatic encoding detection

2. **Text Normalization**:
   - Unicode normalization
   - Whitespace cleaning
   - Special character handling
   - Case normalization

3. **Chunking Strategy**:
   - Semantic section detection
   - Context-aware splitting
   - Overlapping chunks for better context
   - Metadata preservation

### Search and Retrieval

1. **Hybrid Search**:
   - Dense retrieval with FAISS
   - Sparse retrieval with BM25
   - Reciprocal Rank Fusion (RRF) for result combination
   - Cross-encoder reranking

2. **Query Understanding**:
   - Query expansion
   - Synonym handling
   - Stopword removal
   - Stemming/Lemmatization

### Advanced Configuration

1. **Environment Variables**:
   ```
   # Embedding model (default: sentence-transformers/all-MiniLM-L6-v2)
   EMBEDDING_MODEL=model_name
   
   # LLM Configuration
   OPENAI_API_KEY=your_key
   GEMINI_API_KEY=your_key
   
   # Local LLM with Ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3
   
   # Advanced
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   ```

2. **Customization**:
   - Override default models
   - Adjust chunking parameters
   - Customize search parameters
   - Extend with custom parsers

## Development

### Project Structure

```
RAG-Resume/
├── app/
│   ├── parsing.py          # Document parsing utilities
│   ├── rag_engine.py       # Core RAG functionality
│   ├── resume_nlp.py       # NLP utilities
│   ├── streamlit_app.py    # Web interface
│   └── utils.py            # Helper functions
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

### Extending Functionality

1. **Adding New Document Types**:
   - Implement a new parser in `parsing.py`
   - Register it in the document processing pipeline

2. **Customizing Search**:
   - Modify retrieval logic in `rag_engine.py`
   - Add new ranking strategies
   - Implement custom filters

3. **UI Customization**:
   - Edit `streamlit_app.py`
   - Add new UI components
   - Customize the layout and styling

## Contact

For questions or feedback, please open an issue on GitHub.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the LLM framework
- [HuggingFace](https://huggingface.co/) for transformer models
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [Streamlit](https://streamlit.io/) for the web interface
  - Job description parsing and requirements matching
  - Candidate ranking based on skills and experience

- **Analytics Dashboard**:
  - Score distribution visualization
  - Top candidate ranking
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

### Environment Variables

- `RAG_DATA_ROOT`: Data directory (default: `./data`)
- `RAG_EMBED_MODEL`: Embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `TESSERACT_CMD`: Tesseract OCR path (if not in PATH)
- `OPENAI_API_KEY`: For OpenAI integration (optional)
- `GEMINI_API_KEY`: For Gemini integration (optional)
- `OLLAMA_BASE_URL`: For local Ollama server (default: `http://localhost:11434`)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Last Modified: September 15, 2025*