# RAG-Resume – AI-Powered Resume Analysis

A comprehensive, local-first resume analysis tool that leverages Retrieval-Augmented Generation (RAG) to process, analyze, and query resumes without relying on external services. Built with privacy and efficiency in mind, this tool allows you to manage and search through resumes using natural language queries.

## 🚀 Key Features

- **Local-First Architecture**: All processing happens on your machine - no data leaves your system
- **Multi-Format Support**: Process various document types including:
  - PDF documents
  - Word documents (DOCX)
  - Images (PNG, JPG, JPEG) with OCR support
  - Batch processing via ZIP files
- **Hybrid Search Capabilities**:
  - Dense vector search using FAISS
  - Sparse BM25 search for keyword matching
  - Cross-encoder reranking for improved relevance
- **Smart Document Processing**:
  - Automatic text extraction and normalization
  - Intelligent chunking of documents
  - Metadata preservation
- **Flexible Querying**:
  - Natural language search
  - Skill-based ranking
  - Context-aware responses
- **Privacy-Focused**:
  - No external API calls
  - All data remains on your machine
  - Optional Tesseract OCR for local image processing

## 🛠️ Tech Stack

- **Frontend**: Streamlit for interactive web interface
- **Vector Database**: FAISS (via LangChain) for efficient similarity search
- **Embeddings**: 
  - Default: `sentence-transformers/all-MiniLM-L6-v2`
  - Customizable via environment variables
- **Document Processing**:
  - PyMuPDF for PDF text extraction
  - python-docx for Word document parsing
  - Tesseract OCR for image text recognition
- **NLP & ML**:
  - LangChain for document processing pipelines
  - HuggingFace Transformers for embeddings
  - Cross-encoder models for result reranking
- **Optional Advanced Features**:
  - Integration with RAG-Anything for enhanced parsing
  - Support for multiple LLM providers (OpenAI, Gemini, local models)
  - Ollama support for local LLM inference

## 📊 System Architecture

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

## 🚀 Quick Start

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

## 🎯 Usage Guide

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

## 🔍 Advanced Features

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

## 🛠️ Development

### Project Structure

```
RAG-Resume/
├── app/
│   ├── parsing.py          # Document parsing utilities
│   ├── rag_engine.py       # Core RAG functionality
│   ├── resume_nlp.py       # NLP utilities
│   ├── streamlit_app.py    # Web interface
│   └── utils.py            # Helper functions
├── data/                   # Processed data and indexes
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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the LLM framework
- [HuggingFace](https://huggingface.co/) for transformer models
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [Streamlit](https://streamlit.io/) for the web interface
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
*Last Modified: September 15, 2025*