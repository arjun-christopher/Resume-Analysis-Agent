# RAG Resume Analysis System

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap and Dependencies
Install core Python dependencies in the following order:
```bash
# Basic framework and data processing (takes ~2 minutes)
python3 -m pip install streamlit pandas numpy tqdm python-dotenv --break-system-packages

# Document processing libraries (takes ~3 minutes)  
python3 -m pip install pdfplumber python-docx dateparser phonenumbers pypdf --break-system-packages

# Machine learning stack (takes ~15 minutes) - NEVER CANCEL. Set timeout to 20+ minutes.
python3 -m pip install faiss-cpu sentence-transformers rank-bm25 torch transformers --break-system-packages

# NLP and text processing (takes ~5 minutes)
python3 -m pip install spacy textstat --break-system-packages

# Development tools
python3 -m pip install black flake8 --break-system-packages
```

**CRITICAL TIMING:** Full dependency installation takes 20-25 minutes total. NEVER CANCEL builds or long-running pip installs. Use timeout of 30+ minutes for pip commands.

### Build and Run the Application
```bash
# Navigate to repository root
cd /home/runner/work/RAG-Resume/RAG-Resume

# Run the Streamlit application (takes ~6 seconds to initialize)
PYTHONPATH=app streamlit run app/streamlit_app.py --server.headless=true --server.port=8502 --browser.gatherUsageStats=false

# Application will be available at http://localhost:8502
```

**Application Startup Time:** ~6 seconds for import and RAG system initialization. NEVER CANCEL during startup.

### Testing and Validation
Always run these validation steps after making changes:

```bash
# Test basic import (should complete in ~6 seconds)
python3 -c "import sys; sys.path.append('app'); import streamlit_app; print('SUCCESS: App imports correctly')"

# Test RAG system initialization (should complete in ~3 seconds)
python3 -c "
import sys
sys.path.append('app')
from fast_semantic_rag import create_fast_semantic_rag
rag = create_fast_semantic_rag('data/index')
print('SUCCESS: RAG system initializes correctly')
"

# Format code (takes ~10 seconds)
python3 -m black app/ config.py

# Lint code (takes ~5 seconds)  
python3 -m flake8 app/ --max-line-length=100 --ignore=E203,W503
```

### Validation Scenarios
After making changes, ALWAYS test these complete user workflows:

1. **Document Upload Workflow:**
   - Start the Streamlit application: `PYTHONPATH=app streamlit run app/streamlit_app.py --server.headless=true --server.port=8502`
   - Create test resume files in `/tmp/test_resumes/`
   - Upload PDF or DOCX files via the sidebar
   - Verify files are processed and indexed automatically
   - Check that file count and size are displayed correctly

2. **Query Testing Workflow:**
   - After uploading documents, test these queries:
     - "Who has Python experience?"
     - "List all candidates with machine learning skills"
     - "Perform EDA analysis"
     - "Show all email addresses and LinkedIn profiles"
   - Verify responses are generated within 5 seconds
   - Check that chat history persists in the interface

3. **System Reset Workflow:**
   - Click "Clear Session" button
   - Verify all uploaded files are removed
   - Verify chat history is cleared
   - Verify system can accept new uploads immediately

### Internet Connectivity Fallbacks
The system gracefully handles limited internet access:
- **No internet for models:** Falls back to TF-IDF embeddings (logged as warnings, but functional)
- **No Hugging Face access:** Uses built-in scikit-learn TF-IDF vectorizer
- **No LLM access:** Provides pattern-based responses only
- These fallbacks are expected and do not indicate failure

## Key Components and Architecture

### Core Files
- `app/streamlit_app.py` - Main Streamlit web application
- `app/fast_semantic_rag.py` - Core RAG engine with semantic search
- `app/parsing.py` - Document parsing and NLP processing
- `app/utils.py` - File handling and utility functions
- `config.py` - System configuration and model settings
- `requirements.txt` - Python dependencies

### Data Flow
1. Documents uploaded via Streamlit sidebar → `app/streamlit_app.py`
2. Files saved to `data/uploads/` → processed by `app/parsing.py`
3. Text chunks and metadata → indexed by `app/fast_semantic_rag.py`
4. Vector embeddings stored in `data/index/` using FAISS
5. User queries → semantic search → LLM response generation

### Dependencies and Fallback Behavior
The system gracefully degrades when optional dependencies are missing:
- **No internet access:** Falls back to TF-IDF embeddings instead of pre-trained models
- **Missing FAISS:** Falls back to linear search (slower but functional)
- **Missing LLM:** Provides pattern-based responses only
- **Missing spaCy models:** Basic regex-based pattern extraction

### Performance Expectations
- **Dependency Installation:** 20-25 minutes total
- **Application Startup:** ~6 seconds (measured: 5.8s for app import)
- **RAG System Initialization:** ~5 seconds (measured: 4.9s)
- **Document Processing:** ~2-5 seconds per resume
- **Query Response:** 1-5 seconds per query
- **Code Formatting:** ~1 second for entire codebase (measured: 0.17s)
- **Linting:** ~0.3 seconds for entire codebase (measured: 0.26s)

## Common Tasks

### Repository Structure
```
RAG-Resume/
├── .github/
│   └── copilot-instructions.md
├── app/
│   ├── streamlit_app.py      # Main web application
│   ├── fast_semantic_rag.py  # RAG engine
│   ├── parsing.py           # Document processing
│   └── utils.py             # Utility functions
├── data/                    # Created at runtime
│   ├── uploads/            # Uploaded resume files
│   └── index/              # Vector index storage
├── config.py               # System configuration
├── requirements.txt        # Python dependencies
└── README.md              # System documentation
```

### Environment Setup
```bash
# Python version
python3 --version  # Should be 3.8+

# Set Python path for imports
export PYTHONPATH=app

# Create required directories (done automatically by app)
mkdir -p data/uploads data/index
```

### Development Workflow
1. **Make Code Changes:** Edit files in `app/` directory
2. **Format Code:** `python3 -m black app/ config.py`
3. **Lint Code:** `python3 -m flake8 app/ --max-line-length=100 --ignore=E203,W503`
4. **Test Import:** Test basic import as shown in validation section
5. **Test Functionality:** Run complete validation scenarios
6. **Run Application:** Start Streamlit and test in browser

### Key Configuration Files
- **No pytest, unittest, or formal test framework** - use validation scripts instead
- **No CI/CD workflows** - manual testing required
- **No Docker or containerization** - direct Python execution
- **No package management** beyond pip requirements.txt

### Common Error Patterns
- **Import errors:** Check `PYTHONPATH=app` is set
- **Module not found:** Install missing dependencies from requirements.txt
- **Slow performance:** Expected due to ML model initialization
- **Connection errors:** System falls back gracefully, continue with offline mode
- **Memory issues:** Reduce document batch sizes, clear `data/` directories

### Troubleshooting
1. **Application won't start:**
   ```bash
   # Check if all dependencies are installed
   python3 -c "import streamlit, pandas, numpy; print('Core deps OK')"
   
   # Check app import
   PYTHONPATH=app python3 -c "import streamlit_app; print('App import OK')"
   ```

2. **RAG system initialization fails:**
   ```bash
   # Test RAG system separately
   PYTHONPATH=app python3 -c "from fast_semantic_rag import create_fast_semantic_rag; create_fast_semantic_rag('data/index')"
   ```

3. **Document processing errors:**
   ```bash
   # Test document parsing
   PYTHONPATH=app python3 -c "from parsing import extract_docs_to_chunks_and_records; print('Parsing module OK')"
   ```

### Working with Large Files
- **Resume files:** Supports PDF, DOC, DOCX up to 50MB each
- **Batch processing:** System handles multiple file uploads automatically
- **Memory management:** Vector index grows incrementally, monitor `data/index/` size

## Critical Reminders
- **NEVER CANCEL** pip install commands - they may take 15+ minutes
- **NEVER CANCEL** during application startup - initialization takes ~6 seconds
- **ALWAYS** set PYTHONPATH=app before running application code
- **ALWAYS** test complete user workflows after changes
- **ALWAYS** format and lint code before committing changes
- **NEVER** delete `data/` directory contents while application is running