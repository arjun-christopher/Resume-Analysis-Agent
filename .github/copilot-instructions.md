# RAG-Resume Copilot Instructions

RAG-Resume is an advanced AI-powered resume analysis system built with Python and Streamlit. It uses cutting-edge retrieval-augmented generation (RAG) techniques with LangChain, LightRAG, and various open-source language models to analyze and query resume collections intelligently.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap and Setup Process
**CRITICAL: All dependency installations may take 60+ minutes due to large ML packages. NEVER CANCEL. Set timeouts to 120+ minutes.**

1. **Prerequisites Verification**:
   ```bash
   python3 --version  # Should be 3.8+ (3.10+ recommended)
   pip --version
   ```

2. **Network Connectivity Test**:
   ```bash
   # Test PyPI connectivity before installation
   pip install --dry-run requests
   ```
   - **If this fails**: Network issues present, expect installation failures
   - **Workaround**: Use offline installation or cached packages if available

3. **Core Dependencies Installation** (Expected failures in restricted networks):
   ```bash
   cd RAG-Resume
   time pip install -r requirements.txt
   ```
   - **Expected Time**: 45-90 minutes for complete installation  
   - **NEVER CANCEL**: ML packages (torch, transformers, etc.) are very large
   - **Known Issue**: `HTTPSConnectionPool(host='pypi.org', port=443): Read timed out`
   - **Primary Fallback**: `pip install --timeout 600 --retries 5 -r requirements.txt`
   - **Secondary Fallback**: `pip install --no-cache-dir --timeout 900 -r requirements.txt`
   - **For Restricted Networks**: Installation may completely fail - document this limitation

4. **Dependency Installation Verification**:
   ```bash
   # Test if core packages installed successfully
   python3 -c "import streamlit; print('Streamlit: OK')" || echo "Streamlit: FAILED"
   python3 -c "import pandas; print('Pandas: OK')" || echo "Pandas: FAILED" 
   python3 -c "import numpy; print('NumPy: OK')" || echo "NumPy: FAILED"
   python3 -c "import torch; print('PyTorch: OK')" || echo "PyTorch: FAILED"
   ```

5. **NLP Model Downloads** (Required for enhanced functionality):
   ```bash
   # spaCy language models - takes 10-15 minutes. NEVER CANCEL.
   python -m spacy download en_core_web_sm
   python -m spacy download en_core_web_trf
   
   # NLTK data downloads - takes 5-10 minutes
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   
   # Sentence transformers model (auto-downloaded on first use)
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
   ```

6. **Optional: Local LLM Setup with Ollama**:
   ```bash
   # Install Ollama from https://ollama.ai first
   # Download recommended models - each takes 15-30 minutes. NEVER CANCEL.
   ollama pull llama2:7b       # ~3.8GB download
   ollama pull mistral:7b      # ~4.1GB download  
   ollama pull codellama:7b    # ~3.8GB download
   ```

### Running the Application

**IMPORTANT**: Application requires ALL dependencies from requirements.txt to function properly.

1. **Pre-flight Check**:
   ```bash
   # Verify core dependencies are installed
   python3 -c "
   try:
       import streamlit
       print('✓ Streamlit available')
   except ImportError:
       print('✗ Streamlit missing - run pip install -r requirements.txt')
       exit(1)
   "
   ```

2. **Start the Streamlit Application**:
   ```bash
   streamlit run app/streamlit_app.py
   ```
   - **Default URL**: http://localhost:8501
   - **Startup Time**: 30-60 seconds for first launch (model loading)
   - **Expected Behavior**: Web interface opens with file upload area
   - **Expected Error if deps missing**: `ModuleNotFoundError: No module named 'streamlit'`

3. **Alternative: Development Mode**:
   ```bash
   streamlit run app/streamlit_app.py --server.runOnSave true
   ```

4. **Testing Without Dependencies** (for development/debugging):
   ```bash
   # Check import structure without running
   python3 -c "
   import ast
   with open('app/streamlit_app.py') as f:
       tree = ast.parse(f.read())
   print('✓ Python syntax is valid')
   "
   
   # Try running main file to see import errors
   python3 app/streamlit_app.py  # Will fail with import errors but shows structure
   ```

### Environment Configuration

1. **API Keys Setup** (Optional but recommended):
   ```bash
   # Create environment file
   cp .env.template .env  # if template exists, otherwise create manually
   
   # Edit .env with your API credentials:
   # OPENAI_API_KEY=your_openai_key_here
   # ANTHROPIC_API_KEY=your_anthropic_key_here  
   # GEMINI_API_KEY=your_gemini_key_here
   ```

2. **System Environment Variables**:
   ```bash
   # For memory optimization
   export EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   
   # For local Ollama usage
   export OLLAMA_HOST=http://localhost:11434
   ```

## Validation and Testing

### Manual Validation Requirements
**CRITICAL**: Always test complete end-to-end scenarios after making changes:

1. **Basic Application Startup**:
   - Run `streamlit run app/streamlit_app.py`
   - Verify web interface loads at http://localhost:8501
   - Check no import errors in terminal output

2. **File Upload and Processing**:
   - Upload a sample PDF resume through the web interface
   - Verify file appears in session files list
   - Confirm processing completes without errors

3. **Query Functionality**:
   - Use one of the example queries (e.g., "List all email addresses")
   - Verify response is generated within 10-30 seconds
   - Check that source documents are displayed

4. **Data Persistence**:
   - Upload multiple files
   - Clear session using "Clear Session" button
   - Verify all data is properly cleared

### Performance Expectations
- **First Query**: 30-60 seconds (model loading + processing)
- **Subsequent Queries**: 5-15 seconds
- **File Upload**: 2-10 seconds per MB
- **Session Clear**: 1-3 seconds

## Common Tasks and Scenarios

### Development Workflow
1. **Making Code Changes**:
   - Edit Python files in `app/` directory
   - Streamlit auto-reloads on file changes
   - Test functionality immediately in browser

2. **Adding New Dependencies**:
   ```bash
   pip install new-package
   pip freeze > requirements.txt  # Update requirements
   ```

3. **Code Quality and Linting**:
   ```bash
   # Python syntax validation (always works)
   python3 -m py_compile app/streamlit_app.py
   python3 -m py_compile app/parsing.py  
   python3 -m py_compile app/advanced_rag_engine.py
   python3 -m py_compile config.py
   
   # Optional: Install and run common linters
   pip install black flake8 mypy
   black --check app/  # Code formatting check
   flake8 app/        # Style and error checking
   mypy app/          # Type checking (may have many warnings)
   ```
   **Note**: No pre-existing linting configuration found in repository

4. **Debugging Issues**:
   - Check terminal output for error messages
   - Use browser developer tools for frontend issues
   - Monitor system memory usage (app can use 2-4GB RAM)

### File Structure Overview
```
RAG-Resume/
├── app/
│   ├── streamlit_app.py      # Main Streamlit application (175 lines)
│   ├── advanced_rag_engine.py # Core RAG system (1,185 lines)
│   ├── parsing.py           # Resume parsing logic (461 lines)
│   └── utils.py             # Utility functions (29 lines)
├── config.py                # Configuration settings (79 lines)
├── requirements.txt         # 100+ Python dependencies
├── README.md               # Project documentation
├── SETUP.md               # Detailed setup guide
└── data/                  # Created automatically for uploads/indexes
```

### Key Application Components
- **Frontend**: Streamlit web interface with chat-based interaction
- **Backend**: Advanced RAG pipeline with multiple vector stores (FAISS, Chroma, Qdrant)
- **LLM Integration**: Supports Ollama (local), OpenAI, Anthropic, Google APIs
- **Document Processing**: PDF/DOC/DOCX parsing with entity extraction
- **Vector Storage**: Automatic indexing and retrieval of resume content

### Critical Dependencies Analysis
**Core Imports per File**:
- `app/streamlit_app.py`: streamlit, pandas, dotenv, custom modules (utils, parsing, advanced_rag_engine)
- `app/advanced_rag_engine.py`: torch, transformers, sentence_transformers, langchain, faiss, chromadb
- `app/parsing.py`: spacy, nltk, textstat, transformers (NER models)
- `config.py`: Standard library only (os, pathlib)

**Dependency Chain Failures**:
- Without streamlit: `ModuleNotFoundError: No module named 'streamlit'`
- Without pandas: Data processing features fail
- Without torch/transformers: ML/AI functionality completely broken
- Without spacy models: Enhanced NLP features unavailable

## Troubleshooting Common Issues

### Installation Problems
1. **PyPI Timeout Errors** (MOST COMMON):
   ```bash
   # Primary solutions:
   pip install --timeout 600 --retries 5 -r requirements.txt
   pip install --no-cache-dir --timeout 900 -r requirements.txt
   
   # For persistent network issues:
   pip install --find-links https://download.pytorch.org/whl/torch_stable.html torch
   pip install --index-url https://test.pypi.org/simple/ package-name  # Alternative index
   ```
   **Expected Error**: `HTTPSConnectionPool(host='pypi.org', port=443): Read timed out`
   **Solution**: This is a network connectivity issue - document as known limitation

2. **Memory Issues During Installation**:
   ```bash
   pip install --no-cache-dir -r requirements.txt
   ```

3. **Missing System Dependencies**:
   ```bash
   # On Ubuntu/Debian:
   sudo apt-get update
   sudo apt-get install python3-dev build-essential
   ```

### Runtime Issues
1. **Import Errors**:
   ```bash
   # Install missing packages
   pip install langchain langchain-community
   pip install chromadb faiss-cpu
   ```

2. **Model Download Failures**:
   ```bash
   # Manual model installation
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
   ```

3. **Ollama Connection Issues**:
   ```bash
   # Check Ollama service
   ollama list
   ollama serve  # Start if not running
   ```

4. **Memory Issues**:
   ```bash
   # Use smaller models
   export EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   ```

### Application-Specific Issues
1. **Streamlit Port Conflicts**:
   ```bash
   streamlit run app/streamlit_app.py --server.port 8502
   ```

2. **File Upload Failures**:
   - Check file format (only PDF, DOC, DOCX supported)
   - Verify file size < 100MB
   - Ensure sufficient disk space in `data/` directory

3. **Slow Query Performance**:
   - First query always slower (model initialization)
   - Reduce `RETRIEVAL_K` parameter in config
   - Use local models instead of API calls for faster response

## Critical Warnings and Timeouts

**NEVER CANCEL these long-running operations:**
- Initial `pip install -r requirements.txt`: 45-90 minutes
- spaCy model downloads: 10-15 minutes per model
- Ollama model downloads: 15-30 minutes per model (3-4GB each)
- First application startup: 30-60 seconds
- First query processing: 30-60 seconds

**Always set appropriate timeouts:**
- Build commands: 120+ minutes
- Model downloads: 45+ minutes  
- Application startup: 120 seconds
- Query processing: 60 seconds

**Memory Requirements:**
- Minimum: 4GB RAM
- Recommended: 8GB RAM
- With large models: 16GB+ RAM
- Disk space: 5GB+ for models and dependencies

## Common Command Outputs

### Repository Structure After Setup
```bash
ls -la
# Expected output:
total 68
drwxr-xr-x 6 runner runner 4096 Sep 20 12:47 .
drwxr-xr-x 3 runner runner 4096 Sep 20 12:40 ..
drwxrwxr-x 7 runner runner 4096 Sep 20 12:46 .git
drwxrwxr-x 2 runner runner 4096 Sep 20 12:48 .github
-rw-rw-r-- 1 runner runner 4835 Sep 20 12:41 .gitignore
-rw-rw-r-- 1 runner runner 1074 Sep 20 12:41 LICENSE
-rw-rw-r-- 1 runner runner 9576 Sep 20 12:41 README.md
-rw-rw-r-- 1 runner runner 8906 Sep 20 12:41 SETUP.md
drwxrwxr-x 2 runner runner 4096 Sep 20 12:41 app
-rw-rw-r-- 1 runner runner 1967 Sep 20 12:41 config.py
-rw-rw-r-- 1 runner runner 1372 Sep 20 12:41 requirements.txt
```

### Code Quality Check Output
```bash
python3 -m py_compile app/*.py config.py
# Expected: No output = success
# Failure example: SyntaxError: invalid syntax (file.py, line 10)
```

### Dependency Check Output  
```bash
python3 -c "import streamlit; print('✓ Streamlit available')"
# Success: ✓ Streamlit available
# Failure: ModuleNotFoundError: No module named 'streamlit'
```

### Application Startup Expected Output
```bash
streamlit run app/streamlit_app.py
# Expected output (when dependencies available):
#
#   You can now view your Streamlit app in your browser.
#   
#   Local URL: http://localhost:8501
#   Network URL: http://192.168.x.x:8501
#
# Error output (when dependencies missing):
# ModuleNotFoundError: No module named 'streamlit'
```