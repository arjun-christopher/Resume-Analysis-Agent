# RAG Resume Analysis System

AI-powered resume analysis using RAG with section-based chunking and hybrid search.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Resume analysis system using Retrieval-Augmented Generation for intelligent document parsing and natural language querying.

**Features:**
- Variable-sized chunking respecting section boundaries
- Hybrid search: FAISS semantic + BM25 keyword matching
- Multi-LLM: Google Gemini primary, Ollama fallback
- Entity extraction: names, contacts, skills, experience, education
- Multi-format: PDF and DOCX support

## Installation

```bash
git clone https://github.com/arjun-christopher/RAG-Resume.git
cd RAG-Resume
python setup.py
```

**Manual Setup:**
```bash
pip install -r requirements.txt
cat > .env << EOF
LLM_FALLBACK_ORDER=google,ollama
ENABLE_GOOGLE=true
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_MODEL=gemini-pro
ENABLE_OLLAMA=true
OLLAMA_MODEL=qwen2.5:1.5b
MAX_CHUNK_SIZE=1000
MIN_CHUNK_SIZE=100
SEMANTIC_WEIGHT=0.7
BM25_WEIGHT=0.3
EOF
mkdir -p data/{uploads,index,advanced_rag}
```

## Usage

**Web Interface:**
```bash
streamlit run app/streamlit_app.py
```

**Programmatic:**
```python
from app.rag_engine import create_advanced_rag_engine

rag = create_advanced_rag_engine("data/index")
result = rag.add_documents(["Resume content..."], [{"source": "resume.pdf"}])
response = rag.query("Find Python developers with ML experience")
print(response['answer'])
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| GOOGLE_API_KEY | required | Get from https://makersuite.google.com/app/apikey |
| LLM_FALLBACK_ORDER | google,ollama | LLM priority order |
| MAX_CHUNK_SIZE | 1000 | Maximum chunk size |
| SEMANTIC_WEIGHT | 0.7 | Semantic search weight |
| BM25_WEIGHT | 0.3 | Keyword search weight |

**Ollama Setup:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:1.5b
ollama serve
```

## Architecture

```
Document → Section Detection → Variable Chunking → Embeddings → FAISS Index
Query → Hybrid Search (Semantic + BM25) → LLM → Response
```

**Tech Stack:** Streamlit, FastEmbed, FAISS, BM25Okapi, LangChain, Gemini, Ollama

## Features

**Section-Based Chunking:** Preserves semantic boundaries instead of arbitrary splits

**Hybrid Search:** Combines dense semantic vectors with sparse keyword matching

**Query-Adaptive Weighting:**
- Keyword queries: 30% semantic, 70% BM25
- Semantic queries: 80% semantic, 20% BM25
- Balanced: 70% semantic, 30% BM25

**Entity Extraction:** Multi-strategy name detection, social links, context-aware skills

## API

**AdvancedRAGEngine:**
```python
def __init__(self, storage_path: str = "data/advanced_rag")
def add_documents(documents: List[str], metadata: List[Dict] = None) -> Dict
def query(query: str, k: int = 5) -> Dict
```

**SectionBasedChunker:**
```python
def __init__(self, min_chunk_size: int = 100, max_chunk_size: int = 1000)
def chunk_document(self, document: str) -> List[Dict]
```

**HybridSearchEngine:**
```python
def search(query: str, k: int = 5, semantic_weight: float = 0.7) -> List[Dict]
```

## Performance

| Metric | Value |
|--------|-------|
| Search Speed | <50ms |
| Embedding Speed | 100-500 docs/sec |
| LLM Response | 1-3s |
| Memory Usage | 500MB-2GB |

## Project Structure

```
app/
  ├── rag_engine.py          # Core RAG
  ├── streamlit_app.py       # Web UI
  ├── parser.py              # Document parsing
  └── extractors/            # Entity extraction
data/                        # Runtime (gitignored)
requirements.txt             # Dependencies
setup.py                     # Automated setup
.env                         # Config (gitignored)
```

## License

MIT License - See LICENSE file

## Built With

LangChain, FAISS, FastEmbed, Streamlit, Google Gemini, Ollama

## Repository

https://github.com/arjun-christopher/RAG-Resume
