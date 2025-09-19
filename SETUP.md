# RAG-Resume - Complete Setup Guide

## Prerequisites

Before beginning the installation process, ensure your system meets the minimum requirements. Python 3.8 or higher is required, with Python 3.10+ recommended for optimal performance. At least 4GB of RAM is necessary, though 8GB is preferred for handling larger resume collections efficiently.

## Installation Process

### Step 1: Core Dependencies Installation

Begin by installing the essential Python packages that power the RAG-Resume system. These dependencies include LangChain for document processing, various embedding models, and the Streamlit web interface.

```bash
pip install -r requirements.txt
```

### Step 2: Natural Language Processing Models

Enhanced text processing capabilities require additional language models. These models significantly improve entity extraction and text analysis accuracy, particularly for resume parsing and skill identification.

```bash
# Install spaCy language models for advanced NLP
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf

# Download NLTK data for text tokenization and preprocessing
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Step 3: Local Language Model Setup (Optional)

For organizations prioritizing data privacy or requiring offline operation, Ollama provides local language model hosting. This setup enables complete offline functionality without external API dependencies.

```bash
# Install Ollama from https://ollama.ai
# After installation, download recommended models:
ollama pull llama2:7b
ollama pull mistral:7b
ollama pull codellama:7b
```

### Step 4: API Configuration (Optional)

Cloud-based language models offer enhanced analytical capabilities. Configure API keys for services like OpenAI, Google Gemini, or Anthropic Claude to access state-of-the-art language processing features.

```bash
# Create environment configuration file
cp .env.template .env
# Edit .env file with your API credentials
```

### Step 5: Application Launch

Once configuration is complete, launch the RAG-Resume application through Streamlit. The system will automatically initialize all configured components and prepare the document processing pipeline.

```bash
streamlit run app/streamlit_app.py
```

## Advanced Configuration Options

### Environment Variables Configuration

The RAG-Resume system supports extensive customization through environment variables. Create a `.env` file in the project root directory to configure these settings according to your organizational requirements.

```bash
# Language Model API Keys (optional but recommended)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GEMINI_API_KEY=your_gemini_key_here

# Embedding Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
INSTRUCTOR_MODEL=hkunlp/instructor-base

# RAG System Parameters
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=10
RERANK_K=5

# Performance Settings
MAX_TOKENS=2000
TEMPERATURE=0.1
```

### Vector Database Selection

The system architecture supports multiple vector database backends, each optimized for different deployment scenarios and performance requirements.

FAISS serves as the default option, providing high-performance similarity search with local storage capabilities. This configuration works well for development environments and smaller deployments where simplicity is prioritized.

Chroma offers persistent storage with enhanced metadata support, making it suitable for production environments requiring data persistence across application restarts. The system automatically handles index creation and maintenance.

Qdrant provides distributed vector storage capabilities, enabling horizontal scaling for large-scale deployments. This option supports cloud deployment and can handle millions of document vectors efficiently.

Weaviate delivers enterprise-grade features including advanced filtering, hybrid search capabilities, and comprehensive API access. This option is recommended for organizations requiring sophisticated vector operations and integration capabilities.

### Language Model Integration

The system implements a comprehensive fallback strategy for language model integration, ensuring reliable operation across various deployment configurations.

Local model deployment through Ollama supports Llama 2 variants (7B, 13B, 70B parameters), Mistral 7B, and CodeLlama models (7B, 13B parameters). These models provide complete offline operation while maintaining competitive performance for resume analysis tasks.

Cloud API integration enables access to state-of-the-art models including OpenAI GPT-4 and GPT-3.5, Anthropic Claude variants, and Google Gemini models. These services offer enhanced reasoning capabilities and more sophisticated natural language understanding.

HuggingFace model integration supports any compatible text-generation model through the transformers library. This option provides flexibility for organizations with specific model requirements or custom fine-tuned models.

## Features Overview

### Core RAG Components

1. **Advanced Text Splitting**
   - Recursive character splitting
   - Semantic sentence splitting
   - Token-aware chunking

2. **Hybrid Retrieval**
   - Dense vector search (FAISS)
   - Sparse keyword search (BM25)
   - Cross-encoder reranking

3. **Multi-Model Embeddings**
   - Sentence Transformers
   - Instructor embeddings
   - FastEmbed for speed

4. **LightRAG Integration**
   - Graph-based retrieval
   - Advanced reasoning
   - Multi-hop queries

### Resume Analysis Features

1. **Entity Extraction**
   - Names, emails, phones
   - Skills and technologies
   - Experience and education
   - Certifications and locations

2. **Intelligent Querying**
   - Natural language questions
   - Ranking and comparison
   - Skill-based filtering
   - Experience analysis

3. **Advanced Analytics**
   - Candidate profiling
   - Skill distribution
   - Experience mapping
   - Corpus statistics

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Install missing packages
   pip install langchain langchain-community
   pip install chromadb faiss-cpu
   ```

2. **Model Download Failures**
   ```bash
   # Manual model installation
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
   ```

3. **Memory Issues**
   ```bash
   # Use smaller models
   export EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   ```

4. **Ollama Connection**
   ```bash
   # Check Ollama service
   ollama list
   ollama serve
   ```

### Performance Optimization

1. **GPU Acceleration**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Faster Embeddings**
   ```bash
   pip install fastembed
   ```

3. **Memory Management**
   - Use smaller chunk sizes for large documents
   - Limit retrieval_k for faster queries
   - Enable model quantization

## System Requirements

### Minimum
- Python 3.8+
- 4GB RAM
- 2GB disk space

### Recommended
- Python 3.10+
- 8GB RAM
- 5GB disk space
- GPU (optional)

### For Large Scale
- 16GB+ RAM
- SSD storage
- GPU with 8GB+ VRAM
- Vector database (Qdrant/Weaviate)

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│  Advanced RAG    │────│  Vector Store   │
│                 │    │     Engine       │    │   (FAISS/etc)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌────────┴────────┐
                       │                 │
                ┌──────▼──────┐   ┌──────▼──────┐
                │  LangChain  │   │  LightRAG   │
                │   Pipeline  │   │  Integration│
                └─────────────┘   └─────────────┘
                       │                 │
                ┌──────▼──────┐   ┌──────▼──────┐
                │ Local LLMs  │   │ Cloud APIs  │
                │  (Ollama)   │   │ (OpenAI/etc)│
                └─────────────┘   └─────────────┘
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.
