# RAG-Resume - Advanced Setup Guide

## Prerequisites

Before beginning the installation process, ensure your system meets the requirements for this next-generation RAG system. Python 3.9 or higher is required, with Python 3.11+ strongly recommended for optimal performance with the latest AI models. At least 8GB of RAM is necessary for basic operation, though 16GB+ is recommended for enterprise deployments with advanced models.

## Installation Process

### Step 1: Core Dependencies Installation

Install the comprehensive set of dependencies for the advanced RAG system, including cutting-edge embedding models, multiple vector databases, and the latest AI frameworks.

```bash
# Create virtual environment (recommended)
python -m venv rag-resume-env
source rag-resume-env/bin/activate  # On Windows: rag-resume-env\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# For GPU acceleration (optional but recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Advanced NLP Models

Download state-of-the-art language models for enhanced text processing, entity extraction, and semantic understanding.

```bash
# Install spaCy transformer models for advanced NLP
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_lg

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"

# Pre-download embedding models (optional - will download automatically when first used)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/e5-large-v2')"
```

### Step 3: Advanced Local Language Models (Recommended)

Install Ollama for local hosting of cutting-edge open-source models. This provides enhanced privacy, reduced costs, and eliminates API dependencies.

```bash
# Install Ollama from https://ollama.ai
# Download latest advanced models:

# Primary models (choose based on your hardware)
ollama pull qwen2.5:32b          # Best overall performance (requires 32GB+ RAM)
ollama pull llama3.1:70b         # Meta's flagship model (requires 64GB+ RAM)
ollama pull llama3.2:90b         # Latest Llama model

# Efficient models for lower-end hardware
ollama pull qwen2.5:14b          # Good balance of performance and efficiency
ollama pull phi3.5:latest        # Microsoft's efficient model
ollama pull mistral-nemo:latest  # Latest Mistral architecture

# Specialized models
ollama pull deepseek-coder:33b   # Best for code-related queries
ollama pull gemma2:27b           # Google's improved model
```

### Step 4: Advanced API Configuration

Configure cloud APIs for enhanced capabilities with the latest AI models. The system supports multiple providers with intelligent fallback.

```bash
# Create environment configuration file
cp .env.template .env

# Edit .env file with your API credentials:
# - OpenAI API key for GPT-4/GPT-3.5-turbo
# - Anthropic API key for Claude models
# - Google API key for Gemini Pro/Ultra
# - Groq API key for ultra-fast inference
# - HuggingFace API key for hosted models
```

**API Key Setup Guide:**

1. **OpenAI** (https://platform.openai.com/api-keys)
   - Sign up and create API key
   - Add to .env: `OPENAI_API_KEY=sk-...`

2. **Anthropic** (https://console.anthropic.com/)
   - Create account and generate API key
   - Add to .env: `ANTHROPIC_API_KEY=sk-ant-...`

3. **Google Gemini** (https://makersuite.google.com/app/apikey)
   - Create API key in Google AI Studio
   - Add to .env: `GEMINI_API_KEY=AIza...`

4. **Groq** (https://console.groq.com/keys)
   - Sign up for ultra-fast inference
   - Add to .env: `GROQ_API_KEY=gsk_...`

### Step 5: Vector Database Setup (Optional)

For production deployments, configure advanced vector databases for better performance and scalability.

**Qdrant (Recommended for Production):**
```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant

# Or install locally
pip install qdrant-client
# Add to .env: QDRANT_URL=http://localhost:6333
```

**Weaviate (Enterprise Features):**
```bash
# Using Docker
docker run -p 8080:8080 semitechnologies/weaviate:latest

# Add to .env: WEAVIATE_URL=http://localhost:8080
```

**Neo4j (For Graph RAG):**
```bash
# Using Docker
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest

# Add to .env:
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=password
```

### Step 6: Application Launch

Launch the advanced RAG-Resume application with all configured components.

```bash
# Basic launch
streamlit run app/streamlit_app.py

# With custom configuration
streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

# For development with auto-reload
streamlit run app/streamlit_app.py --server.runOnSave true
```

## 🔧 Advanced Configuration Options

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

### Vector Database Comparison

| Database | Use Case | Performance | Features | Scalability |
|----------|----------|-------------|----------|--------------|
| **Qdrant** | Production | Excellent | Filtering, Clustering | Horizontal |
| **Weaviate** | Enterprise | Very Good | GraphQL, ML Models | Horizontal |
| **Pinecone** | Cloud-first | Excellent | Managed Service | Auto-scaling |
| **Chroma** | Development | Good | Persistent Storage | Vertical |
| **FAISS** | Local/Fast | Excellent | In-memory | Single-node |

**Qdrant Configuration (Recommended):**
```bash
# Local deployment
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# Environment variables
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=resume_documents
QDRANT_API_KEY=your_api_key_here  # For cloud deployment
```

**Weaviate Configuration:**
```bash
# Local deployment with modules
docker run -p 8080:8080 -e ENABLE_MODULES=text2vec-transformers semitechnologies/weaviate:latest

# Environment variables
WEAVIATE_URL=http://localhost:8080
WEAVIATE_CLASS_NAME=ResumeDocument
WEAVIATE_API_KEY=your_api_key_here  # For cloud deployment
```

### Advanced Language Model Configuration

**Model Priority and Fallback Strategy:**

The system automatically selects the best available model using this priority order:

1. **OpenAI GPT-4** (if API key provided) - Best reasoning
2. **Anthropic Claude** (if API key provided) - Best analysis
3. **Google Gemini** (if API key provided) - Good balance
4. **Groq** (if API key provided) - Fastest inference
5. **Local Ollama Models** - Privacy-focused

**Local Model Recommendations by Hardware:**

| RAM | Recommended Model | Performance | Use Case |
|-----|------------------|-------------|----------|
| 8-16GB | `phi3.5:latest` | Good | Development |
| 16-32GB | `qwen2.5:14b` | Very Good | Small teams |
| 32-64GB | `qwen2.5:32b` | Excellent | Production |
| 64GB+ | `llama3.1:70b` | Outstanding | Enterprise |

**Custom Model Configuration:**
```bash
# Default model selection
DEFAULT_LLM_MODEL=qwen2.5:32b

# Fallback models (comma-separated)
FALLBACK_MODELS=llama3.1:70b,mistral-nemo:latest,phi3.5:latest

# Model-specific settings
LLAMA_CONTEXT_LENGTH=4096
QWEN_TEMPERATURE=0.1
MISTRAL_TOP_P=0.9
```

## 🚀 Advanced Features Overview

### Next-Generation RAG Components

**🔍 Advanced Retrieval Strategies:**
- **RAPTOR**: Recursive abstractive processing with hierarchical clustering
- **Graph RAG**: Knowledge graph traversal using Neo4j
- **ColBERT**: Late interaction retrieval for precise matching
- **RAG Fusion**: Multi-query generation and result fusion
- **Adaptive Selection**: AI-powered strategy selection based on query type

**🧠 State-of-the-Art Embeddings:**
- **BGE-M3**: Best multilingual embeddings (8192 context)
- **E5-Mistral-7B**: Instruction-tuned embeddings
- **Nomic Embed**: High-performance text embeddings
- **Arctic Embed**: Snowflake's enterprise-grade embeddings
- **Jina v2**: Long-context embeddings (8K tokens)

**🔄 Self-Improving System:**
- **Feedback Learning**: Continuous improvement from user interactions
- **RAGAS Evaluation**: Automated quality assessment
- **Query Pattern Recognition**: Adaptive responses
- **Self-Correction**: Automatic response validation

### 📄 Advanced Document Processing

**🖼️ Multi-Modal Capabilities:**
- **OCR Integration**: Extract text from images using EasyOCR
- **Vision Models**: BLIP/LLaVA for image understanding
- **Layout Analysis**: Preserve document structure
- **Table Extraction**: Parse complex tabular data

**🎯 Intelligent Entity Extraction:**
- **Advanced NER**: Names, contacts, skills, experience
- **Skill Categorization**: Technical vs. soft skills
- **Experience Parsing**: Work history and duration analysis
- **Education Analysis**: Degrees, institutions, certifications
- **Location Intelligence**: Geographic and mobility preferences

**📊 Advanced Analytics:**
- **Candidate Ranking**: Multi-criteria scoring algorithms
- **Skill Gap Analysis**: Identify missing competencies
- **Market Intelligence**: Salary and trend analysis
- **Diversity Metrics**: Comprehensive demographic insights

## 🔧 Troubleshooting Guide

### Common Issues and Solutions

**1. Import Errors**
```bash
# Install missing LangChain packages
pip install langchain langchain-community langchain-openai langchain-anthropic

# Install vector database clients
pip install qdrant-client weaviate-client pinecone-client chromadb

# Install advanced ML packages
pip install sentence-transformers transformers torch faiss-cpu
```

**2. Model Download Failures**
```bash
# Pre-download embedding models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/e5-large-v2')"

# Download with specific cache directory
export TRANSFORMERS_CACHE=/path/to/cache
python -c "from transformers import AutoModel; AutoModel.from_pretrained('BAAI/bge-m3')"
```

**3. Memory and Performance Issues**
```bash
# Use efficient models for limited RAM
export EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
export DEFAULT_LLM_MODEL=phi3.5:latest

# Enable model quantization
export ENABLE_QUANTIZATION=true
export QUANTIZATION_BITS=8

# Reduce batch sizes
export BATCH_SIZE=16
export CHUNK_SIZE=256
```

**4. Ollama Connection Issues**
```bash
# Check Ollama status
ollama list
ollama ps

# Restart Ollama service
ollama serve

# Test model availability
ollama run qwen2.5:32b "Hello, world!"

# Check logs
ollama logs
```

**5. Vector Database Issues**
```bash
# Qdrant connection test
curl http://localhost:6333/health

# Weaviate connection test
curl http://localhost:8080/v1/meta

# Reset vector database
rm -rf data/index/*
# Restart application to rebuild index
```

**6. API Key Issues**
```bash
# Test OpenAI connection
python -c "import openai; print(openai.Model.list())"

# Test Anthropic connection
python -c "import anthropic; client = anthropic.Anthropic(); print('Connected')"

# Verify environment variables
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

### ⚡ Performance Optimization

**1. GPU Acceleration**
```bash
# CUDA 11.8 (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 (latest)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ROCm for AMD GPUs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**2. Advanced Embedding Optimization**
```bash
# Install FastEmbed for high-performance embeddings
pip install fastembed

# Use optimized models
export FASTEMBED_MODEL=nomic-ai/nomic-embed-text-v1.5
export USE_GPU=true
export EMBEDDING_BATCH_SIZE=64
```

**3. Memory and Processing Optimization**
```bash
# Optimal settings for different hardware configurations

# 8GB RAM configuration
export CHUNK_SIZE=256
export TOP_K=5
export BATCH_SIZE=16
export EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# 16GB RAM configuration
export CHUNK_SIZE=512
export TOP_K=10
export BATCH_SIZE=32
export EMBEDDING_MODEL=BAAI/bge-large-en-v1.5

# 32GB+ RAM configuration
export CHUNK_SIZE=1024
export TOP_K=20
export BATCH_SIZE=64
export EMBEDDING_MODEL=BAAI/bge-m3
```

**4. Vector Database Optimization**
```bash
# Qdrant optimization
export QDRANT_HNSW_EF=128
export QDRANT_HNSW_M=16

# Enable compression
export QDRANT_QUANTIZATION=true

# Weaviate optimization
export WEAVIATE_VECTOR_INDEX_TYPE=hnsw
export WEAVIATE_EF_CONSTRUCTION=128
```

**5. Async Processing**
```bash
# Enable async processing for better performance
export USE_ASYNC=true
export MAX_CONCURRENT_REQUESTS=10
export ASYNC_BATCH_SIZE=32
```

## 💻 System Requirements

### Minimum Configuration
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.9+ (3.11+ recommended)
- **RAM**: 8GB (basic operation)
- **Storage**: 10GB free space
- **CPU**: 4 cores, 2.5GHz+

### Recommended Configuration
- **Python**: 3.11+
- **RAM**: 16GB (optimal performance)
- **Storage**: 20GB SSD
- **CPU**: 8 cores, 3.0GHz+
- **GPU**: NVIDIA RTX 3060+ or equivalent (optional)

### Enterprise/Production
- **RAM**: 32GB+ (large-scale deployments)
- **Storage**: 50GB+ NVMe SSD
- **CPU**: 16+ cores, 3.5GHz+
- **GPU**: NVIDIA RTX 4080+ or A100 (recommended)
- **Network**: High-speed internet for cloud APIs
- **Database**: Dedicated vector database server

### Cloud Deployment Requirements
- **AWS**: m5.2xlarge or larger
- **Google Cloud**: n2-standard-8 or larger
- **Azure**: Standard_D8s_v3 or larger
- **Docker**: 8GB+ container memory limit

## 🏗️ Advanced System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Web Interface (Streamlit)                   │
│  📊 Analytics │ 💬 Chat │ 📁 Files │ ⚙️ Config │ 📈 Monitoring  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                 Advanced RAG Engine                             │
│  🧠 Intent │ 🔄 Strategy │ 📈 Learning │ 🔍 Multi-Modal │ ✅ QA │
└─────────────────────────┬───────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼──────┐ ┌────────▼────────┐ ┌──────▼──────┐
│   Retrieval  │ │   Processing    │ │  Learning   │
│   Strategies │ │    Pipeline     │ │   System    │
│              │ │                 │ │             │
│ • RAPTOR     │ │ • OCR/Vision    │ │ • RAGAS     │
│ • Graph RAG  │ │ • NER/Parsing   │ │ • Feedback  │
│ • ColBERT    │ │ • Chunking      │ │ • Analytics │
│ • RAG Fusion │ │ • Embedding     │ │ • Auto-tune │
└──────────────┘ └─────────────────┘ └─────────────┘
        │                 │                 │
┌───────▼─────────────────▼─────────────────▼─────────────────────┐
│                    Data Layer                                   │
│  🚀 Qdrant │ 🌐 Weaviate │ 📌 Pinecone │ 💾 Chroma │ ⚡ FAISS  │
│  🗄️ Neo4j (Graph) │ 📊 Analytics DB │ 🔄 Cache Layer           │
└─────────────────────────┬─────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                   AI Model Layer                               │
│                                                                 │
│  🏠 Local Models          │  ☁️ Cloud APIs                     │
│  • Qwen2.5 (32B/72B)     │  • OpenAI GPT-4/3.5               │
│  • Llama 3.1/3.2         │  • Anthropic Claude               │
│  • Phi-3.5, Mistral      │  • Google Gemini                  │
│  • DeepSeek, Gemma2      │  • Groq (Fast Inference)          │
└─────────────────────────────────────────────────────────────────┘
```

## 🤝 Contributing

We welcome contributions to make RAG-Resume even better!

### Development Setup
```bash
# Clone your fork
git clone https://github.com/your-username/RAG-Resume.git
cd RAG-Resume

# Create development environment
python -m venv dev-env
source dev-env/bin/activate  # Windows: dev-env\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Install pre-commit hooks
pre-commit install
```

### Contribution Guidelines
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Add** comprehensive tests for new functionality
4. **Ensure** all tests pass (`pytest tests/`)
5. **Update** documentation as needed
6. **Submit** a pull request with detailed description

### Areas for Contribution
- 🔍 New retrieval strategies
- 🧠 Additional embedding models
- 📊 Enhanced analytics features
- 🌐 Multi-language support
- 🔧 Performance optimizations
- 📚 Documentation improvements
- 🧪 Test coverage expansion

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Questions or Issues?** 
- 📖 Check the [documentation](README.md)
- 🐛 Report bugs via [GitHub Issues](https://github.com/your-repo/issues)
- 💬 Join our [Discord community](https://discord.gg/your-invite)
- 📧 Email support: support@rag-resume.com
