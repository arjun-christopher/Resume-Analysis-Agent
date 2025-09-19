# RAG-Resume

**Advanced AI-Powered Resume Analysis System**

RAG-Resume is a sophisticated retrieval-augmented generation system that transforms how organizations analyze resumes. Built with cutting-edge natural language processing frameworks, it provides comprehensive entity extraction and intelligent question answering capabilities using LangChain, LightRAG, and various open-source models.

## Features

### Advanced RAG Architecture

The system employs a multi-layered approach to document retrieval and analysis. LangChain integration provides a complete pipeline with intelligent document splitting, sophisticated retrieval chains, and comprehensive question-answering systems. LightRAG support enables graph-based retrieval for handling complex multi-hop queries that require connecting information across multiple documents.

The hybrid retrieval system combines dense vector similarity search using FAISS with sparse keyword matching through BM25, enhanced by cross-encoder reranking for optimal precision. Multiple vector store options are supported, including FAISS for speed, Chroma for persistence, Qdrant for scalability, and Weaviate for enterprise features.

### Open-Source Model Support

The system implements a robust fallback approach for language model integration. Local language models are supported through Ollama integration, including Llama2, Mistral, and CodeLlama variants. HuggingFace models can be integrated through pipeline support, allowing any text-generation model to be utilized.

Multiple embedding strategies are available, including Sentence Transformers for general-purpose embeddings, Instructor embeddings for domain-specific tasks, and FastEmbed for high-performance scenarios. Cloud API integration is optional but recommended, supporting OpenAI, Anthropic, and Google Gemini services.

### Intelligent Resume Processing

The document processing pipeline performs advanced entity extraction, identifying names, email addresses, phone numbers, technical skills, work experience, educational background, and professional certifications. Multi-format document support handles PDF, DOC, and DOCX files seamlessly.

Smart text chunking employs both recursive character splitting and semantic sentence splitting strategies to maintain context while optimizing for retrieval performance. Comprehensive analytics provide candidate profiling, skill distribution analysis, and experience level mapping across the entire resume corpus.

## Quick Start Guide

### Installation and Setup

Getting started with RAG-Resume requires minimal configuration for basic functionality. Clone the repository and install the core dependencies to begin processing resumes immediately.

```bash
git clone <repository-url>
cd RAG-Resume
pip install -r requirements.txt
```

### Application Launch

Launch the web interface through Streamlit to access the complete resume analysis system. The application will automatically initialize with default settings and prepare for document upload and processing.

```bash
streamlit run app/streamlit_app.py
```

The system provides intelligent defaults for immediate use, while advanced configuration options enable customization for specific organizational requirements. Comprehensive setup instructions, including optional language models and cloud API integration, are available in the detailed [SETUP.md](SETUP.md) guide.

## Usage Examples

### Natural Language Queries

The system supports intuitive natural language queries that allow users to extract specific information from resume collections. Users can ask questions like "Rank candidates by Python and AWS experience" to get a prioritized list based on technical skills, or "Who has the most leadership experience?" to identify candidates with management backgrounds.

Contact information extraction is seamless with queries such as "List all email addresses and LinkedIn profiles," while comparative analysis can be performed using questions like "Compare candidates' machine learning skills" or "Find candidates with 5+ years in data science."

### Advanced Analytics

The analytics dashboard provides comprehensive insights through interactive visualizations. Candidate comparison charts display skill sets against experience levels, helping recruiters identify the best matches for specific roles. Skills distribution analysis reveals the most common technologies and competencies across the candidate pool.

Experience level mapping shows the distribution of candidates across different seniority levels, while educational background summaries provide insights into academic qualifications and institutional diversity within the resume collection.

## System Architecture

### Component Overview

The RAG-Resume architecture implements a modular design that enables flexible deployment and scaling according to organizational needs. The system processes documents through multiple specialized components working in coordination.

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

The frontend interface provides document upload capabilities and query interaction through an intuitive web-based dashboard. The advanced RAG engine coordinates document processing, retrieval operations, and response generation across multiple backend services.

LangChain integration handles document splitting, embedding generation, and retrieval chain orchestration. LightRAG provides graph-based retrieval capabilities for complex multi-document reasoning tasks. The hybrid retriever combines dense vector similarity with sparse keyword matching for optimal precision.

## Technical Architecture

### Core Technologies

The frontend interface is built using Streamlit, providing an intuitive web-based experience for document upload and query interaction. The RAG frameworks leverage both LangChain for comprehensive document processing pipelines and LightRAG for advanced graph-based retrieval capabilities.

Vector storage options include FAISS for high-performance similarity search, Chroma for persistent storage, Qdrant for distributed deployments, and Weaviate for enterprise-scale implementations. Embedding generation utilizes Sentence Transformers for general-purpose tasks, Instructor embeddings for domain-specific applications, and FastEmbed for optimized performance scenarios.

Language model integration supports Ollama for local deployment, HuggingFace Transformers for open-source models, and cloud APIs including OpenAI, Anthropic, and Google Gemini services. Natural language processing capabilities are powered by spaCy and NLTK libraries, while document processing handles multiple formats through pdfplumber and python-docx libraries.

### Configuration Options

The system architecture supports flexible deployment configurations to meet various organizational needs. Local model deployment allows complete offline operation using Ollama-hosted language models, ensuring data privacy and reducing external dependencies.

Cloud API integration provides enhanced analytical capabilities through state-of-the-art language models from OpenAI, Anthropic, and Google. Hybrid configurations combine local processing for sensitive data with cloud services for advanced reasoning tasks, optimizing both performance and security requirements.

Detailed configuration instructions and environment setup procedures are available in the comprehensive [SETUP.md](SETUP.md) guide.

## Contributing

Contributions to RAG-Resume are welcome and encouraged. The development process follows standard open-source practices to ensure code quality and maintainability. Contributors should begin by forking the repository and creating a dedicated feature branch for their proposed changes.

New functionality should include appropriate test coverage to maintain system reliability. Once development is complete, contributors can submit a pull request with a detailed description of the changes and their intended impact on the system.

## License

This project is distributed under the MIT License, providing flexibility for both commercial and non-commercial use. Complete license terms and conditions are available in the LICENSE file included with the project distribution.

---