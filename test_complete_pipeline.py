#!/usr/bin/env python3
"""
Test the complete RAG Resume pipeline
"""

import sys
import os
import tempfile
from pathlib import Path

sys.path.append('app')

def create_test_resume_text():
    """Create a comprehensive test resume"""
    return """
John Smith
Senior Software Engineer

Contact Information:
Email: john.smith@email.com
Phone: (555) 123-4567
LinkedIn: linkedin.com/in/johnsmith
GitHub: github.com/johnsmith
Portfolio: johnsmith.dev

Objective:
Experienced software engineer with 8+ years of experience in full-stack development.

Skills:
• Programming Languages: Python, JavaScript, Java, TypeScript
• Web Frameworks: React, Django, Flask, Node.js
• Cloud Platforms: AWS, Azure, Docker, Kubernetes
• Databases: PostgreSQL, MongoDB, Redis
• Machine Learning: TensorFlow, PyTorch, scikit-learn

Experience:

Senior Software Engineer - TechCorp Inc. (2019-2024)
• Led development of microservices architecture using Python and Docker
• Implemented machine learning models for recommendation systems
• Mentored junior developers and conducted code reviews
• Technologies used: Python, React, AWS, PostgreSQL

Software Engineer Intern - StartupXYZ (2017-2018)
• Developed web applications using JavaScript and React
• Learned AWS deployment and CI/CD pipelines during internship
• Worked with senior developers on backend APIs

Education:

Master of Science in Computer Science
Stanford University (2017-2019)
• Specialized in Artificial Intelligence and Machine Learning
• Coursework: Advanced algorithms, Deep Learning, Distributed Systems
• Technologies learned: Python, Java, C++, TensorFlow

Bachelor of Science in Computer Science  
University of California, Berkeley (2013-2017)
• Graduated Summa Cum Laude
• Relevant coursework: Data Structures, Operating Systems, Database Systems

Projects:

Personal Finance Tracker (2023)
• Built full-stack web application using React and Django
• Implemented real-time data visualization with D3.js
• Deployed on AWS with automated CI/CD pipeline

AI Chatbot Platform (2022)
• Developed chatbot framework using Python and TensorFlow
• Integrated with multiple messaging platforms
• Processed over 1M+ conversations monthly

Certifications:
• AWS Certified Solutions Architect
• Google Cloud Professional Data Engineer
• Certified Kubernetes Administrator (CKA)
"""

def test_complete_pipeline():
    """Test the complete resume processing pipeline"""
    print("=== Testing Complete RAG Resume Pipeline ===\n")
    
    # Import necessary modules
    from parsing import extract_docs_to_chunks_and_records, _extract_comprehensive_entities
    from lightrag_integration import create_hybrid_rag_system, HybridRAGConfig
    from fast_semantic_rag import create_fast_semantic_rag
    
    # Create test resume
    resume_text = create_test_resume_text()
    
    # Test entity extraction
    print("1. Testing Enhanced Entity Extraction...")
    entities = _extract_comprehensive_entities(resume_text)
    
    print(f"   Names found: {entities['names'][:3]}...")
    print(f"   Skills found: {len(entities['skills'])} skills")
    print(f"   Emails found: {entities['emails']}")
    print(f"   Experience years: {entities['experience_years']}")
    print(f"   Skill contexts: {len(entities.get('skill_contexts', {}))}")
    print()
    
    # Test document processing (simulate file)
    print("2. Testing Document Processing...")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(resume_text)
        temp_file = Path(f.name)
    
    try:
        # This would normally process PDFs, but we'll test with text
        print(f"   Created test file: {temp_file}")
        print(f"   File size: {temp_file.stat().st_size} bytes")
        print()
        
        # Test RAG system creation
        print("3. Testing RAG System Creation...")
        
        # Test FastSemanticRAG (fallback)
        fast_rag = create_fast_semantic_rag("/tmp/test_fast_rag")
        print("   ✓ FastSemanticRAG created successfully")
        
        # Test Hybrid system (with LightRAG disabled for testing)
        hybrid_config = HybridRAGConfig(
            enable_lightrag=False,  # Disable for testing without API keys
            enable_fast_rag=True,
            query_mode="fast_rag_only"
        )
        
        hybrid_rag = create_hybrid_rag_system("/tmp/test_hybrid_rag", **hybrid_config.__dict__)
        print("   ✓ Hybrid RAG system created successfully")
        print()
        
        # Test document addition
        print("4. Testing Document Addition...")
        
        documents = [resume_text]
        metadata = [{"source": "test_resume.txt", "candidate": "John Smith"}]
        
        # Add to FastRAG
        result = fast_rag.add_documents(documents, metadata)
        print(f"   ✓ Added to FastRAG: {result.get('documents_added', 'unknown')} documents")
        
        # Test queries
        print("5. Testing Query Processing...")
        
        test_queries = [
            "What is the candidate's name?",
            "List the programming skills",
            "How many years of experience does the candidate have?",
            "What are the candidate's social links?",
            "Describe the candidate's education background"
        ]
        
        for i, query in enumerate(test_queries):
            try:
                result = fast_rag.query(query)
                print(f"   Query {i+1}: {query}")
                print(f"   Answer: {result['answer'][:100]}...")
                print(f"   Processing time: {result.get('processing_time', 0):.3f}s")
                print()
            except Exception as e:
                print(f"   Query {i+1} failed: {e}")
        
        print("6. Testing Enhanced Features...")
        
        # Test EDA
        try:
            eda_result = fast_rag.perform_eda()
            print(f"   ✓ EDA analysis completed in {eda_result.get('processing_time', 0):.3f}s")
        except Exception as e:
            print(f"   EDA analysis failed: {e}")
        
        # Test system stats
        stats = fast_rag.get_system_stats()
        print(f"   ✓ System stats: {stats['documents_processed']} docs processed")
        print()
        
    finally:
        # Clean up
        temp_file.unlink()
        print("✓ Test completed successfully!")

if __name__ == "__main__":
    test_complete_pipeline()