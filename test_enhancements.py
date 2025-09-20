#!/usr/bin/env python3
"""
Test script for enhanced RAG-Resume functionality
"""

import sys
import os
sys.path.append('app')

def test_name_extraction():
    """Test enhanced name extraction"""
    from parsing import _extract_names_from_resume
    
    test_cases = [
        "John Smith\nSoftware Engineer\nEmail: john@email.com",
        "Name: Jane Doe\nPosition: Data Scientist",
        "Dr. Michael Johnson\nPh.D. Computer Science",
        "Sarah Chen\nLinkedIn: linkedin.com/in/sarah-chen"
    ]
    
    print("=== Testing Name Extraction ===")
    for i, text in enumerate(test_cases):
        names = _extract_names_from_resume(text)
        print(f"Test {i+1}: {repr(text[:30])}...")
        print(f"Extracted: {names}")
        print()

def test_social_links():
    """Test enhanced social links extraction"""
    from parsing import _extract_social_links_enhanced
    
    test_text = """
    Contact Information:
    Email: john.doe@email.com
    LinkedIn: linkedin.com/in/johndoe
    GitHub: github.com/johndoe
    Portfolio: johndoe.dev
    Twitter: @johndoe
    """
    
    print("=== Testing Social Links Extraction ===")
    urls = []
    socials = _extract_social_links_enhanced(test_text, urls)
    print("Extracted social links:")
    for social in socials:
        print(f"  - {social}")
    print()

def test_skills_with_context():
    """Test skills extraction with context"""
    from parsing import _extract_skills_with_context
    
    test_text = """
    Skills:
    - Python, JavaScript, React
    - AWS, Docker, Kubernetes
    
    Experience:
    Software Engineer at TechCorp
    - Developed applications using Python and Django
    - Worked with Docker containers during internship
    
    Education:
    Computer Science Degree
    - Learned Java and C++ programming
    """
    
    print("=== Testing Skills with Context ===")
    skill_analysis = _extract_skills_with_context(test_text)
    print("Skills found:")
    for skill in skill_analysis['skills']:
        contexts = skill_analysis['contexts'].get(skill, [])
        print(f"  - {skill}: {contexts}")
    print()

def test_dynamic_chunking():
    """Test dynamic chunk sizing"""
    from parsing import _adaptive_params
    
    print("=== Testing Dynamic Chunking ===")
    test_cases = [
        (50 * 1024, 1),      # 50KB, 1 file
        (500 * 1024, 3),     # 500KB, 3 files  
        (5 * 1024 * 1024, 5), # 5MB, 5 files
        (50 * 1024 * 1024, 10) # 50MB, 10 files
    ]
    
    for size, count in test_cases:
        params = _adaptive_params(size, count)
        print(f"Size: {size//1024}KB, Files: {count} -> Chunk: {params['chunk']}, Overlap: {params['overlap']}")
    print()

def test_lightrag_integration():
    """Test LightRAG integration"""
    print("=== Testing LightRAG Integration ===")
    try:
        from lightrag_integration import HybridRAGConfig, create_hybrid_rag_system
        
        config = HybridRAGConfig(
            enable_lightrag=False,  # Disable for testing without API keys
            enable_fast_rag=True,
            query_mode="fast_rag_only"
        )
        
        system = create_hybrid_rag_system("/tmp/test_rag", **config.__dict__)
        print("✓ Hybrid RAG system created successfully")
        
        stats = system.get_system_stats()
        print(f"✓ System stats: {stats}")
        
    except Exception as e:
        print(f"✗ LightRAG integration test failed: {e}")
    print()

if __name__ == "__main__":
    print("Running Enhanced RAG-Resume Tests...\n")
    
    test_name_extraction()
    test_social_links()
    test_skills_with_context()
    test_dynamic_chunking()
    test_lightrag_integration()
    
    print("All tests completed!")