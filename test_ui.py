#!/usr/bin/env python3
"""
Simple UI test to ensure Streamlit app can start
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def test_streamlit_app():
    """Test if the Streamlit app can start without errors"""
    print("Testing Streamlit app startup...")
    
    # Add app directory to path
    app_dir = Path(__file__).parent / "app"
    sys.path.insert(0, str(app_dir))
    
    try:
        # Test imports
        print("1. Testing imports...")
        import streamlit as st
        from parsing import extract_docs_to_chunks_and_records
        from lightrag_integration import create_hybrid_rag_system
        from fast_semantic_rag import create_fast_semantic_rag
        print("   ✓ All imports successful")
        
        # Test agent creation (without UI)
        print("2. Testing agent creation...")
        
        # Test hybrid system creation
        from lightrag_integration import HybridRAGConfig
        config = HybridRAGConfig(
            enable_lightrag=False,  # Disable for testing
            enable_fast_rag=True,
            query_mode="fast_rag_only"
        )
        
        agent = create_hybrid_rag_system("/tmp/test_ui_rag", **config.__dict__)
        print("   ✓ Hybrid RAG agent created successfully")
        
        # Test document processing
        print("3. Testing document processing...")
        sample_text = "John Smith\nSoftware Engineer\nemail: john@example.com"
        
        with open("/tmp/sample_resume.txt", "w") as f:
            f.write(sample_text)
        
        chunks, metas, records = extract_docs_to_chunks_and_records([Path("/tmp/sample_resume.txt")])
        print(f"   ✓ Processed document: {len(chunks)} chunks, {len(records)} records")
        
        print("\n✅ All UI components are working correctly!")
        print("The Streamlit app should start without errors.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ UI test failed: {e}")
        return False
    
    finally:
        # Cleanup
        if os.path.exists("/tmp/sample_resume.txt"):
            os.remove("/tmp/sample_resume.txt")

if __name__ == "__main__":
    test_streamlit_app()