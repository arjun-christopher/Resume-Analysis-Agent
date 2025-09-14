# app/streamlit_app.py - Clean UI without custom colors and minimal emojis
import os
import time
from pathlib import Path
from zipfile import ZipFile, is_zipfile
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
import plotly.express as px
import pandas as pd

from utils import human_size, safe_listdir, init_session_paths, max_upload_guard, clear_dir
from parsing import extract_docs_to_chunks_and_records, file_sha256
from rag_engine import AutoHybridRAG, IndustryClassifier

# Load environment and configure page
load_dotenv()
st.set_page_config(
    page_title="RAG-Resume — Resume Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Remove custom styling for cleaner look

# Initialize directories
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR = DATA_DIR / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Initialize session state
def init_session_state():
    defaults = {
        "manifest": {},  # {filepath: sha256}
        "rag": AutoHybridRAG(str(INDEX_DIR)),
        "history": [],  # [(role, text, metadata)]
        "upload_stats": {
            "total_files": 0,
            "total_size": 0,
            "file_types": {},
            "processing_time": 0
        },
        "analysis_stats": {
            "total_queries": 0,
            "avg_response_time": 0,
            "query_history": []
        }
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()

# Header
st.title("RAG-Resume — Resume Analysis")
st.write("Upload up to 50 resumes in PDF, DOCX, or image formats. Get instant insights across all industries and roles.")

# Sidebar
def render_sidebar():
    with st.sidebar:
        st.markdown("### Session Overview")
        
        # Current session stats
        if hasattr(st.session_state.rag, 'session_stats'):
            stats = st.session_state.rag.session_stats
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats["total_documents"])
                st.metric("Text Chunks", stats["total_chunks"])
            with col2:
                st.metric("Industries", len(stats["industries_detected"]))
                st.metric("Unique Skills", len(stats["unique_skills"]))
            
            # Industry breakdown
            if stats["industries_detected"]:
                st.markdown("**Industries Detected:**")
                for industry in sorted(stats["industries_detected"]):
                    st.markdown(f"• {industry.replace('_', ' ').title()}")
        
        st.markdown("---")
        
        # File management
        st.markdown("### Uploaded Files")
        files = [p for p in safe_listdir(UPLOAD_DIR) 
                if p.suffix.lower() in {".pdf", ".docx", ".png", ".jpg", ".jpeg"}]
        
        if files:
            total_size = sum(p.stat().st_size for p in files)
            st.markdown(f"**{len(files)} files** • **{human_size(total_size)}**")
            
            # Show files with details
            for p in files[:10]:  # Show first 10
                with st.expander(f"{p.name}", expanded=False):
                    st.write(f"**Size:** {human_size(p.stat().st_size)}")
                    
                    # Try to show file stats if available
                    file_path = str(p)
                    if hasattr(st.session_state.rag, 'document_registry'):
                        reg = st.session_state.rag.document_registry
                        if file_path in reg:
                            doc_info = reg[file_path]
                            st.write(f"**Chunks:** {doc_info['chunks']}")
                            st.write(f"**Industry:** {doc_info['industry'].title()}")
                            if doc_info['skills']:
                                skills_preview = list(doc_info['skills'])[:5]
                                st.write(f"**Skills:** {', '.join(skills_preview)}")
            
            if len(files) > 10:
                st.write(f"... and {len(files) - 10} more files")
        else:
            st.info("No files uploaded yet")
        
        st.markdown("---")
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear All", use_container_width=True):
                clear_session()
                st.rerun()
        
        with col2:
            if st.button("Analytics", use_container_width=True):
                st.session_state.show_analytics = not st.session_state.get('show_analytics', False)
                st.rerun()

def clear_session():
    """Clear all session data"""
    # Clear files
    for p in safe_listdir(UPLOAD_DIR):
        p.unlink(missing_ok=True)
    
    clear_dir(str(INDEX_DIR))
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    # Reset session state
    st.session_state.manifest.clear()
    st.session_state.history.clear()
    st.session_state.rag = AutoHybridRAG(str(INDEX_DIR))
    st.session_state.upload_stats = {
        "total_files": 0,
        "total_size": 0,
        "file_types": {},
        "processing_time": 0
    }
    
    st.success("Session cleared successfully!")

def save_and_expand_files(uploaded_files):
    """Save uploaded files and handle ZIP extraction"""
    if not uploaded_files:
        return []
    
    start_time = time.time()
    saved_files = []
    
    # Check total limit before processing
    existing_files = list(safe_listdir(UPLOAD_DIR))
    if len(existing_files) + len(uploaded_files) > 50:
        st.error(f"Maximum 50 files allowed per session. Current: {len(existing_files)}, Trying to add: {len(uploaded_files)}")
        return []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for i, file in enumerate(uploaded_files):
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {file.name}...")
            
            # Save file
            dest_path = UPLOAD_DIR / file.name
            dest_path.write_bytes(file.read())
            saved_files.append(dest_path)
            
            # Update stats
            file_ext = dest_path.suffix.lower()
            st.session_state.upload_stats["file_types"][file_ext] = \
                st.session_state.upload_stats["file_types"].get(file_ext, 0) + 1
            st.session_state.upload_stats["total_size"] += dest_path.stat().st_size
        
        # Handle ZIP files
        zip_files = [f for f in saved_files if f.suffix.lower() == '.zip']
        for zip_file in zip_files:
            if is_zipfile(zip_file):
                status_text.text(f"Extracting {zip_file.name}...")
                try:
                    with ZipFile(zip_file, 'r') as z:
                        z.extractall(UPLOAD_DIR)
                    zip_file.unlink()  # Remove ZIP after extraction
                    status_text.text(f"Extracted {zip_file.name}")
                except Exception as e:
                    st.error(f"Failed to extract {zip_file.name}: {str(e)}")
        
        # Get final list of supported files
        final_files = [p for p in UPLOAD_DIR.iterdir() 
                      if p.is_file() and p.suffix.lower() in {".pdf", ".docx", ".png", ".jpg", ".jpeg"}]
        
        processing_time = time.time() - start_time
        st.session_state.upload_stats["processing_time"] = processing_time
        st.session_state.upload_stats["total_files"] = len(final_files)
        
        progress_bar.progress(1.0)
        status_text.text("Upload completed!")
        
        return final_files
    
    except Exception as e:
        st.error(f"Upload failed: {str(e)}")
        return []
    finally:
        progress_bar.empty()
        status_text.empty()

def auto_index_files(file_paths: List[Path]):
    """Automatically index uploaded files"""
    if not file_paths:
        return
    
    # Check for changes
    changed_files = []
    for p in file_paths:
        current_hash = file_sha256(p)
        if st.session_state.manifest.get(str(p)) != current_hash:
            changed_files.append(p)
            st.session_state.manifest[str(p)] = current_hash
    
    if not changed_files:
        return
    
    with st.spinner(f"Indexing {len(changed_files)} files..."):
        start_time = time.time()
        
        try:
            # Extract and process documents
            chunks, metas, records = extract_docs_to_chunks_and_records(changed_files)
            
            if chunks:
                st.session_state.rag.build_or_update_index(chunks, metas)
                
                processing_time = time.time() - start_time
                
                st.success(f"Indexed {len(changed_files)} files with {len(chunks)} text chunks in {processing_time:.2f}s")
                
                # Show processing summary
                with st.expander("Processing Summary", expanded=False):
                    df_records = pd.DataFrame(records)
                    if not df_records.empty:
                        st.dataframe(df_records[['file', 'total_chunks', 'total_pages', 'file_size']], use_container_width=True)
                        
                        # Skills summary
                        all_skills = []
                        for record in records:
                            all_skills.extend(record.get('skills', []))
                        
                        if all_skills:
                            skill_counts = pd.Series(all_skills).value_counts().head(10)
                            fig = px.bar(
                                x=skill_counts.values,
                                y=skill_counts.index,
                                orientation='h',
                                title="Top 10 Skills Found",
                                labels={'x': 'Frequency', 'y': 'Skills'}
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Indexing failed: {str(e)}")

def render_analytics():
    """Render analytics dashboard"""
    if not st.session_state.get('show_analytics', False):
        return
    
    st.markdown("## Analytics Dashboard")
    
    # Get session summary
    summary = st.session_state.rag.get_session_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", summary["total_documents"])
    with col2:
        st.metric("Text Chunks", summary["total_chunks"])
    with col3:
        st.metric("Industries", len(summary["industries_detected"]))
    with col4:
        st.metric("Unique Skills", summary["unique_skills_count"])
    
    # Industry distribution
    if summary["industries_detected"]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Industry pie chart
            industry_counts = {}
            for doc in summary["documents"]:
                industry = doc["industry"]
                industry_counts[industry] = industry_counts.get(industry, 0) + 1
            
            fig = px.pie(
                values=list(industry_counts.values()),
                names=[name.replace('_', ' ').title() for name in industry_counts.keys()],
                title="Industry Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top skills bar chart
            if summary["top_skills"]:
                skills_data = summary["top_skills"][:15]
                fig = px.bar(
                    x=skills_data,
                    y=list(range(len(skills_data))),
                    orientation='h',
                    title="Top Skills Across All Resumes",
                    labels={'x': 'Skill', 'y': 'Index'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Query analytics
    if st.session_state.analysis_stats["query_history"]:
        st.markdown("### Query Analytics")
        
        query_df = pd.DataFrame(st.session_state.analysis_stats["query_history"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Queries", len(query_df))
            st.metric("Avg Response Time", f"{st.session_state.analysis_stats['avg_response_time']:.2f}s")
        
        with col2:
            if not query_df.empty:
                # Query frequency over time
                fig = px.histogram(
                    query_df, 
                    x="timestamp", 
                    title="Query Frequency Over Time",
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)

def render_upload_section():
    """Render the file upload section"""
    st.markdown("## Upload Resumes")
    
    # Upload interface
    uploaded_files = st.file_uploader(
        "Drop your resume files here",
        accept_multiple_files=True,
        type=["pdf", "docx", "png", "jpg", "jpeg", "zip"],
        help="Supported formats: PDF, DOCX, PNG, JPG, JPEG, ZIP (containing supported files)"
    )
    
    if uploaded_files:
        # Show upload preview
        with st.expander(f"Upload Preview ({len(uploaded_files)} files)", expanded=True):
            total_size = sum(len(f.read()) for f in uploaded_files)
            for f in uploaded_files:
                f.seek(0)  # Reset file pointer
            
            st.write(f"**Total files:** {len(uploaded_files)}")
            st.write(f"**Total size:** {human_size(total_size)}")
            
            # Show file list
            for file in uploaded_files:
                st.write(f"• {file.name} ({human_size(len(file.read()))})")
                file.seek(0)
        
        if st.button("Process Files", type="primary", use_container_width=True):
            # Process uploaded files
            processed_files = save_and_expand_files(uploaded_files)
            if processed_files:
                auto_index_files(processed_files)
                st.rerun()

def render_chat_interface():
    """Render the chat interface"""
    st.markdown("## AI Resume Analysis")
    
    # Quick query suggestions
    if not st.session_state.history:
        st.markdown("### Try these example queries:")
        
        example_queries = [
            "Rank all candidates by Python experience and provide evidence",
            "Which candidates have both AWS and machine learning skills?",
            "Find candidates suitable for a senior frontend developer role",
            "Compare the educational backgrounds of all candidates",
            "Who has the most relevant experience for a data scientist position?",
            "List all candidates with project management experience",
            "Find candidates with healthcare industry experience",
            "Which candidates have the strongest leadership background?"
        ]
        
        cols = st.columns(2)
        for i, query in enumerate(example_queries):
            with cols[i % 2]:
                if st.button(f"{query}", key=f"example_{i}", use_container_width=True):
                    st.session_state.current_query = query
                    st.rerun()
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, (role, message, metadata) in enumerate(st.session_state.history):
            with st.chat_message(role):
                if role == "user":
                    st.write(message)
                else:
                    st.markdown(message)
                    
                    # Show metadata if available
                    if metadata and metadata.get("hits_meta"):
                        with st.expander(f"Sources ({len(metadata['hits_meta'])} documents)", expanded=False):
                            hits_df = pd.DataFrame(metadata["hits_meta"])
                            if not hits_df.empty:
                                st.dataframe(hits_df, use_container_width=True)
                            
                            if metadata.get("retrieval_stats"):
                                stats = metadata["retrieval_stats"]
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Documents Found", stats["total_hits"])
                                with col2:
                                    st.metric("Avg Relevance", f"{stats['avg_score']:.3f}")
                                with col3:
                                    st.metric("Industry Context", metadata.get("industry_context", "general").title())
    
    # Chat input
    query_input = st.chat_input(
        placeholder="Ask anything about the resumes (e.g., 'Rank candidates by Python skills')",
        key="chat_input"
    )
    
    # Handle current query from examples
    if st.session_state.get('current_query'):
        query_input = st.session_state.current_query
        del st.session_state.current_query
    
    if query_input:
        handle_user_query(query_input)

def handle_user_query(query: str):
    """Handle user query and generate response"""
    # Check if we have any documents
    if not st.session_state.manifest:
        st.error("Please upload some resumes first!")
        return
    
    # Add user message to history
    st.session_state.history.append(("user", query, {}))
    
    # Show user message immediately
    with st.chat_message("user"):
        st.write(query)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing resumes..."):
            start_time = time.time()
            
            try:
                # Get answer from RAG system
                result = st.session_state.rag.answer(query, style="detailed analysis with bullet points")
                
                response_time = time.time() - start_time
                
                # Update analytics
                st.session_state.analysis_stats["total_queries"] += 1
                st.session_state.analysis_stats["query_history"].append({
                    "query": query,
                    "response_time": response_time,
                    "timestamp": time.time(),
                    "industry_context": result.get("industry_context", "general")
                })
                
                # Update average response time
                total_time = sum(q["response_time"] for q in st.session_state.analysis_stats["query_history"])
                st.session_state.analysis_stats["avg_response_time"] = total_time / len(st.session_state.analysis_stats["query_history"])
                
                # Display response
                st.markdown(result["text"])
                
                # Add to history with metadata
                st.session_state.history.append(("assistant", result["text"], result))
                
                # Show quick stats
                if result.get("retrieval_stats"):
                    stats = result["retrieval_stats"]
                    st.info(f"Found {stats['total_hits']} relevant documents • Response time: {response_time:.2f}s")
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.history.append(("assistant", error_msg, {}))

def render_main_content():
    """Render the main content area"""
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Upload & Process", "AI Analysis", "Analytics"])
    
    with tab1:
        render_upload_section()
        
        # Show current session summary
        if st.session_state.manifest:
            st.markdown("---")
            summary = st.session_state.rag.get_session_summary()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Documents", summary["total_documents"])
            with col2:
                st.metric("Text Chunks", summary["total_chunks"])
            with col3:
                st.metric("Industries", len(summary["industries_detected"]))
            with col4:
                st.metric("Skills Found", summary["unique_skills_count"])
            
            # Show top skills
            if summary["top_skills"]:
                st.markdown("Most Common Skills")
                skills_html = " ".join([f'<span class="skill-tag">{skill}</span>' for skill in summary["top_skills"][:20]])
                st.markdown(skills_html, unsafe_allow_html=True)
    
    with tab2:
        render_chat_interface()
    
    with tab3:
        render_analytics()

# Main app layout
def main():
    # Render sidebar
    render_sidebar()
    
    # Main content
    render_main_content()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>RAG-Resume</strong> - AI-Powered Resume Analysis System</p>
        <p>Built with Streamlit • Powered by Advanced RAG Technology</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()