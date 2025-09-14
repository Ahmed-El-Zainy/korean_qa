import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import time
import json
from typing import List, Dict, Any, Optional
import logging

#

try:
    from logger.custom_logger import CustomLoggerTracker
    custom_log = CustomLoggerTracker()
    logger = custom_log.get_logger("rag_demo")

except ImportError:
    # Fallback to standard logging if custom logger not available
    logger = logging.getLogger("rag_demo")



# Import RAG components
try:
    from src.config import Config
    from src.ingestion_pipeline import DocumentIngestionPipeline, IngestionResult
    from src.rag_engine import RAGEngine, RAGResponse
    from src.metadata_manager import MetadataManager
    from src.document_processor import ProcessingStatus
except ImportError as e:
    st.error(f"Failed to import RAG components: {e}")
    st.stop()


class RAGDemo:
    """
    Streamlit demo application for the Manufacturing RAG Agent.
    
    This demo provides a user-friendly interface for document upload,
    question answering, and result visualization.
    """
    
    def __init__(self):
        """Initialize the RAG demo application."""
        self.config = None
        self.ingestion_pipeline = None
        self.rag_engine = None
        self.metadata_manager = None
        
        # Initialize session state
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.documents = []
            st.session_state.chat_history = []
    
    def initialize_system(self):
        """Initialize the RAG system components."""
        try:
            # Load configuration
            self.config = Config("src/config.yaml")
            
            # Initialize components
            config_dict = {
                'silicon_flow_api_key': self.config.silicon_flow_api_key,
                'groq_api_key': self.config.groq_api_key,
                'qdrant_url': self.config.qdrant_url,
                'qdrant_api_key': self.config.qdrant_api_key,
                **self.config.rag_config,
                **self.config.document_processing_config,
                **self.config.storage_config
            }
            
            self.ingestion_pipeline = DocumentIngestionPipeline(config_dict)
            self.rag_engine = RAGEngine(config_dict)
            self.metadata_manager = MetadataManager(config_dict)
            
            st.session_state.initialized = True
            return True
            
        except Exception as e:
            st.error(f"Failed to initialize RAG system: {e}")
            return False
    
    def run(self):
        """Run the Streamlit demo application."""
        st.set_page_config(
            page_title="Manufacturing RAG Agent",
            page_icon="🏭",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🏭 Manufacturing RAG Agent")
        st.markdown("*Intelligent document analysis for manufacturing data*")
        
        # Initialize system if not already done
        if not st.session_state.initialized:
            with st.spinner("Initializing RAG system..."):
                if not self.initialize_system():
                    st.stop()
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["📄 Document Upload", "❓ Ask Questions", "📊 Analytics", "⚙️ System Status"]
        )
        
        # Route to appropriate page
        if page == "📄 Document Upload":
            self.document_upload_page()
        elif page == "❓ Ask Questions":
            self.question_answering_page()
        elif page == "📊 Analytics":
            self.analytics_page()
        elif page == "⚙️ System Status":
            self.system_status_page()
    
    def document_upload_page(self):
        """Document upload and management page."""
        st.header("📄 Document Upload & Management")
        
        # File upload section
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['pdf', 'xlsx', 'xls', 'xlsm', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Supported formats: PDF, Excel (.xlsx, .xls, .xlsm), Images (.png, .jpg, .jpeg)"
        )
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                self.process_uploaded_files(uploaded_files)
        
        # Document management section
        st.subheader("Document Library")
        self.display_document_library()
    
    def process_uploaded_files(self, uploaded_files):
        """Process uploaded files through the ingestion pipeline."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
        
        results = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Process document
                result = self.ingestion_pipeline.ingest_document(tmp_file_path)
                results.append(result)
                
                # Clean up temporary file
                Path(tmp_file_path).unlink()
                
            except Exception as e:
                st.error(f"Failed to process {uploaded_file.name}: {e}")
                results.append(IngestionResult(
                    document_id="error",
                    filename=uploaded_file.name,
                    success=False,
                    processing_time=0.0,
                    chunks_created=0,
                    chunks_indexed=0,
                    error_message=str(e)
                ))
        
        # Display results
        status_text.text("Processing complete!")
        self.display_processing_results(results, results_container)
        
        # Refresh document library
        st.rerun()
    
    def display_processing_results(self, results: List[IngestionResult], container):
        """Display processing results in a formatted way."""
        with container:
            st.subheader("Processing Results")
            
            # Summary metrics
            successful = sum(1 for r in results if r.success)
            total_chunks = sum(r.chunks_indexed for r in results if r.success)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Documents Processed", f"{successful}/{len(results)}")
            with col2:
                st.metric("Total Chunks Created", total_chunks)
            with col3:
                avg_time = sum(r.processing_time for r in results) / len(results)
                st.metric("Avg Processing Time", f"{avg_time:.2f}s")
            
            # Detailed results
            for result in results:
                with st.expander(f"📄 {result.filename} - {'✅ Success' if result.success else '❌ Failed'}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Document ID:** {result.document_id}")
                        st.write(f"**Processing Time:** {result.processing_time:.2f}s")
                        st.write(f"**Chunks Created:** {result.chunks_created}")
                        st.write(f"**Chunks Indexed:** {result.chunks_indexed}")
                    
                    with col2:
                        if result.success:
                            st.success("Document processed successfully!")
                        else:
                            st.error(f"Processing failed: {result.error_message}")
                        
                        if result.warnings:
                            for warning in result.warnings:
                                st.warning(warning)
    
    def display_document_library(self):
        """Display the document library with management options."""
        try:
            # Get document list
            documents = self.metadata_manager.list_documents(limit=100)
            
            if not documents:
                st.info("No documents uploaded yet. Use the upload section above to add documents.")
                return
            
            # Create DataFrame for display
            doc_data = []
            for doc in documents:
                doc_data.append({
                    'Filename': doc.filename,
                    'Type': doc.file_type.upper(),
                    'Status': doc.processing_status.value.title(),
                    'Chunks': doc.total_chunks,
                    'Size': self.format_file_size(doc.file_size),
                    'Uploaded': doc.upload_timestamp.strftime('%Y-%m-%d %H:%M'),
                    'Processing Time': f"{doc.processing_time:.2f}s" if doc.processing_time else "N/A",
                    'Document ID': doc.document_id
                })
            
            df = pd.DataFrame(doc_data)
            
            # Display with selection
            selected_indices = st.dataframe(
                df.drop('Document ID', axis=1),
                use_container_width=True,
                selection_mode="multi-row",
                on_select="rerun"
            ).selection.rows
            
            # Management actions
            if selected_indices:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🔄 Reprocess Selected", help="Reprocess selected documents"):
                        self.reprocess_documents([doc_data[i]['Document ID'] for i in selected_indices])
                
                with col2:
                    if st.button("🗑️ Delete Selected", help="Delete selected documents", type="secondary"):
                        self.delete_documents([doc_data[i]['Document ID'] for i in selected_indices])
                
                with col3:
                    if st.button("📋 View Details", help="View detailed information"):
                        self.show_document_details([doc_data[i]['Document ID'] for i in selected_indices])
            
        except Exception as e:
            st.error(f"Failed to load document library: {e}")
    
    def question_answering_page(self):
        """Question answering interface."""
        st.header("❓ Ask Questions")
        
        # Check if documents are available
        try:
            documents = self.metadata_manager.list_documents(
                status=ProcessingStatus.COMPLETED, 
                limit=1
            )
            if not documents:
                st.warning("No processed documents available. Please upload and process documents first.")
                return
        except Exception as e:
            st.error(f"Failed to check document availability: {e}")
            return
        
        # Question input
        question = st.text_input(
            "Enter your question about the manufacturing documents:",
            placeholder="e.g., What is the average production yield for Q3?",
            help="Ask questions about processes, metrics, specifications, or any content in your uploaded documents."
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                max_results = st.slider("Max Context Chunks", 1, 10, 5)
                similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, 0.1)
            
            with col2:
                document_filter = st.selectbox(
                    "Filter by Document Type",
                    ["All", "PDF", "Excel", "Image"]
                )
                enable_reranking = st.checkbox("Enable Reranking", value=True)
        
        # Ask question
        if st.button("🔍 Ask Question", type="primary", disabled=not question):
            self.process_question(question, max_results, similarity_threshold, document_filter, enable_reranking)
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("💬 Recent Questions")
            for i, (q, response) in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.expander(f"Q: {q[:100]}..." if len(q) > 100 else f"Q: {q}"):
                    self.display_rag_response(response)
    
    def process_question(self, question: str, max_results: int, similarity_threshold: float, 
                        document_filter: str, enable_reranking: bool):
        """Process a question through the RAG engine."""
        with st.spinner("Searching documents and generating answer..."):
            try:
                # Prepare filters
                filters = {}
                if document_filter != "All":
                    filters["document_type"] = document_filter.lower()
                
                # Update RAG engine config temporarily
                original_config = {
                    'final_top_k': self.rag_engine.final_top_k,
                    'similarity_threshold': self.rag_engine.similarity_threshold
                }
                
                self.rag_engine.final_top_k = max_results
                self.rag_engine.similarity_threshold = similarity_threshold
                
                # Get response
                response = self.rag_engine.answer_question(question, filters if filters else None)
                
                # Restore original config
                self.rag_engine.final_top_k = original_config['final_top_k']
                self.rag_engine.similarity_threshold = original_config['similarity_threshold']
                
                # Add to chat history
                st.session_state.chat_history.append((question, response))
                
                # Display response
                self.display_rag_response(response)
                
            except Exception as e:
                st.error(f"Failed to process question: {e}")
    
    def display_rag_response(self, response: RAGResponse):
        """Display a RAG response with formatting and citations."""
        if not response.success:
            st.error(f"Failed to generate answer: {response.error_message}")
            return
        
        # Main answer
        st.markdown("### 📝 Answer")
        st.markdown(response.answer)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Confidence", f"{response.confidence_score:.2f}")
        with col2:
            st.metric("Processing Time", f"{response.processing_time:.2f}s")
        with col3:
            st.metric("Sources Used", len(response.citations))
        with col4:
            st.metric("Chunks Retrieved", response.total_chunks_retrieved)
        
        # Citations
        if response.citations:
            st.markdown("### 📚 Sources & Citations")
            
            for i, citation in enumerate(response.citations):
                with st.expander(f"Source {i+1}: {citation.source_file} (Confidence: {citation.confidence:.2f})"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("**Text Snippet:**")
                        st.markdown(f"*{citation.text_snippet}*")
                    
                    with col2:
                        st.markdown("**Citation Details:**")
                        if citation.page_number:
                            st.write(f"📄 Page: {citation.page_number}")
                        if citation.worksheet_name:
                            st.write(f"📊 Sheet: {citation.worksheet_name}")
                        if citation.cell_range:
                            st.write(f"📍 Range: {citation.cell_range}")
                        if citation.section_title:
                            st.write(f"📑 Section: {citation.section_title}")
        
        # Performance breakdown
        with st.expander("⚡ Performance Details"):
            perf_data = {
                'Stage': ['Retrieval', 'Reranking', 'Generation', 'Total'],
                'Time (s)': [
                    response.retrieval_time,
                    response.rerank_time,
                    response.generation_time,
                    response.processing_time
                ]
            }
            
            fig = px.bar(
                perf_data, 
                x='Stage', 
                y='Time (s)',
                title="Processing Time Breakdown"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def analytics_page(self):
        """Analytics and statistics page."""
        st.header("📊 Analytics Dashboard")
        
        try:
            # Get system statistics
            pipeline_stats = self.ingestion_pipeline.get_pipeline_stats()
            metadata_stats = self.metadata_manager.get_statistics()
            
            # Overview metrics
            st.subheader("📈 Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Documents", metadata_stats.get('total_documents', 0))
            with col2:
                st.metric("Total Chunks", metadata_stats.get('total_chunks', 0))
            with col3:
                st.metric("Total File Size", self.format_file_size(metadata_stats.get('total_file_size', 0)))
            with col4:
                vector_stats = pipeline_stats.get('vector_store', {})
                st.metric("Vector Points", vector_stats.get('total_points', 0))
            
            # Document type distribution
            st.subheader("📄 Document Types")
            type_counts = metadata_stats.get('documents_by_type', {})
            if type_counts:
                fig = px.pie(
                    values=list(type_counts.values()),
                    names=list(type_counts.keys()),
                    title="Documents by Type"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Processing status distribution
            st.subheader("⚙️ Processing Status")
            status_counts = metadata_stats.get('documents_by_status', {})
            if status_counts:
                fig = px.bar(
                    x=list(status_counts.keys()),
                    y=list(status_counts.values()),
                    title="Documents by Processing Status"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Recent activity
            st.subheader("🕒 Recent Activity")
            recent_docs = self.metadata_manager.list_documents(limit=10)
            if recent_docs:
                activity_data = []
                for doc in recent_docs:
                    activity_data.append({
                        'Document': doc.filename,
                        'Status': doc.processing_status.value.title(),
                        'Chunks': doc.total_chunks,
                        'Upload Time': doc.upload_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    })
                
                st.dataframe(pd.DataFrame(activity_data), use_container_width=True)
            
        except Exception as e:
            st.error(f"Failed to load analytics: {e}")
    
    def system_status_page(self):
        """System status and health check page."""
        st.header("⚙️ System Status")
        
        # Health checks
        st.subheader("🏥 Health Checks")
        
        try:
            # RAG engine health
            rag_health = self.rag_engine.health_check()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status = "✅ Healthy" if rag_health.get('vector_store', False) else "❌ Unhealthy"
                st.metric("Vector Store", status)
            
            with col2:
                status = "✅ Healthy" if rag_health.get('llm_system', False) else "❌ Unhealthy"
                st.metric("LLM System", status)
            
            with col3:
                status = "✅ Healthy" if rag_health.get('embedding_system', False) else "❌ Unhealthy"
                st.metric("Embedding System", status)
            
            # Pipeline health
            pipeline_health = self.ingestion_pipeline.health_check()
            
            st.subheader("🔧 Pipeline Components")
            for component, healthy in pipeline_health.items():
                status = "✅ Healthy" if healthy else "❌ Unhealthy"
                st.write(f"**{component.replace('_', ' ').title()}:** {status}")
            
            # Configuration display
            st.subheader("⚙️ Configuration")
            with st.expander("View Current Configuration"):
                config_display = {
                    "RAG Settings": {
                        "Max Context Chunks": self.rag_engine.max_context_chunks,
                        "Similarity Threshold": self.rag_engine.similarity_threshold,
                        "Rerank Top K": self.rag_engine.rerank_top_k,
                        "Final Top K": self.rag_engine.final_top_k
                    },
                    "Pipeline Settings": {
                        "Chunk Size": self.ingestion_pipeline.chunk_size,
                        "Chunk Overlap": self.ingestion_pipeline.chunk_overlap,
                        "Batch Size": self.ingestion_pipeline.batch_size,
                        "Max Workers": self.ingestion_pipeline.max_workers
                    }
                }
                
                st.json(config_display)
            
        except Exception as e:
            st.error(f"Failed to check system status: {e}")
    
    def format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f}{size_names[i]}"
    
    def reprocess_documents(self, document_ids: List[str]):
        """Reprocess selected documents."""
        with st.spinner(f"Reprocessing {len(document_ids)} documents..."):
            for doc_id in document_ids:
                try:
                    result = self.ingestion_pipeline.reprocess_document(doc_id)
                    if result.success:
                        st.success(f"Successfully reprocessed {result.filename}")
                    else:
                        st.error(f"Failed to reprocess {result.filename}: {result.error_message}")
                except Exception as e:
                    st.error(f"Error reprocessing document {doc_id}: {e}")
        
        st.rerun()
    
    def delete_documents(self, document_ids: List[str]):
        """Delete selected documents."""
        if st.confirm(f"Are you sure you want to delete {len(document_ids)} documents? This action cannot be undone."):
            with st.spinner(f"Deleting {len(document_ids)} documents..."):
                for doc_id in document_ids:
                    try:
                        success = self.ingestion_pipeline.delete_document(doc_id)
                        if success:
                            st.success(f"Successfully deleted document {doc_id}")
                        else:
                            st.error(f"Failed to delete document {doc_id}")
                    except Exception as e:
                        st.error(f"Error deleting document {doc_id}: {e}")
            
            st.rerun()
    
    def show_document_details(self, document_ids: List[str]):
        """Show detailed information for selected documents."""
        for doc_id in document_ids:
            try:
                metadata = self.metadata_manager.get_document_metadata(doc_id)
                if metadata:
                    with st.expander(f"📄 {metadata.filename} Details"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Document ID:** {metadata.document_id}")
                            st.write(f"**File Type:** {metadata.file_type}")
                            st.write(f"**File Size:** {self.format_file_size(metadata.file_size)}")
                            st.write(f"**Total Chunks:** {metadata.total_chunks}")
                        
                        with col2:
                            st.write(f"**Upload Time:** {metadata.upload_timestamp}")
                            st.write(f"**Processing Status:** {metadata.processing_status.value}")
                            st.write(f"**Processing Time:** {metadata.processing_time:.2f}s" if metadata.processing_time else "N/A")
                            st.write(f"**Checksum:** {metadata.checksum[:16]}..." if metadata.checksum else "N/A")
                        
                        if metadata.error_message:
                            st.error(f"Error: {metadata.error_message}")
                        
                        if metadata.metadata_json:
                            with st.expander("Raw Metadata"):
                                try:
                                    metadata_dict = json.loads(metadata.metadata_json)
                                    st.json(metadata_dict)
                                except:
                                    st.text(metadata.metadata_json)
            except Exception as e:
                st.error(f"Failed to load details for document {doc_id}: {e}")


def main():
    """Main function to run the Streamlit demo."""
    demo = RAGDemo()
    demo.run()


if __name__ == "__main__":
    main()