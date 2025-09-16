import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import time
import json
import logging
import os
import sys
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


try:
    from src.config import Config
    from src.ingestion_pipeline import DocumentIngestionPipeline, IngestionResult
    from src.rag_engine import RAGEngine, RAGResponse
    from src.metadata_manager import MetadataManager
    from src.document_processor import ProcessingStatus
    from src.embedding_system import EmbeddingSystem
    from src.vector_store import QdrantVectorStore
    from src.groq_client import LLMSystem
    from logger.custom_logger import CustomLoggerTracker
    custom_log = CustomLoggerTracker()
    logger = custom_log.get_logger("gradio_demo")

except ImportError:
    # Fallback to standard logging if custom logger not available
    logger = logging.getLogger("gradio_demo")


class RAGGradioDemo:
    """
    Gradio demo application for the Manufacturing RAG Agent.
    This demo provides a user-friendly interface for document upload,
    question answering, and result visualization using Gradio.
    """
    def __init__(self):
        """Initialize the RAG demo application."""
        self.config = None
        self.ingestion_pipeline = None
        self.rag_engine = None
        self.metadata_manager = None
        self.embedding_system = None
        self.vector_store = None
        self.llm_system = None
        
        # Demo state
        self.chat_history = []
        self.documents = []
        self.system_initialized = False
        
    def initialize_system(self) -> Tuple[bool, str]:
        """
        Initialize the RAG system components.
        
        Returns:
            Tuple of (success, message)
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(current_dir)
        try:
            # Check if required modules are imported
            if Config is None:
                return False, "RAG modules not imported. Please ensure all src/ modules are available and properly structured."
            
            # Check for config file in multiple locations
            config_paths = [
                "config.yaml",
                "src/config.yaml", 
                os.path.join(current_dir, "config.yaml"),
                os.path.join(src_dir, "config.yaml")
            ]
            
            config_path = None
            for path in config_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            
            if not config_path:
                available_files = []
                for search_dir in [current_dir, src_dir]:
                    if os.path.exists(search_dir):
                        files = [f for f in os.listdir(search_dir) if f.endswith('.yaml') or f.endswith('.yml')]
                        if files:
                            available_files.extend([os.path.join(search_dir, f) for f in files])
                
                error_msg = f"Configuration file not found. Searched: {config_paths}"
                if available_files:
                    error_msg += f"\nAvailable config files: {available_files}"
                return False, error_msg
            
            
            logger.info(f"Using config file: {config_path}")
            
            # Load configuration
            self.config = Config(config_path)
            
            # Initialize components
            config_dict = {
                'siliconflow_api_key': self.config.siliconflow_api_key,
                'groq_api_key': self.config.groq_api_key,
                'qdrant_url': self.config.qdrant_url,
                'qdrant_api_key': self.config.qdrant_api_key,
                **self.config.rag_config,
                **self.config.document_processing_config,
                **self.config.storage_config
            }
            
            # Initialize core components
            self.embedding_system = EmbeddingSystem(config_dict)
            self.vector_store = QdrantVectorStore(config_dict)
            self.llm_system = LLMSystem(config_dict)
            self.ingestion_pipeline = DocumentIngestionPipeline(config_dict)
            self.rag_engine = RAGEngine(config_dict)
            self.metadata_manager = MetadataManager(config_dict)
            
            self.system_initialized = True
            return True, "RAG system initialized successfully!"
            
        except Exception as e:
            error_msg = f"Failed to initialize RAG system: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Error details: {type(e).__name__}")
            return False, error_msg
    
    def process_uploaded_files(self, files) -> Tuple[str, pd.DataFrame]:
        """
        Process uploaded files through the ingestion pipeline.
        
        Args:
            files: List of uploaded file objects
            
        Returns:
            Tuple of (status_message, results_dataframe)
        """
        if not self.system_initialized:
            return "❌ System not initialized. Please initialize first.", pd.DataFrame()
        
        if not files:
            return "No files uploaded.", pd.DataFrame()
        
        results = []
        total_files = len(files)
        
        try:
            for i, file in enumerate(files):
                # Save uploaded file temporarily
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
                        tmp_file.write(file.read())
                        temp_path = tmp_file.name
                    
                    # Process document
                    result = self.ingestion_pipeline.ingest_document(temp_path)
                    
                    # Add result info
                    results.append({
                        'Filename': file.name,
                        'Status': '✅ Success' if result.success else '❌ Failed',
                        'Chunks Created': result.chunks_created,
                        'Chunks Indexed': result.chunks_indexed,
                        'Processing Time (s)': f"{result.processing_time:.2f}",
                        'Error Message': result.error_message or 'None'
                    })
                    
                except Exception as e:
                    results.append({
                        'Filename': file.name,
                        'Status': '❌ Failed',
                        'Chunks Created': 0,
                        'Chunks Indexed': 0,
                        'Processing Time (s)': '0.00',
                        'Error Message': str(e)
                    })
                
                finally:
                    # Clean up temporary file
                    if temp_path and os.path.exists(temp_path):
                        os.unlink(temp_path)
            
            # Create results summary
            successful = sum(1 for r in results if 'Success' in r['Status'])
            total_chunks = sum(r['Chunks Indexed'] for r in results if isinstance(r['Chunks Indexed'], int))
            
            status_msg = f"✅ Processing Complete: {successful}/{total_files} files processed successfully. Total chunks indexed: {total_chunks}"
            
            return status_msg, pd.DataFrame(results)
            
        except Exception as e:
            error_msg = f"❌ Batch processing failed: {str(e)}"
            logger.error(error_msg)
            return error_msg, pd.DataFrame(results) if results else pd.DataFrame()
    
    def ask_question(self, question: str, max_results: int = 5, 
                    similarity_threshold: float = 0.7, document_filter: str = "All") -> Tuple[str, str, pd.DataFrame]:
        """
        Process a question through the RAG engine.
        
        Args:
            question: Question to answer
            max_results: Maximum context chunks
            similarity_threshold: Similarity threshold for retrieval
            document_filter: Document type filter
            
        Returns:
            Tuple of (answer, citations_info, performance_dataframe)
        """
        if not self.system_initialized:
            return "❌ System not initialized. Please initialize first.", "", pd.DataFrame()
        
        if not question.strip():
            return "Please enter a question.", "", pd.DataFrame()
        
        try:
            # Check if documents are available
            documents = self.metadata_manager.list_documents(
                status=ProcessingStatus.COMPLETED, 
                limit=1
            )
            if not documents:
                return "⚠️ No processed documents available. Please upload and process documents first.", "", pd.DataFrame()
            
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
            self.chat_history.append((question, response))
            
            # Format answer
            if not response.success:
                return f"❌ Failed to generate answer: {response.error_message}", "", pd.DataFrame()
            
            # Create citations info
            citations_info = self._format_citations(response.citations)
            
            # Create performance dataframe
            performance_data = {
                'Metric': ['Confidence Score', 'Processing Time (s)', 'Retrieval Time (s)', 
                          'Generation Time (s)', 'Rerank Time (s)', 'Sources Used', 'Chunks Retrieved'],
                'Value': [
                    f"{response.confidence_score:.3f}",
                    f"{response.processing_time:.3f}",
                    f"{response.retrieval_time:.3f}",
                    f"{response.generation_time:.3f}",
                    f"{response.rerank_time:.3f}",
                    len(response.citations),
                    response.total_chunks_retrieved
                ]
            }
            
            performance_df = pd.DataFrame(performance_data)
            
            return response.answer, citations_info, performance_df
            
        except Exception as e:
            error_msg = f"❌ Question processing failed: {str(e)}"
            logger.error(error_msg)
            return error_msg, "", pd.DataFrame()
    
    def _format_citations(self, citations) -> str:
        """Format citations for display."""
        if not citations:
            return "No citations available."
        
        citation_text = "## 📚 Sources & Citations\n\n"
        
        for i, citation in enumerate(citations):
            citation_text += f"**Source {i+1}:** {citation.source_file} (Confidence: {citation.confidence:.3f})\n"
            
            # Add specific location info
            location_parts = []
            if citation.page_number:
                location_parts.append(f"📄 Page: {citation.page_number}")
            if citation.worksheet_name:
                location_parts.append(f"📊 Sheet: {citation.worksheet_name}")
            if citation.cell_range:
                location_parts.append(f"🔢 Range: {citation.cell_range}")
            if citation.section_title:
                location_parts.append(f"📑 Section: {citation.section_title}")
            
            if location_parts:
                citation_text += f"*Location:* {' | '.join(location_parts)}\n"
            
            citation_text += f"*Excerpt:* \"{citation.text_snippet}\"\n\n"
        
        return citation_text
    
    def get_document_library(self) -> pd.DataFrame:
        """Get document library as DataFrame."""
        if not self.system_initialized:
            return pd.DataFrame({'Message': ['System not initialized']})
        
        try:
            documents = self.metadata_manager.list_documents(limit=100)
            
            if not documents:
                return pd.DataFrame({'Message': ['No documents uploaded yet']})
            
            doc_data = []
            for doc in documents:
                doc_data.append({
                    'Filename': doc.filename,
                    'Type': doc.file_type.upper(),
                    'Status': doc.processing_status.value.title(),
                    'Chunks': doc.total_chunks,
                    'Size': self._format_file_size(doc.file_size),
                    'Uploaded': doc.upload_timestamp.strftime('%Y-%m-%d %H:%M'),
                    'Processing Time (s)': f"{doc.processing_time:.2f}" if doc.processing_time else "N/A"
                })
            
            return pd.DataFrame(doc_data)
            
        except Exception as e:
            logger.error(f"Failed to load document library: {e}")
            return pd.DataFrame({'Error': [str(e)]})
    
    def get_system_status(self) -> Tuple[str, pd.DataFrame]:
        """Get system status and health information."""
        if not self.system_initialized:
            return "❌ System not initialized", pd.DataFrame()
        
        try:
            # Health checks
            rag_health = self.rag_engine.health_check()
            pipeline_health = self.ingestion_pipeline.health_check()
            
            # Create status message
            status_parts = []
            for component, healthy in rag_health.items():
                status = "✅ Healthy" if healthy else "❌ Unhealthy"
                status_parts.append(f"**{component.replace('_', ' ').title()}:** {status}")
            
            status_message = "## 🏥 System Health\n" + "\n".join(status_parts)
            
            # Create detailed status table
            all_health = {**rag_health, **pipeline_health}
            health_data = []
            
            for component, healthy in all_health.items():
                health_data.append({
                    'Component': component.replace('_', ' ').title(),
                    'Status': '✅ Healthy' if healthy else '❌ Unhealthy',
                    'Last Checked': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            
            return status_message, pd.DataFrame(health_data)
            
        except Exception as e:
            error_msg = f"❌ Failed to check system status: {str(e)}"
            logger.error(error_msg)
            return error_msg, pd.DataFrame()
    
    def get_analytics_data(self) -> Tuple[str, Dict[str, Any]]:
        """Get analytics data for visualization."""
        if not self.system_initialized:
            return "❌ System not initialized", {}
        
        try:
            # Get system statistics
            pipeline_stats = self.ingestion_pipeline.get_pipeline_stats()
            metadata_stats = self.metadata_manager.get_statistics()
            
            # Create summary message
            total_docs = metadata_stats.get('total_documents', 0)
            total_chunks = metadata_stats.get('total_chunks', 0)
            total_size = metadata_stats.get('total_file_size', 0)
            
            summary = f"""## 📊 Analytics Overview
            
**Total Documents:** {total_docs}
**Total Chunks:** {total_chunks}
**Total File Size:** {self._format_file_size(total_size)}
**Vector Points:** {pipeline_stats.get('vector_store', {}).get('total_points', 0)}
"""
            
            # Prepare data for charts
            analytics_data = {
                'document_types': metadata_stats.get('documents_by_type', {}),
                'processing_status': metadata_stats.get('documents_by_status', {}),
                'pipeline_stats': pipeline_stats,
                'metadata_stats': metadata_stats
            }
            
            return summary, analytics_data
            
        except Exception as e:
            error_msg = f"❌ Failed to load analytics: {str(e)}"
            logger.error(error_msg)
            return error_msg, {}
    
    def create_document_type_chart(self, analytics_data: Dict[str, Any]):
        """Create document type distribution chart."""
        if not analytics_data or 'document_types' not in analytics_data:
            return None
        
        type_counts = analytics_data['document_types']
        if not type_counts:
            return None
        
        fig = px.pie(
            values=list(type_counts.values()),
            names=list(type_counts.keys()),
            title="Documents by Type"
        )
        return fig
    
    def create_status_chart(self, analytics_data: Dict[str, Any]):
        """Create processing status chart."""
        if not analytics_data or 'processing_status' not in analytics_data:
            return None
        
        status_counts = analytics_data['processing_status']
        if not status_counts:
            return None
        
        fig = px.bar(
            x=list(status_counts.keys()),
            y=list(status_counts.values()),
            title="Documents by Processing Status"
        )
        return fig
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f}{size_names[i]}"


def create_gradio_interface():
    """Create the main Gradio interface."""
    
    # Initialize demo instance
    demo_instance = RAGGradioDemo()
    
    # Define the interface
    with gr.Blocks(title="Manufacturing RAG Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏭 Manufacturing RAG Agent
        *Intelligent document analysis for manufacturing data*
        
        This system allows you to upload manufacturing documents (PDF, Excel, Images) and ask questions about their content.
        """)
        
        # System Status
        with gr.Row():
            with gr.Column(scale=3):
                system_status = gr.Markdown("**System Status:** Not initialized")
            with gr.Column(scale=1):
                init_btn = gr.Button("🚀 Initialize System", variant="primary")
        
        # Main tabs
        with gr.Tabs():
            # Document Upload Tab
            with gr.TabItem("📄 Document Upload"):
                gr.Markdown("### Upload and Process Documents")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        file_upload = gr.File(
                            file_count="multiple",
                            file_types=[".pdf", ".xlsx", ".xls", ".xlsm", ".png", ".jpg", ".jpeg"],
                            label="Choose files to upload"
                        )
                        upload_btn = gr.Button("🔄 Process Documents", variant="primary")
                    
                    with gr.Column(scale=1):
                        upload_status = gr.Textbox(
                            label="Processing Status",
                            interactive=False,
                            lines=3
                        )
                
                # Results display
                upload_results = gr.Dataframe(
                    label="Processing Results",
                    interactive=False
                )
                
                # Document Library
                gr.Markdown("### 📚 Document Library")
                refresh_docs_btn = gr.Button("🔄 Refresh Library")
                doc_library = gr.Dataframe(
                    label="Uploaded Documents",
                    interactive=False
                )
            
            # Question Answering Tab
            with gr.TabItem("❓ Ask Questions"):
                gr.Markdown("### Ask Questions About Your Documents")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        question_input = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g., What is the average production yield for Q3?",
                            lines=2
                        )
                        
                        with gr.Row():
                            ask_btn = gr.Button("🔍 Ask Question", variant="primary")
                            clear_btn = gr.Button("🗑️ Clear")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### Advanced Options")
                        max_results = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1,
                            label="Max Context Chunks"
                        )
                        similarity_threshold = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.7, step=0.1,
                            label="Similarity Threshold"
                        )
                        doc_filter = gr.Dropdown(
                            choices=["All", "PDF", "Excel", "Image"],
                            value="All",
                            label="Filter by Document Type"
                        )
                
                # Answer display
                answer_output = gr.Markdown(label="Answer")
                citations_output = gr.Markdown(label="Citations")
                
                # Performance metrics
                performance_metrics = gr.Dataframe(
                    label="Performance Metrics",
                    interactive=False
                )
            
            # Analytics Tab
            with gr.TabItem("📊 Analytics"):
                gr.Markdown("### System Analytics")
                
                refresh_analytics_btn = gr.Button("🔄 Refresh Analytics")
                analytics_summary = gr.Markdown("Analytics data will appear here...")
                
                with gr.Row():
                    doc_type_chart = gr.Plot(label="Document Types")
                    status_chart = gr.Plot(label="Processing Status")
            
            # System Status Tab
            with gr.TabItem("⚙️ System Status"):
                gr.Markdown("### System Health & Configuration")
                
                check_health_btn = gr.Button("🔍 Check System Health")
                health_status = gr.Markdown("System health information will appear here...")
                health_details = gr.Dataframe(
                    label="Component Health Details",
                    interactive=False
                )
        
        # Event handlers
        def initialize_system():
            success, message = demo_instance.initialize_system()
            status_color = "green" if success else "red"
            status_icon = "✅" if success else "❌"
            return f"**System Status:** <span style='color: {status_color}'>{status_icon} {message}</span>"
        
        def refresh_document_library():
            return demo_instance.get_document_library()
        
        def refresh_analytics():
            summary, data = demo_instance.get_analytics_data()
            doc_chart = demo_instance.create_document_type_chart(data)
            status_chart_fig = demo_instance.create_status_chart(data)
            return summary, doc_chart, status_chart_fig
        
        def check_system_health():
            status, details = demo_instance.get_system_status()
            return status, details
        
        def clear_question():
            return "", "", gr.Dataframe()
        
        # Connect event handlers
        init_btn.click(
            initialize_system,
            outputs=[system_status]
        )
        
        upload_btn.click(
            demo_instance.process_uploaded_files,
            inputs=[file_upload],
            outputs=[upload_status, upload_results]
        )
        
        refresh_docs_btn.click(
            refresh_document_library,
            outputs=[doc_library]
        )
        
        ask_btn.click(
            demo_instance.ask_question,
            inputs=[question_input, max_results, similarity_threshold, doc_filter],
            outputs=[answer_output, citations_output, performance_metrics]
        )
        
        clear_btn.click(
            clear_question,
            outputs=[question_input, answer_output, performance_metrics]
        )
        
        refresh_analytics_btn.click(
            refresh_analytics,
            outputs=[analytics_summary, doc_type_chart, status_chart]
        )
        
        check_health_btn.click(
            check_system_health,
            outputs=[health_status, health_details]
        )
        
        # Auto-refresh document library on upload
        upload_btn.click(
            refresh_document_library,
            outputs=[doc_library]
        )
    
    return demo

def main():
    """Main function to launch the Gradio demo."""
    try:
        # Create and launch the interface
        demo = create_gradio_interface()
        
        # Launch with configuration
        demo.launch(
            server_name="0.0.0.0",  # Allow external connections
            server_port=7860,       # Default Gradio port
            share=False,            # Set to True to create public link
            debug=True,             # Enable debug mode
            show_error=True,        # Show detailed error messages
            quiet=False             # Enable logging
        )
        
    except Exception as e:
        print(f"Failed to launch Gradio demo: {e}")
        print("Please ensure all dependencies are installed and the src/ directory contains the required modules.")

if __name__ == "__main__":
    main()