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
from typing import Dict, Any, Tuple, List
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.config import Config
    from src.ingestion_pipeline import DocumentIngestionPipeline, IngestionResult
    from src.rag_engine import RAGEngine, RAGResponse
    from src.metadata_manager import MetadataManager
    from src.document_processor import ProcessingStatus, DocumentProcessorFactory, DocumentType
    from src.pdf_processor import PDFProcessor
    from src.excel_processor import ExcelProcessor
    from src.image_processor import ImageProcessor
    
except ImportError as e:
    logger.error(f"Failed to import RAG components: {e}")
    print(f"❌ Import Error: {e}")
    print("Please ensure all src/ modules are properly structured and dependencies are installed")
    sys.exit(1)




class RAGGradioDemo:
    """Fixed Gradio demo application for the Manufacturing RAG Agent."""
    
    def __init__(self):
        """Initialize the RAG demo application."""
        self.config = None
        self.ingestion_pipeline = None
        self.rag_engine = None
        self.metadata_manager = None
        
        # Initialize session state tracking
        self.system_initialized = False
        self.documents = []
        self.chat_history = []
    
    def initialize_system(self) -> Tuple[bool, str]:
        """Initialize the RAG system components with better error handling."""
        try:
            # Find config file
            config_paths = [
                "src/config.yaml",
                "config.yaml", 
                os.path.join(os.path.dirname(__file__), "config.yaml"),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "config.yaml")
            ]
            
            config_path = None
            for path in config_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            
            if not config_path:
                return False, f"Configuration file not found. Searched: {config_paths}"
            
            logger.info(f"Using config file: {config_path}")
            
            # Load configuration
            self.config = Config(config_path)
            
            # Validate API keys
            required_keys = {
                'GROQ_API_KEY': self.config.groq_api_key,
                'SILICONFLOW_API_KEY': self.config.siliconflow_api_key,
                'QDRANT_URL': self.config.qdrant_url
            }
            
            missing_keys = [k for k, v in required_keys.items() if not v]
            if missing_keys:
                return False, f"Missing required environment variables: {', '.join(missing_keys)}"
            
            # Create config dictionary using your config structure
            rag_config = self.config.rag_config
            
            config_dict = {
                # API keys
                'siliconflow_api_key': self.config.siliconflow_api_key,
                'groq_api_key': self.config.groq_api_key,
                
                # Qdrant configuration
                'qdrant_url': self.config.qdrant_url,
                'qdrant_api_key': self.config.qdrant_api_key,
                'qdrant_collection': 'manufacturing_docs',
                
                # Model configuration from your config.yaml
                'embedding_model': rag_config.get('embedding_model', 'Qwen/Qwen3-Embedding-8B'),
                'reranker_model': rag_config.get('reranker_model', 'Qwen/Qwen3-Reranker-8B'),
                'llm_model': rag_config.get('llm_model', 'openai/gpt-oss-120b'),
                
                # Vector configuration
                'vector_size': 1024,  # Adjust based on your embedding model
                
                # RAG parameters from your config
                'max_context_chunks': rag_config.get('max_context_chunks', 5),
                'similarity_threshold': rag_config.get('similarity_threshold', 0.7),
                'rerank_top_k': rag_config.get('rerank_top_k', 20),
                'final_top_k': rag_config.get('final_top_k', 5),
                
                # Text processing
                'chunk_size': rag_config.get('chunk_size', 512),
                'chunk_overlap': rag_config.get('chunk_overlap', 50),
                'max_context_length': 4000,
                
                # Document processing
                'image_processing': True,
                'table_extraction': True,
                'max_file_size_mb': 100,
                
                # Storage
                'metadata_db_path': './data/metadata.db',
                
                # Performance
                'max_retries': 3,
                'batch_size': 32,
                'enable_caching': True,
                'temperature': 0.1,
                'max_tokens': 1024
            }
            
            # Register document processors
            DocumentProcessorFactory.register_processor(DocumentType.PDF, PDFProcessor)
            DocumentProcessorFactory.register_processor(DocumentType.EXCEL, ExcelProcessor)
            DocumentProcessorFactory.register_processor(DocumentType.IMAGE, ImageProcessor)
            
            # Initialize components with error handling
            try:
                self.metadata_manager = MetadataManager(config_dict)
                logger.info("✅ Metadata manager initialized")
                
                self.ingestion_pipeline = DocumentIngestionPipeline(config_dict)
                logger.info("✅ Ingestion pipeline initialized")
                
                self.rag_engine = RAGEngine(config_dict)
                logger.info("✅ RAG engine initialized")
                
            except Exception as e:
                return False, f"Failed to initialize components: {str(e)}"
            
            self.system_initialized = True
            return True, "RAG system initialized successfully!"
            
        except Exception as e:
            error_msg = f"Failed to initialize RAG system: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def process_uploaded_files(self, files) -> Tuple[str, pd.DataFrame]:
        """Process uploaded files with improved error handling."""
        if not self.system_initialized:
            return "❌ System not initialized. Please initialize first.", pd.DataFrame()
        
        if not files:
            return "No files uploaded.", pd.DataFrame()
        
        results = []
        total_files = len(files)
        
        try:
            for i, file in enumerate(files):
                logger.info(f"Processing file {i+1}/{total_files}: {file.name}")
                
                # Save uploaded file temporarily
                temp_path = None
                try:
                    # Create temporary file with proper extension
                    suffix = Path(file.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        # Read file content
                        file_content = file.read()
                        tmp_file.write(file_content)
                        temp_path = tmp_file.name
                    
                    logger.info(f"Saved temp file: {temp_path}")
                    
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
                    
                    logger.info(f"Processing result: {'Success' if result.success else 'Failed'}")
                    
                except Exception as e:
                    logger.error(f"Error processing {file.name}: {e}")
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
                        try:
                            os.unlink(temp_path)
                            logger.info(f"Cleaned up temp file: {temp_path}")
                        except Exception as e:
                            logger.warning(f"Failed to clean up temp file: {e}")
            
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
                    similarity_threshold: float = 0.7) -> Tuple[str, str, pd.DataFrame]:
        """Process a question through the RAG engine with better error handling."""
        if not self.system_initialized:
            return "❌ System not initialized. Please initialize first.", "", pd.DataFrame()
        
        if not question.strip():
            return "Please enter a question.", "", pd.DataFrame()
        
        try:
            try:
                documents = self.metadata_manager.list_documents(
                    status=ProcessingStatus.COMPLETED, 
                    limit=1
                )
                if not documents:
                    return "⚠️ No processed documents available. Please upload and process documents first.", "", pd.DataFrame()
            except Exception as e:
                logger.error(f"Failed to check documents: {e}")
                return "❌ Error checking document availability.", "", pd.DataFrame()
            
            # Update RAG engine config temporarily for this query
            original_final_top_k = self.rag_engine.final_top_k
            original_similarity_threshold = self.rag_engine.similarity_threshold
            
            self.rag_engine.final_top_k = max_results
            self.rag_engine.similarity_threshold = similarity_threshold
            
            # Get response
            logger.info(f"Asking question: {question[:50]}...")
            response = self.rag_engine.answer_question(question)
            
            # Restore original config
            self.rag_engine.final_top_k = original_final_top_k
            self.rag_engine.similarity_threshold = original_similarity_threshold
            
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
    
    
    
    def get_document_library(self):
        if not self.system_initialized:
            return pd.DataFrame({'Message': ['System not initialized']})
        try:
            documents = self.metadata_manager.list_documents(limit=50)
            if not documents:
                return pd.DataFrame({'Message': ['No documents processed yet']})
            doc_data = []
            for doc in documents:
                doc_data.append({
                    'Filename': doc.filename,
                    'Type': doc.file_type.upper(),
                    'Status': doc.processing_status.value.title(),
                    'Chunks': doc.total_chunks,
                    'Size': self._format_size(doc.file_size),
                    'Uploaded': doc.upload_timestamp.strftime('%Y-%m-%d %H:%M')
                })
            return pd.DataFrame(doc_data)
        except Exception as e:
            logger.error(f"Failed to get document library: {e}")
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
            all_health = {**rag_health, **pipeline_health}
            for component, healthy in all_health.items():
                status = "✅ Healthy" if healthy else "❌ Unhealthy"
                status_parts.append(f"**{component.replace('_', ' ').title()}:** {status}")
            
            status_message = "## 🏥 System Health\n" + "\n".join(status_parts)
            
            # Create detailed status table
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
    """Create the main Gradio interface with proper error handling."""
    
    # Initialize demo instance
    demo_instance = RAGGradioDemo()
    
    # Define the interface
    with gr.Blocks(title="Manufacturing RAG Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏭 Manufacturing RAG Agent
        *Intelligent document analysis for manufacturing data*
        
        This system allows you to upload manufacturing documents (PDF, Excel, Images) and ask questions about their content using SiliconFlow embeddings and Groq LLM.
        """)
        
        # System initialization status
        with gr.Row():
            system_status = gr.Markdown("**System Status:** Not initialized")
            init_btn = gr.Button("🚀 Initialize System", variant="primary")
        
        # Main functionality tabs
        with gr.Tabs():
            # Document Upload Tab
            with gr.TabItem("📄 Document Upload"):
                gr.Markdown("### Upload and Process Documents")
                
                with gr.Row():
                    with gr.Column():
                        file_upload = gr.File(
                            file_count="multiple",
                            file_types=[".pdf", ".xlsx", ".xls", ".xlsm", ".png", ".jpg", ".jpeg"],
                            label="Choose files to upload (PDF, Excel, Images)"
                        )
                        upload_btn = gr.Button("🔄 Process Documents", variant="primary")
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
                            placeholder="e.g., What is the production yield mentioned in the documents?",
                            lines=2
                        )
                        ask_btn = gr.Button("🔍 Ask Question", variant="primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### Settings")
                        max_results = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1,
                            label="Max Context Chunks"
                        )
                        similarity_threshold = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.7, step=0.1,
                            label="Similarity Threshold"
                        )
                
                # Answer display
                answer_output = gr.Markdown(label="Answer")
                citations_output = gr.Markdown(label="Citations")
                
                # Performance metrics
                performance_metrics = gr.Dataframe(
                    label="Performance Metrics",
                    interactive=False
                )
            
            # System Status Tab
            with gr.TabItem("⚙️ System Status"):
                gr.Markdown("### System Health & Information")
                
                check_health_btn = gr.Button("🔍 Check System Health")
                health_status = gr.Markdown("Click 'Check System Health' to view status...")
                health_details = gr.Dataframe(
                    label="Component Health Details",
                    interactive=False
                )
        
        # Event handlers
        def initialize_system():
            """Initialize the system and return status."""
            success, message = demo_instance.initialize_system()
            if success:
                return f"**System Status:** <span style='color: green'>✅ {message}</span>"
            else:
                return f"**System Status:** <span style='color: red'>❌ {message}</span>"
        
        def process_files(files):
            """Process uploaded files."""
            if not files:
                return "No files selected", pd.DataFrame()
            return demo_instance.process_uploaded_files(files)
        
        def ask_question(question, max_results, similarity_threshold):
            """Ask a question."""
            if not question.strip():
                return "Please enter a question", "", pd.DataFrame()
            return demo_instance.ask_question(question, max_results, similarity_threshold)
        
        def refresh_library():
            """Refresh document library."""
            return demo_instance.get_document_library()
        
        def check_health():
            """Check system health."""
            return demo_instance.get_system_status()
        
        # Connect events
        init_btn.click(
            initialize_system,
            outputs=[system_status]
        )
        
        upload_btn.click(
            process_files,
            inputs=[file_upload],
            outputs=[upload_status, upload_results]
        )
        
        ask_btn.click(
            ask_question,
            inputs=[question_input, max_results, similarity_threshold],
            outputs=[answer_output, citations_output, performance_metrics]
        )
        
        refresh_docs_btn.click(
            refresh_library,
            outputs=[doc_library]
        )
        
        check_health_btn.click(
            check_health,
            outputs=[health_status, health_details]
        )
        
        # Auto-refresh library after upload
        upload_btn.click(
            refresh_library,
            outputs=[doc_library]
        )
    
    return demo


def main():
    """Main function to launch the Gradio demo."""
    try:
        # Create directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Create and launch the interface
        demo = create_gradio_interface()
        
        # Launch with configuration
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True
        )
        
    except Exception as e:
        print(f"❌ Failed to launch Gradio demo: {e}")
        print("Please check your configuration and dependencies.")


if __name__ == "__main__":
    main()