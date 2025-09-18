import gradio as gr
import pandas as pd
import plotly.express as px
from pathlib import Path
import tempfile
import time
import logging
import os
import sys
import shutil
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
    from src.ingestion_pipeline import DocumentIngestionPipeline
    from src.rag_engine import RAGEngine
    from src.metadata_manager import MetadataManager
    from src.document_processor import ProcessingStatus, DocumentProcessorFactory, DocumentType
    from src.pdf_processor import PDFProcessor
    from src.excel_processor import ExcelProcessor
    from src.image_processor import ImageProcessor
    
except ImportError as e:
    logger.error(f"Failed to import RAG components: {e}")
    print(f"❌ Import Error: {e}")
    print("Please ensure all src/ modules are properly structured")
    sys.exit(1)

class RAGGradioDemo:
    """Fixed Gradio demo for Manufacturing RAG Agent with proper file handling."""
    
    def __init__(self):
        self.system_initialized = False
        self.rag_engine = None
        self.ingestion_pipeline = None
        self.metadata_manager = None
        self.chat_history = []
    
    def initialize_system(self):
        """Initialize the RAG system."""
        try:
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
                return "❌ Configuration file not found. Please ensure src/config.yaml exists."
            
            logger.info(f"Using config file: {config_path}")
            
            # Load configuration
            config = Config(config_path)
            
            # Validate API keys
            if not config.groq_api_key:
                return "❌ Missing GROQ_API_KEY in environment variables"
            if not config.siliconflow_api_key:
                return "❌ Missing SILICONFLOW_API_KEY in environment variables"
            if not config.qdrant_url:
                return "❌ Missing QDRANT_URL in environment variables"
            
            # Create configuration dictionary
            rag_config = config.rag_config
            config_dict = {
                'siliconflow_api_key': config.siliconflow_api_key,
                'groq_api_key': config.groq_api_key,
                'qdrant_url': config.qdrant_url,
                'qdrant_api_key': config.qdrant_api_key,
                'qdrant_collection': 'manufacturing_docs',
                'embedding_model': rag_config.get('embedding_model', 'Qwen/Qwen3-Embedding-8B'),
                'reranker_model': rag_config.get('reranker_model', 'Qwen/Qwen3-Reranker-8B'),
                'llm_model': rag_config.get('llm_model', 'openai/gpt-oss-120b'),
                'vector_size': 1024,  # Updated to match Qwen/Qwen3-Embedding-8B actual dimensions
                'max_context_chunks': rag_config.get('max_context_chunks', 5),
                'similarity_threshold': rag_config.get('similarity_threshold', 0.7),
                'chunk_size': rag_config.get('chunk_size', 512),
                'chunk_overlap': rag_config.get('chunk_overlap', 50),
                'metadata_db_path': './data/metadata.db',
                'max_retries': 3,
                'rerank_top_k': 20,
                'final_top_k': 5
            }
            
            # Register processors
            DocumentProcessorFactory.register_processor(DocumentType.PDF, PDFProcessor)
            DocumentProcessorFactory.register_processor(DocumentType.EXCEL, ExcelProcessor)
            DocumentProcessorFactory.register_processor(DocumentType.IMAGE, ImageProcessor)
            
            # Initialize components
            self.metadata_manager = MetadataManager(config_dict)
            self.ingestion_pipeline = DocumentIngestionPipeline(config_dict)
            self.rag_engine = RAGEngine(config_dict)
            
            self.system_initialized = True
            return "✅ System initialized successfully!"
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return f"❌ Initialization failed: {str(e)}"
    
    def process_files(self, files):
        if not self.system_initialized:
            return "❌ System not initialized", pd.DataFrame()
        
        if not files:
            return "No files uploaded", pd.DataFrame()
        
        results = []
        
        for i, file_obj in enumerate(files):
            try:
                logger.info(f"Processing file {i+1}/{len(files)}: {file_obj}")
                
                # Handle different types of file objects from Gradio
                file_path = None
                temp_path = None
                
                # Check if file_obj is a path string
                if isinstance(file_obj, str):
                    file_path = file_obj
                    filename = os.path.basename(file_path)
                # Check if it's a file-like object with a name
                elif hasattr(file_obj, 'name'):
                    file_path = file_obj.name
                    filename = os.path.basename(file_path)
                # Check if it's a tuple/list (Gradio sometimes returns tuples)
                elif isinstance(file_obj, (tuple, list)) and len(file_obj) > 0:
                    file_path = file_obj[0] if isinstance(file_obj[0], str) else file_obj[0].name
                    filename = os.path.basename(file_path)
                else:
                    logger.error(f"Unknown file object type: {type(file_obj)}")
                    results.append({
                        'Filename': f'Unknown file {i+1}',
                        'Status': '❌ Failed',
                        'Chunks': 0,
                        'Time': '0.00s',
                        'Error': 'Unknown file object type'
                    })
                    continue
                
                if not file_path or not os.path.exists(file_path):
                    logger.error(f"File path does not exist: {file_path}")
                    results.append({
                        'Filename': filename if 'filename' in locals() else f'File {i+1}',
                        'Status': '❌ Failed',
                        'Chunks': 0,
                        'Time': '0.00s',
                        'Error': 'File path not found'
                    })
                    continue
                
                logger.info(f"Processing file: {filename} from path: {file_path}")
                
                # Create a temporary copy if needed (to avoid issues with Gradio's temp files)
                suffix = Path(filename).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    shutil.copy2(file_path, tmp.name)
                    temp_path = tmp.name
                
                # Process the document
                start_time = time.time()
                result = self.ingestion_pipeline.ingest_document(temp_path)
                processing_time = time.time() - start_time
                
                results.append({
                    'Filename': filename,
                    'Status': '✅ Success' if result.success else '❌ Failed',
                    'Chunks': result.chunks_indexed if result.success else 0,
                    'Time': f"{processing_time:.2f}s",
                    'Error': result.error_message if not result.success else 'None'
                })
                
                logger.info(f"{'Success' if result.success else 'Failed'}: {filename}")
                
            except Exception as e:
                logger.error(f"Error processing file {i+1}: {e}")
                results.append({
                    'Filename': f'File {i+1}',
                    'Status': '❌ Failed',
                    'Chunks': 0,
                    'Time': '0.00s',
                    'Error': str(e)
                })
            
            finally:
                # Clean up temp file
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        logger.warning(f"Failed to clean temp file: {e}")
        
        # Create summary
        successful = sum(1 for r in results if 'Success' in r['Status'])
        total_chunks = sum(r['Chunks'] for r in results if isinstance(r['Chunks'], int))
        
        status = f"✅ Processed {successful}/{len(results)} files successfully. Total chunks: {total_chunks}"
        
        return status, pd.DataFrame(results)
    
    def ask_question(self, question, max_results=5, threshold=0.7):
        """Ask a question to the RAG system."""
        if not self.system_initialized:
            return "❌ System not initialized", "", pd.DataFrame()
        
        if not question.strip():
            return "Please enter a question", "", pd.DataFrame()
        
        try:
            # Check for documents
            docs = self.metadata_manager.list_documents(status=ProcessingStatus.COMPLETED, limit=1)
            if not docs:
                return "⚠️ No processed documents available. Please upload documents first.", "", pd.DataFrame()
            
            # Update RAG settings temporarily
            original_final_top_k = self.rag_engine.final_top_k
            original_threshold = self.rag_engine.similarity_threshold
            
            self.rag_engine.final_top_k = max_results
            self.rag_engine.similarity_threshold = threshold
            
            # Get answer
            logger.info(f"Processing question: {question[:50]}...")
            response = self.rag_engine.answer_question(question)
            
            # Restore settings
            self.rag_engine.final_top_k = original_final_top_k
            self.rag_engine.similarity_threshold = original_threshold
            
            if not response.success:
                return f"❌ {response.error_message}", "", pd.DataFrame()
            
            # Format citations
            citations = "## 📚 Sources & Citations\n\n"
            for i, citation in enumerate(response.citations):
                citations += f"**{i+1}.** {citation.source_file}\n"
                if citation.page_number:
                    citations += f"📄 Page {citation.page_number}\n"
                if citation.worksheet_name:
                    citations += f"📊 Sheet: {citation.worksheet_name}\n"
                citations += f"*Excerpt:* \"{citation.text_snippet[:100]}...\"\n\n"
            
            # Performance metrics
            metrics = pd.DataFrame({
                'Metric': ['Confidence Score', 'Processing Time (s)', 'Sources Used', 'Chunks Retrieved'],
                'Value': [
                    f"{response.confidence_score:.3f}",
                    f"{response.processing_time:.2f}",
                    len(response.citations),
                    response.total_chunks_retrieved
                ]
            })
            
            return response.answer, citations, metrics
            
        except Exception as e:
            logger.error(f"Question processing failed: {e}")
            return f"❌ Error: {str(e)}", "", pd.DataFrame()
    
    def get_document_library(self):
        """Get list of processed documents."""
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
    
    def _format_size(self, size_bytes):
        """Format file size."""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f}{size_names[i]}"

def create_interface():
    """Create the Gradio interface."""
    demo = RAGGradioDemo()
    
    with gr.Blocks(title="Manufacturing RAG Agent", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🏭 Manufacturing RAG Agent
        *Upload documents and ask questions about manufacturing data*
        
        **Supports:** PDF, Excel (.xlsx, .xls), Images (.png, .jpg, .jpeg)
        """)
        
        # System initialization
        with gr.Row():
            init_btn = gr.Button("🚀 Initialize System", variant="primary")
            status_display = gr.Textbox("System not initialized", label="System Status", interactive=False)
        
        with gr.Tabs():
            # Document Upload Tab
            with gr.TabItem("📄 Document Upload"):
                gr.Markdown("### Upload and Process Documents")
                
                with gr.Column():
                    file_input = gr.File(
                        file_count="multiple",
                        file_types=[".pdf", ".xlsx", ".xls", ".xlsm", ".png", ".jpg", ".jpeg"],
                        label="Upload Documents"
                    )
                    upload_btn = gr.Button("🔄 Process Documents", variant="primary")
                    
                    upload_status = gr.Textbox(
                        label="Processing Status",
                        interactive=False,
                        lines=2
                    )
                    
                    upload_results = gr.Dataframe(
                        label="Processing Results",
                        interactive=False
                    )
                
                gr.Markdown("### 📚 Document Library")
                refresh_btn = gr.Button("🔄 Refresh Library")
                doc_library = gr.Dataframe(
                    label="Processed Documents",
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
                            lines=3
                        )
                        ask_btn = gr.Button("🔍 Ask Question", variant="primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### Settings")
                        max_results = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1,
                            label="Max Context Chunks"
                        )
                        similarity_threshold = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.7, step=0.05,
                            label="Similarity Threshold"
                        )
                
                # Answer display
                answer_output = gr.Markdown(label="Answer")
                citations_output = gr.Markdown(label="Citations")
                performance_metrics = gr.Dataframe(
                    label="Performance Metrics",
                    interactive=False
                )
        
        # Event handlers
        init_btn.click(
            demo.initialize_system,
            outputs=[status_display]
        )
        
        upload_btn.click(
            demo.process_files,
            inputs=[file_input],
            outputs=[upload_status, upload_results]
        )
        
        ask_btn.click(
            demo.ask_question,
            inputs=[question_input, max_results, similarity_threshold],
            outputs=[answer_output, citations_output, performance_metrics]
        )
        
        refresh_btn.click(
            demo.get_document_library,
            outputs=[doc_library]
        )
        
        # Auto-refresh library after upload
        upload_btn.click(
            demo.get_document_library,
            outputs=[doc_library]
        )
    
    return app


def main():
    """Launch the application."""
    try:
        # Create necessary directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Create interface
        app = create_interface()
        
        # Launch
        print("🏭 Launching Manufacturing RAG Agent...")
        print("📱 Interface will be available at: http://localhost:7860")
        print("🛑 Press Ctrl+C to stop")
        
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            debug=True,
            show_error=True
        )
        
    except Exception as e:
        print(f"❌ Failed to launch: {e}")

if __name__ == "__main__":
    main()