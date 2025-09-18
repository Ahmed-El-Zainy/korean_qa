#!/usr/bin/env python3
"""
Direct Document Loading Script for RAG Pipeline
This script loads documents directly from a data directory into the RAG system
and provides an interactive question-answering interface.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.config import Config
    from src.ingestion_pipeline import DocumentIngestionPipeline, IngestionResult
    from src.rag_engine import RAGEngine, RAGResponse
    from src.metadata_manager import MetadataManager
    from src.vector_store import QdrantVectorStore, QdrantClient
    from src.embedding_system import EmbeddingSystem, RerankResult
    from logger.custom_logger import CustomLoggerTracker
    from src.document_processor import ProcessingStatus, DocumentProcessorFactory, DocumentType
    from src.pdf_processor import PDFProcessor
    from src.excel_processor import ExcelProcessor
    from src.image_processor import ImageProcessor
    
    # Initialize logger
    custom_log = CustomLoggerTracker()
    logger = custom_log.get_logger("direct_rag_loader")
    
except ImportError as e:
    print(f"Failed to import RAG components: {e}")
    print("Please ensure all src/ modules are available and properly structured.")
    sys.exit(1)


class DirectRAGLoader:
    """
    Direct document loader for RAG system.
    Loads documents from a specified directory and enables question answering.
    """
    
    def __init__(self, data_directory: str = "data", config_path: str = "src/config.yaml"):
        """
        Initialize the RAG loader.
        
        Args:
            data_directory: Directory containing documents to load
            config_path: Path to configuration file
        """
        self.data_directory = Path(data_directory)
        self.config_path = config_path
        
        # RAG components
        self.config = None
        self.ingestion_pipeline = None
        self.rag_engine = None
        self.metadata_manager = None
        
        # Document tracking
        self.loaded_documents = []
        self.processing_results = []
        
        logger.info(f"DirectRAGLoader initialized for directory: {self.data_directory}")
    
    def initialize_system(self) -> bool:
        """
        Initialize the RAG system components.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Initializing RAG system...")
            
            # Check if config file exists
            if not Path(self.config_path).exists():
                logger.error(f"Configuration file not found: {self.config_path}")
                return False
            
            # Load configuration
            self.config = Config(self.config_path)
            logger.info("Configuration loaded successfully")
            
            # Initialize components with config
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
            self.ingestion_pipeline = DocumentIngestionPipeline(config_dict)
            self.rag_engine = RAGEngine(config_dict)
            self.metadata_manager = MetadataManager(config_dict)
            # Register document processors
            DocumentProcessorFactory.register_processor(DocumentType.PDF, PDFProcessor)
            DocumentProcessorFactory.register_processor(DocumentType.EXCEL, ExcelProcessor)
            DocumentProcessorFactory.register_processor(DocumentType.IMAGE, ImageProcessor)
            
            logger.info("RAG system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
    
    
    def discover_documents(self) -> List[Path]:
        if not self.data_directory.exists():
            logger.error(f"Data directory does not exist: {self.data_directory}")
            return []
        
        # Supported file extensions
        supported_extensions = ['.pdf', '.xlsx', '.xls', '.xlsm', '.png', '.jpg', '.jpeg', '.csv', '.txt']
        
        documents = []
        for ext in supported_extensions:
            documents.extend(self.data_directory.glob(f"*{ext}"))
            documents.extend(self.data_directory.glob(f"**/*{ext}"))  # Recursive search
        
        # Remove duplicates and sort
        documents = sorted(list(set(documents)))
        
        logger.info(f"Found {len(documents)} documents in {self.data_directory}")
        for doc in documents:
            logger.info(f"  - {doc.name} ({doc.suffix})")
        
        return documents
    
    def load_documents(self, document_paths: Optional[List[Path]] = None) -> bool:
        """
        Load documents into the RAG system.
        
        Args:
            document_paths: Optional list of specific documents to load.
                           If None, loads all discovered documents.
        
        Returns:
            True if at least one document was loaded successfully
        """
        if not self.ingestion_pipeline:
            logger.error("RAG system not initialized. Call initialize_system() first.")
            return False
        
        # Discover documents if not provided
        if document_paths is None:
            document_paths = self.discover_documents()
        
        if not document_paths:
            logger.warning("No documents found to load")
            return False
        
        logger.info(f"Starting batch ingestion of {len(document_paths)} documents...")
        
        # Convert Path objects to strings
        file_paths = [str(path) for path in document_paths]
        
        # Process documents in batch
        start_time = time.time()
        batch_stats = self.ingestion_pipeline.ingest_batch(file_paths, max_workers=2)
        
        # Store results
        self.processing_results = batch_stats
        
        # Log results
        logger.info("=" * 60)
        logger.info("BATCH PROCESSING RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total documents: {batch_stats.total_documents}")
        logger.info(f"Successful: {batch_stats.successful_documents}")
        logger.info(f"Failed: {batch_stats.failed_documents}")
        logger.info(f"Total chunks created: {batch_stats.total_chunks}")
        logger.info(f"Processing time: {batch_stats.total_processing_time:.2f}s")
        logger.info(f"Average time per document: {batch_stats.average_processing_time:.2f}s")
        
        if batch_stats.documents_by_type:
            logger.info("Documents by type:")
            for doc_type, count in batch_stats.documents_by_type.items():
                logger.info(f"  {doc_type}: {count}")
        
        if batch_stats.errors:
            logger.warning("Errors encountered:")
            for error in batch_stats.errors:
                logger.warning(f"  - {error}")
        
        logger.info("=" * 60)
        
        return batch_stats.successful_documents > 0
    
    def ask_question(self, question: str, max_results: int = 5, 
                    show_citations: bool = True) -> Optional[RAGResponse]:
        """
        Ask a question to the RAG system.
        
        Args:
            question: Question to ask
            max_results: Maximum number of context chunks to use
            show_citations: Whether to display citations
        
        Returns:
            RAGResponse object or None if failed
        """
        if not self.rag_engine:
            logger.error("RAG system not initialized. Call initialize_system() first.")
            return None
        
        try:
            logger.info(f"Processing question: {question}")
            
            # Temporarily adjust RAG engine parameters
            original_top_k = self.rag_engine.final_top_k
            self.rag_engine.final_top_k = max_results
            
            # Get response
            response = self.rag_engine.answer_question(question)
            
            # Restore original parameter
            self.rag_engine.final_top_k = original_top_k
            
            # Display response
            self._display_response(response, show_citations)
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to process question: {e}")
            return None
    
    def _display_response(self, response: RAGResponse, show_citations: bool = True):
        """Display RAG response in a formatted way."""
        print("\n" + "="*60)
        print("🤖 RAG SYSTEM RESPONSE")
        print("="*60)
        
        if not response.success:
            print(f"❌ Error: {response.error_message}")
            return
        
        # Main answer
        print(f"📝 Answer:")
        print(f"{response.answer}")
        print()
        
        # Metrics
        print(f"📊 Metrics:")
        print(f"  • Confidence Score: {response.confidence_score:.3f}")
        print(f"  • Processing Time: {response.processing_time:.3f}s")
        print(f"  • Sources Used: {len(response.citations)}")
        print(f"  • Chunks Retrieved: {response.total_chunks_retrieved}")
        print(f"  • Model Used: {response.model_used}")
        print()
        
        # Performance breakdown
        print(f"⚡ Performance Breakdown:")
        print(f"  • Retrieval: {response.retrieval_time:.3f}s")
        print(f"  • Reranking: {response.rerank_time:.3f}s")
        print(f"  • Generation: {response.generation_time:.3f}s")
        print()
        
        # Citations
        if show_citations and response.citations:
            print(f"📚 Sources & Citations:")
            for i, citation in enumerate(response.citations, 1):
                print(f"  [{i}] {citation.source_file}")
                
                # Location details
                location_parts = []
                if citation.page_number:
                    location_parts.append(f"Page {citation.page_number}")
                if citation.worksheet_name:
                    location_parts.append(f"Sheet: {citation.worksheet_name}")
                if citation.cell_range:
                    location_parts.append(f"Range: {citation.cell_range}")
                if citation.section_title:
                    location_parts.append(f"Section: {citation.section_title}")
                
                if location_parts:
                    print(f"      📍 {' | '.join(location_parts)}")
                
                print(f"      📈 Confidence: {citation.confidence:.3f}")
                print(f"      📄 Snippet: {citation.text_snippet[:100]}...")
                print()
        
        print("="*60)
    
    def interactive_qa_session(self):
        """Start an interactive question-answering session."""
        print("\n" + "="*60)
        print("🤖 INTERACTIVE Q&A SESSION")
        print("="*60)
        print("Enter your questions below. Type 'quit', 'exit', or 'q' to stop.")
        print("Type 'status' to see system status.")
        print("Type 'docs' to see loaded documents.")
        print("="*60)
        
        while True:
            try:
                # Get user input
                question = input("\n❓ Your question: ").strip()
                
                if not question:
                    continue
                
                # Check for special commands
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif question.lower() == 'status':
                    self._show_system_status()
                    continue
                elif question.lower() == 'docs':
                    self._show_loaded_documents()
                    continue
                
                # Process question
                print("🔍 Processing your question...")
                response = self.ask_question(question, max_results=5, show_citations=True)
                
                if not response:
                    print("❌ Failed to get response. Please try again.")
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
    
    def _show_system_status(self):
        """Display system status information."""
        print("\n" + "="*50)
        print("⚙️ SYSTEM STATUS")
        print("="*50)
        
        try:
            # RAG engine health check
            if self.rag_engine:
                health = self.rag_engine.health_check()
                for component, status in health.items():
                    status_icon = "✅" if status else "❌"
                    print(f"  {component.replace('_', ' ').title()}: {status_icon}")
            
            # Document statistics
            if self.metadata_manager:
                stats = self.metadata_manager.get_statistics()
                print(f"\n📊 Document Statistics:")
                print(f"  Total Documents: {stats.get('total_documents', 0)}")
                print(f"  Total Chunks: {stats.get('total_chunks', 0)}")
                print(f"  Total File Size: {self._format_file_size(stats.get('total_file_size', 0))}")
                
                # Documents by status
                status_counts = stats.get('documents_by_status', {})
                if status_counts:
                    print(f"  By Status:")
                    for status, count in status_counts.items():
                        print(f"    {status}: {count}")
        
        except Exception as e:
            print(f"❌ Error getting system status: {e}")
        
        print("="*50)
    
    def _show_loaded_documents(self):
        """Display loaded documents information."""
        print("\n" + "="*50)
        print("📚 LOADED DOCUMENTS")
        print("="*50)
        
        try:
            if self.metadata_manager:
                documents = self.metadata_manager.list_documents(limit=50)
                
                if not documents:
                    print("No documents loaded yet.")
                    return
                
                for doc in documents:
                    status_icon = "✅" if doc.processing_status == ProcessingStatus.COMPLETED else "❌"
                    print(f"  {status_icon} {doc.filename}")
                    print(f"     Type: {doc.file_type.upper()}")
                    print(f"     Chunks: {doc.total_chunks}")
                    print(f"     Size: {self._format_file_size(doc.file_size)}")
                    print(f"     Status: {doc.processing_status.value}")
                    if doc.error_message:
                        print(f"     Error: {doc.error_message}")
                    print()
        
        except Exception as e:
            print(f"❌ Error getting document list: {e}")
        
        print("="*50)
    
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


def main():
    """Main function to run the direct RAG loader."""
    print("🏭 Manufacturing RAG Agent - Direct Document Loader")
    print("="*60)
    
    # Configuration
    data_directory = "data/documents/"  # Change this to your documents directory
    config_path = "src/config.yaml"  # Change this to your config file path
    
    # Initialize loader
    loader = DirectRAGLoader(data_directory=data_directory, config_path=config_path)
    
    try:
        # Step 1: Initialize system
        print("🔧 Initializing RAG system...")
        if not loader.initialize_system():
            print("❌ Failed to initialize RAG system. Please check your configuration and API keys.")
            return
        
        print("✅ RAG system initialized successfully!")
        
        # Step 2: Load documents
        print("📚 Loading documents...")
        if not loader.load_documents():
            print("❌ Failed to load documents. Please check your data directory and file formats.")
            return
        
        print("✅ Documents loaded successfully!")
        
        # Step 3: Start interactive session
        loader.interactive_qa_session()
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Application error: {e}")
    
    except KeyboardInterrupt:
        print("\n👋 Application interrupted. Goodbye!")


if __name__ == "__main__":
    main()