import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os 
import sys 


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_processor import (
    DocumentProcessor, DocumentProcessorFactory, ProcessedDocument, 
    DocumentChunk, ProcessingStatus, DocumentType
)
from src.embedding_system import EmbeddingSystem
from src.vector_store import QdrantVectorStore
from src.metadata_manager import MetadataManager, DocumentMetadata
from src.image_processor import ImageProcessor


try:
    from logger.custom_logger import CustomLoggerTracker
    custom_log = CustomLoggerTracker()
    logger = custom_log.get_logger("ingestion_pipeline")

except ImportError:
    # Fallback to standard logging if custom logger not available
    logger = logging.getLogger("ingestion_pipeline")


@dataclass
class IngestionResult:
    """Result of document ingestion."""
    document_id: str
    filename: str
    success: bool
    processing_time: float
    chunks_created: int
    chunks_indexed: int
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class IngestionStats:
    """Statistics for batch ingestion."""
    total_documents: int
    successful_documents: int
    failed_documents: int
    total_chunks: int
    total_processing_time: float
    average_processing_time: float
    documents_by_type: Dict[str, int]
    errors: List[str]



def jina_embeddings(text: str) -> List[float]:
    JINA_API_KEY= "jina_a75b55a8a9524bb697ea016b164211ebF5IduSgA0Ku8lmI0pS9fnXoZ83Su"
    import requests

    headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer jina_a75b55a8a9524bb697ea016b164211ebF5IduSgA0Ku8lmI0pS9fnXoZ83Su'}

    data = {
        "model": "jina-embeddings-v3",
        "task": "retrieval.passage",
        "input": text}

    response = requests.post('https://api.jina.ai/v1/embeddings', headers=headers, json=data)
    return response.json()['data'][0]['embedding']


class DocumentIngestionPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Initialize components
        self.embedding_system = EmbeddingSystem(config)
        self.vector_store = QdrantVectorStore(config)
        self.metadata_manager = MetadataManager(config)
        # Initialize components with correct vector dimensions
        self.vector_size = config.get('vector_size', 1024)  # Match Jina's dimension
        self.config['vector_size'] = self.vector_size  # Update config for other components
        
        # Initialize image processor for OCR
        self.image_processor = ImageProcessor(config)
        
        # Pipeline settings
        self.chunk_size = config.get('chunk_size', 512)
        self.chunk_overlap = config.get('chunk_overlap', 50)
        self.batch_size = config.get('embedding_batch_size', 32)
        self.max_workers = config.get('max_workers', 4)
        self.enable_ocr = config.get('image_processing', True)
        
        logger.info(f"Document ingestion pipeline initialized")
    
    def ingest_document(self, file_path: str, document_id: Optional[str] = None) -> IngestionResult:
        """
        Ingest a single document through the complete pipeline.
        
        Args:
            file_path: Path to the document file
            document_id: Optional custom document ID
            
        Returns:
            IngestionResult with processing details
        """
        start_time = time.time()
        file_path_obj = Path(file_path)
        filename = file_path_obj.name
        
        try:
            logger.info(f"Starting ingestion of document: {filename}")
            
            # Generate document ID if not provided
            if not document_id:
                document_id = self._generate_document_id(file_path)
            
            # Check if document already exists
            existing_metadata = self.metadata_manager.get_document_metadata(document_id)
            if existing_metadata and existing_metadata.processing_status == ProcessingStatus.COMPLETED:
                logger.info(f"Document {filename} already processed, skipping")
                return IngestionResult(
                    document_id=document_id,
                    filename=filename,
                    success=True,
                    processing_time=0.0,
                    chunks_created=existing_metadata.total_chunks,
                    chunks_indexed=existing_metadata.total_chunks,
                    warnings=["Document already processed"]
                )
            
            # Step 1: Process document
            processed_doc = self._process_document(file_path)
            if processed_doc.processing_status == ProcessingStatus.FAILED:
                return IngestionResult(
                    document_id=document_id,
                    filename=filename,
                    success=False,
                    processing_time=time.time() - start_time,
                    chunks_created=0,
                    chunks_indexed=0,
                    error_message=processed_doc.error_message
                )
            
            # Step 2: Process images with OCR if enabled
            if self.enable_ocr and processed_doc.images:
                processed_doc.images = self.image_processor.batch_process_images(processed_doc.images)
            
            # Step 3: Create document chunks
            processor = DocumentProcessorFactory.create_processor(file_path, self.config)
            chunks = processor.extract_chunks(processed_doc, self.chunk_size, self.chunk_overlap)
            
            if not chunks:
                logger.warning(f"No chunks created for document: {filename}")
                return IngestionResult(
                    document_id=document_id,
                    filename=filename,
                    success=False,
                    processing_time=time.time() - start_time,
                    chunks_created=0,
                    chunks_indexed=0,
                    error_message="No content chunks could be created"
                )
                
                
                
            
            # Step 4: Generate embeddings
            chunk_texts = [chunk.content for chunk in chunks]
            logger.info(chunk_texts[:2])
            # embeddings = self.embedding_system.generate_embeddings(chunk_texts)
            embeddings = [jina_embeddings(text) for text in chunk_texts]    
            
            
            if not embeddings or len(embeddings) != len(chunks):
                logger.error(f"Embedding generation failed for document: {filename}")
                return IngestionResult(
                    document_id=document_id,
                    filename=filename,
                    success=False,
                    processing_time=time.time() - start_time,
                    chunks_created=len(chunks),
                    chunks_indexed=0,
                    error_message="Failed to generate embeddings"
                )
            
            # Attach embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            
            
            # Step 5: Store in vector database
            vector_success = self.vector_store.add_documents(chunks)
            if not vector_success:
                logger.error(f"Failed to store vectors for document: {filename}")
                return IngestionResult(
                    document_id=document_id,
                    filename=filename,
                    success=False,
                    processing_time=time.time() - start_time,
                    chunks_created=len(chunks),
                    chunks_indexed=0,
                    error_message="Failed to store document vectors"
                )
            
            # Step 6: Store metadata
            processing_time = time.time() - start_time
            metadata = DocumentMetadata(
                document_id=document_id,
                filename=filename,
                file_path=file_path,
                file_type=processed_doc.document_type.value,
                upload_timestamp=processed_doc.processing_timestamp,
                processing_status=ProcessingStatus.COMPLETED,
                total_chunks=len(chunks),
                file_size=processed_doc.file_size,
                checksum=processed_doc.checksum,
                processing_time=processing_time,
                metadata_json=self._serialize_metadata(processed_doc.metadata)
            )
            
            metadata_success = self.metadata_manager.store_document_metadata(document_id, metadata)
            if not metadata_success:
                logger.warning(f"Failed to store metadata for document: {filename}")
            
            logger.info(f"Successfully ingested document {filename}: {len(chunks)} chunks in {processing_time:.2f}s")
            
            return IngestionResult(
                document_id=document_id,
                filename=filename,
                success=True,
                processing_time=processing_time,
                chunks_created=len(chunks),
                chunks_indexed=len(chunks)
            )
            
        except Exception as e:
            error_msg = f"Ingestion failed for {filename}: {str(e)}"
            logger.error(error_msg)
            
            # Update metadata with error status
            if document_id:
                self.metadata_manager.update_document_status(
                    document_id, 
                    ProcessingStatus.FAILED, 
                    error_msg,
                    time.time() - start_time
                )
            
            return IngestionResult(
                document_id=document_id or "unknown",
                filename=filename,
                success=False,
                processing_time=time.time() - start_time,
                chunks_created=0,
                chunks_indexed=0,
                error_message=error_msg
            )
    
    def ingest_batch(self, file_paths: List[str], max_workers: Optional[int] = None) -> IngestionStats:
        """
        Ingest multiple documents in parallel.
        
        Args:
            file_paths: List of file paths to process
            max_workers: Maximum number of worker threads
            
        Returns:
            IngestionStats with batch processing results
        """
        start_time = time.time()
        max_workers = max_workers or self.max_workers
        
        logger.info(f"Starting batch ingestion of {len(file_paths)} documents with {max_workers} workers")
        
        results = []
        errors = []
        documents_by_type = {}
        
        # Process documents in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.ingest_document, file_path): file_path 
                for file_path in file_paths
            }
            
            # Collect results
            for future in as_completed(future_to_path):
                file_path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Track document types
                    file_ext = Path(file_path).suffix.lower()
                    documents_by_type[file_ext] = documents_by_type.get(file_ext, 0) + 1
                    
                    if not result.success:
                        errors.append(f"{result.filename}: {result.error_message}")
                        
                except Exception as e:
                    error_msg = f"Failed to process {file_path}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
        
        # Calculate statistics
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        total_processing_time = time.time() - start_time
        total_chunks = sum(r.chunks_indexed for r in successful_results)
        avg_processing_time = (
            sum(r.processing_time for r in results) / len(results) 
            if results else 0.0
        )
        
        stats = IngestionStats(
            total_documents=len(file_paths),
            successful_documents=len(successful_results),
            failed_documents=len(failed_results),
            total_chunks=total_chunks,
            total_processing_time=total_processing_time,
            average_processing_time=avg_processing_time,
            documents_by_type=documents_by_type,
            errors=errors
        )
        
        logger.info(f"Batch ingestion completed: {stats.successful_documents}/{stats.total_documents} "
                   f"documents processed successfully in {total_processing_time:.2f}s")
        
        return stats
    
    def reprocess_document(self, document_id: str) -> IngestionResult:
        """
        Reprocess an existing document.
        
        Args:
            document_id: ID of the document to reprocess
            
        Returns:
            IngestionResult with reprocessing details
        """
        # Get existing metadata
        metadata = self.metadata_manager.get_document_metadata(document_id)
        if not metadata:
            return IngestionResult(
                document_id=document_id,
                filename="unknown",
                success=False,
                processing_time=0.0,
                chunks_created=0,
                chunks_indexed=0,
                error_message="Document not found in metadata"
            )
        
        # Delete existing vectors
        self.vector_store.delete_document(document_id)
        
        # Reprocess the document
        return self.ingest_document(metadata.file_path, document_id)
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all associated data.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete from vector store
            vector_success = self.vector_store.delete_document(document_id)
            
            # Delete from metadata
            metadata_success = self.metadata_manager.delete_document(document_id)
            
            success = vector_success and metadata_success
            if success:
                logger.info(f"Successfully deleted document: {document_id}")
            else:
                logger.warning(f"Partial deletion of document: {document_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    def _process_document(self, file_path: str) -> ProcessedDocument:
        try:
            processor = DocumentProcessorFactory.create_processor(file_path, self.config)
            return processor.process_document(file_path)
            
        except Exception as e:
            logger.error(f"Document processing failed for {file_path}: {e}")
            
            # Return failed document
            document_id = self._generate_document_id(file_path)
            return ProcessedDocument(
                document_id=document_id,
                filename=Path(file_path).name,
                file_path=file_path,
                document_type=DocumentType.UNKNOWN,
                content="",
                metadata={},
                processing_status=ProcessingStatus.FAILED,
                error_message=str(e)
            )
    
    def _generate_document_id(self, file_path: str) -> str:
        # Use file path and modification time for uniqueness
        file_path_obj = Path(file_path)
        if file_path_obj.exists():
            mtime = file_path_obj.stat().st_mtime
            content = f"{file_path}_{mtime}"
        else:
            content = f"{file_path}_{time.time()}"
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def _serialize_metadata(self, metadata: Dict[str, Any]) -> str:
        try:
            import json
            return json.dumps(metadata, default=str, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to serialize metadata: {e}")
            return "{}"
        
        
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the ingestion pipeline.
        
        Returns:
            Dictionary with pipeline statistics
        """
        try:
            # Get component statistics
            vector_stats = self.vector_store.get_collection_info()
            metadata_stats = self.metadata_manager.get_statistics()
            embedding_stats = self.embedding_system.get_cache_stats()
            
            return {
                "vector_store": vector_stats.__dict__ if vector_stats else {},
                "metadata_manager": metadata_stats,
                "embedding_system": embedding_stats,
                "pipeline_config": {
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "batch_size": self.batch_size,
                    "max_workers": self.max_workers,
                    "enable_ocr": self.enable_ocr
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get pipeline stats: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, bool]:
        """
        Check health of all pipeline components.
        
        Returns:
            Dictionary with health status of each component
        """
        return {
            "vector_store": self.vector_store.health_check(),
            "metadata_manager": True,  # SQLite is always available if file system works
            "embedding_system": True   # Will be checked during actual usage
        }



if __name__=="__main__":
    logger.info(f"Ingestion Pipe init ..")
    
    ## Example usage
    import yaml
    with open("src/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    pipeline = DocumentIngestionPipeline(config)
    stats = pipeline.get_pipeline_stats()
    logger.info(f"Pipeline stats: {stats}")
    # Example single document ingestion
    result = pipeline.ingest_document("data/documents/3.수불확인등록.xlsx")
    logger.info(f"Ingestion result: {result}")
    # Example batch ingestion
    # batch_result = pipeline.ingest_batch(["sample_data/sample.pdf", "sample_data/sample.docx"])
    # logger.info(f"Batch ingestion stats: {batch_result}")
    