
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import hashlib
import sys 
import os 



sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Ensure current directory is in

try:
    from logger.custom_logger import CustomLoggerTracker
    custom_log = CustomLoggerTracker()
    logger = custom_log.get_logger("documents_processor")

except ImportError:
    # Fallback to standard logging if custom logger not available
    logger = logging.getLogger("documents_processor")



class DocumentType(Enum):
    """Supported document types."""
    PDF = "pdf"
    EXCEL = "excel"
    IMAGE = "image"
    UNKNOWN = "unknown"


class ProcessingStatus(Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class ExtractedImage:
    """Represents an image extracted from a document."""
    image_id: str
    filename: str
    content: bytes
    format: str  # PNG, JPEG, etc.
    width: Optional[int] = None
    height: Optional[int] = None
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    extraction_method: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedTable:
    """Represents a table extracted from a document."""
    table_id: str
    headers: List[str]
    rows: List[List[str]]
    page_number: Optional[int] = None
    worksheet_name: Optional[str] = None
    cell_range: Optional[str] = None
    extraction_confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""
    chunk_id: str
    document_id: str
    chunk_index: int
    page_number: Optional[int] = None
    worksheet_name: Optional[str] = None
    cell_range: Optional[str] = None
    section_title: Optional[str] = None
    image_references: List[str] = field(default_factory=list)
    table_references: List[str] = field(default_factory=list)
    extraction_timestamp: datetime = field(default_factory=datetime.now)
    confidence_score: Optional[float] = None


@dataclass
class DocumentChunk:
    content: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Validate chunk content after initialization."""
        if not self.content.strip():
            logger.warning(f"Empty content in chunk {self.metadata.chunk_id}")
        
        if len(self.content) > 10000:  # Warn for very large chunks
            logger.warning(f"Large chunk detected ({len(self.content)} chars): {self.metadata.chunk_id}")


@dataclass
class ProcessedDocument:
    """Represents a fully processed document with all extracted content."""
    document_id: str
    filename: str
    file_path: str
    document_type: DocumentType
    content: str
    metadata: Dict[str, Any]
    images: List[ExtractedImage] = field(default_factory=list)
    tables: List[ExtractedTable] = field(default_factory=list)
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    processing_timestamp: datetime = field(default_factory=datetime.now)
    file_size: int = 0
    checksum: str = ""
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Generate checksum and validate document after initialization."""
        if not self.checksum and Path(self.file_path).exists():
            self.checksum = self._generate_checksum()
            self.file_size = Path(self.file_path).stat().st_size
    
    def _generate_checksum(self) -> str:
        try:
            hash_md5 = hashlib.md5()
            with open(self.file_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Failed to generate checksum for {self.file_path}: {e}")
            return ""


class DocumentProcessingError(Exception):
    """Base exception for document processing errors."""
    
    def __init__(self, file_path: str, error_type: str, details: str):
        self.file_path = file_path
        self.error_type = error_type
        self.details = details
        super().__init__(f"Document processing error in {file_path}: {error_type} - {details}")


class UnsupportedDocumentTypeError(DocumentProcessingError):
    def __init__(self, file_path: str, detected_type: str):
        super().__init__(
            file_path, 
            "UnsupportedDocumentType", 
            f"Document type '{detected_type}' is not supported"
        )


class DocumentProcessor(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.supported_extensions = self._get_supported_extensions()
        logger.info(f"Initialized {self.__class__.__name__} with config: {config}")
    
    @abstractmethod
    def _get_supported_extensions(self) -> List[str]:
        pass
    
    
    @abstractmethod
    def process_document(self, file_path: str) -> ProcessedDocument:
        
        pass
    
    def can_process(self, file_path: str) -> bool:
        file_extension = Path(file_path).suffix.lower()
        return file_extension in self.supported_extensions
    
    
    def extract_chunks(self, document: ProcessedDocument, chunk_size: int = 512, 
                      chunk_overlap: int = 50) -> List[DocumentChunk]:
        if not document.content.strip():
            logger.warning(f"No content to chunk in document {document.document_id}")
            return []
        
        chunks = []
        content = document.content
        start = 0
        chunk_index = 0
        
        while start < len(content):
            # Calculate end position
            end = min(start + chunk_size, len(content))
            
            # Try to break at word boundary if not at end of content
            if end < len(content):
                # Look for the last space within the chunk
                last_space = content.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            # Extract chunk content
            chunk_content = content[start:end].strip()
            
            if chunk_content:  # Only create chunk if it has content
                # Create chunk metadata
                metadata = ChunkMetadata(
                    chunk_id=f"{document.document_id}_chunk_{chunk_index}",
                    document_id=document.document_id,
                    chunk_index=chunk_index
                )
                
                # Create chunk
                chunk = DocumentChunk(
                    content=chunk_content,
                    metadata=metadata
                )
                
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = max(end - chunk_overlap, start + 1)
            
            # Prevent infinite loop
            if start >= end:
                break
        
        logger.info(f"Created {len(chunks)} chunks from document {document.document_id}")
        return chunks
    
    def _detect_document_type(self, file_path: str) -> DocumentType:
        extension = Path(file_path).suffix.lower()
        
        if extension == '.pdf':
            return DocumentType.PDF
        elif extension in ['.xlsx', '.xls', '.xlsm']:
            return DocumentType.EXCEL
        elif extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
            return DocumentType.IMAGE
        else:
            return DocumentType.UNKNOWN
    
    def _generate_document_id(self, file_path: str) -> str:
        """
        Generate a unique document ID based on file path and timestamp.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Unique document ID string
        """
        file_name = Path(file_path).name
        timestamp = datetime.now().isoformat()
        content = f"{file_name}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def validate_file(self, file_path: str) -> None:
        """
        Validate that a file exists and can be processed.
        
        Args:
            file_path: Path to the file to validate
            
        Raises:
            DocumentProcessingError: If file validation fails
        """
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            raise DocumentProcessingError(
                file_path, 
                "FileNotFound", 
                f"File does not exist: {file_path}"
            )
        
        if not file_path_obj.is_file():
            raise DocumentProcessingError(
                file_path, 
                "NotAFile", 
                f"Path is not a file: {file_path}"
            )
        
        # Check file size
        max_size = self.config.get('max_file_size_mb', 100) * 1024 * 1024  # Convert to bytes
        file_size = file_path_obj.stat().st_size
        
        if file_size > max_size:
            raise DocumentProcessingError(
                file_path, 
                "FileTooLarge", 
                f"File size ({file_size} bytes) exceeds maximum allowed size ({max_size} bytes)"
            )
        
        if not self.can_process(file_path):
            detected_type = self._detect_document_type(file_path)
            raise UnsupportedDocumentTypeError(file_path, detected_type.value)
        
        logger.debug(f"File validation passed for: {file_path}")


class DocumentProcessorFactory:
    """Factory class for creating appropriate document processors."""
    
    _processors = {}
    
    @classmethod
    def register_processor(cls, document_type: DocumentType, processor_class):
        """Register a processor class for a document type."""
        cls._processors[document_type] = processor_class
        logger.info(f"Registered processor {processor_class.__name__} for type {document_type.value}")
    
    @classmethod
    def create_processor(cls, file_path: str, config: Dict[str, Any]) -> DocumentProcessor:
        """
        Create appropriate processor for the given file.
        
        Args:
            file_path: Path to the file to process
            config: Configuration dictionary
            
        Returns:
            DocumentProcessor instance
            
        Raises:
            UnsupportedDocumentTypeError: If no processor is available for the file type
        """
        # Detect document type
        extension = Path(file_path).suffix.lower()
        
        if extension == '.pdf':
            document_type = DocumentType.PDF
        elif extension in ['.xlsx', '.xls', '.xlsm']:
            document_type = DocumentType.EXCEL
        elif extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
            document_type = DocumentType.IMAGE
        else:
            document_type = DocumentType.UNKNOWN
        
        # Get processor class
        processor_class = cls._processors.get(document_type)
        if not processor_class:
            raise UnsupportedDocumentTypeError(file_path, document_type.value)
        
        # Create and return processor instance
        return processor_class(config)
    
    @classmethod
    def get_supported_types(cls) -> List[DocumentType]:
        """Get list of supported document types."""
        return list(cls._processors.keys())




if __name__=="__main__":
    logger.info(f"Docs processor init ..")
    # Example usage (for testing purposes)
    config = {'max_file_size_mb': 50}
    processor = DocumentProcessorFactory.create_processor("example.pdf", config)
    processed_doc = processor.process_document("example.pdf")
    chunks = processor.extract_chunks(processed_doc)
    for chunk in chunks:
        print(chunk)
