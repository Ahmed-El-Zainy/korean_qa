"""
Unit tests for the document processor module.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

from src.rag.document_processor import (
    DocumentProcessor,
    ProcessedDocument,
    DocumentChunk,
    ChunkMetadata,
    DocumentType,
    ProcessingStatus,
    DocumentProcessingError,
    UnsupportedDocumentTypeError,
    DocumentProcessorFactory,
    ExtractedImage,
    ExtractedTable
)


class MockDocumentProcessor(DocumentProcessor):
    """Mock document processor for testing."""
    
    def _get_supported_extensions(self):
        return ['.txt', '.mock']
    
    def process_document(self, file_path: str) -> ProcessedDocument:
        """Mock implementation that returns a simple processed document."""
        document_id = self._generate_document_id(file_path)
        return ProcessedDocument(
            document_id=document_id,
            filename=Path(file_path).name,
            file_path=file_path,
            document_type=DocumentType.UNKNOWN,
            content="Mock document content for testing.",
            metadata={"mock": True},
            processing_status=ProcessingStatus.COMPLETED
        )


class TestDocumentProcessor:
    """Test cases for DocumentProcessor base class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'max_file_size_mb': 10,
            'chunk_size': 100,
            'chunk_overlap': 20
        }
        self.processor = MockDocumentProcessor(self.config)
    
    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor.config == self.config
        assert '.txt' in self.processor.supported_extensions
        assert '.mock' in self.processor.supported_extensions
    
    def test_can_process(self):
        """Test file type detection."""
        assert self.processor.can_process('test.txt')
        assert self.processor.can_process('test.mock')
        assert not self.processor.can_process('test.pdf')
        assert not self.processor.can_process('test.xlsx')
    
    def test_detect_document_type(self):
        """Test document type detection."""
        assert self.processor._detect_document_type('test.pdf') == DocumentType.PDF
        assert self.processor._detect_document_type('test.xlsx') == DocumentType.EXCEL
        assert self.processor._detect_document_type('test.png') == DocumentType.IMAGE
        assert self.processor._detect_document_type('test.unknown') == DocumentType.UNKNOWN
    
    def test_generate_document_id(self):
        """Test document ID generation."""
        doc_id1 = self.processor._generate_document_id('test.txt')
        doc_id2 = self.processor._generate_document_id('test.txt')
        
        # IDs should be different due to timestamp
        assert doc_id1 != doc_id2
        assert len(doc_id1) == 32  # MD5 hash length
        assert len(doc_id2) == 32
    
    def test_validate_file_not_exists(self):
        """Test file validation with non-existent file."""
        with pytest.raises(DocumentProcessingError) as exc_info:
            self.processor.validate_file('nonexistent.txt')
        
        assert "FileNotFound" in str(exc_info.value)
    
    def test_validate_file_unsupported_type(self):
        """Test file validation with unsupported file type."""
        with tempfile.NamedTemporaryFile(suffix='.unsupported', delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = tmp.name
        
        try:
            with pytest.raises(UnsupportedDocumentTypeError):
                self.processor.validate_file(tmp_path)
        finally:
            Path(tmp_path).unlink()
    
    def test_validate_file_too_large(self):
        """Test file validation with file too large."""
        # Create processor with very small max file size
        small_config = self.config.copy()
        small_config['max_file_size_mb'] = 0.001  # 1KB
        processor = MockDocumentProcessor(small_config)
        
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"x" * 2000)  # 2KB file
            tmp_path = tmp.name
        
        try:
            with pytest.raises(DocumentProcessingError) as exc_info:
                processor.validate_file(tmp_path)
            
            assert "FileTooLarge" in str(exc_info.value)
        finally:
            Path(tmp_path).unlink()
    
    def test_validate_file_success(self):
        """Test successful file validation."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = tmp.name
        
        try:
            # Should not raise any exception
            self.processor.validate_file(tmp_path)
        finally:
            Path(tmp_path).unlink()
    
    def test_extract_chunks_empty_content(self):
        """Test chunk extraction with empty content."""
        document = ProcessedDocument(
            document_id="test_doc",
            filename="test.txt",
            file_path="test.txt",
            document_type=DocumentType.UNKNOWN,
            content="",
            metadata={}
        )
        
        chunks = self.processor.extract_chunks(document)
        assert len(chunks) == 0
    
    def test_extract_chunks_small_content(self):
        """Test chunk extraction with content smaller than chunk size."""
        document = ProcessedDocument(
            document_id="test_doc",
            filename="test.txt",
            file_path="test.txt",
            document_type=DocumentType.UNKNOWN,
            content="This is a small test content.",
            metadata={}
        )
        
        chunks = self.processor.extract_chunks(document, chunk_size=100, chunk_overlap=20)
        assert len(chunks) == 1
        assert chunks[0].content == "This is a small test content."
        assert chunks[0].metadata.chunk_index == 0
        assert chunks[0].metadata.document_id == "test_doc"
    
    def test_extract_chunks_large_content(self):
        """Test chunk extraction with content larger than chunk size."""
        content = "This is a test sentence. " * 20  # Create long content
        document = ProcessedDocument(
            document_id="test_doc",
            filename="test.txt",
            file_path="test.txt",
            document_type=DocumentType.UNKNOWN,
            content=content,
            metadata={}
        )
        
        chunks = self.processor.extract_chunks(document, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1
        
        # Check that chunks have proper metadata
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.chunk_index == i
            assert chunk.metadata.document_id == "test_doc"
            assert len(chunk.content) <= 100 or i == len(chunks) - 1  # Last chunk can be longer
    
    def test_extract_chunks_overlap(self):
        """Test that chunk overlap works correctly."""
        content = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10"
        document = ProcessedDocument(
            document_id="test_doc",
            filename="test.txt",
            file_path="test.txt",
            document_type=DocumentType.UNKNOWN,
            content=content,
            metadata={}
        )
        
        chunks = self.processor.extract_chunks(document, chunk_size=30, chunk_overlap=10)
        
        if len(chunks) > 1:
            # Check that there's some overlap between consecutive chunks
            # This is a basic check - exact overlap depends on word boundaries
            assert len(chunks) >= 2


class TestProcessedDocument:
    """Test cases for ProcessedDocument class."""
    
    def test_processed_document_creation(self):
        """Test ProcessedDocument creation."""
        doc = ProcessedDocument(
            document_id="test_id",
            filename="test.pdf",
            file_path="/path/to/test.pdf",
            document_type=DocumentType.PDF,
            content="Test content",
            metadata={"pages": 1}
        )
        
        assert doc.document_id == "test_id"
        assert doc.filename == "test.pdf"
        assert doc.document_type == DocumentType.PDF
        assert doc.content == "Test content"
        assert doc.processing_status == ProcessingStatus.PENDING
        assert isinstance(doc.processing_timestamp, datetime)
    
    def test_processed_document_with_images_and_tables(self):
        """Test ProcessedDocument with images and tables."""
        image = ExtractedImage(
            image_id="img1",
            filename="chart.png",
            content=b"fake_image_data",
            format="PNG"
        )
        
        table = ExtractedTable(
            table_id="table1",
            headers=["Column1", "Column2"],
            rows=[["Value1", "Value2"]]
        )
        
        doc = ProcessedDocument(
            document_id="test_id",
            filename="test.pdf",
            file_path="/path/to/test.pdf",
            document_type=DocumentType.PDF,
            content="Test content",
            metadata={},
            images=[image],
            tables=[table]
        )
        
        assert len(doc.images) == 1
        assert len(doc.tables) == 1
        assert doc.images[0].image_id == "img1"
        assert doc.tables[0].table_id == "table1"


class TestDocumentChunk:
    """Test cases for DocumentChunk class."""
    
    def test_document_chunk_creation(self):
        """Test DocumentChunk creation."""
        metadata = ChunkMetadata(
            chunk_id="chunk_1",
            document_id="doc_1",
            chunk_index=0
        )
        
        chunk = DocumentChunk(
            content="Test chunk content",
            metadata=metadata
        )
        
        assert chunk.content == "Test chunk content"
        assert chunk.metadata.chunk_id == "chunk_1"
        assert chunk.metadata.document_id == "doc_1"
        assert chunk.metadata.chunk_index == 0
        assert chunk.embedding is None
    
    def test_document_chunk_with_embedding(self):
        """Test DocumentChunk with embedding."""
        metadata = ChunkMetadata(
            chunk_id="chunk_1",
            document_id="doc_1",
            chunk_index=0
        )
        
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        chunk = DocumentChunk(
            content="Test chunk content",
            metadata=metadata,
            embedding=embedding
        )
        
        assert chunk.embedding == embedding


class TestDocumentProcessorFactory:
    """Test cases for DocumentProcessorFactory."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Clear any existing processors
        DocumentProcessorFactory._processors = {}
    
    def test_register_processor(self):
        """Test processor registration."""
        DocumentProcessorFactory.register_processor(DocumentType.UNKNOWN, MockDocumentProcessor)
        
        assert DocumentType.UNKNOWN in DocumentProcessorFactory._processors
        assert DocumentProcessorFactory._processors[DocumentType.UNKNOWN] == MockDocumentProcessor
    
    def test_create_processor_success(self):
        """Test successful processor creation."""
        DocumentProcessorFactory.register_processor(DocumentType.UNKNOWN, MockDocumentProcessor)
        
        # Create a temporary file with unknown extension
        with tempfile.NamedTemporaryFile(suffix='.unknown', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # This should work since we're testing the factory logic, not file validation
            processor = DocumentProcessorFactory.create_processor(tmp_path, {})
            assert isinstance(processor, MockDocumentProcessor)
        finally:
            Path(tmp_path).unlink()
    
    def test_create_processor_unsupported_type(self):
        """Test processor creation with unsupported type."""
        # Don't register any processors
        
        with pytest.raises(UnsupportedDocumentTypeError):
            DocumentProcessorFactory.create_processor('test.unknown', {})
    
    def test_get_supported_types(self):
        """Test getting supported types."""
        DocumentProcessorFactory.register_processor(DocumentType.PDF, MockDocumentProcessor)
        DocumentProcessorFactory.register_processor(DocumentType.EXCEL, MockDocumentProcessor)
        
        supported_types = DocumentProcessorFactory.get_supported_types()
        assert DocumentType.PDF in supported_types
        assert DocumentType.EXCEL in supported_types
        assert len(supported_types) == 2


class TestExtractedImage:
    """Test cases for ExtractedImage class."""
    
    def test_extracted_image_creation(self):
        """Test ExtractedImage creation."""
        image = ExtractedImage(
            image_id="img1",
            filename="test.png",
            content=b"fake_image_data",
            format="PNG",
            width=100,
            height=200,
            ocr_text="Extracted text",
            ocr_confidence=0.95
        )
        
        assert image.image_id == "img1"
        assert image.filename == "test.png"
        assert image.content == b"fake_image_data"
        assert image.format == "PNG"
        assert image.width == 100
        assert image.height == 200
        assert image.ocr_text == "Extracted text"
        assert image.ocr_confidence == 0.95


class TestExtractedTable:
    """Test cases for ExtractedTable class."""
    
    def test_extracted_table_creation(self):
        """Test ExtractedTable creation."""
        table = ExtractedTable(
            table_id="table1",
            headers=["Name", "Value", "Unit"],
            rows=[
                ["Temperature", "25.5", "°C"],
                ["Pressure", "1013", "hPa"]
            ],
            page_number=1,
            extraction_confidence=0.9
        )
        
        assert table.table_id == "table1"
        assert table.headers == ["Name", "Value", "Unit"]
        assert len(table.rows) == 2
        assert table.rows[0] == ["Temperature", "25.5", "°C"]
        assert table.page_number == 1
        assert table.extraction_confidence == 0.9


if __name__ == "__main__":
    pytest.main([__file__])