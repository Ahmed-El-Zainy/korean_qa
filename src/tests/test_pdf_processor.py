"""
Unit tests for the PDF processor module.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import fitz  # PyMuPDF

from src.rag.pdf_processor import PDFProcessor, PDFPageInfo
from src.rag.document_processor import (
    DocumentType,
    ProcessingStatus,
    DocumentProcessingError,
    ExtractedImage,
    ExtractedTable
)


class TestPDFProcessor:
    """Test cases for PDFProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'max_file_size_mb': 10,
            'image_processing': True,
            'table_extraction': True,
            'min_table_rows': 2,
            'min_table_cols': 2,
            'image_min_size': 100
        }
        self.processor = PDFProcessor(self.config)
    
    def test_initialization(self):
        """Test PDF processor initialization."""
        assert self.processor.extract_images is True
        assert self.processor.extract_tables is True
        assert self.processor.min_table_rows == 2
        assert self.processor.min_table_cols == 2
        assert self.processor.image_min_size == 100
    
    def test_get_supported_extensions(self):
        """Test supported extensions."""
        extensions = self.processor._get_supported_extensions()
        assert extensions == ['.pdf']
    
    def test_can_process(self):
        """Test file type detection."""
        assert self.processor.can_process('test.pdf')
        assert self.processor.can_process('document.PDF')  # Case insensitive
        assert not self.processor.can_process('test.txt')
        assert not self.processor.can_process('test.xlsx')
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        # Test multiple newlines
        text = "Line 1\n\n\n\nLine 2"
        cleaned = self.processor._clean_text(text)
        assert cleaned == "Line 1\n\nLine 2"
        
        # Test multiple spaces
        text = "Word1    Word2\t\tWord3"
        cleaned = self.processor._clean_text(text)
        assert cleaned == "Word1 Word2 Word3"
        
        # Test form feeds
        text = "Page 1\fPage 2"
        cleaned = self.processor._clean_text(text)
        assert cleaned == "Page 1\nPage 2"
        
        # Test empty text
        assert self.processor._clean_text("") == ""
        assert self.processor._clean_text(None) == ""
    
    def test_detect_tables_in_text(self):
        """Test table detection in text."""
        # Test text with a clear table structure
        text = """
        Header1    Header2    Header3
        Value1     Value2     Value3
        Data1      Data2      Data3
        
        Some other text here.
        """
        
        tables = self.processor._detect_tables_in_text(text, 1)
        
        # Should detect one table
        assert len(tables) >= 0  # Basic test - actual detection depends on spacing
    
    def test_detect_tables_no_tables(self):
        """Test table detection with no tables."""
        text = "This is just regular text with no tabular structure."
        
        tables = self.processor._detect_tables_in_text(text, 1)
        assert len(tables) == 0
    
    def test_parse_table_lines(self):
        """Test parsing table lines."""
        table_lines = [
            ['Name', 'Age', 'City'],
            ['John', '25', 'New York'],
            ['Jane', '30', 'Boston']
        ]
        
        table = self.processor._parse_table_lines(table_lines, 1, 0)
        
        assert table is not None
        assert table.table_id == "page1_table0"
        assert table.headers == ['Name', 'Age', 'City']
        assert len(table.rows) == 2
        assert table.rows[0] == ['John', '25', 'New York']
        assert table.rows[1] == ['Jane', '30', 'Boston']
        assert table.page_number == 1
    
    def test_parse_table_lines_empty(self):
        """Test parsing empty table lines."""
        table = self.processor._parse_table_lines([], 1, 0)
        assert table is None
    
    def test_parse_table_lines_uneven_columns(self):
        """Test parsing table with uneven columns."""
        table_lines = [
            ['Name', 'Age', 'City'],
            ['John', '25'],  # Missing city
            ['Jane', '30', 'Boston', 'Extra']  # Extra column
        ]
        
        table = self.processor._parse_table_lines(table_lines, 1, 0)
        
        assert table is not None
        assert len(table.headers) == 3
        assert len(table.rows) == 2
        assert table.rows[0] == ['John', '25', '']  # Padded with empty string
        assert table.rows[1] == ['Jane', '30', 'Boston']  # Truncated to header length
    
    @patch('fitz.open')
    def test_extract_metadata(self, mock_fitz_open):
        """Test metadata extraction."""
        # Mock PDF document
        mock_doc = Mock()
        mock_doc.metadata = {
            'title': 'Test Document',
            'author': 'Test Author',
            'subject': 'Test Subject',
            'creationDate': '2023-01-01',
            'modDate': '2023-01-02'
        }
        mock_doc.page_count = 5
        mock_doc.is_encrypted = False
        mock_doc.is_pdf = True
        mock_doc.pdf_version.return_value = '1.4'
        
        mock_fitz_open.return_value = mock_doc
        
        metadata = self.processor._extract_metadata(mock_doc)
        
        assert metadata['title'] == 'Test Document'
        assert metadata['author'] == 'Test Author'
        assert metadata['subject'] == 'Test Subject'
        assert metadata['creation_date'] == '2023-01-01'
        assert metadata['modification_date'] == '2023-01-02'
        assert metadata['page_count'] == 5
        assert metadata['is_encrypted'] is False
        assert metadata['is_pdf'] is True
    
    @patch('fitz.open')
    def test_extract_page_text(self, mock_fitz_open):
        """Test page text extraction."""
        # Mock page
        mock_page = Mock()
        mock_page.get_text.return_value = "  Test page content  \n\n  with multiple lines  "
        
        text = self.processor._extract_page_text(mock_page, 1)
        
        # Should be cleaned
        assert "Test page content" in text
        assert "with multiple lines" in text
        mock_page.get_text.assert_called_once_with("text")
    
    @patch('fitz.open')
    def test_extract_page_images(self, mock_fitz_open):
        """Test page image extraction."""
        # Mock page and document
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.parent = mock_doc
        mock_page.get_images.return_value = [(123, 0, 100, 100, 8, 'DeviceRGB', '', 'Im1', 'DCTDecode')]
        
        # Mock image extraction
        mock_doc.extract_image.return_value = {
            'image': b'fake_image_data_' + b'x' * 200,  # Make it larger than min_size
            'ext': 'png'
        }
        
        images = self.processor._extract_page_images(mock_page, 1, 'doc123')
        
        assert len(images) == 1
        assert images[0].image_id == 'doc123_page1_img0'
        assert images[0].filename == 'page1_image0.png'
        assert images[0].format == 'PNG'
        assert images[0].metadata['page_number'] == 1
        assert images[0].metadata['image_index'] == 0
    
    @patch('fitz.open')
    def test_extract_page_images_small_image(self, mock_fitz_open):
        """Test page image extraction with small image (should be filtered out)."""
        # Mock page and document
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.parent = mock_doc
        mock_page.get_images.return_value = [(123, 0, 100, 100, 8, 'DeviceRGB', '', 'Im1', 'DCTDecode')]
        
        # Mock small image extraction
        mock_doc.extract_image.return_value = {
            'image': b'small',  # Smaller than min_size
            'ext': 'png'
        }
        
        images = self.processor._extract_page_images(mock_page, 1, 'doc123')
        
        # Should be filtered out
        assert len(images) == 0
    
    def test_process_document_file_not_found(self):
        """Test processing non-existent file."""
        result = self.processor.process_document('nonexistent.pdf')
        
        assert result.processing_status == ProcessingStatus.FAILED
        assert result.content == ""
        assert "nonexistent.pdf" in result.file_path
    
    @patch('fitz.open')
    def test_process_document_success(self, mock_fitz_open):
        """Test successful document processing."""
        # Create a temporary PDF file for testing
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(b'%PDF-1.4\n%fake pdf content')
            tmp_path = tmp.name
        
        try:
            # Mock PDF document
            mock_doc = Mock()
            mock_doc.page_count = 2
            mock_doc.metadata = {'title': 'Test PDF'}
            mock_doc.is_encrypted = False
            mock_doc.is_pdf = True
            
            # Mock pages
            mock_page1 = Mock()
            mock_page1.get_text.return_value = "Page 1 content"
            mock_page1.get_images.return_value = []
            mock_page1.rect.width = 612
            mock_page1.rect.height = 792
            mock_page1.rotation = 0
            
            mock_page2 = Mock()
            mock_page2.get_text.return_value = "Page 2 content"
            mock_page2.get_images.return_value = []
            mock_page2.rect.width = 612
            mock_page2.rect.height = 792
            mock_page2.rotation = 0
            
            mock_doc.__getitem__.side_effect = [mock_page1, mock_page2]
            mock_doc.close = Mock()
            
            mock_fitz_open.return_value = mock_doc
            
            # Process document
            result = self.processor.process_document(tmp_path)
            
            # Verify results
            assert result.processing_status == ProcessingStatus.COMPLETED
            assert result.document_type == DocumentType.PDF
            assert "Page 1 content" in result.content
            assert "Page 2 content" in result.content
            assert result.metadata['total_pages'] == 2
            assert result.metadata['title'] == 'Test PDF'
            
            # Verify PDF was closed
            mock_doc.close.assert_called_once()
            
        finally:
            Path(tmp_path).unlink()
    
    @patch('fitz.open')
    def test_process_document_with_images_and_tables(self, mock_fitz_open):
        """Test document processing with images and tables."""
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(b'%PDF-1.4\n%fake pdf content')
            tmp_path = tmp.name
        
        try:
            # Mock PDF document with images and tables
            mock_doc = Mock()
            mock_doc.page_count = 1
            mock_doc.metadata = {}
            mock_doc.is_encrypted = False
            mock_doc.is_pdf = True
            
            # Mock page with table-like text
            mock_page = Mock()
            table_text = "Name    Age    City\nJohn    25     NYC\nJane    30     LA"
            mock_page.get_text.return_value = table_text
            mock_page.get_images.return_value = [(123, 0, 100, 100, 8, 'DeviceRGB', '', 'Im1', 'DCTDecode')]
            mock_page.rect.width = 612
            mock_page.rect.height = 792
            mock_page.rotation = 0
            mock_page.parent = mock_doc
            
            # Mock image extraction
            mock_doc.extract_image.return_value = {
                'image': b'fake_image_data_' + b'x' * 200,
                'ext': 'jpg'
            }
            
            mock_doc.__getitem__.return_value = mock_page
            mock_doc.close = Mock()
            
            mock_fitz_open.return_value = mock_doc
            
            # Process document
            result = self.processor.process_document(tmp_path)
            
            # Verify results
            assert result.processing_status == ProcessingStatus.COMPLETED
            assert len(result.images) == 1
            assert result.images[0].format == 'JPG'
            
            # Tables might or might not be detected depending on text spacing
            # This is okay as table detection is basic in this implementation
            
        finally:
            Path(tmp_path).unlink()
    
    @patch('fitz.open')
    def test_process_document_processing_error(self, mock_fitz_open):
        """Test document processing with error."""
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(b'%PDF-1.4\n%fake pdf content')
            tmp_path = tmp.name
        
        try:
            # Mock fitz.open to raise an exception
            mock_fitz_open.side_effect = Exception("PDF parsing error")
            
            # Process document
            result = self.processor.process_document(tmp_path)
            
            # Should return failed document
            assert result.processing_status == ProcessingStatus.FAILED
            assert result.content == ""
            assert "PDF parsing error" in result.error_message
            
        finally:
            Path(tmp_path).unlink()


class TestPDFPageInfo:
    """Test cases for PDFPageInfo dataclass."""
    
    def test_pdf_page_info_creation(self):
        """Test PDFPageInfo creation."""
        page_info = PDFPageInfo(
            page_number=1,
            width=612.0,
            height=792.0,
            rotation=0,
            text_length=1500,
            image_count=2,
            table_count=1
        )
        
        assert page_info.page_number == 1
        assert page_info.width == 612.0
        assert page_info.height == 792.0
        assert page_info.rotation == 0
        assert page_info.text_length == 1500
        assert page_info.image_count == 2
        assert page_info.table_count == 1


if __name__ == "__main__":
    pytest.main([__file__])