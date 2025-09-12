"""
Unit tests for the image processor module.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import io
import numpy as np

from src.rag.image_processor import ImageProcessor, OCRResult, ImageAnalysis
from src.rag.document_processor import (
    DocumentType,
    ProcessingStatus,
    DocumentProcessingError,
    ExtractedImage
)


class TestImageProcessor:
    """Test cases for ImageProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'max_file_size_mb': 10,
            'ocr_engine': 'tesseract',
            'ocr_language': 'eng',
            'image_preprocessing': True,
            'min_ocr_confidence': 30.0,
            'max_image_size': (3000, 3000),
            'enhance_contrast': True,
            'enhance_sharpness': True
        }
        
        # Mock Tesseract verification to avoid requiring actual installation in tests
        with patch.object(ImageProcessor, '_verify_tesseract'):
            self.processor = ImageProcessor(self.config)
    
    def test_initialization(self):
        """Test image processor initialization."""
        assert self.processor.ocr_engine == 'tesseract'
        assert self.processor.ocr_language == 'eng'
        assert self.processor.preprocessing_enabled is True
        assert self.processor.min_confidence == 30.0
        assert self.processor.max_image_size == (3000, 3000)
        assert self.processor.enhance_contrast is True
        assert self.processor.enhance_sharpness is True
    
    def test_get_supported_extensions(self):
        """Test supported extensions."""
        extensions = self.processor._get_supported_extensions()
        expected_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif']
        
        for ext in expected_extensions:
            assert ext in extensions
    
    def test_can_process(self):
        """Test file type detection."""
        assert self.processor.can_process('test.png')
        assert self.processor.can_process('document.JPG')  # Case insensitive
        assert self.processor.can_process('test.jpeg')
        assert self.processor.can_process('test.gif')
        assert self.processor.can_process('test.bmp')
        assert self.processor.can_process('test.tiff')
        assert not self.processor.can_process('test.txt')
        assert not self.processor.can_process('test.pdf')
    
    def test_is_grayscale_rgb_true(self):
        """Test grayscale detection for RGB image that is actually grayscale."""
        # Create a grayscale RGB image (R=G=B for all pixels)
        image_data = np.full((10, 10, 3), 128, dtype=np.uint8)  # Gray image
        image = Image.fromarray(image_data, 'RGB')
        
        result = self.processor._is_grayscale_rgb(image)
        assert result is True
    
    def test_is_grayscale_rgb_false(self):
        """Test grayscale detection for RGB image with color."""
        # Create a color RGB image
        image_data = np.zeros((10, 10, 3), dtype=np.uint8)
        image_data[:, :, 0] = 255  # Red channel
        image = Image.fromarray(image_data, 'RGB')
        
        result = self.processor._is_grayscale_rgb(image)
        assert result is False
    
    def test_clean_ocr_text(self):
        """Test OCR text cleaning."""
        # Test multiple spaces
        text = "Word1    Word2\t\tWord3"
        cleaned = self.processor._clean_ocr_text(text)
        assert cleaned == "Word1 Word2 Word3"
        
        # Test common OCR artifacts
        text = "He||o W0r|d"
        cleaned = self.processor._clean_ocr_text(text)
        assert "I" in cleaned  # | should be replaced with I
        
        # Test empty text
        assert self.processor._clean_ocr_text("") == ""
        assert self.processor._clean_ocr_text(None) == ""
    
    def test_analyze_image(self):
        """Test image analysis."""
        # Create a test image
        image_data = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        image = Image.fromarray(image_data, 'RGB')
        
        # Create temporary file for analysis
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            analysis = self.processor._analyze_image(image, tmp_path)
            
            assert analysis.width == 200
            assert analysis.height == 100
            assert analysis.format == 'PNG'
            assert analysis.mode == 'RGB'
            assert analysis.size_bytes > 0
            assert isinstance(analysis.average_brightness, float)
            assert isinstance(analysis.contrast_level, float)
            
        finally:
            Path(tmp_path).unlink()
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        # Create a test image
        image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image = Image.fromarray(image_data, 'RGB')
        
        processed_image, steps = self.processor._preprocess_image(image)
        
        # Should have applied some preprocessing steps
        assert len(steps) > 0
        assert 'grayscale_conversion' in steps
        
        # Processed image should be grayscale
        assert processed_image.mode == 'L'
    
    def test_preprocess_image_large(self):
        """Test preprocessing of large image (should be resized)."""
        # Create a large test image
        large_size = (4000, 4000)
        image_data = np.random.randint(0, 255, (*large_size, 3), dtype=np.uint8)
        image = Image.fromarray(image_data, 'RGB')
        
        processed_image, steps = self.processor._preprocess_image(image)
        
        # Should have been resized
        assert 'resize' in steps
        assert processed_image.size[0] <= self.processor.max_image_size[0]
        assert processed_image.size[1] <= self.processor.max_image_size[1]
    
    @patch('pytesseract.image_to_data')
    def test_perform_ocr_success(self, mock_image_to_data):
        """Test successful OCR processing."""
        # Mock Tesseract output
        mock_image_to_data.return_value = {
            'text': ['', 'Hello', 'World', 'Test'],
            'conf': ['-1', '95', '90', '85']
        }
        
        # Create a test image
        image_data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        image = Image.fromarray(image_data, 'L')
        
        result = self.processor._perform_ocr(image)
        
        assert isinstance(result, OCRResult)
        assert 'Hello World Test' in result.text
        assert result.confidence > 0
        assert result.word_count == 3
        assert result.processing_time > 0
    
    @patch('pytesseract.image_to_data')
    def test_perform_ocr_failure(self, mock_image_to_data):
        """Test OCR processing failure."""
        # Mock Tesseract to raise an exception
        mock_image_to_data.side_effect = Exception("Tesseract error")
        
        # Create a test image
        image_data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        image = Image.fromarray(image_data, 'L')
        
        result = self.processor._perform_ocr(image)
        
        assert result.text == ""
        assert result.confidence == 0.0
        assert result.word_count == 0
    
    def test_process_extracted_image(self):
        """Test processing an extracted image."""
        # Create a test image
        image_data = np.full((50, 100, 3), 128, dtype=np.uint8)
        image = Image.fromarray(image_data, 'RGB')
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        image_bytes = img_buffer.getvalue()
        
        # Create extracted image
        extracted_image = ExtractedImage(
            image_id='test_img',
            filename='test.png',
            content=image_bytes,
            format='PNG',
            metadata={}
        )
        
        # Mock OCR to avoid requiring Tesseract
        with patch.object(self.processor, '_perform_ocr') as mock_ocr:
            mock_ocr.return_value = OCRResult(
                text='Test OCR Text',
                confidence=85.0,
                word_count=3,
                processing_time=0.5,
                preprocessing_applied=[]
            )
            
            result = self.processor.process_extracted_image(extracted_image)
            
            assert result.ocr_text == 'Test OCR Text'
            assert result.ocr_confidence == 85.0
            assert 'ocr_result' in result.metadata
    
    def test_batch_process_images(self):
        """Test batch processing of images."""
        # Create test images
        images = []
        for i in range(3):
            image_data = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            image = Image.fromarray(image_data, 'RGB')
            
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            image_bytes = img_buffer.getvalue()
            
            extracted_image = ExtractedImage(
                image_id=f'test_img_{i}',
                filename=f'test_{i}.png',
                content=image_bytes,
                format='PNG',
                metadata={}
            )
            images.append(extracted_image)
        
        # Mock OCR processing
        with patch.object(self.processor, 'process_extracted_image') as mock_process:
            def side_effect(img):
                img.ocr_text = f'OCR text for {img.image_id}'
                img.ocr_confidence = 80.0
                return img
            
            mock_process.side_effect = side_effect
            
            results = self.processor.batch_process_images(images)
            
            assert len(results) == 3
            assert mock_process.call_count == 3
            
            for i, result in enumerate(results):
                assert result.ocr_text == f'OCR text for test_img_{i}'
    
    def test_process_document_file_not_found(self):
        """Test processing non-existent file."""
        result = self.processor.process_document('nonexistent.png')
        
        assert result.processing_status == ProcessingStatus.FAILED
        assert result.content == ""
        assert "nonexistent.png" in result.file_path
    
    @patch('PIL.Image.open')
    def test_process_document_success(self, mock_image_open):
        """Test successful document processing."""
        # Create a temporary image file for testing
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            # Create a simple test image
            image_data = np.full((100, 200, 3), 128, dtype=np.uint8)
            test_image = Image.fromarray(image_data, 'RGB')
            test_image.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Mock image loading
            mock_image_open.return_value = test_image
            
            # Mock OCR processing
            with patch.object(self.processor, '_perform_ocr') as mock_ocr:
                mock_ocr.return_value = OCRResult(
                    text='Test document content',
                    confidence=90.0,
                    word_count=3,
                    processing_time=1.0,
                    preprocessing_applied=['grayscale_conversion']
                )
                
                # Process document
                result = self.processor.process_document(tmp_path)
                
                # Verify results
                assert result.processing_status == ProcessingStatus.COMPLETED
                assert result.document_type == DocumentType.IMAGE
                assert result.content == 'Test document content'
                assert len(result.images) == 1
                assert result.images[0].ocr_text == 'Test document content'
                assert result.images[0].ocr_confidence == 90.0
                assert 'image_analysis' in result.metadata
                assert 'ocr_result' in result.metadata
                
        finally:
            Path(tmp_path).unlink()
    
    @patch('PIL.Image.open')
    def test_process_document_processing_error(self, mock_image_open):
        """Test document processing with error."""
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(b'fake image content')
            tmp_path = tmp.name
        
        try:
            # Mock Image.open to raise an exception
            mock_image_open.side_effect = Exception("Image loading error")
            
            # Process document
            result = self.processor.process_document(tmp_path)
            
            # Should return failed document
            assert result.processing_status == ProcessingStatus.FAILED
            assert result.content == ""
            assert "Image loading error" in result.error_message
            
        finally:
            Path(tmp_path).unlink()


class TestOCRResult:
    """Test cases for OCRResult dataclass."""
    
    def test_ocr_result_creation(self):
        """Test OCRResult creation."""
        result = OCRResult(
            text='Test text',
            confidence=85.5,
            word_count=2,
            processing_time=1.5,
            preprocessing_applied=['grayscale', 'contrast']
        )
        
        assert result.text == 'Test text'
        assert result.confidence == 85.5
        assert result.word_count == 2
        assert result.processing_time == 1.5
        assert result.preprocessing_applied == ['grayscale', 'contrast']


class TestImageAnalysis:
    """Test cases for ImageAnalysis dataclass."""
    
    def test_image_analysis_creation(self):
        """Test ImageAnalysis creation."""
        analysis = ImageAnalysis(
            width=800,
            height=600,
            format='PNG',
            mode='RGB',
            size_bytes=1024000,
            is_grayscale=False,
            average_brightness=128.5,
            contrast_level=45.2,
            estimated_dpi=300
        )
        
        assert analysis.width == 800
        assert analysis.height == 600
        assert analysis.format == 'PNG'
        assert analysis.mode == 'RGB'
        assert analysis.size_bytes == 1024000
        assert analysis.is_grayscale is False
        assert analysis.average_brightness == 128.5
        assert analysis.contrast_level == 45.2
        assert analysis.estimated_dpi == 300


if __name__ == "__main__":
    pytest.main([__file__])