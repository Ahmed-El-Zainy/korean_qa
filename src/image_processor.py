import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
import numpy as np
from dataclasses import dataclass
import os 
import sys 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_processor import (
    DocumentProcessor,
    ProcessedDocument,
    DocumentType,
    ProcessingStatus,
    DocumentProcessingError,
    ExtractedImage,
    DocumentProcessorFactory
)


try:
    from logger.custom_logger import CustomLoggerTracker
    custom_log = CustomLoggerTracker()
    logger = custom_log.get_logger("excel_processor")

except ImportError:
    # Fallback to standard logging if custom logger not available
    logger = logging.getLogger("excel_processor")


@dataclass
class OCRResult:
    """Result of OCR processing."""
    text: str
    confidence: float
    word_count: int
    processing_time: float
    preprocessing_applied: List[str]


@dataclass
class ImageAnalysis:
    """Analysis results for an image."""
    width: int
    height: int
    format: str
    mode: str
    size_bytes: int
    is_grayscale: bool
    average_brightness: float
    contrast_level: float
    estimated_dpi: Optional[int] = None


class ImageProcessor(DocumentProcessor):
    """
    Image processor with OCR capabilities using Tesseract.
    
    This processor handles standalone image files and provides OCR text extraction
    with preprocessing to improve accuracy.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the image processor.
        
        Args:
            config: Configuration dictionary containing image processing settings
        """
        super().__init__(config)
        self.ocr_engine = config.get('ocr_engine', 'tesseract')
        self.ocr_language = config.get('ocr_language', 'eng')
        self.preprocessing_enabled = config.get('image_preprocessing', True)
        self.min_confidence = config.get('min_ocr_confidence', 30.0)
        self.max_image_size = config.get('max_image_size', (3000, 3000))
        self.enhance_contrast = config.get('enhance_contrast', True)
        self.enhance_sharpness = config.get('enhance_sharpness', True)
        
        # Verify Tesseract installation
        self._verify_tesseract()
        
        logger.info(f"Image processor initialized with OCR language: {self.ocr_language}, "
                   f"preprocessing: {self.preprocessing_enabled}")
    
    def _get_supported_extensions(self) -> List[str]:
        """Get supported file extensions for image processor."""
        return ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif']
    
    def _verify_tesseract(self) -> None:
        """Verify that Tesseract is properly installed and accessible."""
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
        except Exception as e:
            logger.error(f"Tesseract not found or not properly installed: {e}")
            raise DocumentProcessingError(
                "tesseract", 
                "InstallationError", 
                f"Tesseract OCR engine not found: {e}"
            )
    
    def process_document(self, file_path: str) -> ProcessedDocument:
        """
        Process an image file and extract text using OCR.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            ProcessedDocument with extracted text and metadata
            
        Raises:
            DocumentProcessingError: If image processing fails
        """
        try:
            # Validate file first
            self.validate_file(file_path)
            
            # Generate document ID
            document_id = self._generate_document_id(file_path)
            
            logger.info(f"Processing image document: {file_path}")
            
            # Load and analyze image
            image = Image.open(file_path)
            image_analysis = self._analyze_image(image, file_path)
            
            # Preprocess image if enabled
            processed_image = image
            preprocessing_steps = []
            
            if self.preprocessing_enabled:
                processed_image, preprocessing_steps = self._preprocess_image(image)
            
            # Perform OCR
            ocr_result = self._perform_ocr(processed_image)
            
            # Create extracted image object
            with open(file_path, 'rb') as f:
                image_content = f.read()
            
            extracted_image = ExtractedImage(
                image_id=f"{document_id}_main",
                filename=Path(file_path).name,
                content=image_content,
                format=image_analysis.format,
                width=image_analysis.width,
                height=image_analysis.height,
                ocr_text=ocr_result.text,
                ocr_confidence=ocr_result.confidence,
                extraction_method="tesseract_ocr",
                metadata={
                    'image_analysis': {
                        'mode': image_analysis.mode,
                        'size_bytes': image_analysis.size_bytes,
                        'is_grayscale': image_analysis.is_grayscale,
                        'average_brightness': image_analysis.average_brightness,
                        'contrast_level': image_analysis.contrast_level,
                        'estimated_dpi': image_analysis.estimated_dpi
                    },
                    'ocr_result': {
                        'word_count': ocr_result.word_count,
                        'processing_time': ocr_result.processing_time,
                        'preprocessing_applied': ocr_result.preprocessing_applied
                    }
                }
            )
            
            # Create metadata
            metadata = {
                'image_analysis': image_analysis.__dict__,
                'ocr_result': ocr_result.__dict__,
                'preprocessing_steps': preprocessing_steps,
                'ocr_language': self.ocr_language,
                'ocr_engine': self.ocr_engine
            }
            
            # Create processed document
            processed_doc = ProcessedDocument(
                document_id=document_id,
                filename=Path(file_path).name,
                file_path=file_path,
                document_type=DocumentType.IMAGE,
                content=ocr_result.text,
                metadata=metadata,
                images=[extracted_image],
                processing_status=ProcessingStatus.COMPLETED
            )
            
            logger.info(f"Successfully processed image: {len(ocr_result.text)} characters extracted, "
                       f"confidence: {ocr_result.confidence:.1f}%")
            
            return processed_doc
            
        except Exception as e:
            logger.error(f"Failed to process image {file_path}: {e}")
            
            # Create failed document
            document_id = self._generate_document_id(file_path)
            return ProcessedDocument(
                document_id=document_id,
                filename=Path(file_path).name,
                file_path=file_path,
                document_type=DocumentType.IMAGE,
                content="",
                metadata={},
                processing_status=ProcessingStatus.FAILED,
                error_message=str(e)
            )
    
    def process_extracted_image(self, extracted_image: ExtractedImage) -> ExtractedImage:
        """
        Process an already extracted image (e.g., from PDF or Excel) with OCR.
        
        Args:
            extracted_image: ExtractedImage object to process
            
        Returns:
            Updated ExtractedImage with OCR text
        """
        try:
            logger.debug(f"Processing extracted image: {extracted_image.image_id}")
            
            # Load image from bytes
            image = Image.open(io.BytesIO(extracted_image.content))
            
            # Preprocess image if enabled
            processed_image = image
            preprocessing_steps = []
            
            if self.preprocessing_enabled:
                processed_image, preprocessing_steps = self._preprocess_image(image)
            
            # Perform OCR
            ocr_result = self._perform_ocr(processed_image)
            
            # Update extracted image with OCR results
            extracted_image.ocr_text = ocr_result.text
            extracted_image.ocr_confidence = ocr_result.confidence
            
            # Update metadata
            if 'ocr_result' not in extracted_image.metadata:
                extracted_image.metadata['ocr_result'] = {}
            
            extracted_image.metadata['ocr_result'].update({
                'word_count': ocr_result.word_count,
                'processing_time': ocr_result.processing_time,
                'preprocessing_applied': preprocessing_steps,
                'ocr_language': self.ocr_language,
                'ocr_engine': self.ocr_engine
            })
            
            logger.debug(f"OCR completed for {extracted_image.image_id}: "
                        f"{len(ocr_result.text)} characters, confidence: {ocr_result.confidence:.1f}%")
            
            return extracted_image
            
        except Exception as e:
            logger.warning(f"Failed to process extracted image {extracted_image.image_id}: {e}")
            
            # Return original image with error info
            extracted_image.metadata['ocr_error'] = str(e)
            return extracted_image
    
    def _analyze_image(self, image: Image.Image, file_path: str) -> ImageAnalysis:
        """
        Analyze image properties and characteristics.
        
        Args:
            image: PIL Image object
            file_path: Path to the image file
            
        Returns:
            ImageAnalysis object with image properties
        """
        try:
            # Basic properties
            width, height = image.size
            format_name = image.format or Path(file_path).suffix[1:].upper()
            mode = image.mode
            
            # File size
            size_bytes = Path(file_path).stat().st_size
            
            # Convert to grayscale for analysis
            if image.mode != 'L':
                gray_image = image.convert('L')
            else:
                gray_image = image
            
            # Calculate brightness and contrast
            np_image = np.array(gray_image)
            average_brightness = np.mean(np_image)
            contrast_level = np.std(np_image)
            
            # Check if image is grayscale
            is_grayscale = mode in ['L', '1'] or (mode == 'RGB' and self._is_grayscale_rgb(image))
            
            # Estimate DPI if available
            estimated_dpi = None
            if hasattr(image, 'info') and 'dpi' in image.info:
                estimated_dpi = image.info['dpi'][0] if isinstance(image.info['dpi'], tuple) else image.info['dpi']
            
            return ImageAnalysis(
                width=width,
                height=height,
                format=format_name,
                mode=mode,
                size_bytes=size_bytes,
                is_grayscale=is_grayscale,
                average_brightness=float(average_brightness),
                contrast_level=float(contrast_level),
                estimated_dpi=estimated_dpi
            )
            
        except Exception as e:
            logger.warning(f"Failed to analyze image: {e}")
            # Return basic analysis
            return ImageAnalysis(
                width=image.size[0],
                height=image.size[1],
                format=image.format or "UNKNOWN",
                mode=image.mode,
                size_bytes=0,
                is_grayscale=False,
                average_brightness=128.0,
                contrast_level=50.0
            )
    
    def _is_grayscale_rgb(self, image: Image.Image) -> bool:
        """
        Check if an RGB image is actually grayscale.
        
        Args:
            image: PIL Image object in RGB mode
            
        Returns:
            True if image is grayscale, False otherwise
        """
        try:
            # Sample a few pixels to check if R=G=B
            sample_size = min(100, image.size[0] * image.size[1])
            pixels = list(image.getdata())
            
            # Check first 'sample_size' pixels
            for i in range(0, min(sample_size, len(pixels))):
                r, g, b = pixels[i][:3]  # Handle RGBA by taking only RGB
                if r != g or g != b:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _preprocess_image(self, image: Image.Image) -> Tuple[Image.Image, List[str]]:
        """
        Preprocess image to improve OCR accuracy.
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (processed_image, list_of_applied_steps)
        """
        processed_image = image.copy()
        applied_steps = []
        
        try:
            # Resize if image is too large
            if image.size[0] > self.max_image_size[0] or image.size[1] > self.max_image_size[1]:
                processed_image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
                applied_steps.append("resize")
            
            # Convert to grayscale if not already
            if processed_image.mode != 'L':
                processed_image = processed_image.convert('L')
                applied_steps.append("grayscale_conversion")
            
            # Enhance contrast if enabled
            if self.enhance_contrast:
                enhancer = ImageEnhance.Contrast(processed_image)
                processed_image = enhancer.enhance(1.5)  # Increase contrast by 50%
                applied_steps.append("contrast_enhancement")
            
            # Enhance sharpness if enabled
            if self.enhance_sharpness:
                enhancer = ImageEnhance.Sharpness(processed_image)
                processed_image = enhancer.enhance(1.2)  # Increase sharpness by 20%
                applied_steps.append("sharpness_enhancement")
            
            # Apply noise reduction
            processed_image = processed_image.filter(ImageFilter.MedianFilter(size=3))
            applied_steps.append("noise_reduction")
            
        except Exception as e:
            logger.warning(f"Error during image preprocessing: {e}")
            # Return original image if preprocessing fails
            return image, ["preprocessing_failed"]
        
        return processed_image, applied_steps
    
    def _perform_ocr(self, image: Image.Image) -> OCRResult:
        """
        Perform OCR on the processed image.
        
        Args:
            image: PIL Image object
            
        Returns:
            OCRResult with extracted text and metadata
        """
        import time
        
        start_time = time.time()
        
        try:
            # Configure Tesseract
            custom_config = r'--oem 3 --psm 6'  # Use LSTM OCR Engine Mode with uniform text block
            
            # Get text with confidence scores
            data = pytesseract.image_to_data(
                image, 
                lang=self.ocr_language, 
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text and calculate average confidence
            words = []
            confidences = []
            
            for i, conf in enumerate(data['conf']):
                if int(conf) > 0:  # Only include words with confidence > 0
                    word = data['text'][i].strip()
                    if word:  # Only include non-empty words
                        words.append(word)
                        confidences.append(int(conf))
            
            # Combine words into text
            extracted_text = ' '.join(words)
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Clean up text
            extracted_text = self._clean_ocr_text(extracted_text)
            
            return OCRResult(
                text=extracted_text,
                confidence=avg_confidence,
                word_count=len(words),
                processing_time=processing_time,
                preprocessing_applied=[]  # Will be filled by caller
            )
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            processing_time = time.time() - start_time
            
            return OCRResult(
                text="",
                confidence=0.0,
                word_count=0,
                processing_time=processing_time,
                preprocessing_applied=[]
            )
    
    def _clean_ocr_text(self, text: str) -> str:
        """
        Clean and normalize OCR extracted text.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common OCR artifacts
        text = text.replace('|', 'I')  # Common misrecognition
        text = text.replace('0', 'O')  # In some contexts
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def batch_process_images(self, image_list: List[ExtractedImage]) -> List[ExtractedImage]:
        """
        Process multiple extracted images in batch.
        
        Args:
            image_list: List of ExtractedImage objects
            
        Returns:
            List of processed ExtractedImage objects with OCR text
        """
        processed_images = []
        
        logger.info(f"Starting batch OCR processing for {len(image_list)} images")
        
        for i, extracted_image in enumerate(image_list):
            try:
                logger.debug(f"Processing image {i+1}/{len(image_list)}: {extracted_image.image_id}")
                processed_image = self.process_extracted_image(extracted_image)
                processed_images.append(processed_image)
                
            except Exception as e:
                logger.warning(f"Failed to process image {extracted_image.image_id}: {e}")
                # Add original image with error info
                extracted_image.metadata['batch_processing_error'] = str(e)
                processed_images.append(extracted_image)
        
        logger.info(f"Completed batch OCR processing: {len(processed_images)} images processed")
        return processed_images


# Register the Image processor
DocumentProcessorFactory.register_processor(DocumentType.IMAGE, ImageProcessor)