import logging
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
from dataclasses import dataclass
import os 
import sys 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_processor import (
    DocumentProcessor,
    ProcessedDocument,
    DocumentType,
    ProcessingStatus,
    ExtractedImage,
    ExtractedTable,
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
class PDFPageInfo:
    """Information about a PDF page."""
    page_number: int
    width: float
    height: float
    rotation: int
    text_length: int
    image_count: int
    table_count: int


class PDFProcessor(DocumentProcessor):
    """
    PDF document processor using PyMuPDF.
    
    This processor extracts text, images, tables, and metadata from PDF files,
    maintaining proper citations with page numbers and section information.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the PDF processor.
        
        Args:
            config: Configuration dictionary containing PDF processing settings
        """
        super().__init__(config)
        self.extract_images = config.get('image_processing', True)
        self.extract_tables = config.get('table_extraction', True)
        self.min_table_rows = config.get('min_table_rows', 2)
        self.min_table_cols = config.get('min_table_cols', 2)
        self.image_min_size = config.get('image_min_size', 100)  # pixels
        
        logger.info(f"PDF processor initialized with image_processing={self.extract_images}, "
                   f"table_extraction={self.extract_tables}")
    
    def _get_supported_extensions(self) -> List[str]:
        """Get supported file extensions for PDF processor."""
        return ['.pdf']
    
    def process_document(self, file_path: str) -> ProcessedDocument:
        """
        Process a PDF document and extract all content.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            ProcessedDocument with extracted content and metadata
            
        Raises:
            DocumentProcessingError: If PDF processing fails
        """
        try:
            # Validate file first
            self.validate_file(file_path)
            
            # Generate document ID
            document_id = self._generate_document_id(file_path)
            
            logger.info(f"Processing PDF document: {file_path}")
            
            # Open PDF document
            pdf_document = fitz.open(file_path)
            
            try:
                # Extract metadata
                metadata = self._extract_metadata(pdf_document)
                
                # Process all pages
                all_text = []
                all_images = []
                all_tables = []
                page_info = []
                
                for page_num in range(pdf_document.page_count):
                    page = pdf_document[page_num]
                    
                    # Extract text from page
                    page_text = self._extract_page_text(page, page_num + 1)
                    if page_text.strip():
                        all_text.append(f"[Page {page_num + 1}]\n{page_text}")
                    
                    # Extract images if enabled
                    if self.extract_images:
                        page_images = self._extract_page_images(page, page_num + 1, document_id)
                        all_images.extend(page_images)
                    
                    # Extract tables if enabled
                    if self.extract_tables:
                        page_tables = self._extract_page_tables(page, page_num + 1)
                        all_tables.extend(page_tables)
                    
                    # Collect page info
                    page_info.append(PDFPageInfo(
                        page_number=page_num + 1,
                        width=page.rect.width,
                        height=page.rect.height,
                        rotation=page.rotation,
                        text_length=len(page_text),
                        image_count=len(page_images) if self.extract_images else 0,
                        table_count=len(page_tables) if self.extract_tables else 0
                    ))
                
                # Combine all text
                full_content = "\n\n".join(all_text)
                
                # Update metadata with processing info
                metadata.update({
                    'total_pages': pdf_document.page_count,
                    'total_images': len(all_images),
                    'total_tables': len(all_tables),
                    'total_text_length': len(full_content),
                    'page_info': [
                        {
                            'page_number': info.page_number,
                            'width': info.width,
                            'height': info.height,
                            'rotation': info.rotation,
                            'text_length': info.text_length,
                            'image_count': info.image_count,
                            'table_count': info.table_count
                        }
                        for info in page_info
                    ]
                })
                
                # Create processed document
                processed_doc = ProcessedDocument(
                    document_id=document_id,
                    filename=Path(file_path).name,
                    file_path=file_path,
                    document_type=DocumentType.PDF,
                    content=full_content,
                    metadata=metadata,
                    images=all_images,
                    tables=all_tables,
                    processing_status=ProcessingStatus.COMPLETED
                )
                
                logger.info(f"Successfully processed PDF: {pdf_document.page_count} pages, "
                           f"{len(all_images)} images, {len(all_tables)} tables")
                
                return processed_doc
                
            finally:
                pdf_document.close()
                
        except Exception as e:
            logger.error(f"Failed to process PDF {file_path}: {e}")
            
            # Create failed document
            document_id = self._generate_document_id(file_path)
            return ProcessedDocument(
                document_id=document_id,
                filename=Path(file_path).name,
                file_path=file_path,
                document_type=DocumentType.PDF,
                content="",
                metadata={},
                processing_status=ProcessingStatus.FAILED,
                error_message=str(e)
            )
    
    def _extract_metadata(self, pdf_document: fitz.Document) -> Dict[str, Any]:
        """
        Extract metadata from PDF document.
        
        Args:
            pdf_document: PyMuPDF document object
            
        Returns:
            Dictionary containing PDF metadata
        """
        metadata = {}
        
        try:
            # Get document metadata
            pdf_metadata = pdf_document.metadata
            
            # Standard metadata fields
            standard_fields = ['title', 'author', 'subject', 'keywords', 'creator', 'producer']
            for field in standard_fields:
                value = pdf_metadata.get(field, '').strip()
                if value:
                    metadata[field] = value
            
            # Creation and modification dates
            if 'creationDate' in pdf_metadata:
                metadata['creation_date'] = pdf_metadata['creationDate']
            if 'modDate' in pdf_metadata:
                metadata['modification_date'] = pdf_metadata['modDate']
            
            # Document properties
            metadata['page_count'] = pdf_document.page_count
            metadata['is_encrypted'] = pdf_document.is_encrypted
            metadata['is_pdf'] = pdf_document.is_pdf
            
            # PDF version
            if hasattr(pdf_document, 'pdf_version'):
                metadata['pdf_version'] = pdf_document.pdf_version()
            
        except Exception as e:
            logger.warning(f"Failed to extract PDF metadata: {e}")
            metadata['metadata_extraction_error'] = str(e)
        
        return metadata
    
    def _extract_page_text(self, page: fitz.Page, page_number: int) -> str:
        """
        Extract text from a PDF page.
        
        Args:
            page: PyMuPDF page object
            page_number: Page number (1-based)
            
        Returns:
            Extracted text content
        """
        try:
            # Extract text with layout preservation
            text = page.get_text("text")
            
            # Clean up text
            text = self._clean_text(text)
            
            return text
            
        except Exception as e:
            logger.warning(f"Failed to extract text from page {page_number}: {e}")
            return ""
    
    def _extract_page_images(self, page: fitz.Page, page_number: int, document_id: str) -> List[ExtractedImage]:
        """
        Extract images from a PDF page.
        
        Args:
            page: PyMuPDF page object
            page_number: Page number (1-based)
            document_id: Document ID for image naming
            
        Returns:
            List of ExtractedImage objects
        """
        images = []
        
        try:
            # Get image list from page
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get image reference
                    xref = img[0]
                    
                    # Extract image data
                    base_image = page.parent.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Check image size
                    if len(image_bytes) < self.image_min_size:
                        continue
                    
                    # Create image object
                    image_id = f"{document_id}_page{page_number}_img{img_index}"
                    filename = f"page{page_number}_image{img_index}.{image_ext}"
                    
                    extracted_image = ExtractedImage(
                        image_id=image_id,
                        filename=filename,
                        content=image_bytes,
                        format=image_ext.upper(),
                        extraction_method="pymupdf",
                        metadata={
                            'page_number': page_number,
                            'image_index': img_index,
                            'xref': xref,
                            'size_bytes': len(image_bytes)
                        }
                    )
                    
                    images.append(extracted_image)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_index} from page {page_number}: {e}")
                    continue
        
        except Exception as e:
            logger.warning(f"Failed to extract images from page {page_number}: {e}")
        
        return images
    
    def _extract_page_tables(self, page: fitz.Page, page_number: int) -> List[ExtractedTable]:
        """
        Extract tables from a PDF page.
        
        Args:
            page: PyMuPDF page object
            page_number: Page number (1-based)
            
        Returns:
            List of ExtractedTable objects
        """
        tables = []
        
        try:
            # Try to find tables using text analysis
            # This is a basic implementation - more sophisticated table detection
            # could use libraries like camelot-py or tabula-py
            
            text = page.get_text("text")
            potential_tables = self._detect_tables_in_text(text, page_number)
            tables.extend(potential_tables)
            
        except Exception as e:
            logger.warning(f"Failed to extract tables from page {page_number}: {e}")
        
        return tables
    
    def _detect_tables_in_text(self, text: str, page_number: int) -> List[ExtractedTable]:
        """
        Detect tables in text using pattern matching.
        
        This is a basic implementation that looks for tabular patterns in text.
        For production use, consider using specialized table extraction libraries.
        
        Args:
            text: Text content to analyze
            page_number: Page number for metadata
            
        Returns:
            List of detected tables
        """
        tables = []
        
        try:
            lines = text.split('\n')
            current_table_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    # Empty line might end a table
                    if len(current_table_lines) >= self.min_table_rows:
                        table = self._parse_table_lines(current_table_lines, page_number, len(tables))
                        if table:
                            tables.append(table)
                    current_table_lines = []
                    continue
                
                # Check if line looks like a table row (has multiple columns separated by whitespace)
                columns = re.split(r'\s{2,}', line)  # Split on 2+ spaces
                if len(columns) >= self.min_table_cols:
                    current_table_lines.append(columns)
                else:
                    # Line doesn't look like table data
                    if len(current_table_lines) >= self.min_table_rows:
                        table = self._parse_table_lines(current_table_lines, page_number, len(tables))
                        if table:
                            tables.append(table)
                    current_table_lines = []
            
            # Check for table at end of text
            if len(current_table_lines) >= self.min_table_rows:
                table = self._parse_table_lines(current_table_lines, page_number, len(tables))
                if table:
                    tables.append(table)
        
        except Exception as e:
            logger.warning(f"Failed to detect tables in text: {e}")
        
        return tables
    
    def _parse_table_lines(self, table_lines: List[List[str]], page_number: int, table_index: int) -> Optional[ExtractedTable]:
        """
        Parse table lines into an ExtractedTable object.
        
        Args:
            table_lines: List of table rows (each row is a list of columns)
            page_number: Page number for metadata
            table_index: Table index on the page
            
        Returns:
            ExtractedTable object or None if parsing fails
        """
        try:
            if not table_lines:
                return None
            
            # Use first row as headers (this is a simple assumption)
            headers = [col.strip() for col in table_lines[0]]
            
            # Remaining rows are data
            rows = []
            for row_data in table_lines[1:]:
                # Pad row to match header length
                padded_row = row_data + [''] * (len(headers) - len(row_data))
                rows.append([col.strip() for col in padded_row[:len(headers)]])
            
            # Create table object
            table_id = f"page{page_number}_table{table_index}"
            
            return ExtractedTable(
                table_id=table_id,
                headers=headers,
                rows=rows,
                page_number=page_number,
                extraction_confidence=0.7,  # Basic text-based extraction
                metadata={
                    'extraction_method': 'text_pattern_matching',
                    'table_index': table_index
                }
            )
        
        except Exception as e:
            logger.warning(f"Failed to parse table lines: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double newline
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        
        # Remove page breaks and form feeds
        text = text.replace('\f', '\n')
        text = text.replace('\x0c', '\n')
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text


# Register the PDF processor
DocumentProcessorFactory.register_processor(DocumentType.PDF, PDFProcessor)




if __name__=="__main__":
    logger.info(f"PDF processor init ..")
    
    ## Test code (for demonstration purposes)
    config = {'image_processing': True, 'table_extraction': True}
    processor = DocumentProcessorFactory.create_processor("/Users/ahmedmostafa/Downloads/eval_Korean_qa/data/documents/원재료사용현황.pdf", config)
    processed_doc = processor.process_document("/Users/ahmedmostafa/Downloads/eval_Korean_qa/data/documents/원재료사용현황.pdf")
    chunks = processor.extract_chunks(processed_doc)
    for chunk in chunks:
        print(chunk)