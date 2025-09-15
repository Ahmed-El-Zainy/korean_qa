import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os 
import sys 



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_processor import ProcessingStatus, DocumentType


try:
    from logger.custom_logger import CustomLoggerTracker
    custom_log = CustomLoggerTracker()
    logger = custom_log.get_logger("meta_manager")

except ImportError:
    # Fallback to standard logging if custom logger not available
    logger = logging.getLogger("meta_manager")


@dataclass
class DocumentMetadata:
    """Metadata for a processed document."""
    document_id: str
    filename: str
    file_path: str
    file_type: str
    upload_timestamp: datetime
    processing_status: ProcessingStatus
    total_chunks: int
    file_size: int
    checksum: str
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    metadata_json: Optional[str] = None  # Additional metadata as JSON


@dataclass
class CitationInfo:
    """Citation information for a document chunk."""
    chunk_id: str
    document_id: str
    source_document: str
    location_reference: str
    extraction_method: str
    confidence_level: float
    page_number: Optional[int] = None
    worksheet_name: Optional[str] = None
    cell_range: Optional[str] = None
    section_title: Optional[str] = None


class MetadataManager:
    """
    SQLite-based metadata manager for document tracking and citation management.
    
    This manager provides persistent storage for document metadata, processing status,
    and citation information with efficient querying capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the metadata manager.
        
        Args:
            config: Configuration dictionary containing database settings
        """
        self.config = config
        self.db_path = config.get('metadata_db_path', './data/metadata.db')
        
        # Ensure database directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Metadata manager initialized with database: {self.db_path}")
    
    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create documents table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS documents (
                        document_id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        upload_timestamp TEXT NOT NULL,
                        processing_status TEXT NOT NULL,
                        total_chunks INTEGER DEFAULT 0,
                        file_size INTEGER DEFAULT 0,
                        checksum TEXT,
                        error_message TEXT,
                        processing_time REAL,
                        metadata_json TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create citations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS citations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        chunk_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        source_document TEXT NOT NULL,
                        location_reference TEXT NOT NULL,
                        extraction_method TEXT NOT NULL,
                        confidence_level REAL NOT NULL,
                        page_number INTEGER,
                        worksheet_name TEXT,
                        cell_range TEXT,
                        section_title TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (document_id) REFERENCES documents (document_id)
                    )
                ''')
                
                # Create indexes for efficient querying
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (processing_status)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_type ON documents (file_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_citations_document ON citations (document_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_citations_chunk ON citations (chunk_id)')
                
                conn.commit()
                logger.debug("Database tables initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def store_document_metadata(self, doc_id: str, metadata: DocumentMetadata) -> bool:
        """
        Store document metadata in the database.
        
        Args:
            doc_id: Document ID
            metadata: DocumentMetadata object
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert datetime to ISO string
                upload_timestamp = metadata.upload_timestamp.isoformat()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO documents (
                        document_id, filename, file_path, file_type, upload_timestamp,
                        processing_status, total_chunks, file_size, checksum,
                        error_message, processing_time, metadata_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    doc_id,
                    metadata.filename,
                    metadata.file_path,
                    metadata.file_type,
                    upload_timestamp,
                    metadata.processing_status.value,
                    metadata.total_chunks,
                    metadata.file_size,
                    metadata.checksum,
                    metadata.error_message,
                    metadata.processing_time,
                    metadata.metadata_json,
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                logger.debug(f"Stored metadata for document: {doc_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store document metadata: {e}")
            return False
    
    def get_document_metadata(self, doc_id: str) -> Optional[DocumentMetadata]:
        """
        Retrieve document metadata by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            DocumentMetadata object or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT document_id, filename, file_path, file_type, upload_timestamp,
                           processing_status, total_chunks, file_size, checksum,
                           error_message, processing_time, metadata_json
                    FROM documents WHERE document_id = ?
                ''', (doc_id,))
                
                row = cursor.fetchone()
                if row:
                    return DocumentMetadata(
                        document_id=row[0],
                        filename=row[1],
                        file_path=row[2],
                        file_type=row[3],
                        upload_timestamp=datetime.fromisoformat(row[4]),
                        processing_status=ProcessingStatus(row[5]),
                        total_chunks=row[6],
                        file_size=row[7],
                        checksum=row[8],
                        error_message=row[9],
                        processing_time=row[10],
                        metadata_json=row[11]
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get document metadata: {e}")
            return None
    
    def update_document_status(self, doc_id: str, status: ProcessingStatus, 
                              error_message: Optional[str] = None,
                              processing_time: Optional[float] = None) -> bool:
        """
        Update document processing status.
        
        Args:
            doc_id: Document ID
            status: New processing status
            error_message: Optional error message
            processing_time: Optional processing time
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE documents 
                    SET processing_status = ?, error_message = ?, processing_time = ?, updated_at = ?
                    WHERE document_id = ?
                ''', (
                    status.value,
                    error_message,
                    processing_time,
                    datetime.now().isoformat(),
                    doc_id
                ))
                
                conn.commit()
                logger.debug(f"Updated status for document {doc_id}: {status.value}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update document status: {e}")
            return False
    
    def store_citation_info(self, citation: CitationInfo) -> bool:
        """
        Store citation information.
        
        Args:
            citation: CitationInfo object
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO citations (
                        chunk_id, document_id, source_document, location_reference,
                        extraction_method, confidence_level, page_number,
                        worksheet_name, cell_range, section_title
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    citation.chunk_id,
                    citation.document_id,
                    citation.source_document,
                    citation.location_reference,
                    citation.extraction_method,
                    citation.confidence_level,
                    citation.page_number,
                    citation.worksheet_name,
                    citation.cell_range,
                    citation.section_title
                ))
                
                conn.commit()
                logger.debug(f"Stored citation for chunk: {citation.chunk_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store citation info: {e}")
            return False
    
    def get_citation_info(self, chunk_id: str) -> Optional[CitationInfo]:
        """
        Retrieve citation information by chunk ID.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            CitationInfo object or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT chunk_id, document_id, source_document, location_reference,
                           extraction_method, confidence_level, page_number,
                           worksheet_name, cell_range, section_title
                    FROM citations WHERE chunk_id = ?
                ''', (chunk_id,))
                
                row = cursor.fetchone()
                if row:
                    return CitationInfo(
                        chunk_id=row[0],
                        document_id=row[1],
                        source_document=row[2],
                        location_reference=row[3],
                        extraction_method=row[4],
                        confidence_level=row[5],
                        page_number=row[6],
                        worksheet_name=row[7],
                        cell_range=row[8],
                        section_title=row[9]
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get citation info: {e}")
            return None
    
    def list_documents(self, status: Optional[ProcessingStatus] = None,
                      file_type: Optional[str] = None,
                      limit: int = 100) -> List[DocumentMetadata]:
        """
        List documents with optional filtering.
        
        Args:
            status: Optional status filter
            file_type: Optional file type filter
            limit: Maximum number of results
            
        Returns:
            List of DocumentMetadata objects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = '''
                    SELECT document_id, filename, file_path, file_type, upload_timestamp,
                           processing_status, total_chunks, file_size, checksum,
                           error_message, processing_time, metadata_json
                    FROM documents
                '''
                
                conditions = []
                params = []
                
                if status:
                    conditions.append('processing_status = ?')
                    params.append(status.value)
                
                if file_type:
                    conditions.append('file_type = ?')
                    params.append(file_type)
                
                if conditions:
                    query += ' WHERE ' + ' AND '.join(conditions)
                
                query += ' ORDER BY upload_timestamp DESC LIMIT ?'
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                documents = []
                for row in rows:
                    documents.append(DocumentMetadata(
                        document_id=row[0],
                        filename=row[1],
                        file_path=row[2],
                        file_type=row[3],
                        upload_timestamp=datetime.fromisoformat(row[4]),
                        processing_status=ProcessingStatus(row[5]),
                        total_chunks=row[6],
                        file_size=row[7],
                        checksum=row[8],
                        error_message=row[9],
                        processing_time=row[10],
                        metadata_json=row[11]
                    ))
                
                return documents
                
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete document and all associated citations.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete citations first (foreign key constraint)
                cursor.execute('DELETE FROM citations WHERE document_id = ?', (doc_id,))
                
                # Delete document
                cursor.execute('DELETE FROM documents WHERE document_id = ?', (doc_id,))
                
                conn.commit()
                logger.info(f"Deleted document and citations: {doc_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count documents by status
                cursor.execute('''
                    SELECT processing_status, COUNT(*) 
                    FROM documents 
                    GROUP BY processing_status
                ''')
                status_counts = dict(cursor.fetchall())
                
                # Count documents by type
                cursor.execute('''
                    SELECT file_type, COUNT(*) 
                    FROM documents 
                    GROUP BY file_type
                ''')
                type_counts = dict(cursor.fetchall())
                
                # Total statistics
                cursor.execute('SELECT COUNT(*) FROM documents')
                total_documents = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM citations')
                total_citations = cursor.fetchone()[0]
                
                cursor.execute('SELECT SUM(total_chunks) FROM documents')
                total_chunks = cursor.fetchone()[0] or 0
                
                cursor.execute('SELECT SUM(file_size) FROM documents')
                total_file_size = cursor.fetchone()[0] or 0
                
                return {
                    'total_documents': total_documents,
                    'total_citations': total_citations,
                    'total_chunks': total_chunks,
                    'total_file_size': total_file_size,
                    'documents_by_status': status_counts,
                    'documents_by_type': type_counts,
                    'database_path': self.db_path
                }
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {'error': str(e)}
    
    def cleanup_orphaned_citations(self) -> int:
        """
        Clean up citations that reference non-existent documents.
        
        Returns:
            Number of orphaned citations removed
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    DELETE FROM citations 
                    WHERE document_id NOT IN (SELECT document_id FROM documents)
                ''')
                
                removed_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleaned up {removed_count} orphaned citations")
                return removed_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup orphaned citations: {e}")
            return 0



if __name__=="__main__":
    logger.info(f"metadata init ..")