import logging
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import time
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import os 
import sys 
from dotenv import load_dotenv
load_dotenv()


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.document_processor import DocumentChunk, ChunkMetadata
try:
    from logger.custom_logger import CustomLoggerTracker
    custom_log = CustomLoggerTracker()
    logger = custom_log.get_logger("vector_store")

except ImportError:
    # Fallback to standard logging if custom logger not available
    logger = logging.getLogger("vector_store")




@dataclass
class SearchResult:
    """Result of vector similarity search."""
    chunk: DocumentChunk
    similarity_score: float
    rerank_score: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
            
            
@dataclass
class IndexStats:
    """Statistics about the vector index."""
    total_points: int
    collection_name: str
    vector_size: int
    distance_metric: str
    indexed_documents: int
    last_updated: str


class QdrantVectorStore:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.url = config.get('qdrant_url', 'http://localhost:6333')
        self.api_key = config.get('qdrant_api_key')
        self.collection_name = config.get('qdrant_collection', 'manufacturing_docs')
        self.vector_size = config.get('vector_size', 1024)
        self.distance_metric = Distance.COSINE
        
        # Initialize Qdrant client
        logger.info(f"Connecting to Qdrant at URL: {os.environ['QDRANT_URL']}")
        self.client = QdrantClient(
            url="https://50f53cc8-bbb0-4939-8254-8f025a577222.us-west-2-0.aws.cloud.qdrant.io:6333", 
            api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.gHOXbfqPucRwhczrW8s3VSZbconqQ6Rk49Uaz9ZChdE",)
        
        self._ensure_collection_exists()
        logger.info(f"Qdrant vector store initialized: {os.environ['QDRANT_URL']}, collection: {self.collection_name}")
    
    
    def _ensure_collection_exists(self):
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                # Create collection with vector configuration
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance_metric
                    )
                )
                
                # Create payload indexes for efficient filtering
                self._create_payload_indexes()
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.debug(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise
    
    
    
    def _create_payload_indexes(self):
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_id",
                field_schema=models.KeywordIndexParams())
            # Index on document type for filtering by file type
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_type",
                field_schema=models.KeywordIndexParams())
            
            # Index on page_number for PDF citations
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="page_number",
                field_schema=models.IntegerIndexParams())
            
            # Index on worksheet_name for Excel citations
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="worksheet_name",
                field_schema=models.KeywordIndexParams())
            
            logger.debug("Payload indexes created successfully")
        except Exception as e:
            logger.warning(f"Failed to create payload indexes: {e}")
    
    def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        if not chunks:
            logger.warning("No chunks provided for indexing")
            return True
        try:
            points = []
            for chunk in chunks:
                if not chunk.embedding:
                    logger.warning(f"Chunk {chunk.metadata.chunk_id} has no embedding, skipping")
                    continue
                
                # Create point for Qdrant
                point = PointStruct(
                    id=str(uuid.uuid4()),  # Generate unique ID
                    vector=chunk.embedding,
                    payload={
                        # Chunk metadata
                        "chunk_id": chunk.metadata.chunk_id,
                        "document_id": chunk.metadata.document_id,
                        "chunk_index": chunk.metadata.chunk_index,
                        "content": chunk.content,
                        
                        # Citation information
                        "page_number": chunk.metadata.page_number,
                        "worksheet_name": chunk.metadata.worksheet_name,
                        "cell_range": chunk.metadata.cell_range,
                        "section_title": chunk.metadata.section_title,
                        
                        # References
                        "image_references": chunk.metadata.image_references,
                        "table_references": chunk.metadata.table_references,
                        
                        # Timestamps and confidence
                        "extraction_timestamp": chunk.metadata.extraction_timestamp.isoformat(),
                        "confidence_score": chunk.metadata.confidence_score,
                        
                        # Additional metadata
                        "content_length": len(chunk.content),
                        "indexed_at": time.time()
                    }
                )
                
                points.append(point)
            if not points:
                logger.warning("No valid points to index")
                return True
            
            # Upload points to Qdrant
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                points=points)
            logger.info(f"Successfully indexed {len(points)} chunks to Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            return False
    
    def similarity_search(self, query_embedding: List[float], k: int = 10, 
                         filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        try:
            # Build filter conditions
            filter_conditions = self._build_filter_conditions(filters) if filters else None
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                query_filter=filter_conditions,
                with_payload=True,
                with_vectors=False  # Don't return vectors to save bandwidth
            )
            
            # Convert to SearchResult objects
            results = []
            for result in search_results:
                payload = result.payload
                
                # Reconstruct chunk metadata
                metadata = ChunkMetadata(
                    chunk_id=payload.get("chunk_id", ""),
                    document_id=payload.get("document_id", ""),
                    chunk_index=payload.get("chunk_index", 0),
                    page_number=payload.get("page_number"),
                    worksheet_name=payload.get("worksheet_name"),
                    cell_range=payload.get("cell_range"),
                    section_title=payload.get("section_title"),
                    image_references=payload.get("image_references", []),
                    table_references=payload.get("table_references", []),
                    confidence_score=payload.get("confidence_score"))
                
                # Reconstruct document chunk
                chunk = DocumentChunk(
                    content=payload.get("content", ""),
                    metadata=metadata,
                    embedding=None  # Don't include embedding in results
                )
                
                # Create search result
                search_result = SearchResult(
                    chunk=chunk,
                    similarity_score=result.score,
                    metadata={
                        "qdrant_id": result.id,
                        "content_length": payload.get("content_length", 0),
                        "indexed_at": payload.get("indexed_at"),
                        "extraction_timestamp": payload.get("extraction_timestamp")
                    }
                )
                
                results.append(search_result)
            
            logger.debug(f"Found {len(results)} similar chunks")
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def filtered_search(self, query_embedding: List[float], filters: Dict[str, Any], 
                       k: int = 10) -> List[SearchResult]:
        return self.similarity_search(query_embedding, k, filters)
    
    def delete_document(self, document_id: str) -> bool:
        try:
            # Delete points with matching document_id
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="document_id",
                                match=MatchValue(value=document_id)
                            )
                        ]
                    )
                )
            )
            
            logger.info(f"Deleted all chunks for document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    
    def get_collection_info(self) -> Optional[IndexStats]:
        try:
            collection_info = self.client.get_collection(self.collection_name)
            # Count unique documents
            # This is a simplified count - in production you might want to use aggregation
            search_results = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust based on your needs
                with_payload=["document_id"],
                with_vectors=False
            )
            
            unique_documents = set()
            for point in search_results[0]:
                if point.payload and "document_id" in point.payload:
                    unique_documents.add(point.payload["document_id"])
            
            return IndexStats(
                total_points=collection_info.points_count,
                collection_name=self.collection_name,
                vector_size=collection_info.config.params.vectors.size,
                distance_metric=collection_info.config.params.vectors.distance.name,
                indexed_documents=len(unique_documents),
                last_updated=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None
    
    def _build_filter_conditions(self, filters: Dict[str, Any]) -> Filter:
        """
        Build Qdrant filter conditions from filter dictionary.
        
        Args:
            filters: Dictionary of filter conditions
            
        Returns:
            Qdrant Filter object
        """
        conditions = []
        
        # Document ID filter
        if "document_id" in filters:
            conditions.append(
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=filters["document_id"])
                )
            )
        
        # Document type filter
        if "document_type" in filters:
            conditions.append(
                FieldCondition(
                    key="document_type",
                    match=MatchValue(value=filters["document_type"])
                )
            )
        
        # Page number filter
        if "page_number" in filters:
            conditions.append(
                FieldCondition(
                    key="page_number",
                    match=MatchValue(value=filters["page_number"])
                )
            )
        
        # Worksheet name filter
        if "worksheet_name" in filters:
            conditions.append(
                FieldCondition(
                    key="worksheet_name",
                    match=MatchValue(value=filters["worksheet_name"])
                )
            )
        
        # Content length range filter
        if "min_content_length" in filters:
            conditions.append(
                FieldCondition(
                    key="content_length",
                    range=models.Range(gte=filters["min_content_length"])
                )
            )
        
        if "max_content_length" in filters:
            conditions.append(
                FieldCondition(
                    key="content_length",
                    range=models.Range(lte=filters["max_content_length"])
                )
            )
        
        return Filter(must=conditions) if conditions else None
    
    def health_check(self) -> bool:
        """
        Check if the vector store is healthy and accessible.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to get collection info
            self.client.get_collection(self.collection_name)
            return True
            
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return False
    
    def create_collection(self, vector_size: int, distance_metric: Distance = Distance.COSINE) -> bool:
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance_metric
                )
            )
            
            # Update instance variables
            self.vector_size = vector_size
            self.distance_metric = distance_metric
            
            self._create_payload_indexes()
            logger.info(f"Created collection {self.collection_name} with vector size {vector_size}")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False
    
    def delete_collection(self) -> bool:
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
        
        
        
        
        
if __name__=="__main__":
    logger.info(f"Vector store init ..")
    config = {
        'qdrant_url': os.getenv('QDRANT_URL', 'http://localhost:6333'),
        'qdrant_api_key': os.getenv('QDRANT_API_KEY'),
        'qdrant_collection': 'manufacturing_docs',
        'vector_size': 1024
    }
    vector_store = QdrantVectorStore(config)
    health = vector_store.health_check()
    if health:
        logger.info("Vector store is healthy and ready.")
    else:
        logger.error("Vector store is not accessible.")