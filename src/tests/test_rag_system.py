"""
Integration tests for the complete RAG system.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

from src.rag.embedding_system import EmbeddingSystem, EmbeddingResult, RerankResult
from src.rag.vector_store import QdrantVectorStore, SearchResult
from src.rag.groq_client import GroqClient, LLMResponse, LLMSystem
from src.rag.rag_engine import RAGEngine, RAGResponse, Citation
from src.rag.metadata_manager import MetadataManager, DocumentMetadata
from src.rag.ingestion_pipeline import DocumentIngestionPipeline, IngestionResult
from src.rag.document_processor import DocumentChunk, ChunkMetadata, ProcessingStatus


class TestEmbeddingSystem:
    """Test cases for EmbeddingSystem."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'silicon_flow_api_key': 'test_key',
            'embedding_model': 'test-model',
            'reranker_model': 'test-reranker',
            'batch_size': 2,
            'max_retries': 2,
            'enable_embedding_cache': True
        }
    
    @patch('src.rag.embedding_system.SiliconFlowEmbeddingClient')
    def test_embedding_system_initialization(self, mock_client_class):
        """Test embedding system initialization."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        embedding_system = EmbeddingSystem(self.config)
        
        assert embedding_system.api_key == 'test_key'
        assert embedding_system.embedding_model == 'test-model'
        assert embedding_system.batch_size == 2
        assert embedding_system.cache_enabled is True
        mock_client_class.assert_called_once_with('test_key')
    
    @patch('src.rag.embedding_system.SiliconFlowEmbeddingClient')
    def test_generate_embeddings_success(self, mock_client_class):
        """Test successful embedding generation."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock successful embedding response
        mock_client.generate_embeddings.return_value = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            model_name='test-model',
            processing_time=1.0,
            token_count=10,
            success=True
        )
        
        embedding_system = EmbeddingSystem(self.config)
        embeddings = embedding_system.generate_embeddings(['text1', 'text2'])
        
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
    
    @patch('src.rag.embedding_system.SiliconFlowEmbeddingClient')
    def test_rerank_results(self, mock_client_class):
        """Test reranking functionality."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock reranking response
        mock_client.rerank_documents.return_value = [
            RerankResult(text='doc2', score=0.9, index=1),
            RerankResult(text='doc1', score=0.7, index=0)
        ]
        
        embedding_system = EmbeddingSystem(self.config)
        results = embedding_system.rerank_results('query', ['doc1', 'doc2'])
        
        assert len(results) == 2
        assert results[0].text == 'doc2'
        assert results[0].score == 0.9
        assert results[1].text == 'doc1'
        assert results[1].score == 0.7


class TestGroqClient:
    """Test cases for GroqClient."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.api_key = 'test_groq_key'
        self.client = GroqClient(self.api_key)
    
    @patch('requests.Session.post')
    def test_generate_response_success(self, mock_post):
        """Test successful response generation."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {'content': 'Test response'},
                'finish_reason': 'stop'
            }],
            'usage': {'total_tokens': 50}
        }
        mock_post.return_value = mock_response
        
        messages = [{'role': 'user', 'content': 'Test question'}]
        result = self.client.generate_response(messages)
        
        assert result.success is True
        assert result.text == 'Test response'
        assert result.token_count == 50
        assert result.finish_reason == 'stop'
    
    @patch('requests.Session.post')
    def test_answer_question(self, mock_post):
        """Test question answering functionality."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {'content': 'Based on the context, the answer is...'},
                'finish_reason': 'stop'
            }],
            'usage': {'total_tokens': 75}
        }
        mock_post.return_value = mock_response
        
        result = self.client.answer_question('What is the yield?', 'Context: Yield is 95%')
        
        assert result.success is True
        assert 'answer is' in result.text
        assert result.token_count == 75


class TestRAGEngine:
    """Test cases for RAGEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'silicon_flow_api_key': 'test_key',
            'groq_api_key': 'test_groq_key',
            'qdrant_url': 'http://localhost:6333',
            'qdrant_api_key': 'test_qdrant_key',
            'qdrant_collection': 'test_collection',
            'max_context_chunks': 3,
            'similarity_threshold': 0.7,
            'rerank_top_k': 10,
            'final_top_k': 3
        }
    
    @patch('src.rag.rag_engine.EmbeddingSystem')
    @patch('src.rag.rag_engine.QdrantVectorStore')
    @patch('src.rag.rag_engine.LLMSystem')
    def test_rag_engine_initialization(self, mock_llm, mock_vector, mock_embedding):
        """Test RAG engine initialization."""
        rag_engine = RAGEngine(self.config)
        
        assert rag_engine.max_context_chunks == 3
        assert rag_engine.similarity_threshold == 0.7
        assert rag_engine.rerank_top_k == 10
        assert rag_engine.final_top_k == 3
        
        mock_embedding.assert_called_once()
        mock_vector.assert_called_once()
        mock_llm.assert_called_once()
    
    @patch('src.rag.rag_engine.EmbeddingSystem')
    @patch('src.rag.rag_engine.QdrantVectorStore')
    @patch('src.rag.rag_engine.LLMSystem')
    def test_answer_question_success(self, mock_llm, mock_vector, mock_embedding):
        """Test successful question answering."""
        # Mock components
        mock_embedding_instance = Mock()
        mock_vector_instance = Mock()
        mock_llm_instance = Mock()
        
        mock_embedding.return_value = mock_embedding_instance
        mock_vector.return_value = mock_vector_instance
        mock_llm.return_value = mock_llm_instance
        
        # Mock embedding generation
        mock_embedding_instance.generate_query_embedding.return_value = [0.1, 0.2, 0.3]
        
        # Mock search results
        mock_chunk = DocumentChunk(
            content="Test content about manufacturing",
            metadata=ChunkMetadata(
                chunk_id="test_chunk_1",
                document_id="test_doc_1",
                chunk_index=0,
                page_number=1
            )
        )
        
        mock_search_result = SearchResult(
            chunk=mock_chunk,
            similarity_score=0.9
        )
        
        mock_vector_instance.similarity_search.return_value = [mock_search_result]
        
        # Mock reranking
        mock_embedding_instance.rerank_results.return_value = [
            RerankResult(text="Test content about manufacturing", score=0.95, index=0)
        ]
        
        # Mock LLM response
        mock_llm_instance.answer_question.return_value = "The manufacturing process shows good results."
        
        # Test question answering
        rag_engine = RAGEngine(self.config)
        response = rag_engine.answer_question("What are the manufacturing results?")
        
        assert response.success is True
        assert "manufacturing process" in response.answer
        assert len(response.citations) == 1
        assert response.citations[0].confidence == 0.9
        assert response.total_chunks_retrieved == 1


class TestMetadataManager:
    """Test cases for MetadataManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'metadata_db_path': str(Path(self.temp_dir) / 'test_metadata.db')
        }
        self.metadata_manager = MetadataManager(self.config)
    
    def test_store_and_retrieve_document_metadata(self):
        """Test storing and retrieving document metadata."""
        from datetime import datetime
        
        # Create test metadata
        metadata = DocumentMetadata(
            document_id='test_doc_1',
            filename='test.pdf',
            file_path='/path/to/test.pdf',
            file_type='pdf',
            upload_timestamp=datetime.now(),
            processing_status=ProcessingStatus.COMPLETED,
            total_chunks=10,
            file_size=1024,
            checksum='abc123',
            processing_time=5.5
        )
        
        # Store metadata
        success = self.metadata_manager.store_document_metadata('test_doc_1', metadata)
        assert success is True
        
        # Retrieve metadata
        retrieved = self.metadata_manager.get_document_metadata('test_doc_1')
        assert retrieved is not None
        assert retrieved.document_id == 'test_doc_1'
        assert retrieved.filename == 'test.pdf'
        assert retrieved.processing_status == ProcessingStatus.COMPLETED
        assert retrieved.total_chunks == 10
        assert retrieved.processing_time == 5.5
    
    def test_update_document_status(self):
        """Test updating document status."""
        from datetime import datetime
        
        # First store a document
        metadata = DocumentMetadata(
            document_id='test_doc_2',
            filename='test2.pdf',
            file_path='/path/to/test2.pdf',
            file_type='pdf',
            upload_timestamp=datetime.now(),
            processing_status=ProcessingStatus.PENDING,
            total_chunks=0,
            file_size=2048,
            checksum='def456'
        )
        
        self.metadata_manager.store_document_metadata('test_doc_2', metadata)
        
        # Update status
        success = self.metadata_manager.update_document_status(
            'test_doc_2', 
            ProcessingStatus.COMPLETED,
            processing_time=3.2
        )
        assert success is True
        
        # Verify update
        retrieved = self.metadata_manager.get_document_metadata('test_doc_2')
        assert retrieved.processing_status == ProcessingStatus.COMPLETED
        assert retrieved.processing_time == 3.2
    
    def test_list_documents(self):
        """Test listing documents with filters."""
        from datetime import datetime
        
        # Store multiple documents
        for i in range(3):
            metadata = DocumentMetadata(
                document_id=f'test_doc_{i}',
                filename=f'test{i}.pdf',
                file_path=f'/path/to/test{i}.pdf',
                file_type='pdf',
                upload_timestamp=datetime.now(),
                processing_status=ProcessingStatus.COMPLETED if i < 2 else ProcessingStatus.FAILED,
                total_chunks=i * 5,
                file_size=1024 * (i + 1),
                checksum=f'hash{i}'
            )
            self.metadata_manager.store_document_metadata(f'test_doc_{i}', metadata)
        
        # List all documents
        all_docs = self.metadata_manager.list_documents()
        assert len(all_docs) == 3
        
        # List only completed documents
        completed_docs = self.metadata_manager.list_documents(status=ProcessingStatus.COMPLETED)
        assert len(completed_docs) == 2
        
        # List by file type
        pdf_docs = self.metadata_manager.list_documents(file_type='pdf')
        assert len(pdf_docs) == 3
    
    def test_get_statistics(self):
        """Test getting database statistics."""
        from datetime import datetime
        
        # Store some test documents
        for i in range(2):
            metadata = DocumentMetadata(
                document_id=f'stats_doc_{i}',
                filename=f'stats{i}.pdf',
                file_path=f'/path/to/stats{i}.pdf',
                file_type='pdf',
                upload_timestamp=datetime.now(),
                processing_status=ProcessingStatus.COMPLETED,
                total_chunks=5,
                file_size=1000,
                checksum=f'stats_hash{i}'
            )
            self.metadata_manager.store_document_metadata(f'stats_doc_{i}', metadata)
        
        # Get statistics
        stats = self.metadata_manager.get_statistics()
        
        assert stats['total_documents'] >= 2
        assert stats['total_chunks'] >= 10
        assert stats['total_file_size'] >= 2000
        assert 'pdf' in stats['documents_by_type']
        assert ProcessingStatus.COMPLETED.value in stats['documents_by_status']


class TestIngestionPipeline:
    """Test cases for DocumentIngestionPipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'silicon_flow_api_key': 'test_key',
            'groq_api_key': 'test_groq_key',
            'qdrant_url': 'http://localhost:6333',
            'qdrant_api_key': 'test_qdrant_key',
            'qdrant_collection': 'test_collection',
            'metadata_db_path': str(Path(self.temp_dir) / 'test_metadata.db'),
            'chunk_size': 100,
            'chunk_overlap': 20,
            'max_workers': 2,
            'image_processing': True
        }
    
    @patch('src.rag.ingestion_pipeline.EmbeddingSystem')
    @patch('src.rag.ingestion_pipeline.QdrantVectorStore')
    @patch('src.rag.ingestion_pipeline.MetadataManager')
    @patch('src.rag.ingestion_pipeline.ImageProcessor')
    def test_pipeline_initialization(self, mock_image, mock_metadata, mock_vector, mock_embedding):
        """Test pipeline initialization."""
        pipeline = DocumentIngestionPipeline(self.config)
        
        assert pipeline.chunk_size == 100
        assert pipeline.chunk_overlap == 20
        assert pipeline.max_workers == 2
        assert pipeline.enable_ocr is True
        
        mock_embedding.assert_called_once()
        mock_vector.assert_called_once()
        mock_metadata.assert_called_once()
        mock_image.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])