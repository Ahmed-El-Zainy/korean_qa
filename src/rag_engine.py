import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
import os 
import sys 


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedding_system import EmbeddingSystem, RerankResult
from src.vector_store import QdrantVectorStore, SearchResult
from src.groq_client import LLMSystem
from src.document_processor import DocumentChunk
from src.utilites import load_yaml_config


try:
    from logger.custom_logger import CustomLoggerTracker
    custom_log = CustomLoggerTracker()
    logger = custom_log.get_logger("rag_engine")

except ImportError:
    # Fallback to standard logging if custom logger not available
    logger = logging.getLogger("excel_processor")


@dataclass
class Citation:
    """Citation information for a source."""
    source_file: str
    page_number: Optional[int] = None
    worksheet_name: Optional[str] = None
    cell_range: Optional[str] = None
    section_title: Optional[str] = None
    text_snippet: str = ""
    confidence: float = 0.0
    chunk_id: str = ""


@dataclass
class RAGResponse:
    """Response from RAG system."""
    answer: str
    confidence_score: float
    citations: List[Citation] = field(default_factory=list)
    context_chunks: List[DocumentChunk] = field(default_factory=list)
    processing_time: float = 0.0
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    rerank_time: float = 0.0
    total_chunks_retrieved: int = 0
    total_chunks_reranked: int = 0
    model_used: str = ""
    success: bool = True
    error_message: Optional[str] = None


class RAGEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.embedding_system = EmbeddingSystem(config)
        self.vector_store = QdrantVectorStore(config)
        self.llm_system = LLMSystem(config)
        
        # RAG parameters
        self.max_context_chunks = config.get('max_context_chunks', 5)
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        self.rerank_top_k = config.get('rerank_top_k', 20)
        self.final_top_k = config.get('final_top_k', 5)
        self.max_context_length = config.get('max_context_length', 4000)
        logger.info(f"RAG engine initialized with max_context_chunks={self.max_context_chunks}")
    

    def answer_question(self, question: str, filters: Optional[Dict[str, Any]] = None) -> RAGResponse:
        start_time = time.time()
        try:
            logger.info(f"Processing question: {question[:100]}...")
            # Step 1: Generate query embedding
            query_embedding = self.embedding_system.generate_embeddings(question)
            if not query_embedding:
                return RAGResponse(
                    answer="I apologize, but I'm unable to process your question due to an embedding generation error.",
                    confidence_score=0.0,
                    success=False,
                    error_message="Failed to generate query embedding")
            
            # Step 2: Retrieve relevant chunks
            retrieval_start = time.time()
            search_results = self.vector_store.similarity_search(
                query_embedding=query_embedding,
                k=self.rerank_top_k,
                filters=filters)
            retrieval_time = time.time() - retrieval_start
            
            if not search_results:
                return RAGResponse(
                    answer="I couldn't find any relevant information to answer your question. Please try rephrasing your question or check if the relevant documents have been uploaded.",
                    confidence_score=0.0,
                    retrieval_time=retrieval_time,
                    processing_time=time.time() - start_time,
                    success=True)
            logger.info(f"Retrieved {len(search_results)} chunks from vector store in {retrieval_time:.2f}s")
            # Step 3: Rerank results
            rerank_start = time.time()
            reranked_chunks = self._rerank_chunks(question, search_results)
            rerank_time = time.time() - rerank_start
            
            # Step 4: Select top chunks and build context
            context_chunks = reranked_chunks[:self.final_top_k]
            context_text = self._build_context(context_chunks)
            logger.info(f"Built context from top {len(context_chunks)} chunks")
            
            # Step 5: Generate answer
            generation_start = time.time()
            answer = self.llm_system.answer_question(question, context_text)
            generation_time = time.time() - generation_start
            if not answer:
                return RAGResponse(
                    answer="I apologize, but I was unable to generate an answer to your question.",
                    confidence_score=0.0,
                    retrieval_time=retrieval_time,
                    generation_time=generation_time,
                    rerank_time=rerank_time,
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message="LLM failed to generate an answer")
                
            logger.info(f"Generated answer in {generation_time:.2f}s")
            # Step 6: Extract citations
            citations = self._extract_citations(context_chunks)
            logger.info(f"Extracted {len(citations)} citations")
            
            # Step 7: Calculate confidence score
            confidence_score = self._calculate_confidence_score(search_results, answer)
            logger.info(f"Calculated confidence score: {confidence_score:.2f}")
            
            total_time = time.time() - start_time
            
            response = RAGResponse(
                answer=answer,
                confidence_score=confidence_score,
                citations=citations,
                context_chunks=[result.chunk for result in context_chunks],
                processing_time=total_time,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                rerank_time=rerank_time,
                total_chunks_retrieved=len(search_results),
                total_chunks_reranked=len(reranked_chunks),
                model_used=self.llm_system.default_model,
                success=True)
            logger.info(f"Question answered successfully in {total_time:.2f}s")
            return response
            
        except Exception as e:
            error_msg = f"RAG processing failed: {str(e)}"
            logger.error(error_msg)
            
            return RAGResponse(
                answer="I apologize, but I encountered an error while processing your question. Please try again.",
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                success=False,
                error_message=error_msg
            )

    
    
    def get_relevant_context(self, question: str, k: int = 5, 
                           filters: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        try:
            # Generate query embedding
            query_embedding = self.embedding_system.generate_query_embedding(question)
            if not query_embedding:
                return []
            
            # Retrieve and rerank
            search_results = self.vector_store.similarity_search(
                query_embedding=query_embedding,
                k=min(k * 2, self.rerank_top_k),  # Get more for reranking
                filters=filters)
            if not search_results:
                return []
            
            # Rerank and return top k
            reranked_chunks = self._rerank_chunks(question, search_results)
            return [result.chunk for result in reranked_chunks[:k]]
            
        except Exception as e:
            logger.error(f"Failed to get relevant context: {e}")
            return []
    
    def _rerank_chunks(self, question: str, search_results: List[SearchResult]) -> List[SearchResult]:
        try:
            if len(search_results) <= 1:
                return search_results
            
            # Extract documents for reranking
            documents = [result.chunk.content for result in search_results]
            
            # Perform reranking
            rerank_results = self.embedding_system.rerank_results(
                query=question,
                documents=documents,
                top_k=len(documents)
            )
            
            # Map rerank results back to search results
            reranked_search_results = []
            for rerank_result in rerank_results:
                # Find corresponding search result
                original_index = rerank_result.index
                if 0 <= original_index < len(search_results):
                    search_result = search_results[original_index]
                    search_result.rerank_score = rerank_result.score
                    reranked_search_results.append(search_result)
            
            logger.debug(f"Reranked {len(search_results)} chunks")
            return reranked_search_results
            
        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            return search_results
    
    def _build_context(self, search_results: List[SearchResult]) -> str:
        """
        Build context text from search results.
        
        Args:
            search_results: List of search results
            
        Returns:
            Formatted context text
        """
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(search_results):
            chunk = result.chunk
            
            # Create context entry with citation info
            citation_info = self._format_citation_info(chunk)
            content = f"[Source {i+1}: {citation_info}]\n{chunk.content}\n"
            
            # Check if adding this chunk would exceed max context length
            if current_length + len(content) > self.max_context_length:
                # Try to fit a truncated version
                remaining_space = self.max_context_length - current_length - len(f"[Source {i+1}: {citation_info}]\n") - 20
                if remaining_space > 100:  # Only add if we have reasonable space
                    truncated_content = chunk.content[:remaining_space] + "..."
                    content = f"[Source {i+1}: {citation_info}]\n{truncated_content}\n"
                    context_parts.append(content)
                break
            
            context_parts.append(content)
            current_length += len(content)
        
        return "\n".join(context_parts)
    
    def _format_citation_info(self, chunk: DocumentChunk) -> str:
        """
        Format citation information for a chunk.
        
        Args:
            chunk: Document chunk
            
        Returns:
            Formatted citation string
        """
        parts = []
        
        # Add document ID or filename if available
        if hasattr(chunk.metadata, 'document_id'):
            parts.append(f"Doc: {chunk.metadata.document_id}")
        
        # Add page number for PDFs
        if chunk.metadata.page_number:
            parts.append(f"Page {chunk.metadata.page_number}")
        
        # Add worksheet info for Excel
        if chunk.metadata.worksheet_name:
            parts.append(f"Sheet: {chunk.metadata.worksheet_name}")
            if chunk.metadata.cell_range:
                parts.append(f"Range: {chunk.metadata.cell_range}")
        
        # Add section title if available
        if chunk.metadata.section_title:
            parts.append(f"Section: {chunk.metadata.section_title}")
        
        return ", ".join(parts) if parts else "Unknown source"
    
    def _extract_citations(self, search_results: List[SearchResult]) -> List[Citation]:
        citations = []
        for result in search_results:
            chunk = result.chunk
            
            # Create citation
            citation = Citation(
                source_file=getattr(chunk.metadata, 'document_id', 'Unknown'),
                page_number=chunk.metadata.page_number,
                worksheet_name=chunk.metadata.worksheet_name,
                cell_range=chunk.metadata.cell_range,
                section_title=chunk.metadata.section_title,
                text_snippet=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                confidence=result.similarity_score,
                chunk_id=chunk.metadata.chunk_id
            )
            
            citations.append(citation)
        
        return citations
    
    def _calculate_confidence_score(self, search_results: List[SearchResult], answer: str) -> float:
        if not search_results:
            return 0.0
        
        # Base confidence on similarity scores
        similarity_scores = [result.similarity_score for result in search_results]
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        
        # Adjust based on number of sources
        source_factor = min(len(search_results) / self.final_top_k, 1.0)
        
        # Adjust based on answer length (very short answers might be less reliable)
        length_factor = min(len(answer) / 100, 1.0) if answer else 0.0
        
        # Combine factors
        confidence = (avg_similarity * 0.6 + source_factor * 0.2 + length_factor * 0.2)
        
        return min(max(confidence, 0.0), 1.0)  # Clamp to [0, 1]
    
    
    
    def health_check(self) -> Dict[str, bool]:
        return {
            "vector_store": self.vector_store.health_check(),
            "llm_system": self.llm_system.client.health_check(),
            "embedding_system": True  # Silicon Flow doesn't have a direct health check
        }
    
    def get_stats(self) -> Dict[str, Any]:
        try:
            vector_stats = self.vector_store.get_collection_info()
            embedding_stats = self.embedding_system.get_cache_stats()
            
            return {
                "vector_store": vector_stats.__dict__ if vector_stats else {},
                "embedding_cache": embedding_stats,
                "config": {
                    "max_context_chunks": self.max_context_chunks,
                    "similarity_threshold": self.similarity_threshold,
                    "rerank_top_k": self.rerank_top_k,
                    "final_top_k": self.final_top_k
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get RAG stats: {e}")
            return {"error": str(e)}
        


if __name__ == "__main__":
    from src.utilites import validate_api_keys
    validation_results = validate_api_keys()
    if not validation_results['valid']:
        logger.error("Missing required API keys. Please set them in the environment variables.")
    else:
        logger.info("All required API keys are present.")
    ## Example usage
    config = load_yaml_config("src/config.yaml")
    rag_engine = RAGEngine(config)
    response = rag_engine.answer_question("What is the capital of France?")
    logger.info(f"Answer: {response.answer}, Confidence: {response.confidence_score}")
    rag_engine.health_check()
    rag_engine.get_stats()
