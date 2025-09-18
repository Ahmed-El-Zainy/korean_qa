import logging
import requests
import time
import os
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from logger.custom_logger import CustomLoggerTracker
    custom_log = CustomLoggerTracker()
    logger = custom_log.get_logger("embedding_system")
except ImportError:
    # Fallback to standard logging if custom logger not available
    logger = logging.getLogger("embedding_system")



SILICONFLOW_API_KEY = os.environ['SILICONFLOW_API_KEY']
@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    embeddings: List[List[float]]
    model_name: str
    processing_time: float
    token_count: int
    success: bool
    error_message: Optional[str] = None


@dataclass
class RerankResult:
    """Result of reranking operation."""
    text: str
    score: float
    index: int


class EmbeddingSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Get API configuration
        self.api_key = SILICONFLOW_API_KEY
        if not self.api_key:
            raise ValueError("SiliconFlow API key is required")
        
        # API endpoints
        self.base_url = "https://api.siliconflow.com/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
        
        # Model configuration from your config
        self.embedding_model = config.get('embedding_model', 'Qwen/Qwen3-Embedding-8B')
        self.reranker_model = config.get('reranker_model', 'Qwen/Qwen3-Reranker-8B')
        
        # Rate limiting
        self.max_requests_per_minute = 60
        self.request_timestamps = []
        
        logger.info(f"EmbeddingSystem initialized with model: {self.embedding_model}")
    
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            logger.warning("No texts provided for embedding generation")
            return []
        
        try:
            self._check_rate_limit()
            
            payload = {
                "model": self.embedding_model,
                "input": texts,
                "encoding_format": "float"
            }
            
            response = self.session.post(
                f"{self.base_url}/embeddings",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                embeddings = [item['embedding'] for item in data.get('data', [])]
                
                if len(embeddings) != len(texts):
                    logger.warning(f"Expected {len(texts)} embeddings, got {len(embeddings)}")
                
                logger.debug(f"Generated {len(embeddings)} embeddings")
                return embeddings
                
            else:
                error_msg = f"SiliconFlow API error {response.status_code}: {response.text}"
                logger.error(error_msg)
                return []
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []
    
    def generate_query_embedding(self, query: str) -> List[float]:
        embeddings = self.generate_embeddings([query])
        return embeddings[0] if embeddings else []
    
    def rerank_documents(self, query: str, documents: List[str], 
                        top_k: Optional[int] = None) -> List[RerankResult]:
        if not documents:
            return []
        
        try:
            self._check_rate_limit()
            
            payload = {
                "model": self.reranker_model,
                "query": query,
                "documents": documents,
                "top_k": top_k or len(documents),
                "return_documents": True
            }
            
            response = self.session.post(
                f"{self.base_url}/rerank",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for item in data.get('results', []):
                    results.append(RerankResult(
                        text=item.get('document', {}).get('text', ''),
                        score=item.get('relevance_score', 0.0),
                        index=item.get('index', 0)
                    ))
                
                # Sort by score (descending)
                results.sort(key=lambda x: x.score, reverse=True)
                logger.debug(f"Reranked {len(results)} documents")
                return results
                
            else:
                error_msg = f"SiliconFlow rerank API error {response.status_code}: {response.text}"
                logger.error(error_msg)
                return []
                
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return []
    
    def rerank_results(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[RerankResult]:
        """Alias for rerank_documents to match the interface expected by rag_engine."""
        return self.rerank_documents(query, documents, top_k)
    
    def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if current_time - ts < 60
        ]
        
        # Check if we're at the rate limit
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            sleep_time = 60 - (current_time - self.request_timestamps[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        # Add current request timestamp
        self.request_timestamps.append(current_time)
    
    def test_api_connection(self) -> Dict[str, Any]:
        """Test the API connection."""
        if not self.api_key:
            return {
                'success': False,
                'error': 'API key not set',
                'details': 'Please set the SILICONFLOW_API_KEY environment variable'
            }
        
        try:
            # Test with a simple embedding request
            test_payload = {
                "model": self.embedding_model,
                "input": ["test connection"],
                "encoding_format": "float"
            }
            
            response = self.session.post(
                f"{self.base_url}/embeddings",
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'message': 'API connection successful',
                    'status_code': response.status_code,
                    'model': self.embedding_model
                }
            else:
                return {
                    'success': False,
                    'error': f'API error {response.status_code}',
                    'details': response.text[:200],
                    'status_code': response.status_code
                }
        
        except Exception as e:
            return {
                'success': False,
                'error': 'Connection failed',
                'details': str(e)
            }
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics (placeholder for compatibility)."""
        return {
            "caching_disabled": True,
            "note": "Caching not implemented in this version"
        }


# Test function
def test_embedding_system():
    """Test the embedding system with your configuration."""
    print("🧪 Testing SiliconFlow Embedding System")
    print("-" * 40)
    
    # Test configuration
    config = {
        'siliconflow_api_key': os.getenv('SILICONFLOW_API_KEY'),
        'embedding_model': 'Qwen/Qwen3-Embedding-8B',
        'reranker_model': 'Qwen/Qwen3-Reranker-8B'
    }
    
    try:
        # Initialize system
        embedding_system = EmbeddingSystem(config)
        print("✅ System initialized")
        
        # Test API connection
        connection_test = embedding_system.test_api_connection()
        if connection_test['success']:
            print("✅ API connection successful")
        else:
            print(f"❌ API connection failed: {connection_test['error']}")
            return
        
        # Test embedding generation
        test_texts = [
            "What is the production yield?",
            "How is quality controlled in manufacturing?",
            "What safety measures are in place?"
        ]
        
        print(f"🔄 Generating embeddings for {len(test_texts)} texts...")
        embeddings = embedding_system.generate_embeddings(test_texts)
        
        if embeddings and len(embeddings) == len(test_texts):
            print(f"✅ Generated {len(embeddings)} embeddings of size {len(embeddings[0])}")
        else:
            print(f"❌ Embedding generation failed. Got {len(embeddings)} embeddings")
            return
        
        # Test reranking
        query = "manufacturing quality control"
        documents = [
            "Quality control processes ensure product reliability",
            "Manufacturing efficiency can be improved through automation",
            "Safety protocols are essential in industrial settings"
        ]
        
        print(f"🔄 Testing reranking with query: '{query}'")
        rerank_results = embedding_system.rerank_documents(query, documents)
        
        if rerank_results:
            print(f"✅ Reranking successful. Top result score: {rerank_results[0].score:.3f}")
            for i, result in enumerate(rerank_results):
                print(f"  {i+1}. Score: {result.score:.3f} - {result.text[:50]}...")
        else:
            print("❌ Reranking failed")
            return
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_embedding_system()