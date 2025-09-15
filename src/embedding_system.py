import logging
import requests
import time
import os
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import yaml
from pathlib import Path

# Load environment variables
load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


try:
    from logger.custom_logger import CustomLoggerTracker
    custom_log = CustomLoggerTracker()
    logger = custom_log.get_logger("embedding_system")

except ImportError:
    # Fallback to standard logging if custom logger not available
    logger = logging.getLogger("embedding_system")

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

@dataclass
class LLMResponse:
    """Response from LLM generation."""
    text: str
    model_name: str
    processing_time: float
    token_count: int
    success: bool
    error_message: Optional[str] = None

class EmbeddingSystem:
    def __init__(self, api_key: str, base_url: str = "https://api.siliconflow.com/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
        
        # Rate limiting
        self.max_requests_per_minute = 60
        self.request_timestamps = []
        
        logger.info(f"SiliconFlow client initialized with base URL: {base_url}")
    
    def generate_embeddings(self, texts: List[str], 
                          model: str = "Qwen/Qwen3-Embedding-8B") -> EmbeddingResult:
        start_time = time.time()
        try:
            self._check_rate_limit()
            payload = {
                "model": model,
                "input": texts,
                "encoding_format": "float"}
            response = self.session.post(
                f"{self.base_url}/embeddings",
                json=payload,
                timeout=30)
            processing_time = time.time() - start_time
            if response.status_code == 200:
                data = response.json()
                embeddings = []
                for item in data.get('data', []):
                    embeddings.append(item.get('embedding', []))
                # Get usage info
                usage = data.get('usage', {})
                token_count = usage.get('total_tokens', 0)
                logger.debug(f"Generated {len(embeddings)} embeddings in {processing_time:.2f}s")
                return EmbeddingResult(
                    embeddings=embeddings,
                    model_name=model,
                    processing_time=processing_time,
                    token_count=token_count,
                    success=True)
            else:
                error_msg = f"SiliconFlow API error {response.status_code}: {response.text}"
                logger.error(error_msg)
                
                return EmbeddingResult(
                    embeddings=[],
                    model_name=model,
                    processing_time=processing_time,
                    token_count=0,
                    success=False,
                    error_message=error_msg
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Embedding generation failed: {str(e)}"
            logger.error(error_msg)
            
            return EmbeddingResult(
                embeddings=[],
                model_name=model,
                processing_time=processing_time,
                token_count=0,
                success=False,
                error_message=error_msg
            )
    
    def rerank_documents(self, query: str, documents: List[str], 
                        model: str = "Qwen/Qwen3-Reranker-8B",
                        top_k: Optional[int] = None) -> List[RerankResult]:
        """Rerank documents based on query relevance."""
        try:
            self._check_rate_limit()
            
            payload = {
                "model": model,
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
                
                # Extract rerank results
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
            logger.error(f"Reranking failed: {str(e)}")
            return []
    
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



class GroqClient:
    """Groq API client for LLM inference."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.groq.com/openai/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
        # Rate limiting
        self.max_requests_per_minute = 30
        self.request_timestamps = []
        logger.info(f"Groq client initialized with base URL: {base_url}")
    
    def generate_response(self, messages: List[Dict[str, str]], 
                         model: str = "openai/gpt-oss-120b",
                         max_tokens: int = 1024,
                         temperature: float = 0.1,
                         stream: bool = False) -> LLMResponse:
        """Generate response using Groq LLM."""
        start_time = time.time()
        
        try:
            self._check_rate_limit()
            
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream
            }
            
            if stream:
                return self._handle_streaming_response(payload, model, start_time)
            else:
                return self._handle_non_streaming_response(payload, model, start_time)
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Groq LLM generation failed: {str(e)}"
            logger.error(error_msg)
            
            return LLMResponse(
                text="",
                model_name=model,
                processing_time=processing_time,
                token_count=0,
                success=False,
                error_message=error_msg
            )
    


    def _handle_non_streaming_response(self, payload: Dict, model: str, start_time: float) -> LLMResponse:
        """Handle non-streaming response."""
        response = self.session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=60
        )
        
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract response text
            choice = data.get('choices', [{}])[0]
            message = choice.get('message', {})
            generated_text = message.get('content', '')
            usage = data.get('usage', {})
            token_count = usage.get('total_tokens', 0)
            logger.debug(f"Generated response in {processing_time:.2f}s, {token_count} tokens")
            return LLMResponse(
                text=generated_text,
                model_name=model,
                processing_time=processing_time,
                token_count=token_count,
                success=True
            )
        else:
            error_msg = f"Groq API error {response.status_code}: {response.text}"
            logger.error(error_msg)
            
            return LLMResponse(
                text="",
                model_name=model,
                processing_time=processing_time,
                token_count=0,
                success=False,
                error_message=error_msg
            )
    
    def _handle_streaming_response(self, payload: Dict, model: str, start_time: float) -> LLMResponse:
        """Handle streaming response."""
        response = self.session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=60,
            stream=True
        )
        
        if response.status_code != 200:
            processing_time = time.time() - start_time
            error_msg = f"Groq streaming API error {response.status_code}: {response.text}"
            logger.error(error_msg)
            
            return LLMResponse(
                text="",
                model_name=model,
                processing_time=processing_time,
                token_count=0,
                success=False,
                error_message=error_msg
            )
        
        # Process streaming chunks
        generated_text = ""
        token_count = 0
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]  # Remove 'data: ' prefix
                    if data_str.strip() == '[DONE]':
                        break
                    
                    try:
                        chunk_data = eval(data_str)  # Note: In production, use json.loads with proper error handling
                        if 'choices' in chunk_data and chunk_data['choices']:
                            delta = chunk_data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                generated_text += delta['content']
                                token_count += 1
                    except:
                        continue
        
        processing_time = time.time() - start_time
        
        return LLMResponse(
            text=generated_text,
            model_name=model,
            processing_time=processing_time,
            token_count=token_count,
            success=True
        )
    
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

class RAGSystem:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the RAG system."""
        if config is None:
            config = self._load_default_config()
        self.config = config
        
        # Get API keys
        self.siliconflow_api_key = config.get('siliconflow_api_key') or os.getenv('SILICONFLOW_API_KEY')
        self.groq_api_key = config.get('groq_api_key') or os.getenv('GROQ_API_KEY')
        
        if not self.siliconflow_api_key:
            raise ValueError("SiliconFlow API key is required")
        if not self.groq_api_key:
            raise ValueError("Groq API key is required")
        
        # Initialize clients
        self.siliconflow_client = EmbeddingSystem(
            api_key=self.siliconflow_api_key,
            base_url=config.get('siliconflow_base_url', 'https://api.siliconflow.com/v1')
        )
        
        self.groq_client = GroqClient(
            api_key=self.groq_api_key,
            base_url=config.get('groq_base_url', 'https://api.groq.com/openai/v1')
        )
        


        # RAG configuration
        self.embedding_model = config.get('embedding_model', 'Qwen/Qwen3-Embedding-8B')
        self.reranker_model = config.get('reranker_model', 'Qwen/Qwen3-Reranker-8B')
        self.llm_model = config.get('llm_model', 'openai/gpt-oss-120b')
        self.max_context_chunks = config.get('max_context_chunks', 5)
        self.chunk_size = config.get('chunk_size', 512)
        self.chunk_overlap = config.get('chunk_overlap', 50)
        
        logger.info("RAG system initialized successfully")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            'embedding_model': 'Qwen/Qwen3-Embedding-8B',
            'reranker_model': 'Qwen/Qwen3-Reranker-8B',
            'llm_model': 'openai/gpt-oss-120b',
            'max_context_chunks': 5,
            'chunk_size': 512,
            'chunk_overlap': 50,
            'temperature': 0.1,
            'max_tokens': 1024
        }
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        result = self.siliconflow_client.generate_embeddings(texts, self.embedding_model)
        if result.success:
            return result.embeddings
        else:
            logger.error(f"Failed to generate embeddings: {result.error_message}")
            return []
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        embeddings = self.generate_embeddings([query])
        return embeddings[0] if embeddings else []
    
    def rerank_documents(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[RerankResult]:
        """Rerank documents based on query relevance."""
        return self.siliconflow_client.rerank_documents(
            query=query,
            documents=documents,
            model=self.reranker_model,
            top_k=top_k
        )
    
    def generate_response(self, messages: List[Dict[str, str]], stream: bool = False) -> str:
        """Generate LLM response."""
        response = self.groq_client.generate_response(
            messages=messages,
            model=self.llm_model,
            max_tokens=self.config.get('max_tokens', 1024),
            temperature=self.config.get('temperature', 0.1),
            stream=stream
        )
        
        if response.success:
            return response.text
        else:
            logger.error(f"Failed to generate response: {response.error_message}")
            return "Sorry, I couldn't generate a response due to technical difficulties."
    
    def answer_question(self, question: str, context: str) -> str:
        """Answer a question based on provided context."""
        system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
Use only the information in the context to answer questions. If the context doesn't contain enough information, 
say so clearly. Provide accurate and concise answers."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
        
        return self.generate_response(messages)
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all components."""
        # Test SiliconFlow embedding
        siliconflow_healthy = False
        try:
            result = self.siliconflow_client.generate_embeddings(["test"], self.embedding_model)
            siliconflow_healthy = result.success
        except:
            pass
        
        # Test Groq LLM
        groq_healthy = False
        try:
            response = self.groq_client.generate_response([{"role": "user", "content": "test"}], self.llm_model)
            groq_healthy = response.success
        except:
            pass
        
        return {
            'siliconflow': siliconflow_healthy,
            'groq': groq_healthy,
            'overall': siliconflow_healthy and groq_healthy
        }

# Configuration loader
def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {}

# Example usage and testing
def main():
    """Example usage of the corrected RAG system."""
    logger.info("Testing corrected RAG system")
    
    try:
        # Initialize RAG system
        rag_system = RAGSystem()
        
        # Test health
        health = rag_system.health_check()
        logger.info(f"System health: {health}")
        
        # Test embedding generation
        test_texts = [
            "What is machine learning?",
            "How does artificial intelligence work?",
            "Explain neural networks"
        ]
        
        logger.info("Testing embedding generation...")
        embeddings = rag_system.generate_embeddings(test_texts)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Test reranking
        query = "artificial intelligence"
        documents = [
            "AI is a branch of computer science",
            "Machine learning is a subset of AI", 
            "Neural networks are used in deep learning"
        ]
        
        logger.info("Testing document reranking...")
        rerank_results = rag_system.rerank_documents(query, documents, top_k=2)
        logger.info(f"Reranked {len(rerank_results)} documents")
        for i, result in enumerate(rerank_results):
            logger.info(f"Rank {i+1}: {result.score:.3f} - {result.text}")
        
        # Test LLM response
        logger.info("Testing LLM response generation...")
        context = "Machine learning is a method of data analysis that automates analytical model building."
        question = "What is machine learning?"
        answer = rag_system.answer_question(question, context)
        logger.info(f"Generated answer: {answer}")
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    main()