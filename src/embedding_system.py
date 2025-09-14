import logging
import requests
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import os 
import sys 
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


     
      
from src.utilites import load_yaml_config

try:
    from logger.custom_logger import CustomLoggerTracker
    custom_log = CustomLoggerTracker()
    logger = custom_log.get_logger("embedding_system")

except ImportError:
    # Fallback to standard logging if custom logger not available
    logger = logging.getLogger("embedding_system")


config = load_yaml_config("src/config.yaml")
SILICONFLOW_EMBEDDING_URL = os.environ["SILICONFLOW_EMBEDDING_URL"]
SILICONFLOW_API_KEY = os.environ["SILICONFLOW_API_KEY"]
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

def get_api_key() -> str:
    """Retrieve the API key from environment variables."""
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        raise ValueError("API key not found in environment variables.")
    return api_key


EMBEDDING_URL = os.environ["SILICONFLOW_EMBEDDING_URL"]
RERANKER_URL = os.environ["SILICONFLOW_RERANKING_URL"]
SILICONFLOW_URL = os.environ["SILICONFLOW_URL"]




class SiliconFlowEmbeddingClient:
    def __init__(self, api_key: str, base_url: str = SILICONFLOW_URL):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'})
        self.max_requests_per_minute = 60
        self.request_timestamps = []
        logger.info(f"Silicon Flow embedding client initialized with base URL: {base_url}")
    
    def generate_embeddings(self, texts: List[str]) -> EmbeddingResult:
        start_time = time.time()
        try:
            self._check_rate_limit()
            payload = {
                "model": config["silicon_flow"]["embedding_model"],
                "input": texts,
                "encoding_format": "float"}
            # Make API request
            response = self.session.post(
                f"{self.base_url}/embeddings",
                json=payload,
                timeout=30)
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract embeddings
                embeddings = []
                for item in data.get('data', []):
                    embeddings.append(item.get('embedding', []))
                # Get usage info
                usage = data.get('usage', {})
                token_count = usage.get('total_tokens', 0)
                logger.debug(f"Generated {len(embeddings)} embeddings in {processing_time:.2f}s")
                return EmbeddingResult(
                    embeddings=embeddings,
                    model_name=config["silicon_flow"]["embedding_model"],
                    processing_time=processing_time,
                    token_count=token_count,
                    success=True)
            else:
                error_msg = f"API request failed with status {response.status_code}: {response.text}"
                logger.error(error_msg)
                
                return EmbeddingResult(
                    embeddings=[],
                    model_name=config["silicon_flow"]["embedding_model"],
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
                model_name=config["silicon_flow"]["embedding_model"],
                processing_time=processing_time,
                token_count=0,
                success=False,
                error_message=error_msg
            )

    
    def rerank_documents(self, query: str, documents: List[str], 
                        top_k: Optional[int] = None) -> List[RerankResult]:
        try:
            # Rate limiting check
            self._check_rate_limit()
            # Prepare request
            payload = {
                "model": config["silicon_flow"]["reranker"],
                "query": query,
                "documents": documents,
                "top_k": top_k or len(documents),
                "return_documents": True}
            
            # Make API request
            response = self.session.post(
                f"{self.base_url}/rerank",
                json=payload,
                timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract rerank results
                results = []
                for item in data.get('results', []):
                    results.append(RerankResult(
                        text=item.get('document', {}).get('text', ''),
                        score=item.get('relevance_score', 0.0),
                        index=item.get('index', 0)))
                
                # Sort by score (descending)
                results.sort(key=lambda x: x.score, reverse=True)
                
                logger.debug(f"Reranked {len(results)} documents")
                return results
            else:
                error_msg = f"Rerank request failed with status {response.status_code}: {response.text}"
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
            if current_time - ts < 60]
        
        # Check if we're at the rate limit
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            sleep_time = 60 - (current_time - self.request_timestamps[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        # Add current request timestamp
        self.request_timestamps.append(current_time)


class EmbeddingSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = os.environ["SILICONFLOW_API_KEY"]
        self.embedding_model = config.get('embedding_model', 'BAAI/bge-large-zh-v1.5')
        self.reranker_model = config.get('reranker_model', 'BAAI/bge-reranker-large')
        self.batch_size = config.get('batch_size', 32)
        self.max_retries = config.get('max_retries', 3)
        self.cache_enabled = config.get('enable_embedding_cache', True)
        
        if not self.api_key:
            raise ValueError("Silicon Flow API key is required")
        
        # Initialize client
        self.client = SiliconFlowEmbeddingClient(self.api_key)
        
        # Simple in-memory cache
        self.embedding_cache = {} if self.cache_enabled else None
        
        logger.info(f"Embedding system initialized with model: {self.embedding_model}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Check cache first
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            if self.cache_enabled:
                for j, text in enumerate(batch):
                    cache_key = self._get_cache_key(text)
                    if cache_key in self.embedding_cache:
                        cached_embeddings.append((i + j, self.embedding_cache[cache_key]))
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i + j)
            else:
                uncached_texts = batch
                uncached_indices = list(range(i, i + len(batch)))
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                batch_embeddings = self._generate_batch_embeddings(uncached_texts)
                
                # Cache new embeddings
                if self.cache_enabled and batch_embeddings:
                    for text, embedding in zip(uncached_texts, batch_embeddings):
                        cache_key = self._get_cache_key(text)
                        self.embedding_cache[cache_key] = embedding
                
                # Combine cached and new embeddings
                new_embeddings = [(idx, emb) for idx, emb in zip(uncached_indices, batch_embeddings)]
                all_batch_embeddings = cached_embeddings + new_embeddings
            else:
                all_batch_embeddings = cached_embeddings
            
            # Sort by original index and extract embeddings
            all_batch_embeddings.sort(key=lambda x: x[0])
            batch_result = [emb for _, emb in all_batch_embeddings]
            all_embeddings.extend(batch_result)
        
        logger.info(f"Generated embeddings for {len(texts)} texts")
        return all_embeddings
    
    def generate_query_embedding(self, query: str) -> List[float]:
        embeddings = self.generate_embeddings([query])
        return embeddings[0] if embeddings else []
    
    def rerank_results(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[RerankResult]:
        if not documents:
            return []
        try:
            results = self.client.rerank_documents(
                query=query,
                documents=documents,
                model=self.reranker_model,
                top_k=top_k)
            
            logger.debug(f"Reranked {len(documents)} documents, returning top {len(results)}")
            return results
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original order with default scores
            return [
                RerankResult(text=doc, score=1.0 - (i * 0.1), index=i)
                for i, doc in enumerate(documents[:top_k] if top_k else documents)]
    
    
    def _generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        for attempt in range(self.max_retries):
            try:
                result = self.client.generate_embeddings(texts)
                if result.success:
                    return result.embeddings
                else:
                    logger.warning(f"Embedding generation failed (attempt {attempt + 1}): {result.error_message}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.warning(f"Embedding generation error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        
        # Return empty embeddings if all attempts failed
        logger.error(f"Failed to generate embeddings after {self.max_retries} attempts")
        return [[] for _ in texts]
    
    def _get_cache_key(self, text: str) -> str:
        import hashlib
        return hashlib.md5(f"{self.embedding_model}:{text}".encode()).hexdigest()
    
    def clear_cache(self):
        """Clear the embedding cache."""
        if self.cache_enabled:
            self.embedding_cache.clear()
            logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        if not self.cache_enabled:
            return {"cache_enabled": False}
        return {
            "cache_enabled": True,
            "cache_size": len(self.embedding_cache),
            "model": self.embedding_model}




# ─── Chunker & Embed ──────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=config["chunking"]["chunk_size"],
    chunk_overlap=config["chunking"]["chunk_size"],
    separators=config["chunking"]["chunk_size"],)


def embed_texts(texts: list[str], batch_size: int = 50) -> list[list[float]]:
    all_embeddings = []
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"}
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        logger.info(f"Embedding Model: {config['silicon_flow']['embeding']}")
        payload = {
            "model": config["silicon_flow"]["embeding"],
            "input": batch}
        response = requests.post(
            SILICONFLOW_EMBEDDING_URL, json=payload, headers=headers)
        # response.raise_for_status()
        data = response.json()
        if "data" not in data:
            raise ValueError(f"Invalid response format: {data}")
        batch_embs = [item["embedding"] for item in data["data"]]
        all_embeddings.extend(batch_embs)
        # Fill with empty embeddings in case of failure
        all_embeddings.extend([[] for _ in batch])
    return all_embeddings




        
if __name__=="__main__":
    logger.info("Start Embedding and Reraning Module")
    text = "TAB S10 도장 공정 수율이 어떻게 되나요?"
    embedding_system = EmbeddingSystem(config)
    embedding = embedding_system.generate_query_embedding(text)
    logger.info(embedding)
    logger.info(embedding_system.get_cache_stats())
    embedding_system.clear_cache()
    
        