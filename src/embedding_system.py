import logging
import requests
import time
import os
import sys
import hashlib
import json
import sqlite3
import pickle
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import threading
from collections import defaultdict

# Load environment variables
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.utilites import load_config_yaml

config = load_config_yaml("src/config.yaml")
    
try:
    from logger.custom_logger import CustomLoggerTracker
    custom_log = CustomLoggerTracker()
    logger = custom_log.get_logger("embedding_system")
except ImportError:
    # Fallback to standard logging if custom logger not available
    logger = logging.getLogger("embedding_system")


if config is None:
    logger.error(f"Logger has NULL values")
    
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

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    data: Any
    model_name: str
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int

class EmbeddingCache:
    """Multi-level caching system for embeddings."""
    
    def __init__(self, 
                 cache_dir: str = "cache",
                 max_memory_size: int = 100 * 1024 * 1024,  # 100MB
                 max_disk_size: int = 1024 * 1024 * 1024,   # 1GB
                 ttl_hours: int = 24 * 7):  # 7 days
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.max_memory_size = max_memory_size
        self.max_disk_size = max_disk_size
        self.ttl = timedelta(hours=ttl_hours)
        
        # In-memory cache (LRU-like)
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.memory_size = 0
        
        # Thread lock for thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'memory_hits': 0,
            'disk_hits': 0,
            'evictions': 0,
            'total_size': 0
        }
        
        # Initialize SQLite database for metadata
        self.db_path = self.cache_dir / "cache_metadata.db"
        self._init_database()
        
        # Clean expired entries on startup
        self._cleanup_expired_entries()
        
        logger.info(f"Cache initialized: {cache_dir}, memory limit: {max_memory_size//1024//1024}MB, "
                   f"disk limit: {max_disk_size//1024//1024}MB, TTL: {ttl_hours}h")
    
    def _init_database(self):
        """Initialize SQLite database for cache metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    model_name TEXT,
                    created_at TIMESTAMP,
                    last_accessed TIMESTAMP,
                    access_count INTEGER,
                    size_bytes INTEGER,
                    file_path TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed 
                ON cache_entries(last_accessed)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_name 
                ON cache_entries(model_name)
            """)
    
    def _generate_cache_key(self, texts: List[str], model: str) -> str:
        """Generate a unique cache key for the input."""
        # Create a deterministic hash of the input
        content = json.dumps({
            'texts': sorted(texts),  # Sort to ensure consistent ordering
            'model': model
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_file_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        # Use first 2 chars as subdirectory for better file system performance
        subdir = self.cache_dir / key[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{key}.pkl"
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for storage."""
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data from storage."""
        return pickle.loads(data)
    
    def get(self, texts: List[str], model: str) -> Optional[EmbeddingResult]:
        """Get embeddings from cache."""
        key = self._generate_cache_key(texts, model)
        
        with self._lock:
            # Check memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if self._is_entry_valid(entry):
                    entry.last_accessed = datetime.now()
                    entry.access_count += 1
                    self.stats['hits'] += 1
                    self.stats['memory_hits'] += 1
                    logger.debug(f"Cache hit (memory): {key[:8]}...")
                    return entry.data
                else:
                    # Remove expired entry
                    self._remove_from_memory(key)
            
            # Check disk cache
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT * FROM cache_entries WHERE key = ?", (key,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        key_db, model_name, created_at, last_accessed, access_count, size_bytes, file_path = row
                        created_at = datetime.fromisoformat(created_at)
                        last_accessed = datetime.fromisoformat(last_accessed)
                        
                        # Check if entry is still valid
                        if datetime.now() - created_at <= self.ttl:
                            file_path = Path(file_path)
                            if file_path.exists():
                                try:
                                    with open(file_path, 'rb') as f:
                                        data = self._deserialize_data(f.read())
                                    
                                    # Update access metadata
                                    last_accessed = datetime.now()
                                    access_count += 1
                                    
                                    conn.execute(
                                        "UPDATE cache_entries SET last_accessed = ?, access_count = ? WHERE key = ?",
                                        (last_accessed.isoformat(), access_count, key)
                                    )
                                    
                                    # Add to memory cache if there's space
                                    self._add_to_memory(key, CacheEntry(
                                        key=key,
                                        data=data,
                                        model_name=model_name,
                                        created_at=created_at,
                                        last_accessed=last_accessed,
                                        access_count=access_count,
                                        size_bytes=size_bytes
                                    ))
                                    
                                    self.stats['hits'] += 1
                                    self.stats['disk_hits'] += 1
                                    logger.debug(f"Cache hit (disk): {key[:8]}...")
                                    return data
                                    
                                except Exception as e:
                                    logger.warning(f"Failed to read cache file {file_path}: {e}")
                        
                        # Remove expired or corrupted entry
                        self._remove_from_disk(key)
            
            except Exception as e:
                logger.warning(f"Database error during cache lookup: {e}")
            
            self.stats['misses'] += 1
            logger.debug(f"Cache miss: {key[:8]}...")
            return None
    
    def put(self, texts: List[str], model: str, result: EmbeddingResult):
        """Store embeddings in cache."""
        if not result.success:
            return  # Don't cache failed results
        
        key = self._generate_cache_key(texts, model)
        
        with self._lock:
            try:
                # Serialize data
                serialized_data = self._serialize_data(result)
                size_bytes = len(serialized_data)
                
                now = datetime.now()
                entry = CacheEntry(
                    key=key,
                    data=result,
                    model_name=model,
                    created_at=now,
                    last_accessed=now,
                    access_count=1,
                    size_bytes=size_bytes
                )
                
                # Store to disk
                file_path = self._get_file_path(key)
                with open(file_path, 'wb') as f:
                    f.write(serialized_data)
                
                # Update database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """INSERT OR REPLACE INTO cache_entries 
                           (key, model_name, created_at, last_accessed, access_count, size_bytes, file_path)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (key, model, now.isoformat(), now.isoformat(), 1, size_bytes, str(file_path))
                    )
                
                # Add to memory cache
                self._add_to_memory(key, entry)
                
                # Check disk space and cleanup if needed
                self._cleanup_disk_space()
                
                logger.debug(f"Cached embeddings: {key[:8]}, size: {size_bytes} bytes")
                
            except Exception as e:
                logger.error(f"Failed to cache embeddings: {e}")
    
    def _is_entry_valid(self, entry: CacheEntry) -> bool:
        """Check if cache entry is still valid."""
        return datetime.now() - entry.created_at <= self.ttl
    
    def _add_to_memory(self, key: str, entry: CacheEntry):
        """Add entry to memory cache with LRU eviction."""
        # Remove if already exists
        if key in self.memory_cache:
            self._remove_from_memory(key)
        
        # Evict entries if memory limit would be exceeded
        while (self.memory_size + entry.size_bytes > self.max_memory_size and 
               len(self.memory_cache) > 0):
            self._evict_lru_memory()
        
        # Add new entry
        self.memory_cache[key] = entry
        self.memory_size += entry.size_bytes
    
    def _remove_from_memory(self, key: str):
        """Remove entry from memory cache."""
        if key in self.memory_cache:
            entry = self.memory_cache.pop(key)
            self.memory_size -= entry.size_bytes
    
    def _evict_lru_memory(self):
        """Evict least recently used entry from memory."""
        if not self.memory_cache:
            return
        
        # Find LRU entry
        lru_key = min(self.memory_cache.keys(), 
                     key=lambda k: self.memory_cache[k].last_accessed)
        
        self._remove_from_memory(lru_key)
        self.stats['evictions'] += 1
        logger.debug(f"Evicted from memory: {lru_key[:8]}")
    
    def _remove_from_disk(self, key: str):
        """Remove entry from disk cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT file_path FROM cache_entries WHERE key = ?", (key,))
                row = cursor.fetchone()
                if row:
                    file_path = Path(row[0])
                    if file_path.exists():
                        file_path.unlink()
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
        except Exception as e:
            logger.warning(f"Failed to remove cache entry {key}: {e}")
    
    def _cleanup_disk_space(self):
        """Cleanup disk space by removing old entries."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get total cache size
                cursor = conn.execute("SELECT SUM(size_bytes) FROM cache_entries")
                total_size = cursor.fetchone()[0] or 0
                
                if total_size > self.max_disk_size:
                    # Remove oldest entries until under limit
                    cursor = conn.execute(
                        "SELECT key, size_bytes FROM cache_entries ORDER BY last_accessed ASC"
                    )
                    
                    for key, size_bytes in cursor.fetchall():
                        if total_size <= self.max_disk_size * 0.8:  # Clean to 80% of limit
                            break
                        
                        self._remove_from_disk(key)
                        self._remove_from_memory(key)  # Also remove from memory if present
                        total_size -= size_bytes
                        self.stats['evictions'] += 1
                        
                    logger.info(f"Cleaned disk cache, new size: {total_size} bytes")
        
        except Exception as e:
            logger.error(f"Failed to cleanup disk space: {e}")
    
    def _cleanup_expired_entries(self):
        """Remove expired entries from cache."""
        try:
            cutoff_time = datetime.now() - self.ttl
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT key FROM cache_entries WHERE created_at < ?",
                    (cutoff_time.isoformat(),)
                )
                
                expired_keys = [row[0] for row in cursor.fetchall()]
                
                for key in expired_keys:
                    self._remove_from_disk(key)
                    self._remove_from_memory(key)
                
                if expired_keys:
                    logger.info(f"Cleaned {len(expired_keys)} expired cache entries")
        
        except Exception as e:
            logger.error(f"Failed to cleanup expired entries: {e}")
    
    def clear(self, model: Optional[str] = None):
        """Clear cache entries, optionally for a specific model."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    if model:
                        cursor = conn.execute(
                            "SELECT key FROM cache_entries WHERE model_name = ?", (model,)
                        )
                    else:
                        cursor = conn.execute("SELECT key FROM cache_entries")
                    
                    keys_to_remove = [row[0] for row in cursor.fetchall()]
                    
                    for key in keys_to_remove:
                        self._remove_from_disk(key)
                        self._remove_from_memory(key)
                    
                    logger.info(f"Cleared {len(keys_to_remove)} cache entries" + 
                              (f" for model {model}" if model else ""))
            
            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT COUNT(*), SUM(size_bytes), AVG(access_count) FROM cache_entries"
                    )
                    total_entries, total_size, avg_access = cursor.fetchone()
                    
                    # Get model breakdown
                    cursor = conn.execute(
                        "SELECT model_name, COUNT(*), SUM(size_bytes) FROM cache_entries GROUP BY model_name"
                    )
                    model_stats = {}
                    for model, count, size in cursor.fetchall():
                        model_stats[model] = {'entries': count, 'size_bytes': size or 0}
                
                hit_rate = (self.stats['hits'] / (self.stats['hits'] + self.stats['misses']) 
                           if (self.stats['hits'] + self.stats['misses']) > 0 else 0)
                
                return {
                    'total_entries': total_entries or 0,
                    'total_size_bytes': total_size or 0,
                    'total_size_mb': (total_size or 0) / (1024 * 1024),
                    'memory_entries': len(self.memory_cache),
                    'memory_size_bytes': self.memory_size,
                    'memory_size_mb': self.memory_size / (1024 * 1024),
                    'hit_rate': hit_rate,
                    'hits': self.stats['hits'],
                    'misses': self.stats['misses'],
                    'memory_hits': self.stats['memory_hits'],
                    'disk_hits': self.stats['disk_hits'],
                    'evictions': self.stats['evictions'],
                    'average_access_count': avg_access or 0,
                    'model_breakdown': model_stats
                }
            
            except Exception as e:
                logger.error(f"Failed to get cache stats: {e}")
                return {'error': str(e)}





class EmbeddingSystem:
    def __init__(self, 
                 api_key: str = os.getenv('SILICONFLOW_API_KEY'), 
                 base_url: str = os.environ["SILICONFLOW_URL"],
                 enable_cache: bool = True,
                 cache_dir: str = "cache",
                 cache_ttl_hours: int = 24 * 7):
        
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
        
        # Initialize cache
        self.enable_cache = enable_cache
        if self.enable_cache:
            self.cache = EmbeddingCache(
                cache_dir=cache_dir,
                ttl_hours=cache_ttl_hours
            )
        else:
            self.cache = None
        
        logger.info(f"SiliconFlow client initialized with base URL: {base_url}, caching: {enable_cache}")
    
    def generate_embeddings(self, texts: List[str], 
                          model: str = "Qwen/Qwen3-Embedding-8B") -> EmbeddingResult:
        
        # Try cache first
        if self.enable_cache and self.cache:
            cached_result = self.cache.get(texts, model)
            if cached_result is not None:
                return cached_result
        
        start_time = time.time()
        try:
            # Validate API key
            if not self.api_key:
                error_msg = "SiliconFlow API key is not set. Please check your SILICONFLOW_API_KEY environment variable."
                logger.error(error_msg)
                return EmbeddingResult(
                    embeddings=[],
                    model_name=model,
                    processing_time=0,
                    token_count=0,
                    success=False,
                    error_message=error_msg
                )
            
            self._check_rate_limit()
            payload = {
                "model": model,
                "input": texts,
                "encoding_format": "float"
            }
            
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
                
                # Validate embeddings
                if not embeddings or not all(embeddings):
                    error_msg = "Empty or invalid embeddings received from API"
                    logger.error(error_msg)
                    return EmbeddingResult(
                        embeddings=[],
                        model_name=model,
                        processing_time=processing_time,
                        token_count=0,
                        success=False,
                        error_message=error_msg)
                
                # Get usage info
                usage = data.get('usage', {})
                token_count = usage.get('total_tokens', 0)
                
                result = EmbeddingResult(
                    embeddings=embeddings,
                    model_name=model,
                    processing_time=processing_time,
                    token_count=token_count,
                    success=True)
                
                # Cache the result
                if self.enable_cache and self.cache:
                    self.cache.put(texts, model, result)
                
                logger.debug(f"Generated {len(embeddings)} embeddings in {processing_time:.2f}s")
                return result
            elif response.status_code == 401:
                error_msg = "Invalid SiliconFlow API token. Please check your SILICONFLOW_API_KEY environment variable."
                logger.error(error_msg)
                return EmbeddingResult(
                    embeddings=[],
                    model_name=model,
                    processing_time=processing_time,
                    token_count=0,
                    success=False,
                    error_message=error_msg
                )
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
                timeout=30)
            
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
        
        # This creates a sliding window by filtering out timestamps older than one minute,
        # ensuring the rate limit is based only on recent requests.
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if current_time - ts < 60
        ]
        
        # If the request count within the window meets or exceeds the limit,
        # the system must pause.
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            # The sleep duration is calculated as the time remaining until the
            # oldest request in the window "expires" (falls out of the 60-second window).
            sleep_time = 60 - (current_time - self.request_timestamps[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        # The timestamp of the current request is added to the window to track it.
        self.request_timestamps.append(current_time)

    def test_api_connection(self) -> Dict[str, Any]:
        """Test the API connection and return status."""
        if not self.api_key:
            return {
                'success': False,
                'error': 'API key not set',
                'details': 'Please set the SILICONFLOW_API_KEY environment variable'
            }
        
        try:
            # Test with a simple embedding request
            test_payload = {
                "model": "Qwen/Qwen3-Embedding-8B",
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
                    'status_code': response.status_code
                }
            elif response.status_code == 401:
                return {
                    'success': False,
                    'error': 'Invalid API token',
                    'details': 'Check your SILICONFLOW_API_KEY environment variable',
                    'status_code': response.status_code
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
        """Get statistics about the embedding cache."""
        if not self.enable_cache or not self.cache:
            return {"caching_disabled": True}
        
        return self.cache.get_stats()
    
    def clear_cache(self, model: Optional[str] = None):
        """Clear cache entries, optionally for a specific model."""
        if self.enable_cache and self.cache:
            self.cache.clear(model)
            logger.info(f"Cache cleared" + (f" for model {model}" if model else ""))
        else:
            logger.warning("Cache is not enabled")
    
    def preload_embeddings(self, texts_list: List[List[str]], 
                          model: str = "Qwen/Qwen3-Embedding-8B") -> Dict[str, bool]:
        """Preload embeddings for multiple text batches."""
        results = {}
        
        for i, texts in enumerate(texts_list):
            try:
                result = self.generate_embeddings(texts, model)
                results[f"batch_{i}"] = result.success
                
                if result.success:
                    logger.info(f"Preloaded embeddings for batch {i}: {len(texts)} texts")
                else:
                    logger.error(f"Failed to preload batch {i}: {result.error_message}")
                    
                # Small delay to respect rate limits
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error preloading batch {i}: {e}")
                results[f"batch_{i}"] = False
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Initialize the embedding system with caching
    embedding_system = EmbeddingSystem(
        enable_cache=True,
        cache_dir="./embedding_cache",
        cache_ttl_hours=24 * 7  # 1 week TTL
    )
    
    # Test embeddings
    texts = [
        "This is a test document about machine learning.",
        "Another document about natural language processing.",
        "A third document about artificial intelligence."
    ]
    
    print("First request (should hit API):")
    start_time = time.time()
    result1 = embedding_system.generate_embeddings(texts)
    print(f"Time: {time.time() - start_time:.2f}s, Success: {result1.success}")
    
    print("\nSecond request (should hit cache):")
    start_time = time.time()
    result2 = embedding_system.generate_embeddings(texts)
    print(f"Time: {time.time() - start_time:.2f}s, Success: {result2.success}")
    
    # Print cache statistics
    print("\nCache Statistics:")
    stats = embedding_system.get_cache_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        elif isinstance(value, dict):
            print(f"{key}:")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        else:
            print(f"{key}: {value}")



class GroqClient:
    
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
        response = self.session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=60)
        
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
        result = self.siliconflow_client.generate_embeddings(texts, self.embedding_model)
        if result.success:
            return result.embeddings
        else:
            logger.error(f"Failed to generate embeddings: {result.error_message}")
            return []
    
    
    
    def generate_query_embedding(self, query: str) -> List[float]:
        embeddings = self.generate_embeddings([query])
        return embeddings[0] if embeddings else []
    
    def rerank_documents(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[RerankResult]:
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



  

import requests
def embed_texts(texts: list[str], batch_size: int = 50) -> list[list[float]]:
    all_embeddings = []
    headers = {
        "Authorization": f"Bearer {os.environ.get('SILICONFLOW_API_KEY')}",
        "Content-Type": "application/json"}
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload = {
            "model": config["models"]["embedding_model"],
            "input": batch}
        response = requests.post(
            os.environ.get('SILICONFLOW_EMBEDDING_URL'), json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        if "data" not in data:
            raise ValueError(f"Invalid response format: {data}")
        batch_embs = [item["embedding"] for item in data["data"]]
        all_embeddings.extend(batch_embs)
        # Fill with empty embeddings in case of failure
        all_embeddings.extend([[] for _ in batch])
    return all_embeddings



if __name__ == "__main__":
    # Initialize the embedding system with caching
    embedding_system = EmbeddingSystem(
        enable_cache=True,
        cache_dir="./embedding_cache",
        cache_ttl_hours=24 * 7  # 1 week TTL
        )
    
    # Test embeddings
    texts = [
        "This is a test document about machine learning.",
        "Another document about natural language processing.",
        "A third document about artificial intelligence."]
    
    embeddings = embed_texts(texts)
    print("Embeddings from embed_texts function:")
    print(embeddings)
    print("=" * 40)
    print("\n")
    print("=" * 40)
    print("Embeddings from EmbeddingSystem:")
    embedddings = embedding_system.generate_embeddings(texts).embeddings  
    print(embedddings)
    
    # print("First request (should hit API):")
    # start_time = time.time()
    # result1 = embedding_system.generate_embeddings(texts)
    
    
    # print(f"Time: {time.time() - start_time:.2f}s, Success: {result1.success}")
    
    # print("\nSecond request (should hit cache):")
    # start_time = time.time()
    # result2 = embedding_system.generate_embeddings(texts)
    # print(f"Time: {time.time() - start_time:.2f}s, Success: {result2.success}")
    
    # # Print cache statistics
    # print("\nCache Statistics:")
    # stats = embedding_system.get_cache_stats()
    # for key, value in stats.items():
    #     if isinstance(value, float):
    #         print(f"{key}: {value:.4f}")
    #     elif isinstance(value, dict):
    #         print(f"{key}:")
    #         for subkey, subvalue in value.items():
    #             print(f"  {subkey}: {subvalue}")
    #     else:
    #         print(f"{key}: {value}")
    # # main()