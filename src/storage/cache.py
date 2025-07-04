"""
Advanced caching system with local cache and optional Redis backend
"""

import json
import pickle
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Union, List
from datetime import datetime, timedelta
import threading
from collections import OrderedDict
import hashlib
import logging

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ..core.constants import (
    CACHE_TTL_SECONDS,
    LOCAL_CACHE_SIZE,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    REDIS_PASSWORD,
    USE_REDIS_CACHE,
    USE_LOCAL_ONLY,
    REDIS_MAX_MEMORY,
    REDIS_EVICTION_POLICY
)

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Abstract base class for cache backends"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass


class LocalCache(CacheBackend):
    """Thread-safe local memory cache with LRU eviction"""
    
    def __init__(self, max_size: int = LOCAL_CACHE_SIZE, default_ttl: int = CACHE_TTL_SECONDS):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict = OrderedDict()
        self._expiry: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0
        }
    
    def _is_expired(self, key: str) -> bool:
        """Check if key has expired"""
        if key not in self._expiry:
            return False
        return time.time() > self._expiry[key]
    
    def _evict_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, expiry_time in self._expiry.items()
            if current_time > expiry_time
        ]
        for key in expired_keys:
            self._remove_key(key)
    
    def _remove_key(self, key: str):
        """Remove key from cache and expiry tracking"""
        if key in self._cache:
            del self._cache[key]
        if key in self._expiry:
            del self._expiry[key]
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if self._cache:
            lru_key = next(iter(self._cache))
            self._remove_key(lru_key)
            self._stats['evictions'] += 1
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            # Clean expired entries
            self._evict_expired()
            
            if key not in self._cache or self._is_expired(key):
                self._stats['misses'] += 1
                return None
            
            # Move to end (mark as recently used)
            value = self._cache.pop(key)
            self._cache[key] = value
            self._stats['hits'] += 1
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL"""
        with self._lock:
            try:
                # Remove existing key if present
                if key in self._cache:
                    self._remove_key(key)
                
                # Evict if at capacity
                while len(self._cache) >= self.max_size:
                    self._evict_lru()
                
                # Set value and expiry
                self._cache[key] = value
                if ttl is None:
                    ttl = self.default_ttl
                if ttl > 0:
                    self._expiry[key] = time.time() + ttl
                
                self._stats['sets'] += 1
                return True
            
            except Exception as e:
                logger.error(f"Failed to set cache key {key}: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self._lock:
            if key in self._cache:
                self._remove_key(key)
                self._stats['deletes'] += 1
                return True
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._expiry.clear()
            return True
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        with self._lock:
            self._evict_expired()
            return key in self._cache and not self._is_expired(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            self._evict_expired()
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'backend': 'local',
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'sets': self._stats['sets'],
                'deletes': self._stats['deletes'],
                'evictions': self._stats['evictions']
            }


class RedisCache(CacheBackend):
    """Redis-based distributed cache optimized for large data volumes"""
    
    def __init__(self, host: str = REDIS_HOST, port: int = REDIS_PORT, 
                 db: int = REDIS_DB, password: Optional[str] = REDIS_PASSWORD,
                 default_ttl: int = CACHE_TTL_SECONDS,
                 max_memory: str = REDIS_MAX_MEMORY,
                 eviction_policy: str = REDIS_EVICTION_POLICY):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis package not available. Install with: pip install redis")
        
        self.default_ttl = default_ttl
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False,  # We handle serialization ourselves
            socket_connect_timeout=10,
            socket_timeout=10,
            retry_on_timeout=True,
            max_connections=20,  # Connection pooling for high volume
            health_check_interval=30
        )
        
        # Test connection and configure for large data
        try:
            self.redis_client.ping()
            
            # Configure Redis for large data volumes
            try:
                self.redis_client.config_set('maxmemory', max_memory)
                self.redis_client.config_set('maxmemory-policy', eviction_policy)
                logger.info(f"Configured Redis: maxmemory={max_memory}, policy={eviction_policy}")
            except Exception as config_error:
                logger.warning(f"Could not configure Redis memory settings: {config_error}")
            
            logger.info(f"Connected to Redis at {host}:{port} (optimized for large data)")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage with compression for large data"""
        try:
            import gzip
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Compress if data is large (>1KB)
            if len(data) > 1024:
                compressed = gzip.compress(data)
                # Use compression if it actually reduces size
                if len(compressed) < len(data):
                    return b"GZIP:" + compressed
            
            return b"RAW:" + data
        except Exception as e:
            logger.error(f"Failed to serialize value: {e}")
            raise
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage with compression support"""
        try:
            import gzip
            
            if data.startswith(b"GZIP:"):
                # Decompress data
                compressed_data = data[5:]  # Remove "GZIP:" prefix
                decompressed = gzip.decompress(compressed_data)
                return pickle.loads(decompressed)
            elif data.startswith(b"RAW:"):
                # Raw data
                raw_data = data[4:]  # Remove "RAW:" prefix
                return pickle.loads(raw_data)
            else:
                # Legacy format (no prefix)
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Failed to deserialize data: {e}")
            raise
    
    def _chunk_data(self, data: bytes, chunk_size: int = 100 * 1024 * 1024) -> List[bytes]:
        """Split large data into chunks (default 100MB chunks)"""
        chunks = []
        for i in range(0, len(data), chunk_size):
            chunks.append(data[i:i + chunk_size])
        return chunks
    
    def _unchunk_data(self, chunks: List[bytes]) -> bytes:
        """Reassemble chunked data"""
        return b''.join(chunks)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with chunking support for large data"""
        try:
            # First try to get as single key
            data = self.redis_client.get(key)
            if data is not None:
                return self._deserialize(data)
            
            # Check if it's chunked data
            chunk_info = self.redis_client.get(f"{key}:chunks")
            if chunk_info is None:
                return None
            
            # Get chunk information
            chunk_count = int(chunk_info)
            chunks = []
            
            # Retrieve all chunks
            for i in range(chunk_count):
                chunk_key = f"{key}:chunk:{i}"
                chunk_data = self.redis_client.get(chunk_key)
                if chunk_data is None:
                    logger.error(f"Missing chunk {i} for key {key}")
                    return None
                chunks.append(chunk_data)
            
            # Reassemble and deserialize
            full_data = self._unchunk_data(chunks)
            return self._deserialize(full_data)
            
        except Exception as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with chunking for large data"""
        try:
            data = self._serialize(value)
            if ttl is None:
                ttl = self.default_ttl
            
            # For data larger than 50MB, use chunking
            max_single_size = 50 * 1024 * 1024  # 50MB
            
            if len(data) <= max_single_size:
                # Store as single key
                if ttl > 0:
                    return bool(self.redis_client.setex(key, ttl, data))
                else:
                    return bool(self.redis_client.set(key, data))
            else:
                # Store as chunked data
                chunks = self._chunk_data(data)
                
                # Use pipeline for atomic operation
                pipe = self.redis_client.pipeline()
                
                # Store chunk count
                if ttl > 0:
                    pipe.setex(f"{key}:chunks", ttl, len(chunks))
                else:
                    pipe.set(f"{key}:chunks", len(chunks))
                
                # Store all chunks
                for i, chunk in enumerate(chunks):
                    chunk_key = f"{key}:chunk:{i}"
                    if ttl > 0:
                        pipe.setex(chunk_key, ttl, chunk)
                    else:
                        pipe.set(chunk_key, chunk)
                
                # Execute all operations atomically
                results = pipe.execute()
                return all(results)
                
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache including chunked data"""
        try:
            # Delete main key
            deleted = self.redis_client.delete(key)
            
            # Check for and delete chunked data
            chunk_info = self.redis_client.get(f"{key}:chunks")
            if chunk_info is not None:
                chunk_count = int(chunk_info)
                
                # Use pipeline for efficient deletion
                pipe = self.redis_client.pipeline()
                pipe.delete(f"{key}:chunks")
                
                for i in range(chunk_count):
                    pipe.delete(f"{key}:chunk:{i}")
                
                chunk_results = pipe.execute()
                deleted += sum(chunk_results)
            
            return bool(deleted)
        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            return bool(self.redis_client.flushdb())
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Failed to check cache key {key}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            info = self.redis_client.info('memory')
            stats = self.redis_client.info('stats')
            
            return {
                'backend': 'redis',
                'memory_used': info.get('used_memory', 0),
                'memory_used_human': info.get('used_memory_human', '0B'),
                'connected_clients': info.get('connected_clients', 0),
                'total_commands_processed': stats.get('total_commands_processed', 0),
                'keyspace_hits': stats.get('keyspace_hits', 0),
                'keyspace_misses': stats.get('keyspace_misses', 0),
                'hit_rate': stats.get('keyspace_hits', 0) / max(
                    stats.get('keyspace_hits', 0) + stats.get('keyspace_misses', 0), 1
                )
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {'backend': 'redis', 'error': str(e)}


class HybridCache(CacheBackend):
    """Redis-first hybrid cache with local fallback"""
    
    def __init__(self, use_local_only: bool = USE_LOCAL_ONLY,
                 local_cache_size: int = LOCAL_CACHE_SIZE,
                 redis_config: Optional[Dict[str, Any]] = None):
        self.redis_cache = None
        self.local_cache = LocalCache(max_size=local_cache_size)
        self.redis_primary = True
        
        # Always try Redis first unless explicitly disabled
        if not use_local_only and REDIS_AVAILABLE:
            try:
                redis_config = redis_config or {}
                self.redis_cache = RedisCache(**redis_config)
                logger.info("Hybrid cache initialized with Redis as primary datastore")
            except Exception as e:
                logger.warning(f"Redis unavailable, falling back to local cache: {e}")
                self.redis_primary = False
        else:
            logger.info("Local-only cache mode enabled")
            self.redis_primary = False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (Redis primary, local fallback for small data)"""
        if self.redis_cache and self.redis_primary:
            # Try Redis first (primary datastore)
            value = self.redis_cache.get(key)
            if value is not None:
                # Optionally populate local cache for small frequently accessed data
                try:
                    # Only cache small objects locally (< 1MB)
                    import sys
                    if sys.getsizeof(value) < 1024 * 1024:
                        self.local_cache.set(key, value, ttl=300)  # 5 min local cache
                except:
                    pass  # Ignore errors in local caching
                return value
            
            # If not in Redis, check local cache as fallback
            return self.local_cache.get(key)
        else:
            # Local-only mode or Redis unavailable
            return self.local_cache.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value prioritizing Redis as primary datastore"""
        if self.redis_cache and self.redis_primary:
            # Redis is primary - must succeed
            redis_success = self.redis_cache.set(key, value, ttl)
            
            # Optionally set in local cache for small data
            try:
                import sys
                if sys.getsizeof(value) < 1024 * 1024:  # < 1MB
                    self.local_cache.set(key, value, min(ttl or 300, 300))
            except:
                pass  # Local cache is optional
            
            return redis_success
        else:
            # Local-only mode
            return self.local_cache.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete key from primary datastore"""
        if self.redis_cache and self.redis_primary:
            # Delete from Redis (primary)
            redis_success = self.redis_cache.delete(key)
            # Also delete from local cache
            self.local_cache.delete(key)
            return redis_success
        else:
            # Local-only mode
            return self.local_cache.delete(key)
    
    def clear(self) -> bool:
        """Clear primary datastore"""
        if self.redis_cache and self.redis_primary:
            # Clear Redis (primary)
            redis_success = self.redis_cache.clear()
            # Also clear local cache
            self.local_cache.clear()
            return redis_success
        else:
            # Local-only mode
            return self.local_cache.clear()
    
    def exists(self, key: str) -> bool:
        """Check if key exists in primary datastore"""
        if self.redis_cache and self.redis_primary:
            # Check Redis first, then local as fallback
            if self.redis_cache.exists(key):
                return True
            return self.local_cache.exists(key)
        else:
            # Local-only mode
            return self.local_cache.exists(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics"""
        local_stats = self.local_cache.get_stats()
        
        if self.redis_cache:
            redis_stats = self.redis_cache.get_stats()
            return {
                'backend': 'hybrid',
                'local': local_stats,
                'redis': redis_stats
            }
        else:
            return {
                'backend': 'hybrid',
                'local': local_stats,
                'redis': None
            }


class CacheManager:
    """Singleton cache manager for the entire application"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'cache'):
            self.cache = HybridCache()
            self._key_prefix = "wagehood:"
    
    def _make_key(self, namespace: str, key: str) -> str:
        """Create namespaced cache key"""
        return f"{self._key_prefix}{namespace}:{key}"
    
    def get(self, namespace: str, key: str) -> Optional[Any]:
        """Get value from cache"""
        cache_key = self._make_key(namespace, key)
        return self.cache.get(cache_key)
    
    def set(self, namespace: str, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        cache_key = self._make_key(namespace, key)
        return self.cache.set(cache_key, value, ttl)
    
    def delete(self, namespace: str, key: str) -> bool:
        """Delete key from cache"""
        cache_key = self._make_key(namespace, key)
        return self.cache.delete(cache_key)
    
    def clear_namespace(self, namespace: str) -> bool:
        """Clear all keys in namespace (Redis only)"""
        if hasattr(self.cache, 'redis_cache') and self.cache.redis_cache:
            try:
                pattern = f"{self._key_prefix}{namespace}:*"
                keys = self.cache.redis_cache.redis_client.keys(pattern)
                if keys:
                    return bool(self.cache.redis_cache.redis_client.delete(*keys))
                return True
            except Exception as e:
                logger.error(f"Failed to clear namespace {namespace}: {e}")
                return False
        else:
            logger.warning("Namespace clearing only supported with Redis backend")
            return False
    
    def exists(self, namespace: str, key: str) -> bool:
        """Check if key exists"""
        cache_key = self._make_key(namespace, key)
        return self.cache.exists(cache_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()
    
    def cache_key_hash(self, *args) -> str:
        """Generate cache key from arguments"""
        key_string = "|".join(str(arg) for arg in args)
        return hashlib.md5(key_string.encode()).hexdigest()


# Global cache manager instance
cache_manager = CacheManager()


def cached(namespace: str, ttl: Optional[int] = None, key_func=None):
    """Decorator for caching function results"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache_manager.cache_key_hash(func.__name__, *args, *sorted(kwargs.items()))
            
            # Try to get from cache
            result = cache_manager.get(namespace, cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(namespace, cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator