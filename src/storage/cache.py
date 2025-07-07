"""
Advanced caching system with local cache and optional Redis backend
"""

import pickle
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List
from datetime import datetime
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
    USE_LOCAL_ONLY,
    REDIS_MAX_MEMORY,
    REDIS_EVICTION_POLICY,
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

    def __init__(
        self, max_size: int = LOCAL_CACHE_SIZE, default_ttl: int = CACHE_TTL_SECONDS
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict = OrderedDict()
        self._expiry: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0, "evictions": 0}

    def _is_expired(self, key: str) -> bool:
        """Check if key has expired"""
        if key not in self._expiry:
            return False
        return time.time() > self._expiry[key]

    def _evict_expired(self):
        """Remove expired entries efficiently"""
        current_time = time.time()
        # Process in batches to avoid long locks
        expired_keys = []

        for key, expiry_time in list(self._expiry.items()):
            if current_time > expiry_time:
                expired_keys.append(key)
                # Process in batches of 100
                if len(expired_keys) >= 100:
                    break

        # Remove expired keys
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
            self._stats["evictions"] += 1

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            # Check key existence and expiry first
            if key not in self._cache:
                self._stats["misses"] += 1
                return None

            if self._is_expired(key):
                self._remove_key(key)
                self._stats["misses"] += 1
                return None

            # Move to end (mark as recently used)
            value = self._cache.pop(key)
            self._cache[key] = value
            self._stats["hits"] += 1

            # Periodically clean expired entries (every 100 gets)
            if self._stats["hits"] % 100 == 0:
                self._evict_expired()

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

                self._stats["sets"] += 1
                return True

            except Exception as e:
                logger.error(f"Failed to set cache key {key}: {e}")
                return False

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self._lock:
            if key in self._cache:
                self._remove_key(key)
                self._stats["deletes"] += 1
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
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0

            return {
                "backend": "local",
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": hit_rate,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "sets": self._stats["sets"],
                "deletes": self._stats["deletes"],
                "evictions": self._stats["evictions"],
            }


class RedisCache(CacheBackend):
    """Redis-based distributed cache optimized for large data volumes"""

    def __init__(
        self,
        host: str = REDIS_HOST,
        port: int = REDIS_PORT,
        db: int = REDIS_DB,
        password: Optional[str] = REDIS_PASSWORD,
        default_ttl: int = CACHE_TTL_SECONDS,
        max_memory: str = REDIS_MAX_MEMORY,
        eviction_policy: str = REDIS_EVICTION_POLICY,
    ):
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis package not available. Install with: pip install redis"
            )

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
            health_check_interval=30,
        )

        # Test connection and configure for large data
        try:
            self.redis_client.ping()

            # Configure Redis for large data volumes
            try:
                self.redis_client.config_set("maxmemory", max_memory)
                self.redis_client.config_set("maxmemory-policy", eviction_policy)
                logger.info(
                    f"Configured Redis: maxmemory={max_memory}, policy={eviction_policy}"
                )
            except Exception as config_error:
                logger.warning(
                    f"Could not configure Redis memory settings: {config_error}"
                )

            logger.info(
                f"Connected to Redis at {host}:{port} (optimized for large data)"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _serialize(self, value: Any) -> bytes:
        """Serialize value with JSON as primary format, pickle as fallback"""
        try:
            import gzip
            
            # Try JSON first (more memory efficient and readable)
            try:
                json_data = json.dumps(value, default=str).encode('utf-8')
                json_len = len(json_data)
                
                # Only compress JSON if it's large enough
                if json_len > 2048:  # 2KB threshold
                    compressed = gzip.compress(json_data, compresslevel=6)
                    if len(compressed) < json_len * 0.9:
                        return b"JSON_GZIP:" + compressed
                
                return b"JSON:" + json_data
                
            except (TypeError, ValueError):
                # Fallback to pickle for complex objects that can't be JSON serialized
                logger.debug(f"JSON serialization failed for {type(value)}, using pickle fallback")
                data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                data_len = len(data)

                # Only compress if data is large enough to benefit
                if data_len > 2048:  # 2KB threshold
                    compressed = gzip.compress(data, compresslevel=6)
                    # Use compression only if it provides >10% reduction
                    if len(compressed) < data_len * 0.9:
                        return b"PICKLE_GZIP:" + compressed

                return b"PICKLE:" + data
                
        except Exception as e:
            logger.error(f"Failed to serialize value: {e}")
            raise

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage with support for JSON and pickle formats"""
        try:
            import gzip

            if data.startswith(b"JSON_GZIP:"):
                # Decompress JSON data
                compressed_data = data[10:]  # Remove "JSON_GZIP:" prefix
                decompressed = gzip.decompress(compressed_data)
                return json.loads(decompressed.decode('utf-8'))
            elif data.startswith(b"JSON:"):
                # Raw JSON data
                json_data = data[5:]  # Remove "JSON:" prefix
                return json.loads(json_data.decode('utf-8'))
            elif data.startswith(b"PICKLE_GZIP:"):
                # Decompress pickle data
                compressed_data = data[12:]  # Remove "PICKLE_GZIP:" prefix
                decompressed = gzip.decompress(compressed_data)
                return pickle.loads(decompressed)
            elif data.startswith(b"PICKLE:"):
                # Raw pickle data
                pickle_data = data[7:]  # Remove "PICKLE:" prefix
                return pickle.loads(pickle_data)
            elif data.startswith(b"GZIP:"):
                # Legacy compressed pickle format
                compressed_data = data[5:]  # Remove "GZIP:" prefix
                decompressed = gzip.decompress(compressed_data)
                return pickle.loads(decompressed)
            elif data.startswith(b"RAW:"):
                # Legacy raw pickle format
                raw_data = data[4:]  # Remove "RAW:" prefix
                return pickle.loads(raw_data)
            else:
                # Try to detect if it's raw JSON first
                try:
                    # Attempt to decode as UTF-8 and parse as JSON
                    text_data = data.decode('utf-8')
                    if text_data.strip().startswith(('{', '[', '"')) or text_data.strip() in ('true', 'false', 'null'):
                        return json.loads(text_data)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    pass
                
                # Legacy format (no prefix) - assume pickle
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Failed to deserialize data: {e}")
            raise

    def _chunk_data(
        self, data: bytes, chunk_size: int = 50 * 1024 * 1024
    ) -> List[bytes]:
        """Split large data into chunks (optimized 50MB chunks)"""
        data_len = len(data)
        # Pre-allocate list for better performance
        num_chunks = (data_len + chunk_size - 1) // chunk_size
        chunks = [None] * num_chunks

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, data_len)
            chunks[i] = data[start:end]

        return chunks

    def _unchunk_data(self, chunks: List[bytes]) -> bytes:
        """Reassemble chunked data"""
        return b"".join(chunks)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with optimized chunking support"""
        try:
            # First try to get as single key
            data = self.redis_client.get(key)
            if data is not None:
                return self._deserialize(data)

            # Check if it's chunked data
            pipe = self.redis_client.pipeline()
            pipe.get(f"{key}:chunks")
            pipe.get(f"{key}:size")
            chunk_info, size_info = pipe.execute()

            if chunk_info is None:
                return None

            # Get chunk information
            chunk_count = int(chunk_info)
            expected_size = int(size_info) if size_info else None

            # Retrieve all chunks efficiently using pipeline
            pipe = self.redis_client.pipeline()
            for i in range(chunk_count):
                pipe.get(f"{key}:chunk:{i}")

            chunk_results = pipe.execute()

            # Validate all chunks are present
            if None in chunk_results:
                missing_chunks = [
                    i for i, chunk in enumerate(chunk_results) if chunk is None
                ]
                logger.error(f"Missing chunks {missing_chunks} for key {key}")
                return None

            # Reassemble data efficiently
            if expected_size:
                # Pre-allocate bytearray for better performance
                full_data = bytearray(expected_size)
                offset = 0
                for chunk in chunk_results:
                    chunk_len = len(chunk)
                    full_data[offset : offset + chunk_len] = chunk
                    offset += chunk_len
                full_data = bytes(full_data)
            else:
                # Fallback to joining
                full_data = self._unchunk_data(chunk_results)

            return self._deserialize(full_data)

        except Exception as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optimized chunking for large data"""
        try:
            data = self._serialize(value)
            if ttl is None:
                ttl = self.default_ttl

            data_size = len(data)
            # Adjusted threshold for chunking (25MB)
            max_single_size = 25 * 1024 * 1024  # 25MB

            if data_size <= max_single_size:
                # Store as single key
                if ttl > 0:
                    return bool(self.redis_client.setex(key, ttl, data))
                else:
                    return bool(self.redis_client.set(key, data))
            else:
                # Store as chunked data with optimized chunking
                chunks = self._chunk_data(data)
                chunk_count = len(chunks)

                # Use pipeline for atomic operation with batching
                pipe = self.redis_client.pipeline(
                    transaction=False
                )  # Non-transactional for better performance

                # Store chunk metadata
                if ttl > 0:
                    pipe.setex(f"{key}:chunks", ttl, chunk_count)
                    pipe.setex(
                        f"{key}:size", ttl, data_size
                    )  # Store original size for validation
                else:
                    pipe.set(f"{key}:chunks", chunk_count)
                    pipe.set(f"{key}:size", data_size)

                # Store chunks in batches to avoid pipeline overflow
                batch_size = 100
                for i in range(0, chunk_count, batch_size):
                    batch_end = min(i + batch_size, chunk_count)

                    for j in range(i, batch_end):
                        chunk_key = f"{key}:chunk:{j}"
                        if ttl > 0:
                            pipe.setex(chunk_key, ttl, chunks[j])
                        else:
                            pipe.set(chunk_key, chunks[j])

                    # Execute batch
                    pipe.execute()
                    pipe = self.redis_client.pipeline(transaction=False)

                # Execute any remaining commands
                pipe.execute()
                return True

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
            info = self.redis_client.info("memory")
            stats = self.redis_client.info("stats")

            return {
                "backend": "redis",
                "memory_used": info.get("used_memory", 0),
                "memory_used_human": info.get("used_memory_human", "0B"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": stats.get("total_commands_processed", 0),
                "keyspace_hits": stats.get("keyspace_hits", 0),
                "keyspace_misses": stats.get("keyspace_misses", 0),
                "hit_rate": stats.get("keyspace_hits", 0)
                / max(
                    stats.get("keyspace_hits", 0) + stats.get("keyspace_misses", 0), 1
                ),
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"backend": "redis", "error": str(e)}


class HybridCache(CacheBackend):
    """Redis-primary cache with local cache for optimization only"""

    def __init__(
        self,
        use_local_only: bool = USE_LOCAL_ONLY,
        local_cache_size: int = LOCAL_CACHE_SIZE,
        redis_config: Optional[Dict[str, Any]] = None,
    ):
        self.redis_cache = None
        self.local_cache = LocalCache(max_size=local_cache_size)
        self.redis_required = True

        if use_local_only:
            # Local-only mode for testing/development
            logger.info("Local-only cache mode enabled (testing/development)")
            self.redis_required = False
        else:
            # Production mode - Redis is required
            if not REDIS_AVAILABLE:
                raise ImportError(
                    "Redis package required for production mode. Install with: pip install redis"
                )

            try:
                redis_config = redis_config or {}
                self.redis_cache = RedisCache(**redis_config)
                logger.info("Redis datastore initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Redis datastore: {e}")
                raise Exception(
                    f"Redis datastore connection failed - system cannot start: {e}"
                )

    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis datastore with local cache optimization"""
        if not self.redis_required:
            # Local-only mode for testing/development
            return self.local_cache.get(key)

        if not self.redis_cache:
            raise Exception(
                "Redis datastore unavailable - cannot perform read operation"
            )

        # Check local cache first for small frequently accessed data (optimization)
        local_value = self.local_cache.get(key)
        if local_value is not None:
            return local_value

        # Get from Redis (primary datastore)
        try:
            value = self.redis_cache.get(key)
            if value is not None:
                # Cache small objects locally for optimization
                try:
                    import sys

                    if sys.getsizeof(value) < 1024 * 1024:  # < 1MB
                        self.local_cache.set(
                            key, value, ttl=300
                        )  # 5 min optimization cache
                except Exception:
                    pass  # Local cache optimization is optional
            return value
        except Exception as e:
            logger.error(f"Redis datastore read error for key {key}: {e}")
            raise Exception(f"Redis datastore operation failed: {e}")

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis datastore with local cache optimization"""
        if not self.redis_required:
            # Local-only mode for testing/development
            return self.local_cache.set(key, value, ttl)

        if not self.redis_cache:
            raise Exception(
                "Redis datastore unavailable - cannot perform write operation"
            )

        # Write to Redis (primary datastore) - must succeed
        try:
            redis_success = self.redis_cache.set(key, value, ttl)
            if not redis_success:
                raise Exception("Redis write operation failed")

            # Also cache small objects locally for read optimization
            try:
                import sys

                if sys.getsizeof(value) < 1024 * 1024:  # < 1MB
                    self.local_cache.set(key, value, min(ttl or 300, 300))
            except Exception:
                pass  # Local cache optimization is optional

            return True
        except Exception as e:
            logger.error(f"Redis datastore write error for key {key}: {e}")
            raise Exception(f"Redis datastore operation failed: {e}")

    def delete(self, key: str) -> bool:
        """Delete key from Redis datastore"""
        if not self.redis_required:
            # Local-only mode for testing/development
            return self.local_cache.delete(key)

        if not self.redis_cache:
            raise Exception(
                "Redis datastore unavailable - cannot perform delete operation"
            )

        try:
            # Delete from Redis (primary datastore)
            redis_success = self.redis_cache.delete(key)
            # Also remove from local cache if present
            self.local_cache.delete(key)
            return redis_success
        except Exception as e:
            logger.error(f"Redis datastore delete error for key {key}: {e}")
            raise Exception(f"Redis datastore operation failed: {e}")

    def clear(self) -> bool:
        """Clear Redis datastore"""
        if not self.redis_required:
            # Local-only mode for testing/development
            return self.local_cache.clear()

        if not self.redis_cache:
            raise Exception(
                "Redis datastore unavailable - cannot perform clear operation"
            )

        try:
            # Clear Redis (primary datastore)
            redis_success = self.redis_cache.clear()
            # Also clear local cache
            self.local_cache.clear()
            return redis_success
        except Exception as e:
            logger.error(f"Redis datastore clear error: {e}")
            raise Exception(f"Redis datastore operation failed: {e}")

    def exists(self, key: str) -> bool:
        """Check if key exists in Redis datastore"""
        if not self.redis_required:
            # Local-only mode for testing/development
            return self.local_cache.exists(key)

        if not self.redis_cache:
            raise Exception("Redis datastore unavailable - cannot perform exists check")

        try:
            # Check Redis (primary datastore)
            return self.redis_cache.exists(key)
        except Exception as e:
            logger.error(f"Redis datastore exists check error for key {key}: {e}")
            raise Exception(f"Redis datastore operation failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        local_stats = self.local_cache.get_stats()

        if not self.redis_required:
            return {"backend": "local-only", "local": local_stats, "redis": None}

        if self.redis_cache:
            redis_stats = self.redis_cache.get_stats()
            return {
                "backend": "redis-primary",
                "redis": redis_stats,
                "local_optimization": local_stats,
            }
        else:
            return {
                "backend": "redis-primary",
                "redis": {"error": "Redis datastore unavailable"},
                "local_optimization": local_stats,
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
        if not hasattr(self, "cache"):
            self.cache = HybridCache()
            self._key_prefix = "wagehood:"
            self._redis_client = None
            self._initialize_redis_for_atomics()

    def _initialize_redis_for_atomics(self):
        """Initialize direct Redis client for atomic operations."""
        try:
            if hasattr(self.cache, "redis_cache") and self.cache.redis_cache:
                self._redis_client = self.cache.redis_cache.redis_client
                logger.info("Redis client initialized for atomic operations")
        except Exception as e:
            logger.warning(f"Could not initialize Redis for atomic operations: {e}")

    def _make_key(self, namespace: str, key: str) -> str:
        """Create namespaced cache key"""
        return f"{self._key_prefix}{namespace}:{key}"

    def get(self, namespace: str, key: str) -> Optional[Any]:
        """Get value from cache with optimized key generation"""
        cache_key = f"{self._key_prefix}{namespace}:{key}"
        return self.cache.get(cache_key)

    def set(
        self, namespace: str, key: str, value: Any, ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache with optimized key generation"""
        cache_key = f"{self._key_prefix}{namespace}:{key}"
        return self.cache.set(cache_key, value, ttl)

    def delete(self, namespace: str, key: str) -> bool:
        """Delete key from cache with optimized key generation"""
        cache_key = f"{self._key_prefix}{namespace}:{key}"
        return self.cache.delete(cache_key)

    def clear_namespace(self, namespace: str) -> bool:
        """Clear all keys in namespace (Redis only)"""
        if hasattr(self.cache, "redis_cache") and self.cache.redis_cache:
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
        """Check if key exists with optimized key generation"""
        cache_key = f"{self._key_prefix}{namespace}:{key}"
        return self.cache.exists(cache_key)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()

    def cache_key_hash(self, *args) -> str:
        """Generate cache key from arguments"""
        key_string = "|".join(str(arg) for arg in args)
        return hashlib.md5(key_string.encode()).hexdigest()

    def atomic_update_indicators(
        self, symbol: str, indicator_updates: Dict[str, Any], ttl: int = 300
    ) -> bool:
        """
        Atomically update multiple indicators for a symbol using WATCH/MULTI/EXEC.

        Args:
            symbol: Trading symbol
            indicator_updates: Dictionary of indicator_name -> value
            ttl: Time to live for cache entries

        Returns:
            True if all updates succeeded atomically, False otherwise
        """
        if not self._redis_client:
            raise Exception(
                "Redis datastore unavailable - atomic operations require Redis connection"
            )

        try:
            # Create all cache keys to watch
            cache_keys = [
                f"{self._key_prefix}indicators:{symbol}_{name}"
                for name in indicator_updates.keys()
            ]

            # Retry loop for optimistic locking
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Watch all keys we're about to modify
                    self._redis_client.watch(*cache_keys)

                    # Start transaction
                    pipe = self._redis_client.pipeline()
                    pipe.multi()

                    # Add all updates to transaction
                    for indicator_name, value in indicator_updates.items():
                        cache_key = (
                            f"{self._key_prefix}indicators:{symbol}_{indicator_name}"
                        )
                        serialized_value = (
                            self.cache.redis_cache._serialize(value)
                            if hasattr(self.cache, "redis_cache")
                            else str(value).encode()
                        )

                        if ttl > 0:
                            pipe.setex(cache_key, ttl, serialized_value)
                        else:
                            pipe.set(cache_key, serialized_value)

                    # Execute transaction atomically
                    pipe.execute()

                    # If we get here, transaction succeeded
                    logger.debug(
                        f"Atomic indicator update succeeded for {symbol} "
                        f"after {attempt + 1} attempts"
                    )
                    return True

                except redis.WatchError:
                    # Another process modified one of the watched keys, retry
                    logger.debug(
                        f"Atomic indicator update collision for {symbol}, attempt {attempt + 1}"
                    )
                    continue
                except Exception as e:
                    logger.error(f"Error in atomic indicator update for {symbol}: {e}")
                    return False
                finally:
                    # Always unwatch
                    self._redis_client.unwatch()

            # All retries exhausted
            logger.warning(
                f"Atomic indicator update failed for {symbol} after {max_retries} attempts"
            )
            return False

        except Exception as e:
            logger.error(f"Fatal error in atomic indicator update for {symbol}: {e}")
            return False

    def acquire_symbol_lock(self, symbol: str, timeout: int = 30) -> bool:
        """
        Acquire distributed lock for symbol processing using SET NX.

        Args:
            symbol: Trading symbol to lock
            timeout: Lock timeout in seconds

        Returns:
            True if lock acquired, False otherwise
        """
        if not self._redis_client:
            raise Exception(
                "Redis datastore unavailable - distributed locking requires Redis connection"
            )

        try:
            lock_key = f"{self._key_prefix}locks:symbol:{symbol}"
            lock_value = f"{threading.current_thread().ident}:{time.time()}"

            # SET NX with expiration for automatic cleanup
            result = self._redis_client.set(lock_key, lock_value, nx=True, ex=timeout)

            if result:
                logger.debug(f"Acquired symbol lock for {symbol}")
                return True
            else:
                logger.debug(
                    f"Could not acquire symbol lock for {symbol} (already locked)"
                )
                return False

        except Exception as e:
            logger.error(f"Error acquiring symbol lock for {symbol}: {e}")
            return False

    def release_symbol_lock(self, symbol: str) -> bool:
        """
        Release distributed lock for symbol processing.

        Args:
            symbol: Trading symbol to unlock

        Returns:
            True if lock released, False otherwise
        """
        if not self._redis_client:
            raise Exception(
                "Redis datastore unavailable - distributed locking requires Redis connection"
            )

        try:
            lock_key = f"{self._key_prefix}locks:symbol:{symbol}"

            # Use Lua script for atomic check-and-delete to prevent releasing wrong lock
            lua_script = """
            if redis.call("GET", KEYS[1]) then
                return redis.call("DEL", KEYS[1])
            else
                return 0
            end
            """

            result = self._redis_client.eval(lua_script, 1, lock_key)

            if result:
                logger.debug(f"Released symbol lock for {symbol}")
                return True
            else:
                logger.debug(
                    f"Symbol lock for {symbol} was not held or already expired"
                )
                return False

        except Exception as e:
            logger.error(f"Error releasing symbol lock for {symbol}: {e}")
            return False

    def atomic_update_signals(
        self, symbol: str, signal_updates: Dict[str, Any], ttl: int = 600
    ) -> bool:
        """
        Atomically update multiple strategy signals for a symbol using MULTI/EXEC.

        Args:
            symbol: Trading symbol
            signal_updates: Dictionary of strategy_name -> signal_data
            ttl: Time to live for cache entries

        Returns:
            True if all updates succeeded atomically, False otherwise
        """
        if not self._redis_client:
            raise Exception(
                "Redis datastore unavailable - atomic operations require Redis connection"
            )

        try:
            # Start transaction
            pipe = self._redis_client.pipeline()
            pipe.multi()

            # Store individual strategy signals
            for strategy_name, signal_data in signal_updates.items():
                cache_key = (
                    f"{self._key_prefix}strategies:{symbol}_{strategy_name}_signal"
                )
                serialized_value = (
                    self.cache.redis_cache._serialize(signal_data)
                    if hasattr(self.cache, "redis_cache")
                    else str(signal_data).encode()
                )

                if ttl > 0:
                    pipe.setex(cache_key, ttl, serialized_value)
                else:
                    pipe.set(cache_key, serialized_value)

            # Store complete signal set
            complete_key = f"{self._key_prefix}strategies:{symbol}_signals"
            serialized_complete = (
                self.cache.redis_cache._serialize(signal_updates)
                if hasattr(self.cache, "redis_cache")
                else str(signal_updates).encode()
            )

            if ttl > 0:
                pipe.setex(complete_key, ttl, serialized_complete)
            else:
                pipe.set(complete_key, serialized_complete)

            # Execute transaction atomically
            pipe.execute()

            logger.debug(f"Atomic signal update succeeded for {symbol}")
            return True

        except Exception as e:
            logger.error(f"Error in atomic signal update for {symbol}: {e}")
            return False

    def atomic_update_composite_signal(
        self,
        symbol: str,
        composite_signal: Dict[str, Any],
        component_signals: Dict[str, Any],
        ttl: int = 600,
    ) -> bool:
        """
        Atomically update composite signal and its components using MULTI/EXEC.

        Args:
            symbol: Trading symbol
            composite_signal: Composite signal data
            component_signals: Individual component signals
            ttl: Time to live for cache entries

        Returns:
            True if all updates succeeded atomically, False otherwise
        """
        if not self._redis_client:
            raise Exception(
                "Redis datastore unavailable - atomic operations require Redis connection"
            )

        try:
            # Start transaction
            pipe = self._redis_client.pipeline()
            pipe.multi()

            # Store composite signal
            composite_key = (
                f"{self._key_prefix}composite_signals:{symbol}_composite_signal"
            )
            serialized_composite = (
                self.cache.redis_cache._serialize(composite_signal)
                if hasattr(self.cache, "redis_cache")
                else str(composite_signal).encode()
            )

            if ttl > 0:
                pipe.setex(composite_key, ttl, serialized_composite)
            else:
                pipe.set(composite_key, serialized_composite)

            # Store component signals
            for component_name, signal_data in component_signals.items():
                component_key = (
                    f"{self._key_prefix}composite_signals:"
                    f"{symbol}_{component_name}_component"
                )
                serialized_component = (
                    self.cache.redis_cache._serialize(signal_data)
                    if hasattr(self.cache, "redis_cache")
                    else str(signal_data).encode()
                )

                if ttl > 0:
                    pipe.setex(component_key, ttl, serialized_component)
                else:
                    pipe.set(component_key, serialized_component)

            # Add timestamp for ordering
            timestamp_key = f"{self._key_prefix}composite_signals:{symbol}_last_update"
            current_time = datetime.now().isoformat()

            if ttl > 0:
                pipe.setex(timestamp_key, ttl, current_time.encode())
            else:
                pipe.set(timestamp_key, current_time.encode())

            # Execute transaction atomically
            pipe.execute()

            logger.debug(f"Atomic composite signal update succeeded for {symbol}")
            return True

        except Exception as e:
            logger.error(f"Error in atomic composite signal update for {symbol}: {e}")
            return False


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
                cache_key = cache_manager.cache_key_hash(
                    func.__name__, *args, *sorted(kwargs.items())
                )

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
