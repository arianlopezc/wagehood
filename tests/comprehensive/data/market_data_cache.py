"""
Market data caching and management system for testing.
Handles real market data caching, versioning, and efficient retrieval.
"""

import os
import json
import pickle
import hashlib
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum
import logging

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests


class CacheStrategy(Enum):
    """Cache strategy types."""
    MEMORY = "memory"
    DISK = "disk"
    HYBRID = "hybrid"
    DISTRIBUTED = "distributed"


class DataSource(Enum):
    """Data source types."""
    ALPACA = "alpaca"
    YAHOO = "yahoo"
    MOCK = "mock"
    EXTERNAL = "external"


@dataclass
class CacheEntry:
    """Cache entry metadata."""
    key: str
    source: DataSource
    symbol: str
    start_date: datetime
    end_date: datetime
    frequency: str
    data_type: str
    size_bytes: int
    checksum: str
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: int
    metadata: Dict[str, Any]


class MarketDataCache:
    """Advanced market data caching system."""
    
    def __init__(self, 
                 cache_dir: str = None,
                 strategy: CacheStrategy = CacheStrategy.HYBRID,
                 max_memory_mb: int = 256,
                 max_disk_mb: int = 1024,
                 default_ttl: int = 3600):
        """Initialize market data cache."""
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.strategy = strategy
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_disk_bytes = max_disk_mb * 1024 * 1024
        self.default_ttl = default_ttl
        
        # In-memory cache
        self.memory_cache = {}
        self.memory_usage = 0
        
        # Database for metadata
        self.db_path = self.cache_dir / "cache_metadata.db"
        self._init_database()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def _init_database(self):
        """Initialize SQLite database for cache metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    start_date TIMESTAMP NOT NULL,
                    end_date TIMESTAMP NOT NULL,
                    frequency TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    last_accessed TIMESTAMP NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    ttl_seconds INTEGER NOT NULL,
                    metadata TEXT NOT NULL
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON cache_entries(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON cache_entries(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed)")
    
    def _generate_cache_key(self, 
                          symbol: str, 
                          start_date: datetime, 
                          end_date: datetime,
                          frequency: str,
                          data_type: str,
                          source: DataSource) -> str:
        """Generate cache key for data request."""
        key_data = f"{symbol}_{start_date.isoformat()}_{end_date.isoformat()}_{frequency}_{data_type}_{source.value}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _calculate_checksum(self, data: pd.DataFrame) -> str:
        """Calculate checksum for data."""
        return hashlib.md5(str(data.values.tobytes()).encode()).hexdigest()
    
    def _get_file_path(self, cache_key: str) -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _store_memory(self, cache_key: str, data: pd.DataFrame) -> bool:
        """Store data in memory cache."""
        try:
            data_size = data.memory_usage(deep=True).sum()
            
            # Check if we have enough space
            if self.memory_usage + data_size > self.max_memory_bytes:
                # Evict least recently used items
                self._evict_memory_cache(data_size)
            
            self.memory_cache[cache_key] = {
                'data': data,
                'size': data_size,
                'accessed_at': datetime.now()
            }
            
            self.memory_usage += data_size
            return True
        except Exception as e:
            self.logger.error(f"Error storing in memory cache: {e}")
            return False
    
    def _store_disk(self, cache_key: str, data: pd.DataFrame) -> bool:
        """Store data on disk."""
        try:
            file_path = self._get_file_path(cache_key)
            data.to_pickle(file_path)
            return True
        except Exception as e:
            self.logger.error(f"Error storing on disk: {e}")
            return False
    
    def _load_memory(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from memory cache."""
        if cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]
            entry['accessed_at'] = datetime.now()
            return entry['data']
        return None
    
    def _load_disk(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from disk cache."""
        try:
            file_path = self._get_file_path(cache_key)
            if file_path.exists():
                return pd.read_pickle(file_path)
        except Exception as e:
            self.logger.error(f"Error loading from disk: {e}")
        return None
    
    def _evict_memory_cache(self, required_space: int):
        """Evict items from memory cache to make space."""
        # Sort by last accessed time (LRU)
        sorted_entries = sorted(
            self.memory_cache.items(),
            key=lambda x: x[1]['accessed_at']
        )
        
        space_freed = 0
        for cache_key, entry in sorted_entries:
            if space_freed >= required_space:
                break
            
            space_freed += entry['size']
            self.memory_usage -= entry['size']
            del self.memory_cache[cache_key]
            
            self.logger.debug(f"Evicted {cache_key} from memory cache")
    
    def _cleanup_disk_cache(self):
        """Clean up disk cache based on size and TTL."""
        current_time = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            # Get all cache entries
            cursor = conn.execute("""
                SELECT key, created_at, ttl_seconds, size_bytes, last_accessed
                FROM cache_entries
                ORDER BY last_accessed ASC
            """)
            
            entries = cursor.fetchall()
            total_size = sum(entry[3] for entry in entries)
            
            # Remove expired entries
            for entry in entries:
                key, created_at, ttl_seconds, size_bytes, last_accessed = entry
                created_time = datetime.fromisoformat(created_at)
                
                if current_time - created_time > timedelta(seconds=ttl_seconds):
                    self._remove_cache_entry(key)
                    total_size -= size_bytes
            
            # Remove entries if we're over the size limit
            if total_size > self.max_disk_bytes:
                # Remove oldest entries first
                for entry in entries:
                    if total_size <= self.max_disk_bytes * 0.8:
                        break
                    
                    key, _, _, size_bytes, _ = entry
                    self._remove_cache_entry(key)
                    total_size -= size_bytes
    
    def _remove_cache_entry(self, cache_key: str):
        """Remove cache entry completely."""
        try:
            # Remove from memory
            if cache_key in self.memory_cache:
                self.memory_usage -= self.memory_cache[cache_key]['size']
                del self.memory_cache[cache_key]
            
            # Remove from disk
            file_path = self._get_file_path(cache_key)
            if file_path.exists():
                file_path.unlink()
            
            # Remove from database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache_entries WHERE key = ?", (cache_key,))
            
            self.logger.debug(f"Removed cache entry: {cache_key}")
        except Exception as e:
            self.logger.error(f"Error removing cache entry {cache_key}: {e}")
    
    def store(self, 
              symbol: str,
              data: pd.DataFrame,
              start_date: datetime,
              end_date: datetime,
              frequency: str,
              data_type: str,
              source: DataSource,
              ttl_seconds: int = None,
              metadata: Dict[str, Any] = None) -> str:
        """Store data in cache."""
        with self._lock:
            cache_key = self._generate_cache_key(
                symbol, start_date, end_date, frequency, data_type, source
            )
            
            ttl_seconds = ttl_seconds or self.default_ttl
            metadata = metadata or {}
            
            # Calculate data size and checksum
            data_size = data.memory_usage(deep=True).sum()
            checksum = self._calculate_checksum(data)
            
            # Store based on strategy
            stored = False
            
            if self.strategy in [CacheStrategy.MEMORY, CacheStrategy.HYBRID]:
                stored = self._store_memory(cache_key, data)
            
            if self.strategy in [CacheStrategy.DISK, CacheStrategy.HYBRID]:
                stored = self._store_disk(cache_key, data) or stored
            
            if stored:
                # Store metadata
                entry = CacheEntry(
                    key=cache_key,
                    source=source,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency,
                    data_type=data_type,
                    size_bytes=data_size,
                    checksum=checksum,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    access_count=0,
                    ttl_seconds=ttl_seconds,
                    metadata=metadata
                )
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO cache_entries
                        (key, source, symbol, start_date, end_date, frequency, data_type,
                         size_bytes, checksum, created_at, last_accessed, access_count,
                         ttl_seconds, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        entry.key, entry.source.value, entry.symbol, entry.start_date,
                        entry.end_date, entry.frequency, entry.data_type, entry.size_bytes,
                        entry.checksum, entry.created_at, entry.last_accessed,
                        entry.access_count, entry.ttl_seconds, json.dumps(entry.metadata)
                    ))
                
                self.logger.info(f"Stored cache entry: {cache_key}")
                return cache_key
            
            return None
    
    def retrieve(self, 
                 symbol: str,
                 start_date: datetime,
                 end_date: datetime,
                 frequency: str,
                 data_type: str,
                 source: DataSource) -> Optional[pd.DataFrame]:
        """Retrieve data from cache."""
        with self._lock:
            cache_key = self._generate_cache_key(
                symbol, start_date, end_date, frequency, data_type, source
            )
            
            # Check if entry exists and is not expired
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT created_at, ttl_seconds, access_count
                    FROM cache_entries
                    WHERE key = ?
                """, (cache_key,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                created_at, ttl_seconds, access_count = row
                created_time = datetime.fromisoformat(created_at)
                
                # Check if expired
                if datetime.now() - created_time > timedelta(seconds=ttl_seconds):
                    self._remove_cache_entry(cache_key)
                    return None
            
            # Try to load data
            data = None
            
            if self.strategy in [CacheStrategy.MEMORY, CacheStrategy.HYBRID]:
                data = self._load_memory(cache_key)
            
            if data is None and self.strategy in [CacheStrategy.DISK, CacheStrategy.HYBRID]:
                data = self._load_disk(cache_key)
                
                # Store in memory if using hybrid strategy
                if data is not None and self.strategy == CacheStrategy.HYBRID:
                    self._store_memory(cache_key, data)
            
            if data is not None:
                # Update access statistics
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE cache_entries
                        SET last_accessed = ?, access_count = access_count + 1
                        WHERE key = ?
                    """, (datetime.now(), cache_key))
                
                self.logger.debug(f"Retrieved cache entry: {cache_key}")
                return data
            
            return None
    
    def get_or_fetch(self,
                     symbol: str,
                     start_date: datetime,
                     end_date: datetime,
                     frequency: str,
                     data_type: str,
                     source: DataSource,
                     fetch_func: callable,
                     ttl_seconds: int = None) -> Optional[pd.DataFrame]:
        """Get data from cache or fetch if not available."""
        # Try to get from cache first
        data = self.retrieve(symbol, start_date, end_date, frequency, data_type, source)
        
        if data is not None:
            return data
        
        # Fetch data
        try:
            data = fetch_func(symbol, start_date, end_date, frequency)
            
            if data is not None and not data.empty:
                # Store in cache
                self.store(symbol, data, start_date, end_date, frequency, 
                          data_type, source, ttl_seconds)
                return data
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
        
        return None
    
    def invalidate(self, 
                   symbol: str = None,
                   source: DataSource = None,
                   data_type: str = None) -> int:
        """Invalidate cache entries based on criteria."""
        with self._lock:
            query = "SELECT key FROM cache_entries WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if source:
                query += " AND source = ?"
                params.append(source.value)
            
            if data_type:
                query += " AND data_type = ?"
                params.append(data_type)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, params)
                keys = [row[0] for row in cursor.fetchall()]
            
            # Remove entries
            for key in keys:
                self._remove_cache_entry(key)
            
            self.logger.info(f"Invalidated {len(keys)} cache entries")
            return len(keys)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        source,
                        COUNT(*) as count,
                        SUM(size_bytes) as total_size,
                        AVG(access_count) as avg_access_count
                    FROM cache_entries
                    GROUP BY source
                """)
                
                stats = {}
                for row in cursor.fetchall():
                    stats[row[0]] = {
                        'count': row[1],
                        'total_size': row[2],
                        'avg_access_count': row[3]
                    }
                
                # Overall stats
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_entries,
                        SUM(size_bytes) as total_size,
                        AVG(access_count) as avg_access_count
                    FROM cache_entries
                """)
                
                overall = cursor.fetchone()
                stats['overall'] = {
                    'total_entries': overall[0],
                    'total_size': overall[1],
                    'avg_access_count': overall[2],
                    'memory_usage': self.memory_usage,
                    'memory_entries': len(self.memory_cache)
                }
                
                return stats
    
    def cleanup(self):
        """Clean up expired and oversized cache entries."""
        with self._lock:
            self._cleanup_disk_cache()
            self.logger.info("Cache cleanup completed")
    
    def clear_all(self):
        """Clear all cache entries."""
        with self._lock:
            # Clear memory cache
            self.memory_cache.clear()
            self.memory_usage = 0
            
            # Clear disk cache
            for file_path in self.cache_dir.glob("*.pkl"):
                file_path.unlink()
            
            # Clear database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache_entries")
            
            self.logger.info("All cache entries cleared")
    
    def export_cache_stats(self, output_file: str):
        """Export cache statistics to file."""
        stats = self.get_cache_stats()
        
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        self.logger.info(f"Cache stats exported to {output_file}")


class DataPrefetcher:
    """Prefetch and warm up cache with commonly used data."""
    
    def __init__(self, cache: MarketDataCache, max_workers: int = 4):
        """Initialize data prefetcher."""
        self.cache = cache
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
    
    def prefetch_symbols(self, 
                        symbols: List[str],
                        days_back: int = 30,
                        frequencies: List[str] = None,
                        data_types: List[str] = None,
                        source: DataSource = DataSource.MOCK) -> Dict[str, Any]:
        """Prefetch data for multiple symbols."""
        frequencies = frequencies or ['1min', '5min', '1h', '1d']
        data_types = data_types or ['bars', 'quotes']
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        prefetch_tasks = []
        
        for symbol in symbols:
            for frequency in frequencies:
                for data_type in data_types:
                    prefetch_tasks.append({
                        'symbol': symbol,
                        'start_date': start_date,
                        'end_date': end_date,
                        'frequency': frequency,
                        'data_type': data_type,
                        'source': source
                    })
        
        # Execute prefetch tasks
        results = {'success': 0, 'failed': 0, 'cached': 0}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._prefetch_task, task): task
                for task in prefetch_tasks
            }
            
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    if result == 'success':
                        results['success'] += 1
                    elif result == 'cached':
                        results['cached'] += 1
                    else:
                        results['failed'] += 1
                except Exception as e:
                    self.logger.error(f"Prefetch task failed: {e}")
                    results['failed'] += 1
        
        self.logger.info(f"Prefetch completed: {results}")
        return results
    
    def _prefetch_task(self, task: Dict[str, Any]) -> str:
        """Execute a single prefetch task."""
        # Check if already cached
        existing_data = self.cache.retrieve(
            task['symbol'], task['start_date'], task['end_date'],
            task['frequency'], task['data_type'], task['source']
        )
        
        if existing_data is not None:
            return 'cached'
        
        # Generate mock data for prefetching
        try:
            # This would normally call a real data source
            # For now, we'll use mock data
            from .mock_data_generator import AdvancedMockDataGenerator, MockDataConfig
            
            config = MockDataConfig(
                symbols=[task['symbol']],
                start_date=task['start_date'],
                end_date=task['end_date'],
                frequency=task['frequency']
            )
            
            generator = AdvancedMockDataGenerator(config)
            data = generator.generate_market_data()
            
            if data is not None and not data.empty:
                self.cache.store(
                    task['symbol'], data, task['start_date'], task['end_date'],
                    task['frequency'], task['data_type'], task['source']
                )
                return 'success'
            else:
                return 'failed'
        except Exception as e:
            self.logger.error(f"Error generating prefetch data: {e}")
            return 'failed'
    
    def warm_cache_for_tests(self, test_symbols: List[str]) -> bool:
        """Warm cache with data commonly used in tests."""
        try:
            # Prefetch recent data
            self.prefetch_symbols(test_symbols, days_back=7, frequencies=['1min'])
            
            # Prefetch historical data
            self.prefetch_symbols(test_symbols, days_back=30, frequencies=['1h', '1d'])
            
            self.logger.info(f"Cache warmed for {len(test_symbols)} symbols")
            return True
        except Exception as e:
            self.logger.error(f"Error warming cache: {e}")
            return False


class CacheAnalyzer:
    """Analyze cache performance and usage patterns."""
    
    def __init__(self, cache: MarketDataCache):
        """Initialize cache analyzer."""
        self.cache = cache
        self.logger = logging.getLogger(__name__)
    
    def analyze_hit_rate(self, days_back: int = 7) -> Dict[str, float]:
        """Analyze cache hit rate over time."""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        with sqlite3.connect(self.cache.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    source,
                    COUNT(*) as total_entries,
                    SUM(access_count) as total_accesses,
                    AVG(access_count) as avg_access_count
                FROM cache_entries
                WHERE created_at >= ?
                GROUP BY source
            """, (cutoff_date,))
            
            results = {}
            for row in cursor.fetchall():
                source, total_entries, total_accesses, avg_access_count = row
                hit_rate = (total_accesses / total_entries) if total_entries > 0 else 0
                results[source] = {
                    'hit_rate': hit_rate,
                    'total_entries': total_entries,
                    'total_accesses': total_accesses,
                    'avg_access_count': avg_access_count
                }
            
            return results
    
    def analyze_storage_efficiency(self) -> Dict[str, Any]:
        """Analyze storage efficiency and patterns."""
        with sqlite3.connect(self.cache.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    symbol,
                    frequency,
                    COUNT(*) as entry_count,
                    SUM(size_bytes) as total_size,
                    AVG(size_bytes) as avg_size,
                    SUM(access_count) as total_accesses
                FROM cache_entries
                GROUP BY symbol, frequency
                ORDER BY total_accesses DESC
            """)
            
            analysis = {
                'top_symbols': [],
                'size_distribution': {},
                'frequency_usage': {}
            }
            
            for row in cursor.fetchall():
                symbol, frequency, entry_count, total_size, avg_size, total_accesses = row
                
                # Track top symbols by access
                analysis['top_symbols'].append({
                    'symbol': symbol,
                    'frequency': frequency,
                    'entry_count': entry_count,
                    'total_size': total_size,
                    'avg_size': avg_size,
                    'total_accesses': total_accesses
                })
                
                # Track frequency usage
                if frequency not in analysis['frequency_usage']:
                    analysis['frequency_usage'][frequency] = 0
                analysis['frequency_usage'][frequency] += total_accesses
            
            # Sort and limit top symbols
            analysis['top_symbols'] = sorted(
                analysis['top_symbols'], 
                key=lambda x: x['total_accesses'], 
                reverse=True
            )[:20]
            
            return analysis
    
    def generate_optimization_recommendations(self) -> List[str]:
        """Generate cache optimization recommendations."""
        recommendations = []
        
        # Analyze hit rates
        hit_rates = self.analyze_hit_rate()
        for source, stats in hit_rates.items():
            if stats['hit_rate'] < 0.5:
                recommendations.append(
                    f"Low hit rate for {source} ({stats['hit_rate']:.2%}). "
                    f"Consider increasing TTL or prefetching strategies."
                )
        
        # Analyze storage efficiency
        efficiency = self.analyze_storage_efficiency()
        
        # Check for unused data
        unused_entries = [
            entry for entry in efficiency['top_symbols'] 
            if entry['total_accesses'] == 0
        ]
        
        if unused_entries:
            recommendations.append(
                f"Found {len(unused_entries)} unused cache entries. "
                f"Consider implementing more aggressive cleanup policies."
            )
        
        # Check for size distribution
        total_size = sum(entry['total_size'] for entry in efficiency['top_symbols'])
        if total_size > self.cache.max_disk_bytes * 0.8:
            recommendations.append(
                f"Cache size is {total_size / (1024*1024):.1f}MB "
                f"({total_size / self.cache.max_disk_bytes:.1%} of limit). "
                f"Consider increasing cache size or implementing better eviction policies."
            )
        
        return recommendations