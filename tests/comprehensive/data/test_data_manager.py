"""
Comprehensive test data management system for the test framework.
Handles test data storage, versioning, cleanup, and management.
"""

import os
import json
import pickle
import hashlib
import logging
import sqlite3
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


class DataType(Enum):
    """Test data types."""
    MARKET_DATA = "market_data"
    MOCK_DATA = "mock_data"
    FIXTURE_DATA = "fixture_data"
    CACHE_DATA = "cache_data"
    RESULT_DATA = "result_data"


@dataclass
class TestDataRecord:
    """Test data record metadata."""
    id: str
    name: str
    data_type: DataType
    version: str
    created_at: datetime
    updated_at: datetime
    size_bytes: int
    checksum: str
    metadata: Dict[str, Any]
    tags: List[str]
    dependencies: List[str]
    retention_days: int = 30


class TestDataManager:
    """Comprehensive test data management system."""
    
    def __init__(self, base_path: str = None):
        """Initialize the test data manager."""
        self.base_path = Path(base_path) if base_path else Path(__file__).parent
        self.data_dir = self.base_path / "storage"
        self.cache_dir = self.data_dir / "cache"
        self.fixture_dir = self.data_dir / "fixtures"
        self.temp_dir = self.data_dir / "temp"
        self.db_path = self.data_dir / "test_data.db"
        
        # Create directories
        for dir_path in [self.data_dir, self.cache_dir, self.fixture_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize database
        self._init_database()
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def _init_database(self):
        """Initialize the SQLite database for metadata storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_data_records (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    dependencies TEXT NOT NULL,
                    retention_days INTEGER NOT NULL DEFAULT 30
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_data_type ON test_data_records(data_type)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tags ON test_data_records(tags)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON test_data_records(created_at)
            """)
    
    def _calculate_checksum(self, data: Any) -> str:
        """Calculate checksum for data."""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data_str = str(data.values.tobytes())
        elif isinstance(data, np.ndarray):
            data_str = str(data.tobytes())
        else:
            data_str = str(data)
        
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _serialize_data(self, data: Any, file_path: Path) -> int:
        """Serialize data to file and return size."""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data.to_pickle(file_path)
        elif isinstance(data, dict) and all(isinstance(v, (int, float, str, bool, list, dict)) for v in data.values()):
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        
        return file_path.stat().st_size
    
    def _deserialize_data(self, file_path: Path) -> Any:
        """Deserialize data from file."""
        if file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        elif file_path.suffix == '.pkl':
            return pd.read_pickle(file_path)
        else:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
    
    def store_data(self, 
                   name: str, 
                   data: Any, 
                   data_type: DataType,
                   version: str = "1.0",
                   metadata: Dict[str, Any] = None,
                   tags: List[str] = None,
                   dependencies: List[str] = None,
                   retention_days: int = 30) -> str:
        """Store test data with metadata."""
        with self._lock:
            # Generate unique ID
            record_id = f"{name}_{data_type.value}_{version}_{datetime.now().isoformat()}"
            
            # Create file path
            file_path = self.data_dir / f"{record_id}.pkl"
            
            # Serialize data
            size_bytes = self._serialize_data(data, file_path)
            
            # Calculate checksum
            checksum = self._calculate_checksum(data)
            
            # Create record
            record = TestDataRecord(
                id=record_id,
                name=name,
                data_type=data_type,
                version=version,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                size_bytes=size_bytes,
                checksum=checksum,
                metadata=metadata or {},
                tags=tags or [],
                dependencies=dependencies or [],
                retention_days=retention_days
            )
            
            # Store metadata in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO test_data_records 
                    (id, name, data_type, version, created_at, updated_at, 
                     size_bytes, checksum, metadata, tags, dependencies, retention_days)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.id, record.name, record.data_type.value, record.version,
                    record.created_at, record.updated_at, record.size_bytes,
                    record.checksum, json.dumps(record.metadata),
                    json.dumps(record.tags), json.dumps(record.dependencies),
                    record.retention_days
                ))
            
            self.logger.info(f"Stored test data: {record_id}")
            return record_id
    
    def retrieve_data(self, record_id: str) -> Optional[Any]:
        """Retrieve test data by ID."""
        with self._lock:
            file_path = self.data_dir / f"{record_id}.pkl"
            
            if not file_path.exists():
                self.logger.warning(f"Test data not found: {record_id}")
                return None
            
            try:
                data = self._deserialize_data(file_path)
                self.logger.info(f"Retrieved test data: {record_id}")
                return data
            except Exception as e:
                self.logger.error(f"Error retrieving test data {record_id}: {e}")
                return None
    
    def search_data(self, 
                   name: str = None,
                   data_type: DataType = None,
                   tags: List[str] = None,
                   version: str = None) -> List[TestDataRecord]:
        """Search for test data records."""
        with self._lock:
            query = "SELECT * FROM test_data_records WHERE 1=1"
            params = []
            
            if name:
                query += " AND name LIKE ?"
                params.append(f"%{name}%")
            
            if data_type:
                query += " AND data_type = ?"
                params.append(data_type.value)
            
            if version:
                query += " AND version = ?"
                params.append(version)
            
            if tags:
                for tag in tags:
                    query += " AND tags LIKE ?"
                    params.append(f"%{tag}%")
            
            query += " ORDER BY created_at DESC"
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                records = []
                for row in rows:
                    record = TestDataRecord(
                        id=row[0],
                        name=row[1],
                        data_type=DataType(row[2]),
                        version=row[3],
                        created_at=datetime.fromisoformat(row[4]),
                        updated_at=datetime.fromisoformat(row[5]),
                        size_bytes=row[6],
                        checksum=row[7],
                        metadata=json.loads(row[8]),
                        tags=json.loads(row[9]),
                        dependencies=json.loads(row[10]),
                        retention_days=row[11]
                    )
                    records.append(record)
                
                return records
    
    def delete_data(self, record_id: str) -> bool:
        """Delete test data by ID."""
        with self._lock:
            file_path = self.data_dir / f"{record_id}.pkl"
            
            try:
                # Remove file
                if file_path.exists():
                    file_path.unlink()
                
                # Remove from database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM test_data_records WHERE id = ?", (record_id,))
                
                self.logger.info(f"Deleted test data: {record_id}")
                return True
            except Exception as e:
                self.logger.error(f"Error deleting test data {record_id}: {e}")
                return False
    
    def cleanup_expired_data(self) -> int:
        """Clean up expired test data."""
        with self._lock:
            cutoff_date = datetime.now() - timedelta(days=30)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id FROM test_data_records 
                    WHERE datetime(created_at) < ? 
                    OR datetime(created_at) < datetime('now', '-' || retention_days || ' days')
                """, (cutoff_date,))
                
                expired_records = cursor.fetchall()
                
                cleaned_count = 0
                for record in expired_records:
                    if self.delete_data(record[0]):
                        cleaned_count += 1
                
                self.logger.info(f"Cleaned up {cleaned_count} expired test data records")
                return cleaned_count
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        data_type,
                        COUNT(*) as count,
                        SUM(size_bytes) as total_size,
                        AVG(size_bytes) as avg_size
                    FROM test_data_records 
                    GROUP BY data_type
                """)
                
                stats = {}
                for row in cursor.fetchall():
                    stats[row[0]] = {
                        'count': row[1],
                        'total_size': row[2],
                        'avg_size': row[3]
                    }
                
                # Overall stats
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_records,
                        SUM(size_bytes) as total_size
                    FROM test_data_records
                """)
                
                overall = cursor.fetchone()
                stats['overall'] = {
                    'total_records': overall[0],
                    'total_size': overall[1]
                }
                
                return stats
    
    @contextmanager
    def transaction(self):
        """Context manager for transactional operations."""
        with self._lock:
            try:
                yield self
            except Exception as e:
                self.logger.error(f"Transaction error: {e}")
                raise
    
    def export_data(self, record_ids: List[str], export_path: str) -> bool:
        """Export test data to external location."""
        try:
            export_dir = Path(export_path)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            for record_id in record_ids:
                data = self.retrieve_data(record_id)
                if data is not None:
                    export_file = export_dir / f"{record_id}.pkl"
                    self._serialize_data(data, export_file)
            
            self.logger.info(f"Exported {len(record_ids)} data records to {export_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return False
    
    def import_data(self, import_path: str) -> List[str]:
        """Import test data from external location."""
        imported_ids = []
        
        try:
            import_dir = Path(import_path)
            
            for file_path in import_dir.glob("*.pkl"):
                data = self._deserialize_data(file_path)
                record_id = self.store_data(
                    name=file_path.stem,
                    data=data,
                    data_type=DataType.FIXTURE_DATA,
                    metadata={"imported_from": str(file_path)}
                )
                imported_ids.append(record_id)
            
            self.logger.info(f"Imported {len(imported_ids)} data records from {import_path}")
            return imported_ids
        except Exception as e:
            self.logger.error(f"Error importing data: {e}")
            return []


class MockDataGenerator:
    """Generate mock data for testing scenarios."""
    
    def __init__(self, seed: int = 42):
        """Initialize mock data generator."""
        np.random.seed(seed)
        self.seed = seed
    
    def generate_market_data(self, 
                           symbols: List[str],
                           start_date: datetime,
                           end_date: datetime,
                           frequency: str = '1min') -> pd.DataFrame:
        """Generate mock market data."""
        date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
        
        data = []
        for symbol in symbols:
            # Generate realistic price movements
            initial_price = np.random.uniform(10, 500)
            returns = np.random.normal(0, 0.02, len(date_range))
            prices = initial_price * np.exp(np.cumsum(returns))
            
            # Generate OHLCV data
            for i, timestamp in enumerate(date_range):
                open_price = prices[i]
                close_price = prices[i] * np.random.uniform(0.98, 1.02)
                high_price = max(open_price, close_price) * np.random.uniform(1.0, 1.05)
                low_price = min(open_price, close_price) * np.random.uniform(0.95, 1.0)
                volume = np.random.randint(1000, 100000)
                
                data.append({
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
        
        return pd.DataFrame(data)
    
    def generate_indicator_data(self, 
                              market_data: pd.DataFrame,
                              indicators: List[str]) -> pd.DataFrame:
        """Generate mock indicator data."""
        result = market_data.copy()
        
        for indicator in indicators:
            if indicator == 'sma_20':
                result['sma_20'] = result.groupby('symbol')['close'].rolling(20).mean().reset_index(0, drop=True)
            elif indicator == 'rsi':
                result['rsi'] = result.groupby('symbol')['close'].apply(
                    lambda x: 50 + np.random.normal(0, 20, len(x))
                ).reset_index(0, drop=True)
            elif indicator == 'macd':
                result['macd'] = result.groupby('symbol')['close'].apply(
                    lambda x: np.random.normal(0, 1, len(x))
                ).reset_index(0, drop=True)
        
        return result
    
    def generate_signal_data(self, 
                           market_data: pd.DataFrame,
                           strategies: List[str]) -> pd.DataFrame:
        """Generate mock signal data."""
        result = market_data.copy()
        
        for strategy in strategies:
            # Generate random signals
            signals = np.random.choice(['BUY', 'SELL', 'HOLD'], 
                                     size=len(result), 
                                     p=[0.1, 0.1, 0.8])
            result[f'{strategy}_signal'] = signals
            
            # Generate confidence scores
            confidence = np.random.uniform(0.1, 1.0, len(result))
            result[f'{strategy}_confidence'] = confidence
        
        return result


class CacheManager:
    """Manage cached test data."""
    
    def __init__(self, cache_dir: str, max_size_mb: int = 100):
        """Initialize cache manager."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._lock = threading.RLock()
    
    def get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached data."""
        with self._lock:
            cache_file = self.cache_dir / f"{key}.pkl"
            
            if not cache_file.exists():
                return None
            
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Update access time
                cache_file.touch()
                return data
            except Exception:
                return None
    
    def set(self, key: str, data: Any) -> bool:
        """Set cached data."""
        with self._lock:
            cache_file = self.cache_dir / f"{key}.pkl"
            
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
                
                # Clean up if cache is too large
                self._cleanup_cache()
                return True
            except Exception:
                return False
    
    def _cleanup_cache(self):
        """Clean up cache to stay within size limits."""
        total_size = sum(f.stat().st_size for f in self.cache_dir.iterdir())
        
        if total_size > self.max_size_bytes:
            # Remove oldest files first
            files = sorted(self.cache_dir.iterdir(), key=lambda f: f.stat().st_atime)
            
            for file in files:
                file.unlink()
                total_size -= file.stat().st_size
                
                if total_size <= self.max_size_bytes * 0.8:
                    break
    
    def clear(self):
        """Clear all cached data."""
        with self._lock:
            for file in self.cache_dir.iterdir():
                if file.is_file():
                    file.unlink()


class TestDataIsolation:
    """Manage test data isolation and cleanup."""
    
    def __init__(self, test_data_manager: TestDataManager):
        """Initialize test data isolation."""
        self.test_data_manager = test_data_manager
        self._test_data_stack = []
        self._lock = threading.RLock()
    
    @contextmanager
    def isolated_test_data(self, test_name: str):
        """Context manager for isolated test data."""
        with self._lock:
            # Create isolated namespace
            namespace = f"test_{test_name}_{datetime.now().isoformat()}"
            created_data = []
            
            # Store current state
            self._test_data_stack.append({
                'namespace': namespace,
                'created_data': created_data
            })
            
            try:
                # Provide isolated data manager
                yield TestDataContext(self.test_data_manager, namespace, created_data)
            finally:
                # Clean up test data
                context = self._test_data_stack.pop()
                for record_id in context['created_data']:
                    self.test_data_manager.delete_data(record_id)


class TestDataContext:
    """Context for isolated test data operations."""
    
    def __init__(self, test_data_manager: TestDataManager, namespace: str, created_data: List[str]):
        """Initialize test data context."""
        self.test_data_manager = test_data_manager
        self.namespace = namespace
        self.created_data = created_data
    
    def store_data(self, name: str, data: Any, **kwargs) -> str:
        """Store data with namespace isolation."""
        namespaced_name = f"{self.namespace}_{name}"
        record_id = self.test_data_manager.store_data(namespaced_name, data, **kwargs)
        self.created_data.append(record_id)
        return record_id
    
    def retrieve_data(self, record_id: str) -> Optional[Any]:
        """Retrieve data."""
        return self.test_data_manager.retrieve_data(record_id)
    
    def search_data(self, **kwargs) -> List[TestDataRecord]:
        """Search data within namespace."""
        return self.test_data_manager.search_data(**kwargs)