#!/usr/bin/env python3
"""
Test Data Management System

Manages test data creation, cleanup, archival, and validation for the comprehensive
test framework. Handles both live market data and synthetic test data.
"""

import asyncio
import json
import os
import logging
import shutil
import gzip
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import tempfile
import hashlib

from src.storage.cache import cache_manager
from src.core.models import OHLCV, TimeFrame

logger = logging.getLogger(__name__)


@dataclass
class TestDataSet:
    """Test data set configuration."""

    name: str
    symbols: List[str]
    timeframes: List[str]
    start_time: datetime
    end_time: datetime
    data_type: str  # live, synthetic, historical
    description: str
    metadata: Dict[str, Any]


class TestDataManager:
    """Manages test data across the comprehensive test framework."""

    def __init__(self, base_dir: Optional[str] = None):
        """Initialize test data manager."""
        self.base_dir = Path(base_dir or self._get_default_base_dir())
        self.data_dir = self.base_dir / "data"
        self.archive_dir = self.base_dir / "archive"
        self.temp_dir = self.base_dir / "temp"
        self.metadata_file = self.base_dir / "data_catalog.json"

        # Data catalog
        self.data_catalog = {}
        self.active_datasets = {}

        # Test symbols for various scenarios
        self.test_symbols = {
            "large_cap": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            "etf": ["SPY", "QQQ", "IWM", "VTI", "GLD"],
            "volatile": ["GME", "AMC", "MEME", "COIN", "NVDA"],
            "forex": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
            "crypto": ["BTCUSD", "ETHUSD", "ADAUSD", "SOLUSD"],
        }

        self._ensure_directories()
        self._load_catalog()

    def _get_default_base_dir(self) -> str:
        """Get default base directory for test data."""
        return os.path.join(os.path.dirname(__file__), "..", "data")

    def _ensure_directories(self):
        """Ensure all required directories exist."""
        for directory in [self.data_dir, self.archive_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _load_catalog(self):
        """Load data catalog from file."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, "r") as f:
                    self.data_catalog = json.load(f)
                logger.info(
                    f"Loaded data catalog with {len(self.data_catalog)} entries"
                )
            else:
                self.data_catalog = {}
                self._save_catalog()
        except Exception as e:
            logger.error(f"Failed to load data catalog: {e}")
            self.data_catalog = {}

    def _save_catalog(self):
        """Save data catalog to file."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.data_catalog, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save data catalog: {e}")

    async def setup(self):
        """Setup test data manager."""
        logger.info("Setting up test data manager...")

        # Clear temporary data
        await self._clear_temp_data()

        # Initialize Redis namespaces
        self._initialize_redis_namespaces()

        # Validate data integrity
        await self._validate_existing_data()

        logger.info("Test data manager setup completed")

    async def cleanup(self):
        """Cleanup test data manager."""
        logger.info("Cleaning up test data manager...")

        # Archive active datasets
        await self._archive_active_datasets()

        # Clear temporary data
        await self._clear_temp_data()

        # Clear Redis test namespaces
        self._clear_redis_namespaces()

        logger.info("Test data manager cleanup completed")

    def _initialize_redis_namespaces(self):
        """Initialize Redis namespaces for test data."""
        namespaces = [
            "test_market_data",
            "test_indicators",
            "test_signals",
            "test_portfolio",
        ]
        for namespace in namespaces:
            try:
                cache_manager.clear_namespace(namespace)
            except Exception as e:
                logger.debug(f"Failed to clear namespace {namespace}: {e}")

    def _clear_redis_namespaces(self):
        """Clear Redis test namespaces."""
        namespaces = [
            "test_market_data",
            "test_indicators",
            "test_signals",
            "test_portfolio",
        ]
        for namespace in namespaces:
            try:
                cache_manager.clear_namespace(namespace)
            except Exception as e:
                logger.debug(f"Failed to clear namespace {namespace}: {e}")

    async def _clear_temp_data(self):
        """Clear temporary data files."""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir(exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to clear temp data: {e}")

    async def _validate_existing_data(self):
        """Validate integrity of existing data."""
        validation_results = {}

        for dataset_name, metadata in self.data_catalog.items():
            dataset_path = self.data_dir / f"{dataset_name}.pkl.gz"

            if dataset_path.exists():
                try:
                    # Validate file integrity
                    file_size = dataset_path.stat().st_size
                    expected_size = metadata.get("file_size", 0)

                    if file_size != expected_size:
                        logger.warning(
                            f"File size mismatch for {dataset_name}: {file_size} vs {expected_size}"
                        )
                        validation_results[dataset_name] = "size_mismatch"
                    else:
                        validation_results[dataset_name] = "valid"

                except Exception as e:
                    logger.error(f"Validation failed for {dataset_name}: {e}")
                    validation_results[dataset_name] = "error"
            else:
                logger.warning(f"Missing data file for {dataset_name}")
                validation_results[dataset_name] = "missing"

        logger.info(
            f"Data validation completed: {len(validation_results)} datasets checked"
        )
        return validation_results

    async def create_synthetic_dataset(
        self,
        name: str,
        symbols: List[str],
        timeframes: List[str],
        duration_days: int = 30,
        volatility_factor: float = 1.0,
    ) -> TestDataSet:
        """Create synthetic market data for testing."""
        logger.info(f"Creating synthetic dataset '{name}' with {len(symbols)} symbols")

        start_time = datetime.now() - timedelta(days=duration_days)
        end_time = datetime.now()

        synthetic_data = {}

        for symbol in symbols:
            synthetic_data[symbol] = {}

            for timeframe in timeframes:
                # Generate synthetic OHLCV data
                data_points = self._generate_synthetic_ohlcv(
                    symbol, timeframe, start_time, end_time, volatility_factor
                )
                synthetic_data[symbol][timeframe] = data_points

        # Create dataset
        dataset = TestDataSet(
            name=name,
            symbols=symbols,
            timeframes=timeframes,
            start_time=start_time,
            end_time=end_time,
            data_type="synthetic",
            description=f"Synthetic market data for {duration_days} days",
            metadata={
                "duration_days": duration_days,
                "volatility_factor": volatility_factor,
                "created_at": datetime.now().isoformat(),
            },
        )

        # Save dataset
        await self._save_dataset(dataset, synthetic_data)

        logger.info(f"Synthetic dataset '{name}' created successfully")
        return dataset

    def _generate_synthetic_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        volatility_factor: float,
    ) -> List[OHLCV]:
        """Generate synthetic OHLCV data."""
        import random
        import math

        # Determine time delta based on timeframe
        timeframe_deltas = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "1d": timedelta(days=1),
        }

        delta = timeframe_deltas.get(timeframe, timedelta(minutes=1))

        # Base price for symbol
        base_prices = {
            "AAPL": 150.0,
            "MSFT": 300.0,
            "GOOGL": 2500.0,
            "SPY": 400.0,
            "QQQ": 350.0,
            "TSLA": 200.0,
        }
        base_price = base_prices.get(symbol, 100.0)

        data_points = []
        current_time = start_time
        current_price = base_price

        while current_time < end_time:
            # Generate realistic price movement
            volatility = 0.02 * volatility_factor  # 2% base volatility
            price_change = random.gauss(0, volatility) * current_price

            # Trend component (slight upward bias)
            trend = 0.0001 * current_price

            new_price = max(current_price + price_change + trend, 0.01)

            # Generate OHLCV
            high = new_price + random.uniform(0, 0.005 * new_price)
            low = new_price - random.uniform(0, 0.005 * new_price)
            volume = random.randint(100000, 1000000)

            ohlcv = OHLCV(
                timestamp=current_time,
                open_price=current_price,
                high_price=high,
                low_price=low,
                close_price=new_price,
                volume=volume,
            )

            data_points.append(ohlcv)

            current_price = new_price
            current_time += delta

        return data_points

    async def _save_dataset(self, dataset: TestDataSet, data: Dict[str, Any]):
        """Save dataset to file."""
        dataset_path = self.data_dir / f"{dataset.name}.pkl.gz"

        try:
            # Serialize and compress data
            with gzip.open(dataset_path, "wb") as f:
                pickle.dump({"metadata": asdict(dataset), "data": data}, f)

            # Update catalog
            self.data_catalog[dataset.name] = {
                **asdict(dataset),
                "file_path": str(dataset_path),
                "file_size": dataset_path.stat().st_size,
                "checksum": self._calculate_file_checksum(dataset_path),
            }

            self._save_catalog()

        except Exception as e:
            logger.error(f"Failed to save dataset {dataset.name}: {e}")
            raise

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    async def load_dataset(self, name: str) -> Optional[Dict[str, Any]]:
        """Load dataset from file."""
        if name not in self.data_catalog:
            logger.warning(f"Dataset '{name}' not found in catalog")
            return None

        dataset_path = Path(self.data_catalog[name]["file_path"])

        if not dataset_path.exists():
            logger.error(f"Dataset file not found: {dataset_path}")
            return None

        try:
            with gzip.open(dataset_path, "rb") as f:
                dataset_content = pickle.load(f)

            logger.info(f"Loaded dataset '{name}' successfully")
            return dataset_content

        except Exception as e:
            logger.error(f"Failed to load dataset {name}: {e}")
            return None

    async def inject_test_data_to_redis(
        self,
        dataset_name: str,
        namespace: str = "test_market_data",
        symbol_filter: Optional[List[str]] = None,
    ):
        """Inject test data into Redis for testing."""
        dataset = await self.load_dataset(dataset_name)
        if not dataset:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        data = dataset["data"]
        injected_count = 0

        for symbol, symbol_data in data.items():
            if symbol_filter and symbol not in symbol_filter:
                continue

            for timeframe, ohlcv_list in symbol_data.items():
                for ohlcv in ohlcv_list:
                    cache_key = f"{symbol}_{timeframe}_{ohlcv.timestamp.isoformat()}"

                    try:
                        cache_manager.set(
                            namespace,
                            cache_key,
                            {
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "timestamp": ohlcv.timestamp.isoformat(),
                                "open": ohlcv.open_price,
                                "high": ohlcv.high_price,
                                "low": ohlcv.low_price,
                                "close": ohlcv.close_price,
                                "volume": ohlcv.volume,
                            },
                            ttl=3600,  # 1 hour TTL
                        )
                        injected_count += 1

                    except Exception as e:
                        logger.warning(f"Failed to inject data point {cache_key}: {e}")

        logger.info(
            f"Injected {injected_count} data points to Redis namespace '{namespace}'"
        )
        return injected_count

    async def _archive_active_datasets(self):
        """Archive currently active datasets."""
        if not self.active_datasets:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = self.archive_dir / f"test_session_{timestamp}.json"

        try:
            archive_data = {
                "timestamp": timestamp,
                "active_datasets": list(self.active_datasets.keys()),
                "session_metadata": {
                    "total_datasets": len(self.active_datasets),
                    "archive_created": datetime.now().isoformat(),
                },
            }

            with open(archive_file, "w") as f:
                json.dump(archive_data, f, indent=2)

            logger.info(
                f"Archived {len(self.active_datasets)} datasets to {archive_file}"
            )

        except Exception as e:
            logger.error(f"Failed to archive datasets: {e}")

    def get_test_symbols(self, category: str = "large_cap") -> List[str]:
        """Get test symbols for specific category."""
        return self.test_symbols.get(category, self.test_symbols["large_cap"])

    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets."""
        return list(self.data_catalog.keys())

    def get_dataset_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific dataset."""
        return self.data_catalog.get(name)

    async def create_temporary_file(self, content: str, suffix: str = ".tmp") -> str:
        """Create temporary file with content."""
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, dir=self.temp_dir, delete=False
        )

        try:
            temp_file.write(content)
            temp_file.flush()
            return temp_file.name
        finally:
            temp_file.close()

    def cleanup_temporary_files(self):
        """Clean up all temporary files."""
        try:
            for file_path in self.temp_dir.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
            logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary files: {e}")

    def get_data_usage_stats(self) -> Dict[str, Any]:
        """Get data usage statistics."""
        stats = {
            "total_datasets": len(self.data_catalog),
            "total_size_mb": 0,
            "categories": {},
        }

        for name, metadata in self.data_catalog.items():
            file_size = metadata.get("file_size", 0)
            stats["total_size_mb"] += file_size / (1024 * 1024)

            data_type = metadata.get("data_type", "unknown")
            if data_type not in stats["categories"]:
                stats["categories"][data_type] = {"count": 0, "size_mb": 0}

            stats["categories"][data_type]["count"] += 1
            stats["categories"][data_type]["size_mb"] += file_size / (1024 * 1024)

        stats["total_size_mb"] = round(stats["total_size_mb"], 2)

        return stats
