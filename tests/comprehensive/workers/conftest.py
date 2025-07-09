"""
Shared fixtures for worker process tests.
"""

import pytest
import asyncio
import threading
import time
import redis
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from dataclasses import dataclass
import logging
import json
import psutil
import os

from src.realtime.config_manager import (
    ConfigManager,
    SignalProfile,
    AssetConfig,
    SystemConfig,
)
from src.realtime.calculation_engine import SignalDetectionEngine
from src.realtime.data_ingestion import MarketDataIngestionService
from src.realtime.timeframe_manager import TimeframeManager
from src.realtime.signal_engine import SignalEngine
from src.core.models import OHLCV
from src.data.providers.mock_provider import MockProvider


@dataclass
class WorkerTestConfig:
    """Configuration for worker tests."""

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 1  # Use different DB for tests
    test_timeout: int = 30
    max_memory_mb: int = 500
    max_cpu_percent: int = 80


@pytest.fixture(scope="session")
def worker_test_config():
    """Get worker test configuration."""
    return WorkerTestConfig()


@pytest.fixture(scope="session")
def redis_test_client(worker_test_config):
    """Create Redis client for tests."""
    try:
        client = redis.Redis(
            host=worker_test_config.redis_host,
            port=worker_test_config.redis_port,
            db=worker_test_config.redis_db,
            decode_responses=True,
        )
        client.ping()
        yield client
    except redis.ConnectionError:
        pytest.skip("Redis not available for tests")
    finally:
        if "client" in locals():
            client.flushdb()  # Clean test database
            client.close()


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for isolated tests."""
    mock_client = Mock()
    mock_client.ping.return_value = True
    mock_client.xadd.return_value = f"test-{int(time.time() * 1000)}"
    mock_client.xreadgroup.return_value = []
    mock_client.xgroup_create.return_value = True
    mock_client.xack.return_value = 1
    mock_client.xinfo_stream.return_value = {
        "length": 0,
        "groups": 0,
        "last-generated-id": "0-0",
    }
    return mock_client


@pytest.fixture
def test_config_manager():
    """Create test configuration manager."""
    with patch("src.realtime.config_manager.cache_manager"):
        config_manager = ConfigManager()

        # Setup test watchlist
        test_assets = [
            AssetConfig(
                symbol="AAPL",
                enabled=True,
                data_provider="mock",
                timeframes=["1m", "5m", "1h"],
                priority=1,
                enabled_strategies=["macd_rsi_strategy", "rsi_trend_strategy"],
                trading_profile=TradingProfile.SWING_TRADING,
            ),
            AssetConfig(
                symbol="GOOGL",
                enabled=True,
                data_provider="mock",
                timeframes=["5m", "1h", "1d"],
                priority=2,
                enabled_strategies=["ma_crossover_strategy", "rsi_trend_strategy"],
                trading_profile=TradingProfile.POSITION_TRADING,
            ),
        ]

        # Mock the configuration methods
        config_manager.get_watchlist = Mock(return_value=test_assets)
        config_manager.get_enabled_symbols = Mock(return_value=["AAPL", "GOOGL"])
        config_manager.get_enabled_indicators = Mock(
            return_value=config_manager._default_indicators
        )
        config_manager.get_enabled_strategies = Mock(
            return_value=config_manager._default_strategies
        )
        config_manager.get_system_config = Mock(return_value=SystemConfig())
        config_manager.get_timeframe_configs = Mock(
            return_value=config_manager._default_timeframe_configs
        )
        config_manager.get_strategy_matrix = Mock(
            return_value=config_manager._default_strategy_matrix
        )

        return config_manager


@pytest.fixture
def mock_data_provider():
    """Create mock data provider."""
    provider = MockProvider()

    # Create realistic test data
    base_time = datetime.now()
    test_data = []
    base_price = 150.0

    for i in range(100):
        timestamp = base_time - timedelta(minutes=i)
        price_change = (i % 10 - 5) * 0.5  # Oscillating price
        price = base_price + price_change

        ohlcv = OHLCV(
            timestamp=timestamp,
            open=price,
            high=price * 1.01,
            low=price * 0.99,
            close=price + 0.1,
            volume=1000 + i * 10,
        )
        test_data.append(ohlcv)

    provider.get_latest_data = Mock(return_value=test_data[-1])
    return provider


@pytest.fixture
def test_timeframe_manager(test_config_manager):
    """Create test timeframe manager."""
    return TimeframeManager(test_config_manager)


@pytest.fixture
def test_signal_engine(test_config_manager, test_timeframe_manager):
    """Create test signal engine."""
    return SignalEngine(test_config_manager, test_timeframe_manager)


@pytest.fixture
async def test_data_ingestion_service(test_config_manager, mock_redis_client):
    """Create test data ingestion service."""
    with patch(
        "src.realtime.data_ingestion.redis.Redis", return_value=mock_redis_client
    ):
        service = MarketDataIngestionService(test_config_manager)
        yield service
        await service.stop()


@pytest.fixture
async def test_calculation_engine(
    test_config_manager, test_data_ingestion_service, mock_redis_client
):
    """Create test calculation engine."""
    with patch(
        "src.realtime.calculation_engine.redis.Redis", return_value=mock_redis_client
    ):
        engine = CalculationEngine(test_config_manager, test_data_ingestion_service)
        yield engine
        await engine.stop()


@pytest.fixture
def sample_market_data_events():
    """Generate sample market data events."""
    base_time = datetime.now()
    events = []

    symbols = ["AAPL", "GOOGL", "MSFT"]

    for i in range(50):
        for symbol in symbols:
            event_data = {
                "symbol": symbol,
                "timestamp": (base_time - timedelta(seconds=i)).isoformat(),
                "price": str(150.0 + i * 0.1),
                "volume": str(1000 + i * 10),
                "open": str(149.0 + i * 0.1),
                "high": str(151.0 + i * 0.1),
                "low": str(148.0 + i * 0.1),
                "close": str(150.0 + i * 0.1),
                "source": "mock",
            }
            events.append((f"event-{i}-{symbol}", event_data))

    return events


@pytest.fixture
def resource_monitor():
    """Monitor system resources during tests."""

    class ResourceMonitor:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.start_memory = self.get_memory_usage()
            self.start_cpu = self.get_cpu_usage()
            self.peak_memory = self.start_memory
            self.peak_cpu = self.start_cpu

        def get_memory_usage(self):
            """Get current memory usage in MB."""
            return self.process.memory_info().rss / 1024 / 1024

        def get_cpu_usage(self):
            """Get current CPU usage percentage."""
            return self.process.cpu_percent()

        def update_peaks(self):
            """Update peak usage metrics."""
            current_memory = self.get_memory_usage()
            current_cpu = self.get_cpu_usage()

            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
            if current_cpu > self.peak_cpu:
                self.peak_cpu = current_cpu

        def get_summary(self):
            """Get resource usage summary."""
            return {
                "start_memory_mb": self.start_memory,
                "current_memory_mb": self.get_memory_usage(),
                "peak_memory_mb": self.peak_memory,
                "memory_increase_mb": self.get_memory_usage() - self.start_memory,
                "start_cpu_percent": self.start_cpu,
                "current_cpu_percent": self.get_cpu_usage(),
                "peak_cpu_percent": self.peak_cpu,
            }

    return ResourceMonitor()


@pytest.fixture
def performance_thresholds():
    """Define performance thresholds for tests."""
    return {
        "max_startup_time": 5.0,  # seconds
        "max_shutdown_time": 3.0,  # seconds
        "max_processing_time_per_message": 0.1,  # seconds
        "max_memory_increase_mb": 100,  # MB
        "max_cpu_usage_percent": 50,  # %
        "max_error_rate": 0.01,  # 1%
        "min_throughput_messages_per_second": 100,
        "max_latency_ms": 50,
        "max_queue_size": 1000,
    }


@pytest.fixture
def error_injector():
    """Utility for injecting errors during tests."""

    class ErrorInjector:
        def __init__(self):
            self.error_count = 0
            self.error_types = []

        def inject_network_error(self):
            """Inject network connection error."""
            self.error_count += 1
            self.error_types.append("network")
            raise ConnectionError("Simulated network error")

        def inject_redis_error(self):
            """Inject Redis connection error."""
            self.error_count += 1
            self.error_types.append("redis")
            raise redis.ConnectionError("Simulated Redis error")

        def inject_processing_error(self):
            """Inject data processing error."""
            self.error_count += 1
            self.error_types.append("processing")
            raise ValueError("Simulated processing error")

        def inject_timeout_error(self):
            """Inject timeout error."""
            self.error_count += 1
            self.error_types.append("timeout")
            raise TimeoutError("Simulated timeout error")

        def get_error_summary(self):
            """Get error injection summary."""
            return {
                "total_errors": self.error_count,
                "error_types": self.error_types,
                "unique_error_types": list(set(self.error_types)),
            }

        def reset(self):
            """Reset error counters."""
            self.error_count = 0
            self.error_types = []

    return ErrorInjector()


@pytest.fixture
def load_generator():
    """Generate load for stress testing."""

    class LoadGenerator:
        def __init__(self):
            self.active_threads = []
            self.stop_flag = threading.Event()

        def generate_message_load(
            self, target_function, messages_per_second=10, duration=5
        ):
            """Generate message processing load."""

            def worker():
                interval = 1.0 / messages_per_second
                start_time = time.time()

                while (
                    not self.stop_flag.is_set()
                    and (time.time() - start_time) < duration
                ):
                    try:
                        target_function()
                        time.sleep(interval)
                    except Exception as e:
                        logging.error(f"Load generation error: {e}")

            thread = threading.Thread(target=worker)
            self.active_threads.append(thread)
            thread.start()

        def generate_cpu_load(self, cpu_percent=50, duration=5):
            """Generate CPU load."""

            def cpu_worker():
                start_time = time.time()

                while (
                    not self.stop_flag.is_set()
                    and (time.time() - start_time) < duration
                ):
                    # Busy work to consume CPU
                    for _ in range(10000):
                        _ = sum(range(100))
                    time.sleep(0.01)  # Brief pause to allow other processes

            thread = threading.Thread(target=cpu_worker)
            self.active_threads.append(thread)
            thread.start()

        def stop_all(self):
            """Stop all load generation."""
            self.stop_flag.set()
            for thread in self.active_threads:
                thread.join(timeout=2)
            self.active_threads.clear()
            self.stop_flag.clear()

    return LoadGenerator()


@pytest.fixture
def worker_health_checker():
    """Health checker for worker processes."""

    class WorkerHealthChecker:
        def __init__(self):
            self.checks = []

        def check_worker_responsiveness(self, worker, timeout=5):
            """Check if worker responds to status requests."""
            start_time = time.time()
            try:
                if hasattr(worker, "get_stats"):
                    stats = worker.get_stats()
                    response_time = time.time() - start_time
                    self.checks.append(
                        {
                            "check": "responsiveness",
                            "passed": response_time < timeout,
                            "response_time": response_time,
                            "stats": stats,
                        }
                    )
                    return response_time < timeout
                else:
                    self.checks.append(
                        {
                            "check": "responsiveness",
                            "passed": False,
                            "error": "Worker does not have get_stats method",
                        }
                    )
                    return False
            except Exception as e:
                self.checks.append(
                    {"check": "responsiveness", "passed": False, "error": str(e)}
                )
                return False

        def check_worker_state(self, worker):
            """Check worker internal state."""
            try:
                if hasattr(worker, "_running"):
                    is_running = getattr(worker, "_running", False)
                    self.checks.append(
                        {"check": "state", "passed": True, "running": is_running}
                    )
                    return True
                else:
                    self.checks.append(
                        {
                            "check": "state",
                            "passed": False,
                            "error": "Worker does not have _running attribute",
                        }
                    )
                    return False
            except Exception as e:
                self.checks.append({"check": "state", "passed": False, "error": str(e)})
                return False

        def check_memory_leaks(self, worker, threshold_mb=50):
            """Check for memory leaks."""
            try:
                process = psutil.Process(os.getpid())
                memory_usage = process.memory_info().rss / 1024 / 1024

                passed = memory_usage < threshold_mb
                self.checks.append(
                    {
                        "check": "memory_leaks",
                        "passed": passed,
                        "memory_usage_mb": memory_usage,
                        "threshold_mb": threshold_mb,
                    }
                )
                return passed
            except Exception as e:
                self.checks.append(
                    {"check": "memory_leaks", "passed": False, "error": str(e)}
                )
                return False

        def get_health_report(self):
            """Get comprehensive health report."""
            total_checks = len(self.checks)
            passed_checks = sum(
                1 for check in self.checks if check.get("passed", False)
            )

            return {
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "failed_checks": total_checks - passed_checks,
                "health_score": passed_checks / total_checks if total_checks > 0 else 0,
                "checks": self.checks,
            }

    return WorkerHealthChecker()


@pytest.fixture
def message_factory():
    """Factory for creating test messages."""

    class MessageFactory:
        def __init__(self):
            self.message_counter = 0

        def create_market_data_message(self, symbol="AAPL", price=150.0, volume=1000):
            """Create market data message."""
            self.message_counter += 1
            return {
                "event_id": f"test_event_{self.message_counter}",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "price": str(price),
                "volume": str(volume),
                "open": str(price * 0.99),
                "high": str(price * 1.01),
                "low": str(price * 0.98),
                "close": str(price),
                "source": "test",
                "metadata": json.dumps({"test": True, "id": self.message_counter}),
            }

        def create_calculation_event(self, symbol="AAPL", results=None):
            """Create calculation event message."""
            self.message_counter += 1
            return {
                "event_id": f"calc_event_{self.message_counter}",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "results": json.dumps(
                    results or {"test": True, "id": self.message_counter}
                ),
            }

        def create_alert_message(
            self, alert_type="signal", symbol="AAPL", message="Test alert"
        ):
            """Create alert message."""
            self.message_counter += 1
            return {
                "event_id": f"alert_{alert_type}_{self.message_counter}",
                "type": alert_type,
                "symbol": symbol,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "metadata": json.dumps({"test": True, "id": self.message_counter}),
            }

        def create_batch_messages(self, count=10, message_type="market_data", **kwargs):
            """Create batch of messages."""
            messages = []
            for i in range(count):
                if message_type == "market_data":
                    messages.append(self.create_market_data_message(**kwargs))
                elif message_type == "calculation":
                    messages.append(self.create_calculation_event(**kwargs))
                elif message_type == "alert":
                    messages.append(self.create_alert_message(**kwargs))
            return messages

    return MessageFactory()


@pytest.fixture(autouse=True)
async def cleanup_workers():
    """Cleanup workers after each test."""
    yield
    # Cleanup code runs after each test
    pass


@pytest.fixture
def async_timeout():
    """Timeout for async operations."""
    return 10.0  # seconds


@pytest.fixture
def worker_stress_config():
    """Configuration for stress testing."""
    return {
        "max_concurrent_workers": 5,
        "message_burst_size": 100,
        "stress_duration": 10,  # seconds
        "memory_limit_mb": 200,
        "cpu_limit_percent": 70,
        "error_injection_rate": 0.05,  # 5% error rate
    }
