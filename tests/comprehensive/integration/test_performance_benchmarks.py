"""
Performance Benchmarks and Load Testing

This module contains comprehensive performance tests for the real-time
system, including latency measurements, throughput testing, memory usage
monitoring, and system stability under various load conditions.
"""

import pytest
import asyncio
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch, AsyncMock
from collections import defaultdict, deque
import statistics
import json
import gc

# Import test modules
from src.realtime.data_ingestion import MarketDataIngestionService
from src.realtime.config_manager import ConfigManager, AssetConfig, SystemConfig
from src.data.providers.mock_provider import MockProvider
from src.core.models import OHLCV, TimeFrame

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Performance monitoring utility for tests."""
    
    def __init__(self):
        self.metrics = {
            'start_time': None,
            'end_time': None,
            'operations': 0,
            'errors': 0,
            'latencies': [],
            'memory_samples': [],
            'cpu_samples': [],
            'throughput_samples': []
        }
        self.running = False
        self._monitor_task = None
    
    def start(self):
        """Start performance monitoring."""
        self.metrics['start_time'] = time.time()
        self.running = True
        self._monitor_task = asyncio.create_task(self._monitor_system())
    
    async def stop(self):
        """Stop performance monitoring."""
        self.running = False
        self.metrics['end_time'] = time.time()
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_system(self):
        """Monitor system resources."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            
            while self.running:
                try:
                    # Memory usage
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    self.metrics['memory_samples'].append(memory_mb)
                    
                    # CPU usage
                    cpu_percent = process.cpu_percent()
                    self.metrics['cpu_samples'].append(cpu_percent)
                    
                    await asyncio.sleep(0.1)  # Sample every 100ms
                    
                except Exception as e:
                    logger.warning(f"Error monitoring system resources: {e}")
                    await asyncio.sleep(1.0)
                    
        except ImportError:
            logger.warning("psutil not available for system monitoring")
        except asyncio.CancelledError:
            pass
    
    def record_operation(self, latency: float = None, error: bool = False):
        """Record a single operation."""
        self.metrics['operations'] += 1
        if error:
            self.metrics['errors'] += 1
        if latency is not None:
            self.metrics['latencies'].append(latency)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        duration = (self.metrics['end_time'] or time.time()) - (self.metrics['start_time'] or time.time())
        
        summary = {
            'duration_seconds': duration,
            'total_operations': self.metrics['operations'],
            'total_errors': self.metrics['errors'],
            'error_rate': self.metrics['errors'] / max(self.metrics['operations'], 1),
            'operations_per_second': self.metrics['operations'] / max(duration, 0.001)
        }
        
        if self.metrics['latencies']:
            summary.update({
                'avg_latency_ms': statistics.mean(self.metrics['latencies']) * 1000,
                'median_latency_ms': statistics.median(self.metrics['latencies']) * 1000,
                'p95_latency_ms': statistics.quantiles(self.metrics['latencies'], n=20)[18] * 1000,
                'p99_latency_ms': statistics.quantiles(self.metrics['latencies'], n=100)[98] * 1000,
                'max_latency_ms': max(self.metrics['latencies']) * 1000
            })
        
        if self.metrics['memory_samples']:
            summary.update({
                'avg_memory_mb': statistics.mean(self.metrics['memory_samples']),
                'peak_memory_mb': max(self.metrics['memory_samples']),
                'memory_std_mb': statistics.stdev(self.metrics['memory_samples']) if len(self.metrics['memory_samples']) > 1 else 0
            })
        
        if self.metrics['cpu_samples']:
            summary.update({
                'avg_cpu_percent': statistics.mean(self.metrics['cpu_samples']),
                'peak_cpu_percent': max(self.metrics['cpu_samples'])
            })
        
        return summary


class TestPerformanceBenchmarks:
    """
    Comprehensive performance benchmarks for the real-time system.
    
    These tests validate:
    - System latency under various load conditions
    - Throughput capabilities and limitations
    - Memory usage and leak detection
    - CPU utilization and efficiency
    - System stability under sustained load
    - Recovery performance after errors
    """
    
    @pytest.fixture(scope="class")
    def performance_config_manager(self):
        """Create configuration manager optimized for performance testing."""
        config_manager = Mock(spec=ConfigManager)
        
        # High-performance system configuration
        system_config = SystemConfig(
            data_update_interval_seconds=0.1,  # Very frequent updates
            max_concurrent_calculations=50,
            calculation_batch_size=20,
            worker_pool_size=8,
            redis_config={
                'host': 'localhost',
                'port': 6379,
                'db': 0
            }
        )
        config_manager.get_system_config.return_value = system_config
        
        # Large watchlist for load testing
        symbols = [f'TEST{i:03d}' for i in range(100)]  # 100 test symbols
        watchlist = [
            AssetConfig(
                symbol=symbol,
                enabled=True,
                data_provider='mock',
                strategies=['ma_crossover'],
                position_size=0.01
            ) for symbol in symbols
        ]
        
        config_manager.get_watchlist.return_value = watchlist
        config_manager.get_enabled_symbols.return_value = symbols
        
        return config_manager
    
    @pytest.fixture
    def mock_high_performance_redis(self):
        """Create high-performance mock Redis client."""
        mock_redis = Mock()
        
        # Fast in-memory storage
        storage = {}
        streams = defaultdict(deque)
        
        def fast_xadd(stream_name, data, **kwargs):
            message_id = f"{int(time.time() * 1000000)}-{len(streams[stream_name])}"
            streams[stream_name].append({
                'id': message_id,
                'data': data
            })
            
            # Maintain max length
            max_len = kwargs.get('maxlen', 10000)
            while len(streams[stream_name]) > max_len:
                streams[stream_name].popleft()
            
            return message_id.encode()
        
        def fast_set(key, value, **kwargs):
            storage[key] = value
            return True
        
        def fast_get(key):
            return storage.get(key)
        
        mock_redis.xadd = fast_xadd
        mock_redis.set = fast_set
        mock_redis.get = fast_get
        mock_redis.ping.return_value = True
        mock_redis.xgroup_create.return_value = True
        mock_redis.xinfo_stream.return_value = {'length': 0, 'groups': 1, 'last-generated-id': b'0-0'}
        
        return mock_redis
    
    @pytest.fixture
    def high_volume_data_generator(self):
        """Create high-volume data generator for performance testing."""
        class HighVolumeDataGenerator:
            def __init__(self):
                self.base_prices = {}
                self.last_generation_time = time.time()
                self.generation_count = 0
            
            def generate_batch(self, symbols: List[str], batch_size: int = 1000) -> List[OHLCV]:
                """Generate a large batch of data efficiently."""
                data_batch = []
                current_time = datetime.now()
                
                for symbol in symbols:
                    if symbol not in self.base_prices:
                        self.base_prices[symbol] = 100.0 + len(symbol)  # Unique base price
                    
                    for i in range(batch_size):
                        # Fast price generation
                        price_change = (hash(f"{symbol}_{self.generation_count}_{i}") % 1000 - 500) / 100000
                        price = self.base_prices[symbol] * (1 + price_change)
                        
                        volume = 1000 + (hash(f"vol_{symbol}_{i}") % 10000)
                        
                        ohlcv = OHLCV(
                            timestamp=current_time,
                            symbol=symbol,
                            open=price,
                            high=price * 1.001,
                            low=price * 0.999,
                            close=price,
                            volume=volume
                        )
                        data_batch.append(ohlcv)
                        
                        self.generation_count += 1
                
                return data_batch
        
        return HighVolumeDataGenerator()
    
    @pytest.mark.asyncio
    async def test_baseline_latency_benchmarks(self, performance_config_manager, mock_high_performance_redis):
        """Test baseline latency benchmarks for core operations."""
        monitor = PerformanceMonitor()
        monitor.start()
        
        with patch('redis.Redis', return_value=mock_high_performance_redis):
            ingestion_service = MarketDataIngestionService(performance_config_manager)
            ingestion_service._redis_client = mock_high_performance_redis
            
            # Benchmark single operation latency
            symbol = 'AAPL'
            num_operations = 1000
            
            for i in range(num_operations):
                start_time = time.time()
                
                # Create test data
                ohlcv = OHLCV(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.5,
                    volume=1000
                )
                
                # Perform core operation
                await ingestion_service._publish_market_data_event(symbol, ohlcv, 'benchmark')
                
                latency = time.time() - start_time
                monitor.record_operation(latency)
                
                # Small delay to prevent overwhelming
                if i % 100 == 0:
                    await asyncio.sleep(0.001)
        
        await monitor.stop()
        summary = monitor.get_summary()
        
        # Baseline latency requirements
        assert summary['avg_latency_ms'] < 1.0, f"Average latency too high: {summary['avg_latency_ms']:.2f}ms"
        assert summary['p95_latency_ms'] < 5.0, f"P95 latency too high: {summary['p95_latency_ms']:.2f}ms"
        assert summary['p99_latency_ms'] < 10.0, f"P99 latency too high: {summary['p99_latency_ms']:.2f}ms"
        assert summary['max_latency_ms'] < 50.0, f"Maximum latency too high: {summary['max_latency_ms']:.2f}ms"
        
        logger.info(f"✓ Baseline latency benchmark passed:")
        logger.info(f"  Operations: {summary['total_operations']}")
        logger.info(f"  Average latency: {summary['avg_latency_ms']:.2f}ms")
        logger.info(f"  P95 latency: {summary['p95_latency_ms']:.2f}ms")
        logger.info(f"  P99 latency: {summary['p99_latency_ms']:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_throughput_capacity_limits(self, performance_config_manager, mock_high_performance_redis, high_volume_data_generator):
        """Test maximum throughput capacity under ideal conditions."""
        monitor = PerformanceMonitor()
        monitor.start()
        
        with patch('redis.Redis', return_value=mock_high_performance_redis):
            ingestion_service = MarketDataIngestionService(performance_config_manager)
            ingestion_service._redis_client = mock_high_performance_redis
            
            # Test parameters
            test_duration = 5.0  # seconds
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
            target_rate = 10000  # operations per second
            
            start_time = time.time()
            operations_completed = 0
            
            while time.time() - start_time < test_duration:
                batch_start = time.time()
                
                # Generate batch of operations
                batch_size = min(100, target_rate // 10)  # 100ms batches
                
                tasks = []
                for i in range(batch_size):
                    symbol = symbols[i % len(symbols)]
                    
                    async def process_operation(sym):
                        op_start = time.time()
                        
                        ohlcv = OHLCV(
                            timestamp=datetime.now(),
                            symbol=sym,
                            open=100.0 + operations_completed * 0.001,
                            high=101.0 + operations_completed * 0.001,
                            low=99.0 + operations_completed * 0.001,
                            close=100.5 + operations_completed * 0.001,
                            volume=1000
                        )
                        
                        await ingestion_service._publish_market_data_event(sym, ohlcv, 'throughput_test')
                        
                        latency = time.time() - op_start
                        monitor.record_operation(latency)
                        
                        return 1
                    
                    tasks.append(process_operation(symbol))
                
                # Execute batch
                results = await asyncio.gather(*tasks, return_exceptions=True)
                operations_completed += len([r for r in results if not isinstance(r, Exception)])
                
                # Maintain target rate
                batch_time = time.time() - batch_start
                target_batch_time = batch_size / target_rate
                if batch_time < target_batch_time:
                    await asyncio.sleep(target_batch_time - batch_time)
        
        await monitor.stop()
        summary = monitor.get_summary()
        
        # Throughput requirements
        min_throughput = 5000  # operations per second
        assert summary['operations_per_second'] >= min_throughput, f"Throughput too low: {summary['operations_per_second']:.1f} ops/s"
        
        # Latency should remain reasonable under load
        assert summary['avg_latency_ms'] < 5.0, f"Average latency degraded under load: {summary['avg_latency_ms']:.2f}ms"
        assert summary['error_rate'] < 0.01, f"Error rate too high under load: {summary['error_rate']:.2%}"
        
        logger.info(f"✓ Throughput capacity test passed:")
        logger.info(f"  Achieved throughput: {summary['operations_per_second']:.1f} ops/s")
        logger.info(f"  Total operations: {summary['total_operations']}")
        logger.info(f"  Average latency under load: {summary['avg_latency_ms']:.2f}ms")
        logger.info(f"  Error rate: {summary['error_rate']:.2%}")
    
    @pytest.mark.asyncio
    async def test_memory_usage_efficiency(self, performance_config_manager, mock_high_performance_redis):
        """Test memory usage efficiency and leak detection."""
        monitor = PerformanceMonitor()
        monitor.start()
        
        with patch('redis.Redis', return_value=mock_high_performance_redis):
            ingestion_service = MarketDataIngestionService(performance_config_manager)
            ingestion_service._redis_client = mock_high_performance_redis
            
            # Memory baseline
            gc.collect()  # Force garbage collection
            await asyncio.sleep(0.1)  # Let things settle
            
            # Extended operation with memory monitoring
            symbols = [f'MEM{i:03d}' for i in range(50)]
            operations_per_cycle = 100
            cycles = 20
            
            for cycle in range(cycles):
                cycle_start = time.time()
                
                # Perform operations
                for i in range(operations_per_cycle):
                    symbol = symbols[i % len(symbols)]
                    
                    ohlcv = OHLCV(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        open=100.0 + cycle * 10 + i,
                        high=101.0 + cycle * 10 + i,
                        low=99.0 + cycle * 10 + i,
                        close=100.5 + cycle * 10 + i,
                        volume=1000 + i
                    )
                    
                    await ingestion_service._publish_market_data_event(symbol, ohlcv, 'memory_test')
                    monitor.record_operation()
                
                # Periodic garbage collection
                if cycle % 5 == 0:
                    gc.collect()
                
                # Control cycle rate
                cycle_time = time.time() - cycle_start
                if cycle_time < 0.1:
                    await asyncio.sleep(0.1 - cycle_time)
        
        await monitor.stop()
        summary = monitor.get_summary()
        
        # Memory efficiency requirements
        if 'peak_memory_mb' in summary:
            memory_per_operation = summary['peak_memory_mb'] / summary['total_operations'] * 1024  # KB per operation
            assert memory_per_operation < 1.0, f"Memory usage per operation too high: {memory_per_operation:.3f} KB/op"
            
            # Memory growth should be bounded
            if 'memory_std_mb' in summary:
                memory_volatility = summary['memory_std_mb'] / summary['avg_memory_mb'] if summary['avg_memory_mb'] > 0 else 0
                assert memory_volatility < 0.2, f"Memory usage too volatile: {memory_volatility:.2%}"
        
        logger.info(f"✓ Memory efficiency test passed:")
        logger.info(f"  Total operations: {summary['total_operations']}")
        if 'peak_memory_mb' in summary:
            logger.info(f"  Peak memory: {summary['peak_memory_mb']:.1f} MB")
            logger.info(f"  Average memory: {summary['avg_memory_mb']:.1f} MB")
            logger.info(f"  Memory per operation: {summary['peak_memory_mb'] / summary['total_operations'] * 1024:.3f} KB/op")
    
    @pytest.mark.asyncio
    async def test_sustained_load_stability(self, performance_config_manager, mock_high_performance_redis):
        """Test system stability under sustained load."""
        monitor = PerformanceMonitor()
        monitor.start()
        
        with patch('redis.Redis', return_value=mock_high_performance_redis):
            ingestion_service = MarketDataIngestionService(performance_config_manager)
            ingestion_service._redis_client = mock_high_performance_redis
            
            # Sustained load parameters
            test_duration = 10.0  # seconds
            target_rate = 1000   # operations per second
            symbols = ['SUST001', 'SUST002', 'SUST003', 'SUST004', 'SUST005']
            
            # Track stability metrics
            performance_windows = []
            window_size = 1.0  # 1 second windows
            
            start_time = time.time()
            last_window_time = start_time
            window_operations = 0
            
            while time.time() - start_time < test_duration:
                operation_start = time.time()
                
                try:
                    symbol = symbols[monitor.metrics['operations'] % len(symbols)]
                    
                    ohlcv = OHLCV(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        open=100.0 + monitor.metrics['operations'] * 0.001,
                        high=101.0 + monitor.metrics['operations'] * 0.001,
                        low=99.0 + monitor.metrics['operations'] * 0.001,
                        close=100.5 + monitor.metrics['operations'] * 0.001,
                        volume=1000
                    )
                    
                    await ingestion_service._publish_market_data_event(symbol, ohlcv, 'stability_test')
                    
                    latency = time.time() - operation_start
                    monitor.record_operation(latency)
                    window_operations += 1
                    
                    # Check window completion
                    if time.time() - last_window_time >= window_size:
                        window_throughput = window_operations / window_size
                        performance_windows.append(window_throughput)
                        
                        last_window_time = time.time()
                        window_operations = 0
                    
                    # Maintain target rate
                    target_interval = 1.0 / target_rate
                    operation_time = time.time() - operation_start
                    if operation_time < target_interval:
                        await asyncio.sleep(target_interval - operation_time)
                
                except Exception as e:
                    monitor.record_operation(error=True)
                    logger.warning(f"Operation failed during stability test: {e}")
        
        await monitor.stop()
        summary = monitor.get_summary()
        
        # Stability requirements
        assert summary['error_rate'] < 0.005, f"Error rate too high for stability: {summary['error_rate']:.3%}"
        
        # Performance consistency
        if len(performance_windows) > 2:
            throughput_std = statistics.stdev(performance_windows)
            throughput_mean = statistics.mean(performance_windows)
            throughput_cv = throughput_std / throughput_mean if throughput_mean > 0 else 1.0
            
            assert throughput_cv < 0.1, f"Throughput too variable: CV={throughput_cv:.3f}"
            
            # Minimum sustained throughput
            min_window_throughput = min(performance_windows)
            assert min_window_throughput > target_rate * 0.8, f"Throughput dropped too low: {min_window_throughput:.1f} ops/s"
        
        # Latency stability
        if len(monitor.metrics['latencies']) > 100:
            # Check for latency degradation over time
            early_latencies = monitor.metrics['latencies'][:len(monitor.metrics['latencies'])//3]
            late_latencies = monitor.metrics['latencies'][-len(monitor.metrics['latencies'])//3:]
            
            early_avg = statistics.mean(early_latencies)
            late_avg = statistics.mean(late_latencies)
            latency_degradation = (late_avg - early_avg) / early_avg if early_avg > 0 else 0
            
            assert latency_degradation < 0.5, f"Latency degraded too much over time: {latency_degradation:.2%}"
        
        logger.info(f"✓ Sustained load stability test passed:")
        logger.info(f"  Duration: {summary['duration_seconds']:.1f}s")
        logger.info(f"  Average throughput: {summary['operations_per_second']:.1f} ops/s")
        logger.info(f"  Error rate: {summary['error_rate']:.3%}")
        logger.info(f"  Throughput stability: {throughput_cv:.3f} CV" if 'throughput_cv' in locals() else "")
    
    @pytest.mark.asyncio
    async def test_concurrent_symbol_processing(self, performance_config_manager, mock_high_performance_redis):
        """Test performance with concurrent processing of multiple symbols."""
        monitor = PerformanceMonitor()
        monitor.start()
        
        with patch('redis.Redis', return_value=mock_high_performance_redis):
            ingestion_service = MarketDataIngestionService(performance_config_manager)
            ingestion_service._redis_client = mock_high_performance_redis
            
            # Test concurrent processing
            symbols = [f'CONC{i:02d}' for i in range(20)]
            operations_per_symbol = 100
            
            async def process_symbol_stream(symbol: str, operations: int):
                """Process a stream of operations for a single symbol."""
                symbol_start = time.time()
                symbol_operations = 0
                
                for i in range(operations):
                    try:
                        operation_start = time.time()
                        
                        ohlcv = OHLCV(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            open=100.0 + i * 0.1,
                            high=101.0 + i * 0.1,
                            low=99.0 + i * 0.1,
                            close=100.5 + i * 0.1,
                            volume=1000 + i
                        )
                        
                        await ingestion_service._publish_market_data_event(symbol, ohlcv, 'concurrent_test')
                        
                        latency = time.time() - operation_start
                        monitor.record_operation(latency)
                        symbol_operations += 1
                        
                        # Small delay to simulate realistic data arrival
                        await asyncio.sleep(0.01)
                        
                    except Exception as e:
                        monitor.record_operation(error=True)
                        logger.warning(f"Operation failed for {symbol}: {e}")
                
                symbol_duration = time.time() - symbol_start
                return symbol, symbol_operations, symbol_duration
            
            # Start concurrent processing for all symbols
            tasks = [
                process_symbol_stream(symbol, operations_per_symbol)
                for symbol in symbols
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
        await monitor.stop()
        summary = monitor.get_summary()
        
        # Concurrent processing requirements
        successful_symbols = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_symbols) == len(symbols), f"Some symbols failed: {len(successful_symbols)}/{len(symbols)}"
        
        # Performance should not degrade significantly with concurrency
        expected_total_ops = len(symbols) * operations_per_symbol
        completion_ratio = summary['total_operations'] / expected_total_ops
        assert completion_ratio > 0.95, f"Too many operations failed: {completion_ratio:.2%} completion"
        
        # Latency should remain reasonable
        if 'avg_latency_ms' in summary:
            assert summary['avg_latency_ms'] < 10.0, f"Latency too high with concurrency: {summary['avg_latency_ms']:.2f}ms"
        
        # Error rate should be low
        assert summary['error_rate'] < 0.02, f"Error rate too high with concurrency: {summary['error_rate']:.2%}"
        
        logger.info(f"✓ Concurrent symbol processing test passed:")
        logger.info(f"  Symbols processed: {len(successful_symbols)}")
        logger.info(f"  Total operations: {summary['total_operations']}")
        logger.info(f"  Completion ratio: {completion_ratio:.2%}")
        logger.info(f"  Average latency: {summary.get('avg_latency_ms', 'N/A')}")
        logger.info(f"  Error rate: {summary['error_rate']:.2%}")
    
    @pytest.mark.asyncio
    async def test_error_recovery_performance(self, performance_config_manager, mock_high_performance_redis):
        """Test performance during error conditions and recovery."""
        monitor = PerformanceMonitor()
        monitor.start()
        
        with patch('redis.Redis', return_value=mock_high_performance_redis):
            ingestion_service = MarketDataIngestionService(performance_config_manager)
            ingestion_service._redis_client = mock_high_performance_redis
            
            # Inject failures
            failure_rate = 0.1  # 10% failure rate
            failure_count = 0
            recovery_times = []
            
            # Override Redis method to simulate failures
            original_xadd = mock_high_performance_redis.xadd
            
            def failing_xadd(stream_name, data, **kwargs):
                nonlocal failure_count
                if monitor.metrics['operations'] % 10 == 0:  # Every 10th operation fails
                    failure_count += 1
                    raise Exception(f"Simulated Redis failure {failure_count}")
                return original_xadd(stream_name, data, **kwargs)
            
            mock_high_performance_redis.xadd = failing_xadd
            
            symbol = 'ERROR_TEST'
            num_operations = 500
            
            for i in range(num_operations):
                operation_start = time.time()
                
                try:
                    ohlcv = OHLCV(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        open=100.0 + i * 0.01,
                        high=101.0 + i * 0.01,
                        low=99.0 + i * 0.01,
                        close=100.5 + i * 0.01,
                        volume=1000
                    )
                    
                    await ingestion_service._publish_market_data_event(symbol, ohlcv, 'error_test')
                    
                    latency = time.time() - operation_start
                    monitor.record_operation(latency)
                    
                except Exception as e:
                    # Measure recovery time
                    recovery_start = time.time()
                    
                    # Simulate recovery logic
                    await asyncio.sleep(0.01)  # Recovery delay
                    
                    recovery_time = time.time() - recovery_start
                    recovery_times.append(recovery_time)
                    
                    latency = time.time() - operation_start
                    monitor.record_operation(latency, error=True)
                
                # Small operational delay
                await asyncio.sleep(0.002)
        
        await monitor.stop()
        summary = monitor.get_summary()
        
        # Error recovery requirements
        assert summary['error_rate'] > 0.05, "Should have encountered simulated errors"
        assert summary['error_rate'] < 0.15, f"Error rate too high: {summary['error_rate']:.2%}"
        
        # Recovery performance
        if recovery_times:
            avg_recovery_time = statistics.mean(recovery_times)
            max_recovery_time = max(recovery_times)
            
            assert avg_recovery_time < 0.1, f"Average recovery time too slow: {avg_recovery_time*1000:.1f}ms"
            assert max_recovery_time < 0.5, f"Maximum recovery time too slow: {max_recovery_time*1000:.1f}ms"
        
        # System should maintain reasonable performance despite errors
        if 'operations_per_second' in summary:
            min_throughput = 100  # ops/s even with errors
            assert summary['operations_per_second'] > min_throughput, f"Throughput too low with errors: {summary['operations_per_second']:.1f}"
        
        logger.info(f"✓ Error recovery performance test passed:")
        logger.info(f"  Total operations: {summary['total_operations']}")
        logger.info(f"  Error rate: {summary['error_rate']:.2%}")
        logger.info(f"  Recovery times: {len(recovery_times)}")
        if recovery_times:
            logger.info(f"  Average recovery: {statistics.mean(recovery_times)*1000:.1f}ms")
            logger.info(f"  Max recovery: {max(recovery_times)*1000:.1f}ms")
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_efficiency(self, performance_config_manager, mock_high_performance_redis):
        """Test efficiency of resource cleanup and garbage collection."""
        # Force initial garbage collection
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        with patch('redis.Redis', return_value=mock_high_performance_redis):
            # Create and destroy multiple ingestion services
            for cycle in range(5):
                ingestion_service = MarketDataIngestionService(performance_config_manager)
                ingestion_service._redis_client = mock_high_performance_redis
                
                # Perform operations
                for i in range(100):
                    ohlcv = OHLCV(
                        timestamp=datetime.now(),
                        symbol=f'CLEANUP_{cycle}',
                        open=100.0,
                        high=101.0,
                        low=99.0,
                        close=100.5,
                        volume=1000
                    )
                    
                    await ingestion_service._publish_market_data_event(f'CLEANUP_{cycle}', ohlcv, 'cleanup_test')
                
                # Cleanup
                ingestion_service.cleanup()
                del ingestion_service
                
                # Force garbage collection
                gc.collect()
        
        # Final garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Resource cleanup requirements
        object_growth = final_objects - initial_objects
        max_acceptable_growth = 1000  # Maximum number of objects that can remain
        
        assert object_growth < max_acceptable_growth, f"Too many objects retained: {object_growth}"
        
        # Check for potential memory leaks
        growth_ratio = object_growth / initial_objects if initial_objects > 0 else 0
        assert growth_ratio < 0.1, f"Object growth ratio too high: {growth_ratio:.2%}"
        
        logger.info(f"✓ Resource cleanup efficiency test passed:")
        logger.info(f"  Initial objects: {initial_objects}")
        logger.info(f"  Final objects: {final_objects}")
        logger.info(f"  Object growth: {object_growth}")
        logger.info(f"  Growth ratio: {growth_ratio:.2%}")
    
    @pytest.mark.asyncio
    async def test_performance_regression_detection(self, performance_config_manager, mock_high_performance_redis):
        """Test for performance regression detection."""
        # Define performance baselines (these would typically come from previous test runs)
        baselines = {
            'max_avg_latency_ms': 2.0,
            'min_throughput_ops_per_sec': 1000,
            'max_error_rate': 0.01,
            'max_memory_per_op_kb': 0.5,
            'max_p95_latency_ms': 10.0
        }
        
        monitor = PerformanceMonitor()
        monitor.start()
        
        with patch('redis.Redis', return_value=mock_high_performance_redis):
            ingestion_service = MarketDataIngestionService(performance_config_manager)
            ingestion_service._redis_client = mock_high_performance_redis
            
            # Run performance test
            num_operations = 2000
            symbols = ['REGR001', 'REGR002', 'REGR003']
            
            for i in range(num_operations):
                start_time = time.time()
                
                try:
                    symbol = symbols[i % len(symbols)]
                    
                    ohlcv = OHLCV(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        open=100.0 + i * 0.001,
                        high=101.0 + i * 0.001,
                        low=99.0 + i * 0.001,
                        close=100.5 + i * 0.001,
                        volume=1000
                    )
                    
                    await ingestion_service._publish_market_data_event(symbol, ohlcv, 'regression_test')
                    
                    latency = time.time() - start_time
                    monitor.record_operation(latency)
                    
                except Exception as e:
                    monitor.record_operation(error=True)
                
                # Throttle to maintain consistent load
                if i % 100 == 0:
                    await asyncio.sleep(0.01)
        
        await monitor.stop()
        summary = monitor.get_summary()
        
        # Check against baselines
        regressions = []
        
        if 'avg_latency_ms' in summary and summary['avg_latency_ms'] > baselines['max_avg_latency_ms']:
            regressions.append(f"Average latency: {summary['avg_latency_ms']:.2f}ms > {baselines['max_avg_latency_ms']}ms")
        
        if summary['operations_per_second'] < baselines['min_throughput_ops_per_sec']:
            regressions.append(f"Throughput: {summary['operations_per_second']:.1f} < {baselines['min_throughput_ops_per_sec']} ops/s")
        
        if summary['error_rate'] > baselines['max_error_rate']:
            regressions.append(f"Error rate: {summary['error_rate']:.3f} > {baselines['max_error_rate']}")
        
        if 'p95_latency_ms' in summary and summary['p95_latency_ms'] > baselines['max_p95_latency_ms']:
            regressions.append(f"P95 latency: {summary['p95_latency_ms']:.2f}ms > {baselines['max_p95_latency_ms']}ms")
        
        # Memory regression check
        if 'peak_memory_mb' in summary:
            memory_per_op = summary['peak_memory_mb'] / summary['total_operations'] * 1024  # KB
            if memory_per_op > baselines['max_memory_per_op_kb']:
                regressions.append(f"Memory per operation: {memory_per_op:.3f}KB > {baselines['max_memory_per_op_kb']}KB")
        
        # Assert no regressions
        assert len(regressions) == 0, f"Performance regressions detected: {'; '.join(regressions)}"
        
        logger.info(f"✓ Performance regression test passed:")
        logger.info(f"  All metrics within baseline thresholds")
        logger.info(f"  Operations: {summary['total_operations']}")
        logger.info(f"  Throughput: {summary['operations_per_second']:.1f} ops/s")
        if 'avg_latency_ms' in summary:
            logger.info(f"  Average latency: {summary['avg_latency_ms']:.2f}ms")
        logger.info(f"  Error rate: {summary['error_rate']:.3%}")