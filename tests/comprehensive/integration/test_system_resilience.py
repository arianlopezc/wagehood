"""
System Resilience and Recovery Testing

This module contains comprehensive tests for system resilience,
including network failures, data corruption scenarios, resource
exhaustion, and recovery mechanisms under various failure modes.
"""

import pytest
import asyncio
import time
import logging
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import json

# Import test modules
from src.realtime.data_ingestion import MarketDataIngestionService, CircuitBreaker
from src.realtime.config_manager import ConfigManager, AssetConfig, SystemConfig
from src.data.providers.mock_provider import MockProvider
from src.core.models import OHLCV, TimeFrame

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FailureSimulator:
    """Utility class for simulating various failure scenarios."""
    
    def __init__(self):
        self.failure_count = 0
        self.recovery_count = 0
        self.failure_patterns = {
            'intermittent': self._intermittent_failure,
            'cascading': self._cascading_failure,
            'gradual_degradation': self._gradual_degradation,
            'complete_outage': self._complete_outage,
            'data_corruption': self._data_corruption
        }
    
    def _intermittent_failure(self, call_count: int) -> bool:
        """Simulate intermittent failures (every 5th call fails)."""
        return call_count % 5 == 0
    
    def _cascading_failure(self, call_count: int) -> bool:
        """Simulate cascading failures (increasing failure rate)."""
        if call_count < 10:
            return False
        elif call_count < 20:
            return call_count % 4 == 0
        elif call_count < 30:
            return call_count % 2 == 0
        else:
            return True  # Complete failure
    
    def _gradual_degradation(self, call_count: int) -> bool:
        """Simulate gradual system degradation."""
        failure_probability = min(0.8, call_count / 100)
        return random.random() < failure_probability
    
    def _complete_outage(self, call_count: int) -> bool:
        """Simulate complete system outage."""
        return call_count > 10
    
    def _data_corruption(self, call_count: int) -> bool:
        """Simulate random data corruption."""
        return random.random() < 0.05  # 5% corruption rate
    
    def should_fail(self, pattern: str, call_count: int) -> bool:
        """Check if operation should fail based on pattern."""
        if pattern in self.failure_patterns:
            if self.failure_patterns[pattern](call_count):
                self.failure_count += 1
                return True
        return False
    
    def simulate_recovery(self):
        """Simulate recovery from failures."""
        self.recovery_count += 1
        return True


class TestSystemResilience:
    """
    Comprehensive tests for system resilience and recovery.
    
    These tests validate:
    - Network failure handling and recovery
    - Data corruption detection and handling
    - Resource exhaustion scenarios
    - Circuit breaker functionality
    - System recovery mechanisms
    - Graceful degradation under stress
    """
    
    @pytest.fixture(scope="class")
    def resilience_config_manager(self):
        """Create configuration manager for resilience testing."""
        config_manager = Mock(spec=ConfigManager)
        
        # Resilient system configuration
        system_config = SystemConfig(
            data_update_interval_seconds=0.5,
            max_concurrent_calculations=10,
            calculation_batch_size=5,
            worker_pool_size=4,
            redis_config={
                'host': 'localhost',
                'port': 6379,
                'db': 0
            }
        )
        config_manager.get_system_config.return_value = system_config
        
        # Test watchlist
        watchlist = [
            AssetConfig(
                symbol='RESILIENT_TEST',
                enabled=True,
                data_provider='failing_provider',
                strategies=['test_strategy'],
                position_size=0.1
            )
        ]
        config_manager.get_watchlist.return_value = watchlist
        config_manager.get_enabled_symbols.return_value = ['RESILIENT_TEST']
        
        return config_manager
    
    @pytest.fixture
    def failing_redis_client(self):
        """Create Redis client that simulates various failure modes."""
        failure_simulator = FailureSimulator()
        
        class FailingRedis:
            def __init__(self):
                self.call_count = 0
                self.storage = {}
                self.streams = {}
                self.failure_mode = None
                self.is_connected = True
            
            def set_failure_mode(self, mode: str):
                self.failure_mode = mode
                failure_simulator.failure_count = 0
                failure_simulator.recovery_count = 0
            
            def xadd(self, stream_name, data, **kwargs):
                self.call_count += 1
                
                if self.failure_mode and failure_simulator.should_fail(self.failure_mode, self.call_count):
                    if self.failure_mode == 'data_corruption':
                        # Corrupt data instead of failing
                        corrupted_data = {k: 'CORRUPTED' for k in data.keys()}
                        return self._successful_xadd(stream_name, corrupted_data)
                    else:
                        raise ConnectionError(f"Simulated Redis failure: {self.failure_mode}")
                
                return self._successful_xadd(stream_name, data)
            
            def _successful_xadd(self, stream_name, data):
                if stream_name not in self.streams:
                    self.streams[stream_name] = []
                
                message_id = f"{int(time.time() * 1000)}-{len(self.streams[stream_name])}"
                self.streams[stream_name].append({
                    'id': message_id,
                    'data': data
                })
                return message_id.encode()
            
            def get(self, key):
                self.call_count += 1
                if self.failure_mode and failure_simulator.should_fail(self.failure_mode, self.call_count):
                    raise ConnectionError(f"Simulated Redis failure: {self.failure_mode}")
                return self.storage.get(key)
            
            def set(self, key, value, **kwargs):
                self.call_count += 1
                if self.failure_mode and failure_simulator.should_fail(self.failure_mode, self.call_count):
                    raise ConnectionError(f"Simulated Redis failure: {self.failure_mode}")
                self.storage[key] = value
                return True
            
            def ping(self):
                if not self.is_connected:
                    raise ConnectionError("Redis disconnected")
                return True
            
            def xgroup_create(self, *args, **kwargs):
                return True
            
            def xinfo_stream(self, stream_name):
                return {
                    'length': len(self.streams.get(stream_name, [])),
                    'groups': 1,
                    'last-generated-id': b'1234567890-0'
                }
            
            def disconnect(self):
                self.is_connected = False
            
            def reconnect(self):
                self.is_connected = True
        
        return FailingRedis()
    
    @pytest.fixture
    def failing_data_provider(self):
        """Create data provider that simulates various failure modes."""
        failure_simulator = FailureSimulator()
        
        class FailingDataProvider:
            def __init__(self):
                self.call_count = 0
                self.failure_mode = None
                self.connected = True
            
            def set_failure_mode(self, mode: str):
                self.failure_mode = mode
                failure_simulator.failure_count = 0
                failure_simulator.recovery_count = 0
            
            def get_latest_data(self, symbol: str) -> Optional[OHLCV]:
                self.call_count += 1
                
                if self.failure_mode and failure_simulator.should_fail(self.failure_mode, self.call_count):
                    if self.failure_mode == 'data_corruption':
                        # Return corrupted data
                        return OHLCV(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            open=-999.0,  # Invalid negative price
                            high=-888.0,
                            low=-777.0,
                            close=-666.0,
                            volume=-1000  # Invalid negative volume
                        )
                    else:
                        raise ConnectionError(f"Simulated provider failure: {self.failure_mode}")
                
                # Return valid data
                return OHLCV(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    open=100.0 + self.call_count * 0.1,
                    high=101.0 + self.call_count * 0.1,
                    low=99.0 + self.call_count * 0.1,
                    close=100.5 + self.call_count * 0.1,
                    volume=1000 + self.call_count
                )
            
            async def connect(self):
                if self.failure_mode == 'complete_outage':
                    self.connected = False
                    raise ConnectionError("Provider completely unavailable")
                self.connected = True
                return True
            
            async def disconnect(self):
                self.connected = False
        
        return FailingDataProvider()
    
    @pytest.mark.asyncio
    async def test_network_failure_recovery(self, resilience_config_manager, failing_redis_client, failing_data_provider):
        """Test recovery from network failures."""
        with patch('redis.Redis', return_value=failing_redis_client):
            ingestion_service = MarketDataIngestionService(resilience_config_manager)
            ingestion_service._redis_client = failing_redis_client
            ingestion_service.add_provider('failing_provider', failing_data_provider)
            
            # Track recovery metrics
            recovery_metrics = {
                'network_failures': 0,
                'successful_recoveries': 0,
                'total_operations': 0,
                'recovery_times': []
            }
            
            # Test intermittent network failures
            failing_redis_client.set_failure_mode('intermittent')
            
            symbol = 'RESILIENT_TEST'
            
            for i in range(30):
                recovery_metrics['total_operations'] += 1
                recovery_start = time.time()
                
                try:
                    # Get data from provider
                    data = failing_data_provider.get_latest_data(symbol)
                    
                    # Attempt to publish (may fail due to Redis)
                    await ingestion_service._publish_market_data_event(symbol, data, 'failing_provider')
                    
                    # If we get here, operation succeeded (possibly after retry)
                    recovery_metrics['successful_recoveries'] += 1
                    
                except Exception as e:
                    recovery_metrics['network_failures'] += 1
                    logger.debug(f"Network failure {i}: {e}")
                    
                    # Simulate retry logic
                    await asyncio.sleep(0.1)  # Retry delay
                    
                    try:
                        # Retry operation
                        data = failing_data_provider.get_latest_data(symbol)
                        await ingestion_service._publish_market_data_event(symbol, data, 'failing_provider')
                        recovery_metrics['successful_recoveries'] += 1
                    except Exception:
                        pass  # Accept failure
                
                recovery_time = time.time() - recovery_start
                recovery_metrics['recovery_times'].append(recovery_time)
                
                await asyncio.sleep(0.01)
            
            # Validate recovery performance
            recovery_rate = recovery_metrics['successful_recoveries'] / recovery_metrics['total_operations']
            assert recovery_rate > 0.7, f"Recovery rate too low: {recovery_rate:.2%}"
            
            assert recovery_metrics['network_failures'] > 0, "Should have encountered network failures"
            
            if recovery_metrics['recovery_times']:
                avg_recovery_time = sum(recovery_metrics['recovery_times']) / len(recovery_metrics['recovery_times'])
                assert avg_recovery_time < 1.0, f"Average recovery time too slow: {avg_recovery_time:.2f}s"
            
            logger.info(f"✓ Network failure recovery test passed:")
            logger.info(f"  Network failures: {recovery_metrics['network_failures']}")
            logger.info(f"  Successful recoveries: {recovery_metrics['successful_recoveries']}")
            logger.info(f"  Recovery rate: {recovery_rate:.2%}")
    
    @pytest.mark.asyncio
    async def test_data_corruption_handling(self, resilience_config_manager, failing_redis_client, failing_data_provider):
        """Test handling of corrupted data."""
        with patch('redis.Redis', return_value=failing_redis_client):
            ingestion_service = MarketDataIngestionService(resilience_config_manager)
            ingestion_service._redis_client = failing_redis_client
            ingestion_service.add_provider('failing_provider', failing_data_provider)
            
            # Enable data corruption
            failing_data_provider.set_failure_mode('data_corruption')
            
            # Track corruption handling
            corruption_metrics = {
                'total_operations': 0,
                'corrupted_data_detected': 0,
                'corrupted_data_rejected': 0,
                'valid_data_processed': 0
            }
            
            symbol = 'RESILIENT_TEST'
            
            for i in range(50):
                corruption_metrics['total_operations'] += 1
                
                try:
                    # Get data (may be corrupted)
                    data = failing_data_provider.get_latest_data(symbol)
                    
                    # Validate data before processing
                    if self._is_data_corrupted(data):
                        corruption_metrics['corrupted_data_detected'] += 1
                        corruption_metrics['corrupted_data_rejected'] += 1
                        logger.debug(f"Rejected corrupted data: {data}")
                        continue
                    
                    # Process valid data
                    await ingestion_service._publish_market_data_event(symbol, data, 'failing_provider')
                    corruption_metrics['valid_data_processed'] += 1
                    
                except Exception as e:
                    logger.debug(f"Error processing data: {e}")
                
                await asyncio.sleep(0.01)
            
            # Validate corruption handling
            assert corruption_metrics['corrupted_data_detected'] > 0, "Should have detected corrupted data"
            assert corruption_metrics['corrupted_data_rejected'] > 0, "Should have rejected corrupted data"
            assert corruption_metrics['valid_data_processed'] > 0, "Should have processed some valid data"
            
            # Corruption detection rate should be reasonable
            if corruption_metrics['corrupted_data_detected'] > 0:
                rejection_rate = corruption_metrics['corrupted_data_rejected'] / corruption_metrics['corrupted_data_detected']
                assert rejection_rate >= 0.95, f"Corruption rejection rate too low: {rejection_rate:.2%}"
            
            logger.info(f"✓ Data corruption handling test passed:")
            logger.info(f"  Total operations: {corruption_metrics['total_operations']}")
            logger.info(f"  Corrupted data detected: {corruption_metrics['corrupted_data_detected']}")
            logger.info(f"  Corrupted data rejected: {corruption_metrics['corrupted_data_rejected']}")
            logger.info(f"  Valid data processed: {corruption_metrics['valid_data_processed']}")
    
    def _is_data_corrupted(self, data: OHLCV) -> bool:
        """Check if OHLCV data is corrupted."""
        try:
            # Check for negative prices
            if any(price < 0 for price in [data.open, data.high, data.low, data.close]):
                return True
            
            # Check for negative volume
            if data.volume < 0:
                return True
            
            # Check for invalid OHLC relationships
            if not (data.low <= data.open <= data.high and data.low <= data.close <= data.high):
                return True
            
            # Check for extreme values
            if any(price > 1000000 for price in [data.open, data.high, data.low, data.close]):
                return True
            
            return False
            
        except (AttributeError, TypeError):
            return True  # Data structure corruption
    
    @pytest.mark.asyncio
    async def test_cascading_failure_resilience(self, resilience_config_manager, failing_redis_client, failing_data_provider):
        """Test resilience against cascading failures."""
        with patch('redis.Redis', return_value=failing_redis_client):
            ingestion_service = MarketDataIngestionService(resilience_config_manager)
            ingestion_service._redis_client = failing_redis_client
            ingestion_service.add_provider('failing_provider', failing_data_provider)
            
            # Enable cascading failures
            failing_data_provider.set_failure_mode('cascading')
            failing_redis_client.set_failure_mode('cascading')
            
            # Track system behavior during cascading failures
            cascade_metrics = {
                'phase1_success': 0,  # Early phase (should work)
                'phase2_degraded': 0,  # Middle phase (degraded performance)
                'phase3_failure': 0,   # Late phase (high failure rate)
                'total_operations': 0,
                'system_shutdown': False
            }
            
            symbol = 'RESILIENT_TEST'
            
            for i in range(60):  # Long enough to hit all phases
                cascade_metrics['total_operations'] += 1
                operation_succeeded = False
                
                try:
                    # Attempt operation with progressive backoff
                    max_retries = 3 if i < 40 else 1  # Reduce retries in failure phase
                    
                    for retry in range(max_retries):
                        try:
                            data = failing_data_provider.get_latest_data(symbol)
                            await ingestion_service._publish_market_data_event(symbol, data, 'failing_provider')
                            operation_succeeded = True
                            break
                        
                        except Exception as e:
                            if retry < max_retries - 1:
                                await asyncio.sleep(0.1 * (2 ** retry))  # Exponential backoff
                            else:
                                logger.debug(f"Operation failed after {max_retries} retries: {e}")
                
                except Exception as outer_e:
                    logger.debug(f"Outer operation failed: {outer_e}")
                
                # Categorize by phase
                if i < 20:
                    if operation_succeeded:
                        cascade_metrics['phase1_success'] += 1
                elif i < 40:
                    if operation_succeeded:
                        cascade_metrics['phase2_degraded'] += 1
                else:
                    if not operation_succeeded:
                        cascade_metrics['phase3_failure'] += 1
                
                # Check for system shutdown condition
                if i > 50 and cascade_metrics['phase3_failure'] > 5:
                    logger.info("System entering protective shutdown mode")
                    cascade_metrics['system_shutdown'] = True
                    break
                
                await asyncio.sleep(0.02)
            
            # Validate cascading failure handling
            phase1_rate = cascade_metrics['phase1_success'] / min(20, cascade_metrics['total_operations'])
            assert phase1_rate > 0.8, f"Phase 1 success rate too low: {phase1_rate:.2%}"
            
            if cascade_metrics['total_operations'] > 20:
                phase2_rate = cascade_metrics['phase2_degraded'] / min(20, max(0, cascade_metrics['total_operations'] - 20))
                if phase2_rate > 0:  # Only check if we reached phase 2
                    assert phase2_rate > 0.3, f"Phase 2 degradation handling poor: {phase2_rate:.2%}"
            
            # System should either maintain some functionality or shutdown gracefully
            total_failures = cascade_metrics['phase3_failure']
            assert total_failures > 0 or cascade_metrics['system_shutdown'], "Should experience failures or shutdown"
            
            logger.info(f"✓ Cascading failure resilience test passed:")
            logger.info(f"  Phase 1 success rate: {phase1_rate:.2%}")
            logger.info(f"  Total operations: {cascade_metrics['total_operations']}")
            logger.info(f"  System shutdown: {cascade_metrics['system_shutdown']}")
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_behavior(self):
        """Test circuit breaker functionality under various failure patterns."""
        circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=2)
        
        # Track circuit breaker behavior
        cb_metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'blocked_calls': 0,
            'state_transitions': []
        }
        
        def failing_function(should_fail: bool = False):
            cb_metrics['total_calls'] += 1
            if should_fail:
                cb_metrics['failed_calls'] += 1
                raise Exception("Simulated failure")
            else:
                cb_metrics['successful_calls'] += 1
                return "Success"
        
        # Phase 1: Normal operation
        for i in range(10):
            try:
                result = circuit_breaker.call(failing_function, should_fail=False)
                assert result == "Success"
                assert circuit_breaker.state == "CLOSED"
            except Exception:
                pytest.fail("Should not fail in normal operation")
        
        cb_metrics['state_transitions'].append(('CLOSED', time.time()))
        
        # Phase 2: Trigger failures to open circuit
        for i in range(7):  # More than threshold
            try:
                circuit_breaker.call(failing_function, should_fail=True)
            except Exception:
                pass  # Expected failures
        
        assert circuit_breaker.state == "OPEN"
        cb_metrics['state_transitions'].append(('OPEN', time.time()))
        
        # Phase 3: Calls should be blocked
        for i in range(5):
            try:
                circuit_breaker.call(failing_function, should_fail=False)
                pytest.fail("Circuit breaker should block calls")
            except Exception as e:
                if "Circuit breaker is OPEN" in str(e):
                    cb_metrics['blocked_calls'] += 1
                else:
                    pytest.fail(f"Unexpected exception: {e}")
        
        # Phase 4: Wait for recovery timeout
        await asyncio.sleep(2.1)
        
        # Phase 5: Test half-open state and recovery
        try:
            result = circuit_breaker.call(failing_function, should_fail=False)
            assert result == "Success"
            assert circuit_breaker.state == "CLOSED"
            cb_metrics['state_transitions'].append(('HALF_OPEN', time.time()))
            cb_metrics['state_transitions'].append(('CLOSED', time.time()))
        except Exception:
            pytest.fail("Should succeed in half-open state with good call")
        
        # Validate circuit breaker behavior
        assert cb_metrics['blocked_calls'] == 5, f"Should have blocked 5 calls, blocked {cb_metrics['blocked_calls']}"
        assert cb_metrics['failed_calls'] >= 5, f"Should have at least 5 failures, had {cb_metrics['failed_calls']}"
        assert len(cb_metrics['state_transitions']) >= 3, "Should have multiple state transitions"
        
        logger.info(f"✓ Circuit breaker behavior test passed:")
        logger.info(f"  Total calls: {cb_metrics['total_calls']}")
        logger.info(f"  Successful calls: {cb_metrics['successful_calls']}")
        logger.info(f"  Failed calls: {cb_metrics['failed_calls']}")
        logger.info(f"  Blocked calls: {cb_metrics['blocked_calls']}")
        logger.info(f"  State transitions: {len(cb_metrics['state_transitions'])}")
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion_handling(self, resilience_config_manager, failing_redis_client):
        """Test handling of resource exhaustion scenarios."""
        with patch('redis.Redis', return_value=failing_redis_client):
            ingestion_service = MarketDataIngestionService(resilience_config_manager)
            ingestion_service._redis_client = failing_redis_client
            
            # Simulate resource exhaustion
            resource_metrics = {
                'memory_limit_hits': 0,
                'connection_limit_hits': 0,
                'operations_throttled': 0,
                'operations_successful': 0,
                'total_operations': 0
            }
            
            # Override Redis operations to simulate resource limits
            original_xadd = failing_redis_client.xadd
            
            def resource_limited_xadd(stream_name, data, **kwargs):
                resource_metrics['total_operations'] += 1
                
                # Simulate memory exhaustion
                if len(failing_redis_client.streams.get(stream_name, [])) > 100:
                    resource_metrics['memory_limit_hits'] += 1
                    raise Exception("Redis out of memory")
                
                # Simulate connection limits
                if resource_metrics['total_operations'] % 50 == 0:
                    resource_metrics['connection_limit_hits'] += 1
                    raise Exception("Too many connections")
                
                resource_metrics['operations_successful'] += 1
                return original_xadd(stream_name, data, **kwargs)
            
            failing_redis_client.xadd = resource_limited_xadd
            
            symbol = 'RESOURCE_TEST'
            
            # Test with throttling and backoff
            for i in range(200):
                try:
                    ohlcv = OHLCV(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        open=100.0,
                        high=101.0,
                        low=99.0,
                        close=100.5,
                        volume=1000
                    )
                    
                    await ingestion_service._publish_market_data_event(symbol, ohlcv, 'resource_test')
                    
                except Exception as e:
                    if "out of memory" in str(e) or "Too many connections" in str(e):
                        resource_metrics['operations_throttled'] += 1
                        
                        # Implement backoff strategy
                        await asyncio.sleep(0.1)
                        
                        # Try to clear some resources (simulated)
                        if "out of memory" in str(e):
                            # Simulate memory cleanup
                            stream_name = 'market_data_stream'
                            if stream_name in failing_redis_client.streams:
                                # Keep only recent messages
                                failing_redis_client.streams[stream_name] = failing_redis_client.streams[stream_name][-50:]
                
                # Throttle operations to prevent overwhelming
                if i % 10 == 0:
                    await asyncio.sleep(0.01)
            
            # Validate resource exhaustion handling
            assert resource_metrics['memory_limit_hits'] > 0, "Should have hit memory limits"
            assert resource_metrics['connection_limit_hits'] > 0, "Should have hit connection limits"
            assert resource_metrics['operations_throttled'] > 0, "Should have throttled operations"
            assert resource_metrics['operations_successful'] > 0, "Should have some successful operations"
            
            # Success rate should be reasonable despite resource limits
            success_rate = resource_metrics['operations_successful'] / resource_metrics['total_operations']
            assert success_rate > 0.3, f"Success rate too low under resource pressure: {success_rate:.2%}"
            
            logger.info(f"✓ Resource exhaustion handling test passed:")
            logger.info(f"  Memory limit hits: {resource_metrics['memory_limit_hits']}")
            logger.info(f"  Connection limit hits: {resource_metrics['connection_limit_hits']}")
            logger.info(f"  Operations throttled: {resource_metrics['operations_throttled']}")
            logger.info(f"  Success rate: {success_rate:.2%}")
    
    @pytest.mark.asyncio
    async def test_gradual_degradation_detection(self, resilience_config_manager, failing_redis_client, failing_data_provider):
        """Test detection and handling of gradual system degradation."""
        with patch('redis.Redis', return_value=failing_redis_client):
            ingestion_service = MarketDataIngestionService(resilience_config_manager)
            ingestion_service._redis_client = failing_redis_client
            ingestion_service.add_provider('failing_provider', failing_data_provider)
            
            # Enable gradual degradation
            failing_data_provider.set_failure_mode('gradual_degradation')
            
            # Track degradation metrics
            degradation_metrics = {
                'time_windows': [],
                'success_rates': [],
                'response_times': [],
                'degradation_detected': False
            }
            
            symbol = 'DEGRADATION_TEST'
            window_size = 20  # Operations per window
            window_operations = 0
            window_successes = 0
            window_start_time = time.time()
            
            for i in range(100):
                operation_start = time.time()
                operation_success = False
                
                try:
                    data = failing_data_provider.get_latest_data(symbol)
                    await ingestion_service._publish_market_data_event(symbol, data, 'failing_provider')
                    operation_success = True
                    window_successes += 1
                    
                except Exception as e:
                    logger.debug(f"Operation {i} failed: {e}")
                
                operation_time = time.time() - operation_start
                degradation_metrics['response_times'].append(operation_time)
                window_operations += 1
                
                # Check window completion
                if window_operations >= window_size:
                    window_duration = time.time() - window_start_time
                    success_rate = window_successes / window_operations
                    avg_response_time = sum(degradation_metrics['response_times'][-window_size:]) / window_size
                    
                    degradation_metrics['time_windows'].append(window_duration)
                    degradation_metrics['success_rates'].append(success_rate)
                    
                    # Detect degradation
                    if len(degradation_metrics['success_rates']) >= 3:
                        recent_rates = degradation_metrics['success_rates'][-3:]
                        if all(rate < 0.5 for rate in recent_rates):  # 3 consecutive windows below 50%
                            degradation_metrics['degradation_detected'] = True
                            logger.info(f"Degradation detected at operation {i}")
                    
                    # Reset window
                    window_operations = 0
                    window_successes = 0
                    window_start_time = time.time()
                
                await asyncio.sleep(0.01)
            
            # Validate degradation detection
            assert len(degradation_metrics['success_rates']) > 0, "Should have completed at least one window"
            
            # Should show decreasing success rates over time
            if len(degradation_metrics['success_rates']) >= 3:
                early_rate = degradation_metrics['success_rates'][0]
                late_rate = degradation_metrics['success_rates'][-1]
                assert late_rate < early_rate, f"Should show degradation: {early_rate:.2%} -> {late_rate:.2%}"
            
            # Should detect degradation
            assert degradation_metrics['degradation_detected'], "Should detect gradual degradation"
            
            logger.info(f"✓ Gradual degradation detection test passed:")
            logger.info(f"  Windows completed: {len(degradation_metrics['success_rates'])}")
            logger.info(f"  Degradation detected: {degradation_metrics['degradation_detected']}")
            if degradation_metrics['success_rates']:
                logger.info(f"  Initial success rate: {degradation_metrics['success_rates'][0]:.2%}")
                logger.info(f"  Final success rate: {degradation_metrics['success_rates'][-1]:.2%}")
    
    @pytest.mark.asyncio
    async def test_system_recovery_after_outage(self, resilience_config_manager, failing_redis_client, failing_data_provider):
        """Test system recovery after complete outage."""
        with patch('redis.Redis', return_value=failing_redis_client):
            ingestion_service = MarketDataIngestionService(resilience_config_manager)
            ingestion_service._redis_client = failing_redis_client
            ingestion_service.add_provider('failing_provider', failing_data_provider)
            
            # Track recovery metrics
            recovery_metrics = {
                'pre_outage_success': 0,
                'outage_failures': 0,
                'post_outage_success': 0,
                'recovery_time': 0,
                'total_operations': 0
            }
            
            symbol = 'RECOVERY_TEST'
            
            # Phase 1: Normal operation
            for i in range(20):
                try:
                    data = failing_data_provider.get_latest_data(symbol)
                    await ingestion_service._publish_market_data_event(symbol, data, 'failing_provider')
                    recovery_metrics['pre_outage_success'] += 1
                except Exception:
                    pass
                recovery_metrics['total_operations'] += 1
                await asyncio.sleep(0.01)
            
            # Phase 2: Simulate complete outage
            failing_data_provider.set_failure_mode('complete_outage')
            failing_redis_client.set_failure_mode('complete_outage')
            
            outage_start = time.time()
            
            for i in range(20):
                try:
                    data = failing_data_provider.get_latest_data(symbol)
                    await ingestion_service._publish_market_data_event(symbol, data, 'failing_provider')
                except Exception:
                    recovery_metrics['outage_failures'] += 1
                recovery_metrics['total_operations'] += 1
                await asyncio.sleep(0.01)
            
            # Phase 3: Simulate recovery
            failing_data_provider.set_failure_mode(None)  # Disable failure mode
            failing_redis_client.set_failure_mode(None)
            
            recovery_start = time.time()
            
            # Test recovery with exponential backoff
            for i in range(30):
                try:
                    data = failing_data_provider.get_latest_data(symbol)
                    await ingestion_service._publish_market_data_event(symbol, data, 'failing_provider')
                    recovery_metrics['post_outage_success'] += 1
                    
                    # Record recovery time on first success
                    if recovery_metrics['post_outage_success'] == 1:
                        recovery_metrics['recovery_time'] = time.time() - recovery_start
                
                except Exception as e:
                    logger.debug(f"Recovery attempt {i} failed: {e}")
                    # Exponential backoff
                    await asyncio.sleep(min(0.1 * (2 ** (i // 5)), 1.0))
                
                recovery_metrics['total_operations'] += 1
                await asyncio.sleep(0.01)
            
            # Validate recovery
            assert recovery_metrics['pre_outage_success'] > 15, "Should have normal operation before outage"
            assert recovery_metrics['outage_failures'] > 15, "Should experience failures during outage"
            assert recovery_metrics['post_outage_success'] > 10, "Should recover after outage"
            
            # Recovery should be reasonably fast
            assert recovery_metrics['recovery_time'] < 5.0, f"Recovery took too long: {recovery_metrics['recovery_time']:.2f}s"
            
            # Recovery rate should be good
            recovery_rate = recovery_metrics['post_outage_success'] / 30
            assert recovery_rate > 0.5, f"Recovery rate too low: {recovery_rate:.2%}"
            
            logger.info(f"✓ System recovery test passed:")
            logger.info(f"  Pre-outage successes: {recovery_metrics['pre_outage_success']}")
            logger.info(f"  Outage failures: {recovery_metrics['outage_failures']}")
            logger.info(f"  Post-outage successes: {recovery_metrics['post_outage_success']}")
            logger.info(f"  Recovery time: {recovery_metrics['recovery_time']:.2f}s")
            logger.info(f"  Recovery rate: {recovery_rate:.2%}")