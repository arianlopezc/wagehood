"""
Real-time Data Streaming Integration Tests

This module contains comprehensive tests for real-time data streaming
functionality, including WebSocket connections, data flow validation,
and streaming performance under various conditions.
"""

import pytest
import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
from unittest.mock import Mock, patch, AsyncMock
from collections import defaultdict
import json

# Import test modules
from src.data.providers.alpaca_provider import AlpacaProvider
from src.realtime.data_ingestion import MarketDataIngestionService, CircuitBreaker
from src.realtime.config_manager import ConfigManager, AssetConfig, SystemConfig
from src.core.models import OHLCV, TimeFrame

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRealtimeStreaming:
    """
    Comprehensive tests for real-time data streaming.
    
    These tests validate:
    - Real-time WebSocket streaming connections
    - Data flow and processing pipelines
    - Stream reliability and error recovery
    - Performance under load
    - Data quality in streaming mode
    """
    
    @pytest.fixture(scope="class")
    def mock_config_manager(self):
        """Create mock configuration manager."""
        config_manager = Mock(spec=ConfigManager)
        
        # Mock system configuration
        system_config = SystemConfig(
            data_update_interval_seconds=1,
            max_concurrent_calculations=10,
            calculation_batch_size=5,
            worker_pool_size=2,
            redis_config={
                'host': 'localhost',
                'port': 6379,
                'db': 0
            }
        )
        config_manager.get_system_config.return_value = system_config
        
        # Mock watchlist
        watchlist = [
            AssetConfig(
                symbol='AAPL',
                enabled=True,
                data_provider='mock',
                strategies=['ma_crossover'],
                position_size=0.1
            ),
            AssetConfig(
                symbol='MSFT',
                enabled=True,
                data_provider='mock',
                strategies=['rsi_trend'],
                position_size=0.1
            )
        ]
        config_manager.get_watchlist.return_value = watchlist
        config_manager.get_enabled_symbols.return_value = ['AAPL', 'MSFT']
        
        return config_manager
    
    @pytest.fixture
    def mock_redis_client(self):
        """Create mock Redis client for streaming tests."""
        mock_redis = Mock()
        
        # Mock stream operations
        mock_redis.xadd.return_value = b'1234567890-0'
        mock_redis.xinfo_stream.return_value = {
            'length': 100,
            'groups': 1,
            'last-generated-id': b'1234567890-0'
        }
        mock_redis.xgroup_create.return_value = True
        mock_redis.ping.return_value = True
        
        return mock_redis
    
    @pytest.fixture
    def streaming_data_collector(self):
        """Create data collector for streaming tests."""
        class StreamingDataCollector:
            def __init__(self):
                self.received_data = []
                self.bar_data = []
                self.trade_data = []
                self.quote_data = []
                self.errors = []
                self.start_time = None
                self.end_time = None
            
            async def on_bar(self, bar_data):
                if self.start_time is None:
                    self.start_time = time.time()
                self.bar_data.append(bar_data)
                self.received_data.append(('bar', bar_data))
            
            async def on_trade(self, trade_data):
                if self.start_time is None:
                    self.start_time = time.time()
                self.trade_data.append(trade_data)
                self.received_data.append(('trade', trade_data))
            
            async def on_quote(self, quote_data):
                if self.start_time is None:
                    self.start_time = time.time()
                self.quote_data.append(quote_data)
                self.received_data.append(('quote', quote_data))
            
            def on_error(self, error):
                self.errors.append(error)
            
            def stop(self):
                self.end_time = time.time()
            
            def get_stats(self):
                duration = (self.end_time or time.time()) - (self.start_time or time.time())
                return {
                    'total_messages': len(self.received_data),
                    'bar_messages': len(self.bar_data),
                    'trade_messages': len(self.trade_data),
                    'quote_messages': len(self.quote_data),
                    'errors': len(self.errors),
                    'duration': duration,
                    'messages_per_second': len(self.received_data) / max(duration, 0.001)
                }
        
        return StreamingDataCollector()
    
    @pytest.mark.asyncio
    async def test_streaming_connection_establishment(self, mock_config_manager):
        """Test real-time streaming connection establishment."""
        # This test uses mock provider since real streaming requires valid credentials
        from src.data.providers.mock_provider import MockProvider
        
        # Create ingestion service
        ingestion_service = MarketDataIngestionService(mock_config_manager)
        
        # Add mock provider that supports streaming
        mock_provider = MockProvider()
        mock_provider.start_streaming = AsyncMock()
        mock_provider.stop_streaming = AsyncMock()
        mock_provider.is_streaming = Mock(return_value=True)
        
        ingestion_service.add_provider('test_mock', mock_provider)
        
        # Test streaming setup
        symbols = ['AAPL', 'MSFT']
        
        # Mock streaming handlers
        on_bar = AsyncMock()
        on_trade = AsyncMock()
        on_quote = AsyncMock()
        
        # Start streaming
        await mock_provider.start_streaming(
            symbols=symbols,
            on_bar=on_bar,
            on_trade=on_trade,
            on_quote=on_quote
        )
        
        # Verify streaming was started
        mock_provider.start_streaming.assert_called_once()
        assert mock_provider.is_streaming()
        
        # Stop streaming
        await mock_provider.stop_streaming()
        mock_provider.stop_streaming.assert_called_once()
        
        logger.info("✓ Streaming connection establishment test passed")
    
    @pytest.mark.asyncio
    async def test_streaming_data_flow(self, mock_config_manager, streaming_data_collector):
        """Test real-time data flow through streaming pipeline."""
        from src.data.providers.mock_provider import MockProvider
        
        # Create mock provider with streaming capability
        mock_provider = MockProvider()
        
        # Simulate streaming data
        test_symbols = ['AAPL', 'MSFT']
        streaming_duration = 2.0  # seconds
        
        async def simulate_streaming(symbols, on_bar=None, on_trade=None, on_quote=None):
            """Simulate streaming data generation."""
            start_time = time.time()
            message_count = 0
            
            while time.time() - start_time < streaming_duration:
                for symbol in symbols:
                    # Generate mock bar data
                    if on_bar:
                        mock_bar = OHLCV(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            open=100.0 + message_count * 0.1,
                            high=101.0 + message_count * 0.1,
                            low=99.0 + message_count * 0.1,
                            close=100.5 + message_count * 0.1,
                            volume=1000 + message_count * 10
                        )
                        await on_bar(mock_bar)
                    
                    # Generate mock trade data
                    if on_trade:
                        mock_trade = {
                            'symbol': symbol,
                            'price': 100.0 + message_count * 0.1,
                            'size': 100,
                            'timestamp': datetime.now(),
                            'exchange': 'MOCK'
                        }
                        await on_trade(mock_trade)
                    
                    # Generate mock quote data
                    if on_quote:
                        mock_quote = {
                            'symbol': symbol,
                            'bid_price': 99.9 + message_count * 0.1,
                            'ask_price': 100.1 + message_count * 0.1,
                            'bid_size': 100,
                            'ask_size': 100,
                            'timestamp': datetime.now()
                        }
                        await on_quote(mock_quote)
                
                message_count += 1
                await asyncio.sleep(0.1)  # 10 messages per second
        
        # Replace start_streaming with simulation
        mock_provider.start_streaming = simulate_streaming
        
        # Start streaming with data collector
        await mock_provider.start_streaming(
            symbols=test_symbols,
            on_bar=streaming_data_collector.on_bar,
            on_trade=streaming_data_collector.on_trade,
            on_quote=streaming_data_collector.on_quote
        )
        
        streaming_data_collector.stop()
        stats = streaming_data_collector.get_stats()
        
        # Validate streaming performance
        assert stats['total_messages'] > 0, "Should receive streaming messages"
        assert stats['bar_messages'] > 0, "Should receive bar data"
        assert stats['trade_messages'] > 0, "Should receive trade data"
        assert stats['quote_messages'] > 0, "Should receive quote data"
        assert stats['messages_per_second'] > 5, "Should maintain reasonable throughput"
        
        logger.info(f"✓ Streaming data flow test passed:")
        logger.info(f"  Total messages: {stats['total_messages']}")
        logger.info(f"  Messages/sec: {stats['messages_per_second']:.1f}")
        logger.info(f"  Duration: {stats['duration']:.1f}s")
    
    @pytest.mark.asyncio
    async def test_streaming_data_quality(self, streaming_data_collector):
        """Test data quality in streaming mode."""
        from src.data.providers.mock_provider import MockProvider
        
        mock_provider = MockProvider()
        
        # Track data quality metrics
        price_consistency_errors = 0
        timestamp_errors = 0
        data_completeness_errors = 0
        
        async def quality_aware_bar_handler(bar_data):
            nonlocal price_consistency_errors, timestamp_errors, data_completeness_errors
            
            await streaming_data_collector.on_bar(bar_data)
            
            # Check OHLC consistency
            if not (bar_data.low <= bar_data.open <= bar_data.high and
                   bar_data.low <= bar_data.close <= bar_data.high):
                price_consistency_errors += 1
            
            # Check timestamp recency
            age = (datetime.now() - bar_data.timestamp).total_seconds()
            if age > 60:  # Data older than 1 minute
                timestamp_errors += 1
            
            # Check data completeness
            if (bar_data.open <= 0 or bar_data.high <= 0 or 
                bar_data.low <= 0 or bar_data.close <= 0):
                data_completeness_errors += 1
        
        # Simulate streaming with quality checks
        async def quality_streaming(symbols, on_bar=None, **kwargs):
            for i in range(50):  # Generate 50 data points
                for symbol in symbols:
                    if on_bar:
                        # Generate realistic OHLC data
                        base_price = 100.0 + i * 0.1
                        mock_bar = OHLCV(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            open=base_price,
                            high=base_price + 0.5,
                            low=base_price - 0.3,
                            close=base_price + 0.2,
                            volume=1000 + i * 10
                        )
                        await on_bar(mock_bar)
                
                await asyncio.sleep(0.05)  # 20 messages per second
        
        mock_provider.start_streaming = quality_streaming
        
        # Start streaming with quality monitoring
        await mock_provider.start_streaming(
            symbols=['AAPL'],
            on_bar=quality_aware_bar_handler
        )
        
        streaming_data_collector.stop()
        stats = streaming_data_collector.get_stats()
        
        # Validate data quality
        total_bars = stats['bar_messages']
        if total_bars > 0:
            price_error_rate = price_consistency_errors / total_bars
            timestamp_error_rate = timestamp_errors / total_bars
            completeness_error_rate = data_completeness_errors / total_bars
            
            assert price_error_rate < 0.01, f"Price consistency error rate too high: {price_error_rate:.2%}"
            assert timestamp_error_rate < 0.05, f"Timestamp error rate too high: {timestamp_error_rate:.2%}"
            assert completeness_error_rate < 0.01, f"Data completeness error rate too high: {completeness_error_rate:.2%}"
            
            logger.info(f"✓ Streaming data quality validation passed:")
            logger.info(f"  Total bars processed: {total_bars}")
            logger.info(f"  Price consistency errors: {price_consistency_errors}")
            logger.info(f"  Timestamp errors: {timestamp_errors}")
            logger.info(f"  Completeness errors: {data_completeness_errors}")
    
    @pytest.mark.asyncio
    async def test_streaming_error_recovery(self, mock_config_manager):
        """Test error recovery and reconnection in streaming."""
        from src.data.providers.mock_provider import MockProvider
        
        mock_provider = MockProvider()
        
        # Track connection attempts and errors
        connection_attempts = 0
        streaming_errors = 0
        successful_reconnections = 0
        
        async def error_prone_streaming(symbols, on_bar=None, **kwargs):
            nonlocal connection_attempts, streaming_errors, successful_reconnections
            connection_attempts += 1
            
            try:
                # Simulate some successful streaming
                for i in range(10):
                    if on_bar:
                        mock_bar = OHLCV(
                            timestamp=datetime.now(),
                            symbol=symbols[0],
                            open=100.0,
                            high=101.0,
                            low=99.0,
                            close=100.5,
                            volume=1000
                        )
                        await on_bar(mock_bar)
                    
                    # Simulate random connection errors
                    if i == 5 and connection_attempts == 1:
                        raise ConnectionError("Simulated connection lost")
                    
                    await asyncio.sleep(0.1)
                
                successful_reconnections += 1
                
            except Exception as e:
                streaming_errors += 1
                logger.info(f"Simulated streaming error: {e}")
                raise
        
        mock_provider.start_streaming = error_prone_streaming
        
        # Create error recovery wrapper
        async def streaming_with_recovery(provider, symbols, max_retries=3):
            for attempt in range(max_retries):
                try:
                    await provider.start_streaming(
                        symbols=symbols,
                        on_bar=AsyncMock()
                    )
                    return True
                except Exception as e:
                    logger.info(f"Streaming attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1.0)  # Retry delay
                    else:
                        return False
        
        # Test error recovery
        symbols = ['AAPL']
        success = await streaming_with_recovery(mock_provider, symbols)
        
        # Validate recovery behavior
        assert connection_attempts > 1, "Should attempt multiple connections"
        assert streaming_errors > 0, "Should encounter simulated errors"
        assert success, "Should eventually succeed with retry logic"
        
        logger.info(f"✓ Streaming error recovery test passed:")
        logger.info(f"  Connection attempts: {connection_attempts}")
        logger.info(f"  Streaming errors: {streaming_errors}")
        logger.info(f"  Successful reconnections: {successful_reconnections}")
    
    @pytest.mark.asyncio
    async def test_streaming_latency_measurement(self, streaming_data_collector):
        """Test streaming latency and real-time performance."""
        from src.data.providers.mock_provider import MockProvider
        
        mock_provider = MockProvider()
        
        # Track latency metrics
        latencies = []
        processing_times = []
        
        async def latency_aware_handler(bar_data):
            receive_time = time.time()
            
            # Calculate data age (latency)
            data_age = receive_time - bar_data.timestamp.timestamp()
            latencies.append(data_age)
            
            # Measure processing time
            process_start = time.time()
            await streaming_data_collector.on_bar(bar_data)
            processing_time = time.time() - process_start
            processing_times.append(processing_time)
        
        # Simulate low-latency streaming
        async def low_latency_streaming(symbols, on_bar=None, **kwargs):
            for i in range(100):
                if on_bar:
                    # Create data with current timestamp
                    mock_bar = OHLCV(
                        timestamp=datetime.now(),
                        symbol=symbols[0],
                        open=100.0 + i * 0.01,
                        high=100.1 + i * 0.01,
                        low=99.9 + i * 0.01,
                        close=100.05 + i * 0.01,
                        volume=1000
                    )
                    await on_bar(mock_bar)
                
                await asyncio.sleep(0.01)  # 100 messages per second
        
        mock_provider.start_streaming = low_latency_streaming
        
        # Start latency monitoring
        await mock_provider.start_streaming(
            symbols=['AAPL'],
            on_bar=latency_aware_handler
        )
        
        streaming_data_collector.stop()
        
        # Analyze latency metrics
        if latencies and processing_times:
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
            
            avg_processing_time = sum(processing_times) / len(processing_times)
            max_processing_time = max(processing_times)
            
            # Latency assertions
            assert avg_latency < 1.0, f"Average latency too high: {avg_latency:.3f}s"
            assert max_latency < 5.0, f"Maximum latency too high: {max_latency:.3f}s"
            assert p95_latency < 2.0, f"P95 latency too high: {p95_latency:.3f}s"
            
            # Processing time assertions
            assert avg_processing_time < 0.001, f"Average processing time too high: {avg_processing_time:.6f}s"
            assert max_processing_time < 0.01, f"Maximum processing time too high: {max_processing_time:.6f}s"
            
            logger.info(f"✓ Streaming latency measurement passed:")
            logger.info(f"  Average latency: {avg_latency*1000:.1f}ms")
            logger.info(f"  P95 latency: {p95_latency*1000:.1f}ms")
            logger.info(f"  Maximum latency: {max_latency*1000:.1f}ms")
            logger.info(f"  Average processing time: {avg_processing_time*1000000:.1f}μs")
    
    @pytest.mark.asyncio
    async def test_streaming_throughput_under_load(self, streaming_data_collector):
        """Test streaming throughput under high load conditions."""
        from src.data.providers.mock_provider import MockProvider
        
        mock_provider = MockProvider()
        
        # High-throughput streaming simulation
        async def high_throughput_streaming(symbols, on_bar=None, **kwargs):
            start_time = time.time()
            message_count = 0
            target_duration = 5.0  # 5 seconds
            target_rate = 1000  # 1000 messages per second
            
            while time.time() - start_time < target_duration:
                batch_start = time.time()
                
                # Send batch of messages
                for _ in range(10):  # Batch size
                    for symbol in symbols:
                        if on_bar:
                            mock_bar = OHLCV(
                                timestamp=datetime.now(),
                                symbol=symbol,
                                open=100.0 + message_count * 0.001,
                                high=100.01 + message_count * 0.001,
                                low=99.99 + message_count * 0.001,
                                close=100.005 + message_count * 0.001,
                                volume=1000 + message_count
                            )
                            await on_bar(mock_bar)
                            message_count += 1
                
                # Maintain target rate
                batch_time = time.time() - batch_start
                target_batch_time = (len(symbols) * 10) / target_rate
                if batch_time < target_batch_time:
                    await asyncio.sleep(target_batch_time - batch_time)
        
        mock_provider.start_streaming = high_throughput_streaming
        
        # Test with multiple symbols for higher load
        test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        
        start_time = time.time()
        await mock_provider.start_streaming(
            symbols=test_symbols,
            on_bar=streaming_data_collector.on_bar
        )
        streaming_data_collector.stop()
        
        stats = streaming_data_collector.get_stats()
        
        # Validate high-throughput performance
        assert stats['total_messages'] > 1000, f"Should handle high message volume: {stats['total_messages']}"
        assert stats['messages_per_second'] > 500, f"Should maintain high throughput: {stats['messages_per_second']:.1f}"
        assert stats['duration'] >= 4.0, "Should run for reasonable duration"
        
        # Check for message loss (should process most messages)
        expected_messages = 4 * 1000 * 5  # 4 symbols * 1000 msg/s * 5 seconds
        message_loss_rate = 1 - (stats['total_messages'] / expected_messages)
        assert message_loss_rate < 0.1, f"Message loss rate too high: {message_loss_rate:.2%}"
        
        logger.info(f"✓ High-throughput streaming test passed:")
        logger.info(f"  Total messages: {stats['total_messages']}")
        logger.info(f"  Throughput: {stats['messages_per_second']:.1f} msg/s")
        logger.info(f"  Message loss rate: {message_loss_rate:.2%}")
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self):
        """Test circuit breaker behavior in streaming context."""
        # Test circuit breaker with mock failing function
        circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=2)
        
        call_count = 0
        success_count = 0
        failure_count = 0
        
        def failing_function():
            nonlocal call_count, success_count, failure_count
            call_count += 1
            
            if call_count <= 5:  # First 5 calls fail
                failure_count += 1
                raise Exception(f"Simulated failure {call_count}")
            else:  # Subsequent calls succeed
                success_count += 1
                return f"Success {call_count}"
        
        # Test initial failures
        for i in range(5):
            try:
                result = circuit_breaker.call(failing_function)
                pytest.fail("Should have raised exception")
            except Exception as e:
                assert "Simulated failure" in str(e)
        
        # Circuit should be open now
        assert circuit_breaker.state == "OPEN"
        
        # Additional calls should be blocked
        try:
            circuit_breaker.call(failing_function)
            pytest.fail("Circuit breaker should block calls")
        except Exception as e:
            assert "Circuit breaker is OPEN" in str(e)
        
        # Wait for recovery timeout
        await asyncio.sleep(2.1)
        
        # Next call should succeed (half-open state)
        result = circuit_breaker.call(failing_function)
        assert "Success" in result
        assert circuit_breaker.state == "CLOSED"
        
        logger.info(f"✓ Circuit breaker test passed:")
        logger.info(f"  Total calls: {call_count}")
        logger.info(f"  Failures: {failure_count}")
        logger.info(f"  Successes: {success_count}")
        logger.info(f"  Final state: {circuit_breaker.state}")
    
    @pytest.mark.asyncio
    async def test_streaming_memory_management(self, streaming_data_collector):
        """Test memory management during extended streaming."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        from src.data.providers.mock_provider import MockProvider
        mock_provider = MockProvider()
        
        # Memory tracking during streaming
        memory_samples = []
        
        async def memory_monitoring_streaming(symbols, on_bar=None, **kwargs):
            nonlocal memory_samples
            
            for i in range(500):  # Generate many messages
                # Sample memory every 50 messages
                if i % 50 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_samples.append(current_memory)
                
                if on_bar:
                    mock_bar = OHLCV(
                        timestamp=datetime.now(),
                        symbol=symbols[0],
                        open=100.0,
                        high=101.0,
                        low=99.0,
                        close=100.5,
                        volume=1000
                    )
                    await on_bar(mock_bar)
                
                await asyncio.sleep(0.001)  # Fast streaming
        
        mock_provider.start_streaming = memory_monitoring_streaming
        
        # Run streaming with memory monitoring
        await mock_provider.start_streaming(
            symbols=['AAPL'],
            on_bar=streaming_data_collector.on_bar
        )
        
        streaming_data_collector.stop()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Analyze memory usage
        if memory_samples:
            max_memory = max(memory_samples)
            memory_growth = final_memory - initial_memory
            peak_memory_growth = max_memory - initial_memory
            
            # Memory growth should be reasonable
            assert memory_growth < 50, f"Memory growth too large: {memory_growth:.1f} MB"
            assert peak_memory_growth < 100, f"Peak memory growth too large: {peak_memory_growth:.1f} MB"
            
            # Check for memory leaks (final memory should be close to initial)
            memory_leak_threshold = 20  # MB
            if memory_growth > memory_leak_threshold:
                logger.warning(f"Potential memory leak detected: {memory_growth:.1f} MB growth")
            
            logger.info(f"✓ Memory management test passed:")
            logger.info(f"  Initial memory: {initial_memory:.1f} MB")
            logger.info(f"  Final memory: {final_memory:.1f} MB")
            logger.info(f"  Peak memory: {max_memory:.1f} MB")
            logger.info(f"  Memory growth: {memory_growth:.1f} MB")
    
    @pytest.mark.asyncio
    async def test_streaming_data_persistence(self, mock_config_manager, mock_redis_client):
        """Test streaming data persistence to Redis."""
        # Mock Redis for testing
        with patch('redis.Redis', return_value=mock_redis_client):
            ingestion_service = MarketDataIngestionService(mock_config_manager)
            ingestion_service._redis_client = mock_redis_client
            
            # Test data publication
            symbol = 'AAPL'
            mock_ohlcv = OHLCV(
                timestamp=datetime.now(),
                symbol=symbol,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1000
            )
            
            # Publish market data event
            await ingestion_service._publish_market_data_event(
                symbol=symbol,
                data=mock_ohlcv,
                source='test'
            )
            
            # Verify Redis operations
            mock_redis_client.xadd.assert_called_once()
            call_args = mock_redis_client.xadd.call_args
            
            # Check stream name
            stream_name = call_args[0][0]
            assert stream_name == 'market_data_stream'
            
            # Check event data
            event_data = call_args[0][1]
            assert event_data['symbol'] == symbol
            assert event_data['price'] == str(mock_ohlcv.close)
            assert event_data['source'] == 'test'
            
            logger.info("✓ Streaming data persistence test passed")
    
    @pytest.mark.asyncio
    async def test_multi_provider_streaming(self, mock_config_manager):
        """Test streaming from multiple data providers simultaneously."""
        from src.data.providers.mock_provider import MockProvider
        
        # Create multiple mock providers
        provider1 = MockProvider()
        provider2 = MockProvider()
        
        # Track messages from each provider
        provider1_messages = []
        provider2_messages = []
        
        async def provider1_streaming(symbols, on_bar=None, **kwargs):
            for i in range(20):
                if on_bar:
                    mock_bar = OHLCV(
                        timestamp=datetime.now(),
                        symbol='AAPL',
                        open=100.0 + i * 0.1,
                        high=101.0 + i * 0.1,
                        low=99.0 + i * 0.1,
                        close=100.5 + i * 0.1,
                        volume=1000
                    )
                    provider1_messages.append(mock_bar)
                    await on_bar(mock_bar)
                await asyncio.sleep(0.05)
        
        async def provider2_streaming(symbols, on_bar=None, **kwargs):
            for i in range(20):
                if on_bar:
                    mock_bar = OHLCV(
                        timestamp=datetime.now(),
                        symbol='MSFT',
                        open=200.0 + i * 0.1,
                        high=201.0 + i * 0.1,
                        low=199.0 + i * 0.1,
                        close=200.5 + i * 0.1,
                        volume=2000
                    )
                    provider2_messages.append(mock_bar)
                    await on_bar(mock_bar)
                await asyncio.sleep(0.05)
        
        provider1.start_streaming = provider1_streaming
        provider2.start_streaming = provider2_streaming
        
        # Collect all messages
        all_messages = []
        
        async def combined_handler(bar_data):
            all_messages.append(bar_data)
        
        # Start both providers simultaneously
        await asyncio.gather(
            provider1.start_streaming(symbols=['AAPL'], on_bar=combined_handler),
            provider2.start_streaming(symbols=['MSFT'], on_bar=combined_handler)
        )
        
        # Validate multi-provider streaming
        assert len(provider1_messages) == 20, "Provider 1 should send 20 messages"
        assert len(provider2_messages) == 20, "Provider 2 should send 20 messages"
        assert len(all_messages) == 40, "Should receive all messages from both providers"
        
        # Check symbol distribution
        aapl_messages = [msg for msg in all_messages if msg.symbol == 'AAPL']
        msft_messages = [msg for msg in all_messages if msg.symbol == 'MSFT']
        
        assert len(aapl_messages) == 20, "Should receive all AAPL messages"
        assert len(msft_messages) == 20, "Should receive all MSFT messages"
        
        logger.info(f"✓ Multi-provider streaming test passed:")
        logger.info(f"  Provider 1 messages: {len(provider1_messages)}")
        logger.info(f"  Provider 2 messages: {len(provider2_messages)}")
        logger.info(f"  Total messages: {len(all_messages)}")