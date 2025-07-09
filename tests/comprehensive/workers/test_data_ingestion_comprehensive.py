"""
Comprehensive tests for the DataIngestionService worker component.

Tests cover:
- Data fetching from providers
- Redis stream publishing
- Circuit breaker functionality  
- Error handling and recovery
- Performance and throughput
- Provider management
- Stream monitoring
"""

import pytest
import asyncio
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json

from src.realtime.data_ingestion import (
    MarketDataIngestionService, 
    MarketDataEvent, 
    CircuitBreaker,
    MinimalAlpacaProvider
)
from src.core.models import OHLCV


class TestDataIngestionServiceComponents:
    """Test individual DataIngestionService components."""
    
    def test_initialization(self, test_config_manager, mock_redis_client):
        """Test DataIngestionService initialization."""
        with patch('src.realtime.data_ingestion.redis.Redis', return_value=mock_redis_client):
            service = MarketDataIngestionService(test_config_manager)
            
            assert service.config_manager is test_config_manager
            assert service._redis_client is mock_redis_client
            assert not service._running
            assert len(service._tasks) == 0
            assert "market_data" in service.streams
            assert "calculation_events" in service.streams
            assert "alerts" in service.streams
            
    def test_stream_configuration(self, test_data_ingestion_service):
        """Test stream configuration setup."""
        streams = test_data_ingestion_service.streams
        
        # Verify all required streams are configured
        assert "market_data" in streams
        assert "calculation_events" in streams
        assert "alerts" in streams
        
        # Verify stream configuration structure
        market_data_stream = streams["market_data"]
        assert market_data_stream.stream_name == "market_data_stream"
        assert market_data_stream.max_len > 0
        assert market_data_stream.consumer_group is not None
        
    def test_provider_initialization(self, test_data_ingestion_service):
        """Test data provider initialization."""
        providers = test_data_ingestion_service._providers
        
        # Should have at least mock provider
        assert "mock" in providers
        assert hasattr(providers["mock"], "get_latest_data")
        
    def test_circuit_breaker_initialization(self, test_data_ingestion_service):
        """Test circuit breaker setup."""
        # Add a provider to test circuit breaker creation
        mock_provider = Mock()
        test_data_ingestion_service.add_provider("test_provider", mock_provider)
        
        assert "test_provider" in test_data_ingestion_service._circuit_breakers
        
    def test_stats_initialization(self, test_data_ingestion_service):
        """Test performance statistics initialization."""
        stats = test_data_ingestion_service.get_stats()
        
        assert stats["events_published"] == 0
        assert stats["errors"] == 0
        assert stats["circuit_breaker_trips"] == 0
        assert stats["running"] == False


class TestCircuitBreakerFunctionality:
    """Test circuit breaker component."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 60
        assert cb.failure_count == 0
        assert cb.state == "CLOSED"
        
    def test_circuit_breaker_success_calls(self):
        """Test circuit breaker with successful calls."""
        cb = CircuitBreaker(failure_threshold=3)
        
        def success_function():
            return "success"
            
        result = cb.call(success_function)
        assert result == "success"
        assert cb.failure_count == 0
        assert cb.state == "CLOSED"
        
    def test_circuit_breaker_failure_handling(self):
        """Test circuit breaker failure handling."""
        cb = CircuitBreaker(failure_threshold=2)
        
        def failure_function():
            raise Exception("Test failure")
            
        # First failure
        with pytest.raises(Exception):
            cb.call(failure_function)
        assert cb.failure_count == 1
        assert cb.state == "CLOSED"
        
        # Second failure - should open circuit
        with pytest.raises(Exception):
            cb.call(failure_function)
        assert cb.failure_count == 2
        assert cb.state == "OPEN"
        
    def test_circuit_breaker_open_state(self):
        """Test circuit breaker in open state."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1)
        
        def failure_function():
            raise Exception("Test failure")
            
        # Trigger circuit breaker
        with pytest.raises(Exception):
            cb.call(failure_function)
        assert cb.state == "OPEN"
        
        # Should reject calls immediately
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            cb.call(failure_function)
            
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)  # 100ms recovery
        
        def failure_function():
            raise Exception("Test failure")
            
        def success_function():
            return "recovered"
            
        # Trigger circuit breaker
        with pytest.raises(Exception):
            cb.call(failure_function)
        assert cb.state == "OPEN"
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Should allow call and reset on success
        result = cb.call(success_function)
        assert result == "recovered"
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0


class TestMarketDataEventHandling:
    """Test market data event creation and publishing."""
    
    def test_market_data_event_creation(self):
        """Test MarketDataEvent creation."""
        timestamp = datetime.now()
        ohlcv = OHLCV(
            timestamp=timestamp,
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000
        )
        
        event = MarketDataEvent(
            event_id="test_event",
            symbol="AAPL",
            timestamp=timestamp,
            price=151.0,
            volume=1000,
            open_price=150.0,
            high_price=152.0,
            low_price=149.0,
            close_price=151.0,
            source="test",
            metadata={"test": True}
        )
        
        assert event.symbol == "AAPL"
        assert event.price == 151.0
        assert event.volume == 1000
        assert event.source == "test"
        
    @pytest.mark.asyncio
    async def test_market_data_event_publishing(self, test_data_ingestion_service):
        """Test publishing market data events."""
        timestamp = datetime.now()
        ohlcv = OHLCV(
            timestamp=timestamp,
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000
        )
        
        # Should not raise exceptions
        await test_data_ingestion_service._publish_market_data_event("AAPL", ohlcv, "test")
        
        # Verify Redis client was called
        redis_client = test_data_ingestion_service._redis_client
        assert redis_client.xadd.called
        
    @pytest.mark.asyncio  
    async def test_calculation_event_publishing(self, test_data_ingestion_service):
        """Test publishing calculation events."""
        calculation_results = {
            "rsi_14": 45.0,
            "sma_50": 150.0,
            "signals": {"macd_rsi_strategy": {"signal": "buy", "confidence": 0.8}}
        }
        
        await test_data_ingestion_service.publish_calculation_event("AAPL", calculation_results)
        
        # Verify Redis client was called
        redis_client = test_data_ingestion_service._redis_client
        assert redis_client.xadd.called
        
    @pytest.mark.asyncio
    async def test_alert_publishing(self, test_data_ingestion_service):
        """Test publishing alerts."""
        await test_data_ingestion_service.publish_alert(
            alert_type="signal",
            symbol="AAPL", 
            message="Strong buy signal detected",
            metadata={"confidence": 0.9}
        )
        
        # Verify Redis client was called
        redis_client = test_data_ingestion_service._redis_client
        assert redis_client.xadd.called


class TestDataProviderIntegration:
    """Test integration with data providers."""
    
    def test_mock_provider_integration(self, test_data_ingestion_service, mock_data_provider):
        """Test integration with mock data provider."""
        test_data_ingestion_service.add_provider("test_mock", mock_data_provider)
        
        providers = test_data_ingestion_service._providers
        assert "test_mock" in providers
        
        # Test data retrieval
        data = providers["test_mock"].get_latest_data("AAPL")
        assert data is not None
        assert hasattr(data, "close")
        
    @pytest.mark.asyncio
    async def test_provider_circuit_breaker_integration(self, test_data_ingestion_service):
        """Test circuit breaker integration with providers."""
        # Create a failing provider
        failing_provider = Mock()
        failing_provider.get_latest_data.side_effect = Exception("Provider failure")
        
        test_data_ingestion_service.add_provider("failing_provider", failing_provider)
        
        circuit_breaker = test_data_ingestion_service._circuit_breakers["failing_provider"]
        
        # Test circuit breaker protection
        result = await test_data_ingestion_service._fetch_data_with_circuit_breaker(
            circuit_breaker, failing_provider, "AAPL"
        )
        
        assert result is None  # Should return None on failure
        assert circuit_breaker.failure_count > 0
        
    def test_minimal_alpaca_provider_initialization(self):
        """Test MinimalAlpacaProvider initialization."""
        config = {
            'api_key': 'test_key',
            'secret_key': 'test_secret',
            'paper': True
        }
        
        provider = MinimalAlpacaProvider(config)
        
        assert provider.name == "Alpaca"
        assert provider.config == config
        assert not provider._connected
        
    @pytest.mark.asyncio
    async def test_provider_connection_handling(self, test_data_ingestion_service):
        """Test provider connection handling."""
        mock_provider = Mock()
        mock_provider.connect = AsyncMock(return_value=True)
        
        test_data_ingestion_service.add_provider("test_provider", mock_provider)
        
        # Test connection during startup
        providers = test_data_ingestion_service._providers
        
        # Manually test connection
        if hasattr(providers["test_provider"], "connect"):
            connected = await providers["test_provider"].connect()
            assert connected == True


class TestDataIngestionPerformance:
    """Test DataIngestionService performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_ingestion_throughput(self, test_data_ingestion_service, 
                                      performance_thresholds, resource_monitor):
        """Test data ingestion throughput."""
        symbol = "AAPL"
        event_count = 50
        
        # Create test data
        test_data = []
        for i in range(event_count):
            ohlcv = OHLCV(
                timestamp=datetime.now() - timedelta(seconds=i),
                open=150.0 + i * 0.1,
                high=151.0 + i * 0.1,
                low=149.0 + i * 0.1,
                close=150.5 + i * 0.1,
                volume=1000 + i * 10
            )
            test_data.append(ohlcv)
            
        start_time = time.time()
        
        # Publish events
        for ohlcv in test_data:
            await test_data_ingestion_service._publish_market_data_event(symbol, ohlcv, "test")
            
        total_time = time.time() - start_time
        throughput = event_count / total_time
        
        assert throughput > 50  # At least 50 events per second
        
    def test_memory_usage_during_ingestion(self, test_data_ingestion_service, resource_monitor):
        """Test memory usage during data ingestion."""
        initial_memory = resource_monitor.get_memory_usage()
        
        # Process many events
        for i in range(100):
            test_data_ingestion_service._stats["events_published"] += 1
            test_data_ingestion_service._stats["provider_calls"]["mock"] = i
            
        final_memory = resource_monitor.get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal for statistics updates
        assert memory_increase < 10  # Less than 10MB
        
    @pytest.mark.asyncio
    async def test_concurrent_symbol_ingestion(self, test_data_ingestion_service):
        """Test concurrent ingestion for multiple symbols."""
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        
        async def ingest_symbol(symbol):
            ohlcv = OHLCV(
                timestamp=datetime.now(),
                open=150.0,
                high=152.0,
                low=149.0,
                close=151.0,
                volume=1000
            )
            
            await test_data_ingestion_service._publish_market_data_event(symbol, ohlcv, "test")
            
        # Process symbols concurrently
        tasks = [ingest_symbol(symbol) for symbol in symbols]
        await asyncio.gather(*tasks)
        
        # Verify all symbols were processed
        redis_client = test_data_ingestion_service._redis_client
        assert redis_client.xadd.call_count >= len(symbols)


class TestDataIngestionReliability:
    """Test DataIngestionService reliability and error handling."""
    
    @pytest.mark.asyncio
    async def test_redis_connection_error_handling(self, test_config_manager):
        """Test handling of Redis connection errors."""
        mock_redis = Mock()
        mock_redis.ping.side_effect = Exception("Redis connection failed")
        
        with patch('src.realtime.data_ingestion.redis.Redis', return_value=mock_redis):
            # Should raise exception during initialization
            with pytest.raises(Exception):
                MarketDataIngestionService(test_config_manager)
                
    @pytest.mark.asyncio
    async def test_stream_initialization_error_handling(self, test_config_manager, mock_redis_client):
        """Test stream initialization error handling."""
        mock_redis_client.xgroup_create.side_effect = Exception("Stream creation failed")
        
        with patch('src.realtime.data_ingestion.redis.Redis', return_value=mock_redis_client):
            # Should raise exception during stream initialization
            with pytest.raises(Exception):
                MarketDataIngestionService(test_config_manager)
                
    @pytest.mark.asyncio
    async def test_provider_failure_recovery(self, test_data_ingestion_service):
        """Test recovery from provider failures."""
        # Create a provider that fails initially then recovers
        call_count = 0
        
        def failing_then_recovering():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Provider temporarily down")
            return OHLCV(
                timestamp=datetime.now(),
                open=150.0, high=152.0, low=149.0, close=151.0, volume=1000
            )
            
        provider = Mock()
        provider.get_latest_data.side_effect = failing_then_recovering
        
        test_data_ingestion_service.add_provider("recovery_test", provider)
        circuit_breaker = test_data_ingestion_service._circuit_breakers["recovery_test"]
        
        # First calls should fail
        result1 = await test_data_ingestion_service._fetch_data_with_circuit_breaker(
            circuit_breaker, provider, "AAPL"
        )
        assert result1 is None
        
        result2 = await test_data_ingestion_service._fetch_data_with_circuit_breaker(
            circuit_breaker, provider, "AAPL"
        )
        assert result2 is None
        
        # Circuit breaker should be open now
        assert circuit_breaker.state == "OPEN"
        
    @pytest.mark.asyncio
    async def test_message_publishing_error_handling(self, test_data_ingestion_service):
        """Test error handling during message publishing."""
        # Make Redis throw an error
        redis_client = test_data_ingestion_service._redis_client
        redis_client.xadd.side_effect = Exception("Redis publish failed")
        
        ohlcv = OHLCV(
            timestamp=datetime.now(),
            open=150.0, high=152.0, low=149.0, close=151.0, volume=1000
        )
        
        # Should handle error gracefully
        with pytest.raises(Exception):
            await test_data_ingestion_service._publish_market_data_event("AAPL", ohlcv, "test")
            
    def test_latest_price_caching(self, test_data_ingestion_service):
        """Test latest price caching functionality."""
        # Mock the cache manager
        with patch('src.realtime.data_ingestion.cache_manager') as mock_cache:
            mock_cache.get.return_value = {"price": 150.0, "timestamp": datetime.now().isoformat()}
            
            price = test_data_ingestion_service.get_latest_price("AAPL")
            assert price == 150.0
            
            mock_cache.get.assert_called_once()


class TestDataIngestionScalability:
    """Test DataIngestionService scalability characteristics."""
    
    @pytest.mark.asyncio
    async def test_multiple_symbol_ingestion_scaling(self, test_data_ingestion_service, load_generator):
        """Test scaling with multiple symbols."""
        symbols = [f"TEST{i}" for i in range(20)]
        
        async def process_symbol(symbol):
            ohlcv = OHLCV(
                timestamp=datetime.now(),
                open=150.0, high=152.0, low=149.0, close=151.0, volume=1000
            )
            await test_data_ingestion_service._publish_market_data_event(symbol, ohlcv, "test")
            
        # Process many symbols concurrently
        tasks = [process_symbol(symbol) for symbol in symbols]
        start_time = time.time()
        
        await asyncio.gather(*tasks)
        
        processing_time = time.time() - start_time
        
        # Should handle many symbols efficiently
        assert processing_time < 5.0  # Less than 5 seconds for 20 symbols
        
    @pytest.mark.asyncio
    async def test_high_frequency_ingestion(self, test_data_ingestion_service):
        """Test high-frequency data ingestion."""
        symbol = "AAPL"
        frequency_events = 100
        
        async def publish_event():
            ohlcv = OHLCV(
                timestamp=datetime.now(),
                open=150.0, high=152.0, low=149.0, close=151.0, volume=1000
            )
            await test_data_ingestion_service._publish_market_data_event(symbol, ohlcv, "test")
            
        start_time = time.time()
        
        # Publish many events rapidly
        tasks = [publish_event() for _ in range(frequency_events)]
        await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        events_per_second = frequency_events / total_time
        
        # Should handle high frequency
        assert events_per_second > 50  # At least 50 events/second
        
    @pytest.mark.asyncio
    async def test_stream_length_management(self, test_data_ingestion_service):
        """Test Redis stream length management."""
        symbol = "AAPL"
        
        # Publish many events to test stream trimming
        for i in range(50):
            ohlcv = OHLCV(
                timestamp=datetime.now() - timedelta(seconds=i),
                open=150.0, high=152.0, low=149.0, close=151.0, volume=1000
            )
            await test_data_ingestion_service._publish_market_data_event(symbol, ohlcv, "test")
            
        # Verify Redis xadd was called with maxlen parameter
        redis_client = test_data_ingestion_service._redis_client
        
        # Check that xadd was called with maxlen (stream trimming)
        assert redis_client.xadd.called
        call_args = redis_client.xadd.call_args
        assert 'maxlen' in call_args.kwargs or len(call_args.args) > 2


class TestDataIngestionMonitoring:
    """Test DataIngestionService monitoring and diagnostics."""
    
    def test_statistics_collection(self, test_data_ingestion_service):
        """Test statistics collection."""
        # Update some statistics
        test_data_ingestion_service._stats["events_published"] = 100
        test_data_ingestion_service._stats["errors"] = 2
        test_data_ingestion_service._stats["circuit_breaker_trips"] = 1
        test_data_ingestion_service._stats["provider_calls"]["mock"] = 50
        
        stats = test_data_ingestion_service.get_stats()
        
        assert stats["events_published"] == 100
        assert stats["errors"] == 2
        assert stats["circuit_breaker_trips"] == 1
        assert stats["provider_calls"]["mock"] == 50
        assert "enabled_symbols" in stats
        assert "stream_info" in stats
        
    def test_stream_info_monitoring(self, test_data_ingestion_service):
        """Test Redis stream information monitoring."""
        # Mock stream info response
        redis_client = test_data_ingestion_service._redis_client
        redis_client.xinfo_stream.return_value = {
            "length": 10,
            "groups": 1,
            "last-generated-id": "1234567890-0"
        }
        
        stats = test_data_ingestion_service.get_stats()
        stream_info = stats["stream_info"]
        
        # Should have info for all streams
        assert "market_data" in stream_info
        assert "calculation_events" in stream_info
        assert "alerts" in stream_info
        
    def test_provider_statistics_tracking(self, test_data_ingestion_service):
        """Test provider statistics tracking."""
        # Add a provider and simulate calls
        mock_provider = Mock()
        test_data_ingestion_service.add_provider("test_provider", mock_provider)
        
        # Simulate provider calls
        test_data_ingestion_service._stats["provider_calls"]["test_provider"] = 25
        
        stats = test_data_ingestion_service.get_stats()
        assert stats["provider_calls"]["test_provider"] == 25
        
    def test_error_rate_monitoring(self, test_data_ingestion_service):
        """Test error rate monitoring."""
        # Simulate some errors
        test_data_ingestion_service._stats["events_published"] = 100
        test_data_ingestion_service._stats["errors"] = 5
        test_data_ingestion_service._stats["circuit_breaker_trips"] = 2
        
        stats = test_data_ingestion_service.get_stats()
        
        # Calculate error rate
        error_rate = stats["errors"] / max(stats["events_published"], 1)
        assert error_rate == 0.05  # 5% error rate
        
        # Circuit breaker trip rate  
        cb_rate = stats["circuit_breaker_trips"] / max(stats["events_published"], 1)
        assert cb_rate == 0.02  # 2% circuit breaker trip rate


@pytest.mark.asyncio
async def test_data_ingestion_service_lifecycle(test_config_manager, mock_redis_client, 
                                             performance_thresholds):
    """Test complete DataIngestionService lifecycle."""
    with patch('src.realtime.data_ingestion.redis.Redis', return_value=mock_redis_client):
        service = MarketDataIngestionService(test_config_manager)
        
        # Test startup
        start_time = time.time()
        
        # Start service (but cancel quickly to avoid long-running test)
        start_task = asyncio.create_task(service.start())
        await asyncio.sleep(0.1)  # Let it start
        
        startup_time = time.time() - start_time
        assert startup_time < performance_thresholds["max_startup_time"]
        
        # Test running state
        assert service._running == True
        
        # Test shutdown
        stop_start_time = time.time()
        await service.stop()
        shutdown_time = time.time() - stop_start_time
        
        assert shutdown_time < performance_thresholds["max_shutdown_time"]
        assert service._running == False
        
        # Cancel the start task
        start_task.cancel()
        try:
            await start_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_data_ingestion_integration_scenario(test_data_ingestion_service, 
                                                 test_config_manager, message_factory):
    """Test complete data ingestion scenario."""
    # Setup: Enable symbols and providers
    symbols = test_config_manager.get_enabled_symbols()
    assert len(symbols) > 0
    
    # Test: Ingest data for each symbol
    for symbol in symbols[:2]:  # Test first 2 symbols
        # Create realistic market data
        ohlcv = OHLCV(
            timestamp=datetime.now(),
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000
        )
        
        # Publish market data
        await test_data_ingestion_service._publish_market_data_event(symbol, ohlcv, "test")
        
        # Publish calculation event
        calculation_results = {"rsi_14": 45.0, "sma_50": 150.0}
        await test_data_ingestion_service.publish_calculation_event(symbol, calculation_results)
        
        # Publish alert
        await test_data_ingestion_service.publish_alert("signal", symbol, "Test alert")
        
    # Verify: Check statistics
    stats = test_data_ingestion_service.get_stats()
    
    # Should have published events
    redis_client = test_data_ingestion_service._redis_client
    assert redis_client.xadd.call_count >= len(symbols[:2]) * 3  # 3 events per symbol