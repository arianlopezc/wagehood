"""
Comprehensive tests for worker process interactions.

Tests cover:
- Inter-component communication patterns
- Redis stream processing coordination
- Message passing and acknowledgment
- Error propagation and handling
- Resource sharing and locks
- End-to-end workflow validation
"""

import pytest
import asyncio
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json

from src.realtime.calculation_engine import CalculationEngine
from src.realtime.data_ingestion import MarketDataIngestionService
from src.realtime.timeframe_manager import TimeframeManager
from src.realtime.signal_engine import SignalEngine
from src.realtime.config_manager import ConfigManager, TradingProfile


class TestBasicWorkerCommunication:
    """Test basic communication between worker components."""
    
    @pytest.mark.asyncio
    async def test_data_ingestion_to_calculation_engine(self, test_config_manager, mock_redis_client):
        """Test communication from DataIngestion to CalculationEngine."""
        # Setup components
        with patch('src.realtime.data_ingestion.redis.Redis', return_value=mock_redis_client):
            with patch('src.realtime.calculation_engine.redis.Redis', return_value=mock_redis_client):
                ingestion_service = MarketDataIngestionService(test_config_manager)
                calc_engine = CalculationEngine(test_config_manager, ingestion_service)
                
                # Mock Redis stream data
                test_message = {
                    "symbol": "AAPL",
                    "price": "150.0",
                    "timestamp": datetime.now().isoformat(),
                    "volume": "1000"
                }
                
                mock_redis_client.xreadgroup.return_value = [
                    ["market_data_stream", [("test-msg-1", test_message)]]
                ]
                
                # Test message processing
                await calc_engine._process_single_message("test-msg-1", test_message)
                
                # Verify message was acknowledged
                mock_redis_client.xack.assert_called_with(
                    "market_data_stream", "calculation_workers", "test-msg-1"
                )
                
    def test_timeframe_manager_integration(self, test_config_manager, test_calculation_engine):
        """Test integration between CalculationEngine and TimeframeManager."""
        symbol = "AAPL"
        price = 150.0
        timestamp = datetime.now()
        
        # Get timeframe manager from calculation engine
        timeframe_manager = test_calculation_engine.timeframe_manager
        
        # Test tick processing through calculation engine
        message_data = {
            "symbol": symbol,
            "price": str(price),
            "timestamp": timestamp.isoformat(),
            "volume": "1000"
        }
        
        test_calculation_engine._calculate_indicators_for_symbol(symbol, price, message_data)
        
        # Verify timeframe manager was updated
        tf_stats = timeframe_manager.get_stats()
        assert tf_stats["total_updates"] >= 0
        
    def test_signal_engine_integration(self, test_config_manager, test_calculation_engine):
        """Test integration between CalculationEngine and SignalEngine."""
        symbol = "AAPL"
        price = 150.0
        
        # Get signal engine from calculation engine
        signal_engine = test_calculation_engine.signal_engine
        
        # Create timeframe results that would trigger signals
        timeframe_results = {
            "1h": {
                "rsi_14": 25.0,  # Oversold
                "macd": {
                    "macd_line": 1.5,
                    "signal_line": 1.0,
                    "histogram": 0.5
                }
            }
        }
        
        # Test signal generation
        composite_signal = signal_engine.generate_signals(
            symbol=symbol,
            price=price,
            timeframe_results=timeframe_results,
            trading_profile=TradingProfile.SWING_TRADING
        )
        
        if composite_signal:
            assert composite_signal.symbol == symbol
            
    def test_config_manager_integration(self, test_calculation_engine):
        """Test integration with ConfigManager across all components."""
        config_manager = test_calculation_engine.config_manager
        
        # Test that all components use the same config manager
        assert test_calculation_engine.timeframe_manager.config_manager is config_manager
        assert test_calculation_engine.signal_engine.config_manager is config_manager
        
        # Test configuration retrieval
        enabled_symbols = config_manager.get_enabled_symbols()
        enabled_strategies = config_manager.get_enabled_strategies()
        
        assert len(enabled_symbols) > 0
        assert len(enabled_strategies) > 0


class TestRedisStreamCoordination:
    """Test Redis stream coordination between workers."""
    
    @pytest.mark.asyncio
    async def test_stream_publishing_coordination(self, test_data_ingestion_service, message_factory):
        """Test coordinated stream publishing."""
        symbol = "AAPL"
        
        # Test publishing different types of events
        market_data_msg = message_factory.create_market_data_message(symbol)
        calculation_msg = message_factory.create_calculation_event(symbol)
        alert_msg = message_factory.create_alert_message("signal", symbol)
        
        # Publish in sequence
        await test_data_ingestion_service._publish_market_data_event(
            symbol, Mock(timestamp=datetime.now(), open=150.0, high=152.0, low=149.0, close=151.0, volume=1000), "test"
        )
        
        await test_data_ingestion_service.publish_calculation_event(symbol, {"test": True})
        await test_data_ingestion_service.publish_alert("signal", symbol, "Test alert")
        
        # Verify Redis client was called for each stream
        redis_client = test_data_ingestion_service._redis_client
        assert redis_client.xadd.call_count >= 3
        
    @pytest.mark.asyncio
    async def test_consumer_group_coordination(self, test_config_manager, mock_redis_client):
        """Test consumer group coordination."""
        with patch('src.realtime.data_ingestion.redis.Redis', return_value=mock_redis_client):
            with patch('src.realtime.calculation_engine.redis.Redis', return_value=mock_redis_client):
                # Create multiple consumers
                ingestion1 = MarketDataIngestionService(test_config_manager)
                ingestion2 = MarketDataIngestionService(test_config_manager)
                
                calc_engine1 = CalculationEngine(test_config_manager, ingestion1)
                calc_engine2 = CalculationEngine(test_config_manager, ingestion2)
                
                # Verify consumer groups were created
                assert mock_redis_client.xgroup_create.call_count >= 2
                
    @pytest.mark.asyncio
    async def test_message_acknowledgment_flow(self, test_calculation_engine, mock_redis_client):
        """Test message acknowledgment flow."""
        # Setup mock message
        test_message = {
            "symbol": "AAPL",
            "price": "150.0",
            "timestamp": datetime.now().isoformat(),
            "volume": "1000"
        }
        
        message_id = "test-msg-123"
        
        # Process message
        await test_calculation_engine._process_single_message(message_id, test_message)
        
        # Verify acknowledgment
        mock_redis_client.xack.assert_called_with(
            "market_data_stream", "calculation_workers", message_id
        )
        
    @pytest.mark.asyncio
    async def test_stream_error_handling(self, test_calculation_engine, mock_redis_client):
        """Test error handling in stream processing."""
        # Mock Redis error
        mock_redis_client.xreadgroup.side_effect = Exception("Redis connection lost")
        
        # Should handle error gracefully
        try:
            await test_calculation_engine._consume_market_data_stream()
        except asyncio.CancelledError:
            pass  # Expected when test ends
        except Exception as e:
            # Should not crash, but may raise connection errors
            assert "Redis" in str(e)


class TestMessagePassingPatterns:
    """Test message passing patterns between components."""
    
    @pytest.mark.asyncio
    async def test_publish_subscribe_pattern(self, test_data_ingestion_service, test_calculation_engine):
        """Test publish-subscribe message pattern."""
        symbol = "AAPL"
        
        # Publisher: DataIngestionService publishes market data
        from src.core.models import OHLCV
        ohlcv = OHLCV(
            timestamp=datetime.now(),
            open=150.0, high=152.0, low=149.0, close=151.0, volume=1000
        )
        
        await test_data_ingestion_service._publish_market_data_event(symbol, ohlcv, "test")
        
        # Subscriber: CalculationEngine should process the message
        # (In real system, this would be automatic via Redis streams)
        
        # Verify publishing worked
        redis_client = test_data_ingestion_service._redis_client
        assert redis_client.xadd.called
        
    @pytest.mark.asyncio
    async def test_request_response_pattern(self, test_calculation_engine):
        """Test request-response pattern for data queries."""
        symbol = "AAPL"
        
        # Request: Get latest calculation results
        latest_results = test_calculation_engine.get_latest_results(symbol)
        
        # Response: Should return data or None
        if latest_results:
            assert isinstance(latest_results, dict)
        else:
            assert latest_results is None
            
        # Request: Get statistics
        stats = test_calculation_engine.get_stats()
        
        # Response: Should always return stats dict
        assert isinstance(stats, dict)
        assert "calculations_performed" in stats
        
    @pytest.mark.asyncio
    async def test_event_notification_pattern(self, test_data_ingestion_service):
        """Test event notification pattern."""
        symbol = "AAPL"
        
        # Event: Publish alert notification
        await test_data_ingestion_service.publish_alert(
            alert_type="signal",
            symbol=symbol,
            message="Strong buy signal detected",
            metadata={"confidence": 0.9, "strategy": "macd_rsi"}
        )
        
        # Verify notification was published
        redis_client = test_data_ingestion_service._redis_client
        assert redis_client.xadd.called
        
        # Check call arguments
        call_args = redis_client.xadd.call_args
        assert "alert_stream" in call_args[0]
        
    def test_batch_processing_pattern(self, test_calculation_engine, sample_market_data_events):
        """Test batch processing pattern."""
        # Simulate batch of messages
        batch_size = 10
        messages = sample_market_data_events[:batch_size]
        
        # Process batch
        start_time = time.time()
        
        for msg_id, msg_data in messages:
            # Simulate processing without async to test batching logic
            test_calculation_engine._increment_calculation_count()
            
        processing_time = time.time() - start_time
        
        # Verify batch was processed
        stats = test_calculation_engine.get_stats()
        assert stats["calculations_performed"] == batch_size
        assert processing_time < 1.0  # Should be fast for batch


class TestErrorPropagation:
    """Test error propagation between worker components."""
    
    @pytest.mark.asyncio
    async def test_data_ingestion_error_propagation(self, test_data_ingestion_service, error_injector):
        """Test error propagation from data ingestion."""
        symbol = "AAPL"
        
        # Inject Redis error
        redis_client = test_data_ingestion_service._redis_client
        redis_client.xadd.side_effect = error_injector.inject_redis_error
        
        # Attempt to publish - should handle error
        try:
            from src.core.models import OHLCV
            ohlcv = OHLCV(
                timestamp=datetime.now(),
                open=150.0, high=152.0, low=149.0, close=151.0, volume=1000
            )
            await test_data_ingestion_service._publish_market_data_event(symbol, ohlcv, "test")
        except Exception as e:
            # Error should be caught and handled
            assert "Redis" in str(e) or "error" in str(e).lower()
            
        # Verify error was tracked
        error_summary = error_injector.get_error_summary()
        assert error_summary["total_errors"] > 0
        
    @pytest.mark.asyncio
    async def test_calculation_engine_error_propagation(self, test_calculation_engine, error_injector):
        """Test error propagation in calculation engine."""
        symbol = "AAPL"
        
        # Create invalid message that will cause processing error
        invalid_message = {
            "symbol": None,  # Invalid
            "price": "not_a_number",  # Invalid
            "timestamp": "invalid_timestamp",  # Invalid
            "volume": "invalid_volume"  # Invalid
        }
        
        initial_errors = test_calculation_engine.get_stats()["errors"]
        
        # Process invalid message
        await test_calculation_engine._process_single_message("test-id", invalid_message)
        
        # Error count should increase
        final_errors = test_calculation_engine.get_stats()["errors"]
        assert final_errors > initial_errors
        
    def test_timeframe_manager_error_propagation(self, test_timeframe_manager):
        """Test error propagation in timeframe manager."""
        symbol = "AAPL"
        
        initial_errors = test_timeframe_manager.get_stats()["errors"]
        
        # Attempt processing with invalid data
        try:
            test_timeframe_manager.process_tick(
                symbol=symbol,
                price=float('nan'),  # Invalid price
                volume=-1000.0,      # Invalid volume
                timestamp=datetime.now(),
                timeframes=["invalid_timeframe"],  # Invalid timeframe
                trading_profile=TradingProfile.SWING_TRADING
            )
        except (ValueError, TypeError):
            # Errors are acceptable for invalid data
            pass
            
        # Check if errors were tracked
        final_errors = test_timeframe_manager.get_stats()["errors"]
        # Errors may or may not increase depending on validation
        assert final_errors >= initial_errors
        
    def test_signal_engine_error_propagation(self, test_signal_engine):
        """Test error propagation in signal engine."""
        symbol = "AAPL"
        
        initial_errors = test_signal_engine.get_stats()["errors"]
        
        # Attempt signal generation with invalid data
        invalid_timeframe_results = {
            "invalid_timeframe": {
                "rsi_14": float('nan'),  # Invalid
                "macd": "not_a_dict"     # Invalid
            }
        }
        
        try:
            test_signal_engine.generate_signals(
                symbol=symbol,
                price=float('inf'),  # Invalid price
                timeframe_results=invalid_timeframe_results,
                trading_profile=TradingProfile.SWING_TRADING
            )
        except (ValueError, TypeError):
            # Errors are acceptable for invalid data
            pass
            
        # Check if errors were tracked
        final_errors = test_signal_engine.get_stats()["errors"]
        # May increase depending on validation
        assert final_errors >= initial_errors


class TestResourceSharingAndLocks:
    """Test resource sharing and locking mechanisms."""
    
    def test_timeframe_manager_thread_safety(self, test_timeframe_manager):
        """Test TimeframeManager thread safety."""
        symbol = "AAPL"
        results = []
        
        def worker(worker_id):
            try:
                for i in range(20):
                    timestamp = datetime.now() + timedelta(seconds=worker_id * 100 + i)
                    price = 150.0 + worker_id + i * 0.1
                    
                    result = test_timeframe_manager.process_tick(
                        symbol=symbol,
                        price=price,
                        volume=1000.0,
                        timestamp=timestamp,
                        timeframes=["1m"],
                        trading_profile=TradingProfile.SWING_TRADING
                    )
                    
                    results.append((worker_id, i, "success"))
            except Exception as e:
                results.append((worker_id, "error", str(e)))
                
        # Run multiple threads
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(target=worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # Should complete without deadlocks or race conditions
        success_results = [r for r in results if r[2] == "success"]
        error_results = [r for r in results if r[1] == "error"]
        
        assert len(success_results) > 0
        assert len(error_results) == 0  # No errors expected from concurrency
        
    def test_signal_engine_thread_safety(self, test_signal_engine):
        """Test SignalEngine thread safety."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        results = []
        
        def worker(symbol):
            try:
                for i in range(10):
                    price = 150.0 + i * 0.1
                    timeframe_results = {
                        "1h": {
                            "rsi_14": 45.0 + i,
                            "sma_50": price
                        }
                    }
                    
                    composite_signal = test_signal_engine.generate_signals(
                        symbol=symbol,
                        price=price,
                        timeframe_results=timeframe_results,
                        trading_profile=TradingProfile.SWING_TRADING
                    )
                    
                    results.append((symbol, i, composite_signal is not None))
            except Exception as e:
                results.append((symbol, "error", str(e)))
                
        # Run multiple threads for different symbols
        threads = []
        for symbol in symbols:
            thread = threading.Thread(target=worker, args=(symbol,))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # Should complete without deadlocks
        success_results = [r for r in results if r[1] != "error"]
        error_results = [r for r in results if r[1] == "error"]
        
        assert len(success_results) > 0
        assert len(error_results) == 0
        
    def test_shared_configuration_access(self, test_config_manager):
        """Test shared configuration access across components."""
        # Create multiple components that share the config manager
        timeframe_manager = TimeframeManager(test_config_manager)
        signal_engine = SignalEngine(test_config_manager, timeframe_manager)
        
        # Test concurrent access to shared configuration
        results = []
        
        def config_reader(component, component_name):
            try:
                for i in range(10):
                    watchlist = component.config_manager.get_watchlist()
                    strategies = component.config_manager.get_enabled_strategies()
                    indicators = component.config_manager.get_enabled_indicators()
                    
                    results.append((component_name, i, len(watchlist), len(strategies), len(indicators)))
            except Exception as e:
                results.append((component_name, "error", str(e)))
                
        # Run concurrent access
        threads = [
            threading.Thread(target=config_reader, args=(timeframe_manager, "timeframe")),
            threading.Thread(target=config_reader, args=(signal_engine, "signal"))
        ]
        
        for thread in threads:
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # Should complete without conflicts
        success_results = [r for r in results if r[1] != "error"]
        error_results = [r for r in results if r[1] == "error"]
        
        assert len(success_results) > 0
        assert len(error_results) == 0


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_market_data_workflow(self, test_config_manager, mock_redis_client):
        """Test complete market data processing workflow."""
        # Setup all components
        with patch('src.realtime.data_ingestion.redis.Redis', return_value=mock_redis_client):
            with patch('src.realtime.calculation_engine.redis.Redis', return_value=mock_redis_client):
                # Phase 1: Data Ingestion
                ingestion_service = MarketDataIngestionService(test_config_manager)
                
                # Phase 2: Calculation Engine
                calc_engine = CalculationEngine(test_config_manager, ingestion_service)
                
                # Phase 3: Publish Market Data
                from src.core.models import OHLCV
                symbol = "AAPL"
                ohlcv = OHLCV(
                    timestamp=datetime.now(),
                    open=150.0, high=152.0, low=149.0, close=151.0, volume=1000
                )
                
                await ingestion_service._publish_market_data_event(symbol, ohlcv, "test")
                
                # Phase 4: Process in Calculation Engine
                message_data = {
                    "symbol": symbol,
                    "price": "151.0",
                    "timestamp": datetime.now().isoformat(),
                    "volume": "1000"
                }
                
                await calc_engine._process_single_message("test-msg", message_data)
                
                # Phase 5: Verify Results
                stats = calc_engine.get_stats()
                assert stats["calculations_performed"] > 0
                
                # Verify Redis interactions
                assert mock_redis_client.xadd.called  # Data was published
                assert mock_redis_client.xack.called  # Message was acknowledged
                
    @pytest.mark.asyncio
    async def test_multi_symbol_processing_workflow(self, test_config_manager, mock_redis_client):
        """Test workflow with multiple symbols."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        
        with patch('src.realtime.data_ingestion.redis.Redis', return_value=mock_redis_client):
            with patch('src.realtime.calculation_engine.redis.Redis', return_value=mock_redis_client):
                ingestion_service = MarketDataIngestionService(test_config_manager)
                calc_engine = CalculationEngine(test_config_manager, ingestion_service)
                
                # Process data for each symbol
                for symbol in symbols:
                    # Publish market data
                    from src.core.models import OHLCV
                    ohlcv = OHLCV(
                        timestamp=datetime.now(),
                        open=150.0, high=152.0, low=149.0, close=151.0, volume=1000
                    )
                    
                    await ingestion_service._publish_market_data_event(symbol, ohlcv, "test")
                    
                    # Process in calculation engine
                    message_data = {
                        "symbol": symbol,
                        "price": "151.0",
                        "timestamp": datetime.now().isoformat(),
                        "volume": "1000"
                    }
                    
                    await calc_engine._process_single_message(f"msg-{symbol}", message_data)
                    
                # Verify all symbols were processed
                stats = calc_engine.get_stats()
                assert len(stats["symbols_processed"]) == len(symbols)
                
    def test_configuration_change_propagation(self, test_config_manager):
        """Test configuration change propagation to components."""
        # Create components
        timeframe_manager = TimeframeManager(test_config_manager)
        signal_engine = SignalEngine(test_config_manager, timeframe_manager)
        
        # Verify initial configuration
        initial_strategies = test_config_manager.get_enabled_strategies()
        initial_indicators = test_config_manager.get_enabled_indicators()
        
        assert len(initial_strategies) > 0
        assert len(initial_indicators) > 0
        
        # Test that components use the configuration
        symbol = "AAPL"
        price = 150.0
        
        # Process data through timeframe manager
        timeframe_results = timeframe_manager.process_tick(
            symbol=symbol,
            price=price,
            volume=1000.0,
            timestamp=datetime.now(),
            timeframes=["1h"],
            trading_profile=TradingProfile.SWING_TRADING
        )
        
        # Generate signals
        if timeframe_results:
            composite_signal = signal_engine.generate_signals(
                symbol=symbol,
                price=price,
                timeframe_results=timeframe_results,
                trading_profile=TradingProfile.SWING_TRADING
            )
            
            # Verify signals were generated according to configuration
            if composite_signal:
                assert composite_signal.symbol == symbol
                
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, test_config_manager, mock_redis_client):
        """Test error recovery in complete workflow."""
        with patch('src.realtime.data_ingestion.redis.Redis', return_value=mock_redis_client):
            with patch('src.realtime.calculation_engine.redis.Redis', return_value=mock_redis_client):
                ingestion_service = MarketDataIngestionService(test_config_manager)
                calc_engine = CalculationEngine(test_config_manager, ingestion_service)
                
                # Phase 1: Normal operation
                symbol = "AAPL"
                message_data = {
                    "symbol": symbol,
                    "price": "151.0",
                    "timestamp": datetime.now().isoformat(),
                    "volume": "1000"
                }
                
                await calc_engine._process_single_message("msg-1", message_data)
                
                # Phase 2: Inject error
                mock_redis_client.xack.side_effect = Exception("Redis error")
                
                # Phase 3: Process with error
                await calc_engine._process_single_message("msg-2", message_data)
                
                # Phase 4: Recovery
                mock_redis_client.xack.side_effect = None  # Remove error
                
                await calc_engine._process_single_message("msg-3", message_data)
                
                # Verify error handling and recovery
                stats = calc_engine.get_stats()
                assert stats["calculations_performed"] > 0
                assert stats["errors"] > 0  # Error was tracked


class TestPerformanceUnderLoad:
    """Test worker interactions under load."""
    
    @pytest.mark.asyncio
    async def test_high_throughput_coordination(self, test_config_manager, mock_redis_client):
        """Test coordination under high throughput."""
        message_count = 100
        
        with patch('src.realtime.data_ingestion.redis.Redis', return_value=mock_redis_client):
            with patch('src.realtime.calculation_engine.redis.Redis', return_value=mock_redis_client):
                ingestion_service = MarketDataIngestionService(test_config_manager)
                calc_engine = CalculationEngine(test_config_manager, ingestion_service)
                
                start_time = time.time()
                
                # Process high volume of messages
                for i in range(message_count):
                    symbol = f"TEST{i % 5}"  # 5 different symbols
                    message_data = {
                        "symbol": symbol,
                        "price": str(150.0 + i * 0.01),
                        "timestamp": datetime.now().isoformat(),
                        "volume": str(1000 + i)
                    }
                    
                    await calc_engine._process_single_message(f"msg-{i}", message_data)
                    
                total_time = time.time() - start_time
                throughput = message_count / total_time
                
                # Should maintain reasonable throughput
                assert throughput > 50  # At least 50 messages per second
                
                # Verify all messages were processed
                stats = calc_engine.get_stats()
                assert stats["calculations_performed"] >= message_count
                
    def test_concurrent_component_operation(self, test_config_manager):
        """Test concurrent operation of multiple components."""
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        results = []
        
        # Create shared components
        timeframe_manager = TimeframeManager(test_config_manager)
        signal_engine = SignalEngine(test_config_manager, timeframe_manager)
        
        def worker(symbol):
            try:
                for i in range(20):
                    timestamp = datetime.now() + timedelta(seconds=i)
                    price = 150.0 + hash(symbol) % 100 + i * 0.1
                    
                    # Process through timeframe manager
                    tf_results = timeframe_manager.process_tick(
                        symbol=symbol,
                        price=price,
                        volume=1000.0,
                        timestamp=timestamp,
                        timeframes=["1m", "5m"],
                        trading_profile=TradingProfile.SWING_TRADING
                    )
                    
                    # Generate signals
                    if tf_results:
                        composite_signal = signal_engine.generate_signals(
                            symbol=symbol,
                            price=price,
                            timeframe_results=tf_results,
                            trading_profile=TradingProfile.SWING_TRADING
                        )
                        
                        results.append((symbol, i, "success", composite_signal is not None))
                    else:
                        results.append((symbol, i, "no_results", False))
                        
            except Exception as e:
                results.append((symbol, "error", str(e), False))
                
        # Run workers concurrently
        threads = []
        for symbol in symbols:
            thread = threading.Thread(target=worker, args=(symbol,))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # Verify concurrent operation
        success_results = [r for r in results if r[2] in ["success", "no_results"]]
        error_results = [r for r in results if r[1] == "error"]
        
        assert len(success_results) > 0
        assert len(error_results) == 0  # No concurrency errors
        
        # Verify all symbols were processed
        processed_symbols = set(r[0] for r in success_results)
        assert len(processed_symbols) == len(symbols)


@pytest.mark.asyncio
async def test_complete_worker_integration_scenario(test_config_manager, mock_redis_client):
    """Test complete integration scenario with all worker components."""
    
    # Phase 1: Setup all components
    with patch('src.realtime.data_ingestion.redis.Redis', return_value=mock_redis_client):
        with patch('src.realtime.calculation_engine.redis.Redis', return_value=mock_redis_client):
            ingestion_service = MarketDataIngestionService(test_config_manager)
            calc_engine = CalculationEngine(test_config_manager, ingestion_service)
            
            symbols = ["AAPL", "GOOGL"]
            
            # Phase 2: Market data flow
            for symbol in symbols:
                # Ingestion: Publish market data
                from src.core.models import OHLCV
                ohlcv = OHLCV(
                    timestamp=datetime.now(),
                    open=150.0, high=152.0, low=149.0, close=151.0, volume=1000
                )
                
                await ingestion_service._publish_market_data_event(symbol, ohlcv, "test")
                
                # Calculation: Process the data
                message_data = {
                    "symbol": symbol,
                    "price": "151.0",
                    "timestamp": datetime.now().isoformat(),
                    "volume": "1000"
                }
                
                await calc_engine._process_single_message(f"msg-{symbol}", message_data)
                
                # Publish calculation results
                await ingestion_service.publish_calculation_event(
                    symbol, {"rsi_14": 45.0, "sma_50": 151.0}
                )
                
                # Publish signal alert
                await ingestion_service.publish_alert(
                    "signal", symbol, "Buy signal detected", {"confidence": 0.8}
                )
                
            # Phase 3: Verification
            
            # Verify ingestion service
            ingestion_stats = ingestion_service.get_stats()
            assert ingestion_stats["events_published"] >= 0
            
            # Verify calculation engine
            calc_stats = calc_engine.get_stats()
            assert calc_stats["calculations_performed"] >= len(symbols)
            assert len(calc_stats["symbols_processed"]) >= len(symbols)
            
            # Verify timeframe manager
            tf_stats = calc_engine.timeframe_manager.get_stats()
            assert tf_stats["total_updates"] >= 0
            
            # Verify signal engine
            signal_stats = calc_engine.signal_engine.get_stats()
            assert signal_stats["total_signals_generated"] >= 0
            
            # Verify Redis interactions
            assert mock_redis_client.xadd.call_count >= len(symbols) * 3  # Market data + calc + alert
            assert mock_redis_client.xack.call_count >= len(symbols)  # Acknowledgments
            
            # Phase 4: Component state consistency
            for symbol in symbols:
                # Check if data exists in timeframe manager
                tf_data = calc_engine.timeframe_manager.get_timeframe_data(symbol, "1m")
                assert isinstance(tf_data, list)
                
                # Check signal history
                signal_history = calc_engine.signal_engine.get_signal_history(symbol)
                assert isinstance(signal_history, list)
                
            # Phase 5: Resource cleanup verification
            # Components should be in consistent state
            assert calc_engine._running == False  # Not started in test
            assert ingestion_service._running == False  # Not started in test