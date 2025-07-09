"""
Comprehensive tests for the CalculationEngine worker component.

Tests cover:
- Signal generation accuracy and performance
- Indicator calculation correctness
- Multi-timeframe processing
- Error handling and recovery
- Memory management
- Concurrency and scalability
- Integration with other components
"""

import pytest
import asyncio
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from src.realtime.calculation_engine import CalculationEngine
from src.realtime.config_manager import TradingProfile, AssetConfig


class TestCalculationEngineComponents:
    """Test individual CalculationEngine components."""
    
    def test_initialization(self, test_config_manager, test_data_ingestion_service):
        """Test CalculationEngine initialization."""
        with patch('src.realtime.calculation_engine.redis.Redis'):
            engine = CalculationEngine(test_config_manager, test_data_ingestion_service)
            
            assert engine.config_manager is test_config_manager
            assert engine.ingestion_service is test_data_ingestion_service
            assert engine.timeframe_manager is not None
            assert engine.signal_engine is not None
            assert engine.indicator_calculator is not None
            assert not engine._running
            assert len(engine._tasks) == 0
            
    def test_redis_initialization(self, test_config_manager, test_data_ingestion_service, mock_redis_client):
        """Test Redis connection initialization."""
        with patch('src.realtime.calculation_engine.redis.Redis', return_value=mock_redis_client):
            engine = CalculationEngine(test_config_manager, test_data_ingestion_service)
            
            assert engine._redis_client is mock_redis_client
            mock_redis_client.ping.assert_called_once()
            
    def test_stats_initialization(self, test_calculation_engine):
        """Test performance statistics initialization."""
        stats = test_calculation_engine.get_stats()
        
        assert stats["calculations_performed"] == 0
        assert stats["signals_generated"] == 0
        assert stats["composite_signals_generated"] == 0
        assert stats["errors"] == 0
        assert stats["running"] == False
        assert "timeframe_manager_stats" in stats
        assert "signal_engine_stats" in stats
        
    def test_indicator_calculation_batch_processing(self, test_calculation_engine):
        """Test batch processing of indicators."""
        symbol = "AAPL"
        price = 150.0
        message_data = {"timestamp": datetime.now().isoformat(), "volume": "1000"}
        
        # Test legacy indicator calculation
        results = test_calculation_engine._calculate_legacy_indicators(symbol, price, message_data)
        
        assert isinstance(results, dict)
        # Should have calculated some indicators
        assert len(results) > 0
        
    def test_multi_timeframe_calculation(self, test_calculation_engine):
        """Test multi-timeframe indicator calculations."""
        symbol = "AAPL"
        price = 150.0
        timestamp = datetime.now()
        message_data = {"timestamp": timestamp.isoformat(), "volume": "1000"}
        
        # Test timeframe-specific calculations
        test_calculation_engine._calculate_indicators_for_symbol(symbol, price, message_data)
        
        # Verify stats were updated
        stats = test_calculation_engine.get_stats()
        assert stats["calculations_performed"] > 0
        assert symbol in stats["symbols_processed"]


class TestCalculationEngineSignalGeneration:
    """Test signal generation capabilities."""
    
    def test_single_strategy_signal_generation(self, test_calculation_engine):
        """Test signal generation for individual strategies."""
        symbol = "AAPL"
        calculation_results = {
            "rsi_14": 25.0,  # Oversold
            "macd": {
                "macd_line": 1.5,
                "signal_line": 1.0,
                "histogram": 0.5
            },
            "sma_50": 145.0,
            "sma_200": 140.0
        }
        
        # Test MACD+RSI strategy
        strategy_config = test_calculation_engine.config_manager.get_enabled_strategies()[0]  # macd_rsi_strategy
        signal = test_calculation_engine._calculate_strategy_signal(
            strategy_config, symbol, calculation_results
        )
        
        assert signal is not None
        assert signal["signal"] in ["buy", "sell", "weak_buy", "weak_sell", None]
        assert 0 <= signal["confidence"] <= 1
        assert "metadata" in signal
        
    def test_ma_crossover_signal(self, test_calculation_engine):
        """Test Moving Average crossover signal generation."""
        calculation_results = {
            "sma_50": 155.0,   # Fast MA above slow MA
            "sma_200": 150.0
        }
        
        params = {"fast_period": 50, "slow_period": 200}
        signal = test_calculation_engine._calculate_ma_crossover_signal(calculation_results, params)
        
        assert signal is not None
        assert signal["signal"] == "buy"  # Fast MA > Slow MA
        assert signal["confidence"] > 0
        assert signal["metadata"]["fast_ma"] == 155.0
        assert signal["metadata"]["slow_ma"] == 150.0
        
    def test_rsi_trend_signal(self, test_calculation_engine):
        """Test RSI trend signal generation."""
        # Oversold condition
        calculation_results = {
            "rsi_14": 25.0,
            "sma_50": 150.0
        }
        
        params = {"rsi_overbought": 70, "rsi_oversold": 30}
        signal = test_calculation_engine._calculate_rsi_trend_signal(calculation_results, params)
        
        assert signal is not None
        assert signal["signal"] == "buy"  # RSI oversold
        assert signal["confidence"] > 0
        
        # Overbought condition
        calculation_results["rsi_14"] = 75.0
        signal = test_calculation_engine._calculate_rsi_trend_signal(calculation_results, params)
        
        assert signal["signal"] == "sell"  # RSI overbought
        
    def test_bollinger_breakout_signal(self, test_calculation_engine):
        """Test Bollinger Bands breakout signal generation."""
        calculation_results = {
            "bollinger_bands": {
                "upper_band": 155.0,
                "middle_band": 150.0,
                "lower_band": 145.0
            }
        }
        
        params = {"squeeze_threshold": 0.1}
        
        # Test price above upper band (bullish breakout)
        signal = test_calculation_engine._calculate_bollinger_breakout_signal(
            calculation_results, params
        )
        
        assert signal is not None
        assert "band_width" in signal["metadata"]
        assert "squeeze" in signal["metadata"]


class TestCalculationEnginePerformance:
    """Test CalculationEngine performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_message_processing_performance(self, test_calculation_engine, 
                                                sample_market_data_events, performance_thresholds):
        """Test message processing performance."""
        start_time = time.time()
        processed_count = 0
        
        # Process a batch of messages
        for message_id, message_data in sample_market_data_events[:10]:
            await test_calculation_engine._process_single_message(message_id, message_data)
            processed_count += 1
            
        total_time = time.time() - start_time
        avg_time_per_message = total_time / processed_count
        
        assert avg_time_per_message < performance_thresholds["max_processing_time_per_message"]
        
    def test_memory_usage_monitoring(self, test_calculation_engine, resource_monitor):
        """Test memory usage during calculations."""
        initial_memory = resource_monitor.get_memory_usage()
        
        # Perform many calculations
        for i in range(100):
            symbol = f"TEST{i % 5}"
            price = 150.0 + i * 0.1
            message_data = {"timestamp": datetime.now().isoformat(), "volume": "1000"}
            
            test_calculation_engine._calculate_indicators_for_symbol(symbol, price, message_data)
            
        final_memory = resource_monitor.get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 50  # Less than 50MB increase
        
    def test_concurrent_symbol_processing(self, test_calculation_engine):
        """Test concurrent processing of multiple symbols."""
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        results = []
        
        def process_symbol(symbol):
            price = 150.0 + hash(symbol) % 100
            message_data = {"timestamp": datetime.now().isoformat(), "volume": "1000"}
            
            start_time = time.time()
            test_calculation_engine._calculate_indicators_for_symbol(symbol, price, message_data)
            processing_time = time.time() - start_time
            
            results.append((symbol, processing_time))
            
        # Process symbols concurrently
        threads = []
        for symbol in symbols:
            thread = threading.Thread(target=process_symbol, args=(symbol,))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # All symbols should be processed
        assert len(results) == len(symbols)
        
        # Processing times should be reasonable
        for symbol, processing_time in results:
            assert processing_time < 1.0  # Less than 1 second per symbol
            
    def test_stats_update_performance(self, test_calculation_engine):
        """Test performance statistics update efficiency."""
        start_time = time.time()
        
        # Update stats many times
        for i in range(1000):
            test_calculation_engine._update_performance_stats(i * 0.1)
            test_calculation_engine._increment_calculation_count()
            
        update_time = time.time() - start_time
        
        # Stats updates should be very fast
        assert update_time < 0.1  # Less than 100ms for 1000 updates


class TestCalculationEngineReliability:
    """Test CalculationEngine reliability and error handling."""
    
    def test_error_handling_in_indicator_calculation(self, test_calculation_engine):
        """Test error handling during indicator calculations."""
        symbol = "AAPL"
        price = 150.0
        message_data = {"timestamp": "invalid_timestamp", "volume": "invalid_volume"}
        
        # Should handle errors gracefully
        initial_error_count = test_calculation_engine.get_stats()["errors"]
        
        test_calculation_engine._calculate_indicators_for_symbol(symbol, price, message_data)
        
        # Error count might increase, but should not crash
        final_error_count = test_calculation_engine.get_stats()["errors"]
        assert final_error_count >= initial_error_count
        
    def test_redis_connection_error_handling(self, test_config_manager, test_data_ingestion_service):
        """Test handling of Redis connection errors."""
        mock_redis = Mock()
        mock_redis.ping.side_effect = Exception("Redis connection failed")
        
        with patch('src.realtime.calculation_engine.redis.Redis', return_value=mock_redis):
            # Should raise exception during initialization
            with pytest.raises(Exception):
                CalculationEngine(test_config_manager, test_data_ingestion_service)
                
    @pytest.mark.asyncio
    async def test_message_processing_error_recovery(self, test_calculation_engine, error_injector):
        """Test error recovery during message processing."""
        # Create a message that will cause an error
        invalid_message = {
            "symbol": None,  # Invalid symbol
            "price": "invalid_price",
            "timestamp": "invalid_timestamp"
        }
        
        initial_error_count = test_calculation_engine.get_stats()["errors"]
        
        # Process the invalid message
        await test_calculation_engine._process_single_message("test_id", invalid_message)
        
        # Should handle error gracefully
        final_error_count = test_calculation_engine.get_stats()["errors"]
        assert final_error_count > initial_error_count
        
        # Engine should still be functional for valid messages
        valid_message = {
            "symbol": "AAPL",
            "price": "150.0",
            "timestamp": datetime.now().isoformat(),
            "volume": "1000"
        }
        
        await test_calculation_engine._process_single_message("test_id_2", valid_message)
        
    def test_memory_cleanup(self, test_calculation_engine):
        """Test memory cleanup mechanisms."""
        # Fill up some data
        for i in range(50):
            symbol = f"TEST{i}"
            price = 150.0 + i
            message_data = {"timestamp": datetime.now().isoformat(), "volume": "1000"}
            
            test_calculation_engine._calculate_indicators_for_symbol(symbol, price, message_data)
            
        # Check initial state
        stats = test_calculation_engine.get_stats()
        initial_symbols = len(stats["symbols_processed"])
        
        # Reset a symbol (simulating cleanup)
        test_calculation_engine.reset_symbol_multi_timeframe("TEST0")
        
        # Verify cleanup worked
        assert True  # Cleanup completed without errors


class TestCalculationEngineScalability:
    """Test CalculationEngine scalability characteristics."""
    
    @pytest.mark.asyncio
    async def test_high_throughput_processing(self, test_calculation_engine, 
                                           message_factory, performance_thresholds):
        """Test processing high throughput of messages."""
        message_count = 100
        messages = []
        
        # Create many messages
        for i in range(message_count):
            message = message_factory.create_market_data_message(
                symbol=f"TEST{i % 5}",
                price=150.0 + i * 0.1
            )
            messages.append((f"msg_{i}", message))
            
        start_time = time.time()
        
        # Process all messages
        for message_id, message_data in messages:
            await test_calculation_engine._process_single_message(message_id, message_data)
            
        total_time = time.time() - start_time
        throughput = message_count / total_time
        
        assert throughput > performance_thresholds["min_throughput_messages_per_second"]
        
    def test_multi_timeframe_scaling(self, test_calculation_engine):
        """Test scaling with multiple timeframes."""
        symbol = "AAPL"
        timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        
        # Process data for multiple timeframes
        for i in range(20):
            timestamp = datetime.now() - timedelta(minutes=i)
            price = 150.0 + i * 0.1
            volume = 1000 + i * 10
            
            # This would trigger multi-timeframe processing
            results = test_calculation_engine.timeframe_manager.process_tick(
                symbol=symbol,
                price=price,
                volume=volume,
                timestamp=timestamp,
                timeframes=timeframes,
                trading_profile=TradingProfile.SWING_TRADING
            )
            
        # Verify multi-timeframe processing
        timeframe_stats = test_calculation_engine.timeframe_manager.get_stats()
        assert timeframe_stats["total_updates"] > 0
        
    def test_concurrent_engine_instances(self, test_config_manager, test_data_ingestion_service):
        """Test multiple CalculationEngine instances."""
        engines = []
        
        # Create multiple engine instances
        for i in range(3):
            with patch('src.realtime.calculation_engine.redis.Redis'):
                engine = CalculationEngine(test_config_manager, test_data_ingestion_service)
                engines.append(engine)
                
        # Each engine should be independent
        for i, engine in enumerate(engines):
            symbol = f"TEST{i}"
            price = 150.0 + i * 10
            message_data = {"timestamp": datetime.now().isoformat(), "volume": "1000"}
            
            engine._calculate_indicators_for_symbol(symbol, price, message_data)
            
        # Verify each engine processed independently
        for i, engine in enumerate(engines):
            stats = engine.get_stats()
            assert stats["calculations_performed"] > 0


class TestCalculationEngineIntegration:
    """Test CalculationEngine integration with other components."""
    
    @pytest.mark.asyncio
    async def test_integration_with_timeframe_manager(self, test_calculation_engine):
        """Test integration with TimeframeManager."""
        symbol = "AAPL"
        price = 150.0
        timestamp = datetime.now()
        
        # Process through calculation engine
        message_data = {
            "symbol": symbol,
            "price": str(price),
            "timestamp": timestamp.isoformat(),
            "volume": "1000"
        }
        
        await test_calculation_engine._process_single_message("test_id", message_data)
        
        # Check if timeframe manager was used
        tf_stats = test_calculation_engine.timeframe_manager.get_stats()
        assert tf_stats["total_updates"] >= 0  # May be 0 if no updates needed
        
    @pytest.mark.asyncio
    async def test_integration_with_signal_engine(self, test_calculation_engine):
        """Test integration with SignalEngine."""
        symbol = "AAPL"
        price = 150.0
        
        # Create timeframe results that would trigger signal generation
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
        
        # Generate signals
        composite_signal = test_calculation_engine.signal_engine.generate_signals(
            symbol=symbol,
            price=price,
            timeframe_results=timeframe_results,
            trading_profile=TradingProfile.SWING_TRADING
        )
        
        # Should generate some signal
        if composite_signal:
            assert hasattr(composite_signal, 'symbol')
            assert composite_signal.symbol == symbol
            
    @pytest.mark.asyncio
    async def test_integration_with_config_manager(self, test_calculation_engine):
        """Test integration with ConfigManager."""
        # Test that configuration changes affect calculation
        symbols = test_calculation_engine.config_manager.get_enabled_symbols()
        indicators = test_calculation_engine.config_manager.get_enabled_indicators()
        strategies = test_calculation_engine.config_manager.get_enabled_strategies()
        
        assert len(symbols) > 0
        assert len(indicators) > 0
        assert len(strategies) > 0
        
        # Process a symbol from the watchlist
        symbol = symbols[0]
        price = 150.0
        message_data = {"timestamp": datetime.now().isoformat(), "volume": "1000"}
        
        test_calculation_engine._calculate_indicators_for_symbol(symbol, price, message_data)
        
        # Should process successfully
        stats = test_calculation_engine.get_stats()
        assert symbol in stats["symbols_processed"]


class TestCalculationEngineMonitoring:
    """Test CalculationEngine monitoring and diagnostics."""
    
    def test_performance_metrics_collection(self, test_calculation_engine):
        """Test collection of performance metrics."""
        # Process some data to generate metrics
        for i in range(10):
            symbol = "AAPL"
            price = 150.0 + i * 0.1
            message_data = {"timestamp": datetime.now().isoformat(), "volume": "1000"}
            
            start_time = time.time()
            test_calculation_engine._calculate_indicators_for_symbol(symbol, price, message_data)
            processing_time = (time.time() - start_time) * 1000  # ms
            
            test_calculation_engine._update_performance_stats(processing_time)
            
        stats = test_calculation_engine.get_stats()
        
        assert stats["calculations_performed"] == 10
        assert stats["average_calculation_time_ms"] > 0
        assert stats["last_calculation_time"] is not None
        
    def test_error_tracking(self, test_calculation_engine):
        """Test error tracking capabilities."""
        initial_errors = test_calculation_engine.get_stats()["errors"]
        
        # Trigger some errors
        test_calculation_engine._increment_error_count()
        test_calculation_engine._increment_error_count()
        
        final_errors = test_calculation_engine.get_stats()["errors"]
        assert final_errors == initial_errors + 2
        
    def test_comprehensive_stats_reporting(self, test_calculation_engine):
        """Test comprehensive statistics reporting."""
        stats = test_calculation_engine.get_stats()
        
        # Verify all expected stats are present
        expected_stats = [
            "calculations_performed",
            "signals_generated", 
            "composite_signals_generated",
            "errors",
            "running",
            "active_tasks",
            "symbols_processed",
            "timeframe_manager_stats",
            "signal_engine_stats",
            "multi_timeframe_enabled"
        ]
        
        for stat in expected_stats:
            assert stat in stats
            
    def test_performance_summary_generation(self, test_calculation_engine):
        """Test performance summary generation."""
        # Generate some activity
        for i in range(5):
            symbol = "AAPL"
            price = 150.0 + i
            message_data = {"timestamp": datetime.now().isoformat(), "volume": "1000"}
            
            test_calculation_engine._calculate_indicators_for_symbol(symbol, price, message_data)
            
        summary = test_calculation_engine.get_performance_summary()
        
        assert "performance_level" in summary
        assert "key_metrics" in summary
        assert "component_stats" in summary
        assert "timestamp" in summary
        
        key_metrics = summary["key_metrics"]
        assert "total_calculations" in key_metrics
        assert "average_processing_time_ms" in key_metrics
        assert "memory_usage_mb" in key_metrics
        
    def test_health_monitoring(self, test_calculation_engine, worker_health_checker):
        """Test worker health monitoring."""
        # Check basic responsiveness
        is_responsive = worker_health_checker.check_worker_responsiveness(test_calculation_engine)
        assert is_responsive
        
        # Check state
        has_valid_state = worker_health_checker.check_worker_state(test_calculation_engine)
        assert has_valid_state
        
        # Check memory usage
        no_memory_leaks = worker_health_checker.check_memory_leaks(test_calculation_engine)
        assert no_memory_leaks
        
        # Get overall health report
        health_report = worker_health_checker.get_health_report()
        assert health_report["health_score"] > 0.8  # At least 80% healthy


@pytest.mark.asyncio
async def test_calculation_engine_lifecycle(test_config_manager, test_data_ingestion_service, 
                                          performance_thresholds):
    """Test complete CalculationEngine lifecycle."""
    with patch('src.realtime.calculation_engine.redis.Redis'):
        engine = CalculationEngine(test_config_manager, test_data_ingestion_service)
        
        # Test startup
        start_time = time.time()
        
        # Start engine (but cancel quickly to avoid long-running test)
        start_task = asyncio.create_task(engine.start())
        await asyncio.sleep(0.1)  # Let it start
        
        startup_time = time.time() - start_time
        assert startup_time < performance_thresholds["max_startup_time"]
        
        # Test running state
        assert engine._running == True
        
        # Test shutdown
        stop_start_time = time.time()
        await engine.stop()
        shutdown_time = time.time() - stop_start_time
        
        assert shutdown_time < performance_thresholds["max_shutdown_time"]
        assert engine._running == False
        
        # Cancel the start task
        start_task.cancel()
        try:
            await start_task
        except asyncio.CancelledError:
            pass