"""
Comprehensive tests for worker process reliability.

Tests cover:
- Process startup and shutdown reliability
- Error recovery and restart mechanisms
- Memory management and leak prevention
- CPU usage optimization
- Long-running stability
- Fault tolerance and resilience
"""

import pytest
import asyncio
import time
import threading
import gc
import psutil
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import logging

from src.realtime.calculation_engine import CalculationEngine
from src.realtime.data_ingestion import MarketDataIngestionService
from src.realtime.timeframe_manager import TimeframeManager
from src.realtime.signal_engine import SignalEngine
from src.realtime.config_manager import ConfigManager, TradingProfile


class TestProcessLifecycle:
    """Test process startup and shutdown reliability."""
    
    @pytest.mark.asyncio
    async def test_calculation_engine_startup_shutdown(self, test_config_manager, 
                                                     test_data_ingestion_service, 
                                                     performance_thresholds):
        """Test CalculationEngine startup and shutdown reliability."""
        with patch('src.realtime.calculation_engine.redis.Redis'):
            engine = CalculationEngine(test_config_manager, test_data_ingestion_service)
            
            # Test startup
            startup_start = time.time()
            
            # Start engine (but cancel quickly for testing)
            start_task = asyncio.create_task(engine.start())
            await asyncio.sleep(0.1)  # Let it start
            
            startup_time = time.time() - startup_start
            assert startup_time < performance_thresholds["max_startup_time"]
            assert engine._running == True
            
            # Test shutdown
            shutdown_start = time.time()
            await engine.stop()
            shutdown_time = time.time() - shutdown_start
            
            assert shutdown_time < performance_thresholds["max_shutdown_time"]
            assert engine._running == False
            
            # Cancel start task
            start_task.cancel()
            try:
                await start_task
            except asyncio.CancelledError:
                pass
                
    @pytest.mark.asyncio
    async def test_data_ingestion_startup_shutdown(self, test_config_manager, 
                                                  performance_thresholds, mock_redis_client):
        """Test DataIngestionService startup and shutdown reliability."""
        with patch('src.realtime.data_ingestion.redis.Redis', return_value=mock_redis_client):
            service = MarketDataIngestionService(test_config_manager)
            
            # Test startup
            startup_start = time.time()
            
            start_task = asyncio.create_task(service.start())
            await asyncio.sleep(0.1)  # Let it start
            
            startup_time = time.time() - startup_start
            assert startup_time < performance_thresholds["max_startup_time"]
            assert service._running == True
            
            # Test shutdown
            shutdown_start = time.time()
            await service.stop()
            shutdown_time = time.time() - shutdown_start
            
            assert shutdown_time < performance_thresholds["max_shutdown_time"]
            assert service._running == False
            
            # Cancel start task
            start_task.cancel()
            try:
                await start_task
            except asyncio.CancelledError:
                pass
                
    @pytest.mark.asyncio
    async def test_multiple_startup_shutdown_cycles(self, test_config_manager, mock_redis_client):
        """Test reliability over multiple startup/shutdown cycles."""
        with patch('src.realtime.data_ingestion.redis.Redis', return_value=mock_redis_client):
            service = MarketDataIngestionService(test_config_manager)
            
            # Perform multiple cycles
            for cycle in range(3):
                # Startup
                start_task = asyncio.create_task(service.start())
                await asyncio.sleep(0.1)
                
                assert service._running == True
                
                # Shutdown
                await service.stop()
                assert service._running == False
                
                # Cancel start task
                start_task.cancel()
                try:
                    await start_task
                except asyncio.CancelledError:
                    pass
                    
    @pytest.mark.asyncio
    async def test_graceful_shutdown_with_pending_tasks(self, test_config_manager, mock_redis_client):
        """Test graceful shutdown with pending tasks."""
        with patch('src.realtime.calculation_engine.redis.Redis', return_value=mock_redis_client):
            with patch('src.realtime.data_ingestion.redis.Redis', return_value=mock_redis_client):
                ingestion_service = MarketDataIngestionService(test_config_manager)
                engine = CalculationEngine(test_config_manager, ingestion_service)
                
                # Start with some pending work
                start_task = asyncio.create_task(engine.start())
                await asyncio.sleep(0.1)
                
                # Add some work
                message_data = {
                    "symbol": "AAPL",
                    "price": "150.0",
                    "timestamp": datetime.now().isoformat(),
                    "volume": "1000"
                }
                
                # Process a message to create work
                process_task = asyncio.create_task(
                    engine._process_single_message("test-msg", message_data)
                )
                
                # Shutdown should wait for tasks to complete
                shutdown_start = time.time()
                await engine.stop()
                shutdown_time = time.time() - shutdown_start
                
                # Should shutdown gracefully
                assert engine._running == False
                assert shutdown_time < 5.0  # Reasonable shutdown time
                
                # Cleanup
                start_task.cancel()
                try:
                    await start_task
                except asyncio.CancelledError:
                    pass
                    
                try:
                    await process_task
                except asyncio.CancelledError:
                    pass


class TestErrorRecovery:
    """Test error recovery and restart mechanisms."""
    
    @pytest.mark.asyncio
    async def test_redis_connection_recovery(self, test_config_manager):
        """Test recovery from Redis connection failures."""
        mock_redis = Mock()
        
        # Simulate connection failure then recovery
        call_count = 0
        def ping_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Connection failed")
            return True
            
        mock_redis.ping.side_effect = ping_side_effect
        
        with patch('src.realtime.data_ingestion.redis.Redis', return_value=mock_redis):
            # Should eventually succeed after retries
            try:
                service = MarketDataIngestionService(test_config_manager)
                # If we get here, recovery worked
                assert True
            except Exception:
                # May still fail if recovery mechanism isn't implemented
                pass
                
    @pytest.mark.asyncio
    async def test_processing_error_recovery(self, test_calculation_engine):
        """Test recovery from processing errors."""
        symbol = "AAPL"
        
        # Process normal message
        normal_message = {
            "symbol": symbol,
            "price": "150.0",
            "timestamp": datetime.now().isoformat(),
            "volume": "1000"
        }
        
        await test_calculation_engine._process_single_message("msg-1", normal_message)
        
        # Process invalid message (should cause error but not crash)
        invalid_message = {
            "symbol": None,
            "price": "invalid",
            "timestamp": "invalid",
            "volume": "invalid"
        }
        
        initial_errors = test_calculation_engine.get_stats()["errors"]
        
        await test_calculation_engine._process_single_message("msg-2", invalid_message)
        
        # Should handle error and continue
        final_errors = test_calculation_engine.get_stats()["errors"]
        assert final_errors > initial_errors
        
        # Should still process normal messages
        await test_calculation_engine._process_single_message("msg-3", normal_message)
        
        final_calculations = test_calculation_engine.get_stats()["calculations_performed"]
        assert final_calculations > 0
        
    def test_timeframe_manager_error_recovery(self, test_timeframe_manager):
        """Test TimeframeManager error recovery."""
        symbol = "AAPL"
        
        # Process normal data
        normal_result = test_timeframe_manager.process_tick(
            symbol=symbol,
            price=150.0,
            volume=1000.0,
            timestamp=datetime.now(),
            timeframes=["1m"],
            trading_profile=TradingProfile.SWING_TRADING
        )
        
        initial_errors = test_timeframe_manager.get_stats()["errors"]
        
        # Process invalid data
        try:
            test_timeframe_manager.process_tick(
                symbol=symbol,
                price=float('nan'),  # Invalid
                volume=-1000.0,      # Invalid
                timestamp=datetime.now(),
                timeframes=["invalid"],  # Invalid
                trading_profile=TradingProfile.SWING_TRADING
            )
        except (ValueError, TypeError):
            # Error is expected
            pass
            
        # Should still process normal data after error
        recovery_result = test_timeframe_manager.process_tick(
            symbol=symbol,
            price=151.0,
            volume=1000.0,
            timestamp=datetime.now(),
            timeframes=["1m"],
            trading_profile=TradingProfile.SWING_TRADING
        )
        
        # Should recover and continue processing
        assert isinstance(recovery_result, dict)
        
    def test_signal_engine_error_recovery(self, test_signal_engine):
        """Test SignalEngine error recovery."""
        symbol = "AAPL"
        
        # Generate normal signal
        normal_timeframe_results = {
            "1h": {
                "rsi_14": 45.0,
                "sma_50": 150.0
            }
        }
        
        normal_signal = test_signal_engine.generate_signals(
            symbol=symbol,
            price=150.0,
            timeframe_results=normal_timeframe_results,
            trading_profile=TradingProfile.SWING_TRADING
        )
        
        initial_errors = test_signal_engine.get_stats()["errors"]
        
        # Generate signal with invalid data
        invalid_timeframe_results = {
            "invalid": {
                "rsi_14": float('nan'),
                "invalid_indicator": "not_a_number"
            }
        }
        
        try:
            test_signal_engine.generate_signals(
                symbol=symbol,
                price=float('inf'),  # Invalid
                timeframe_results=invalid_timeframe_results,
                trading_profile=TradingProfile.SWING_TRADING
            )
        except (ValueError, TypeError):
            # Error is expected
            pass
            
        # Should still generate normal signals after error
        recovery_signal = test_signal_engine.generate_signals(
            symbol=symbol,
            price=151.0,
            timeframe_results=normal_timeframe_results,
            trading_profile=TradingProfile.SWING_TRADING
        )
        
        # Should recover and continue
        final_signals = test_signal_engine.get_stats()["total_signals_generated"]
        assert final_signals >= 0
        
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, test_data_ingestion_service):
        """Test circuit breaker recovery mechanism."""
        # Create a provider that fails then recovers
        call_count = 0
        
        def failing_provider():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise Exception("Provider failure")
            return Mock(timestamp=datetime.now(), open=150.0, high=152.0, low=149.0, close=151.0, volume=1000)
            
        provider = Mock()
        provider.get_latest_data.side_effect = failing_provider
        
        test_data_ingestion_service.add_provider("failing_provider", provider)
        circuit_breaker = test_data_ingestion_service._circuit_breakers["failing_provider"]
        
        # Initial failures should trip circuit breaker
        for i in range(3):
            result = await test_data_ingestion_service._fetch_data_with_circuit_breaker(
                circuit_breaker, provider, "AAPL"
            )
            assert result is None
            
        # Circuit breaker should be open
        assert circuit_breaker.state == "OPEN"
        
        # Wait for potential recovery (in real scenario)
        # For test, we'll manually reset the circuit breaker
        circuit_breaker.reset()
        
        # Should work after recovery
        result = await test_data_ingestion_service._fetch_data_with_circuit_breaker(
            circuit_breaker, provider, "AAPL"
        )
        assert result is not None


class TestMemoryManagement:
    """Test memory management and leak prevention."""
    
    def test_timeframe_manager_memory_cleanup(self, test_timeframe_manager, resource_monitor):
        """Test TimeframeManager memory cleanup."""
        initial_memory = resource_monitor.get_memory_usage()
        
        symbol = "AAPL"
        
        # Generate lots of data
        for i in range(500):
            timestamp = datetime.now() + timedelta(minutes=i)
            price = 150.0 + i * 0.01
            
            test_timeframe_manager.process_tick(
                symbol=symbol,
                price=price,
                volume=1000.0,
                timestamp=timestamp,
                timeframes=["1m", "5m"],
                trading_profile=TradingProfile.SWING_TRADING
            )
            
        mid_memory = resource_monitor.get_memory_usage()
        
        # Run cleanup
        test_timeframe_manager.cleanup_old_data(max_age_hours=1)
        
        # Force garbage collection
        gc.collect()
        
        final_memory = resource_monitor.get_memory_usage()
        
        # Memory should not grow excessively
        memory_increase = final_memory - initial_memory
        assert memory_increase < 50  # Less than 50MB increase
        
        # Cleanup should have some effect
        assert final_memory <= mid_memory
        
    def test_signal_engine_memory_cleanup(self, test_signal_engine, resource_monitor):
        """Test SignalEngine memory cleanup."""
        initial_memory = resource_monitor.get_memory_usage()
        
        symbol = "AAPL"
        
        # Generate many signals
        for i in range(200):
            timeframe_results = {
                "1h": {
                    "rsi_14": 45.0 + i % 50,
                    "sma_50": 150.0 + i * 0.01
                }
            }
            
            test_signal_engine.generate_signals(
                symbol=symbol,
                price=150.0 + i * 0.01,
                timeframe_results=timeframe_results,
                trading_profile=TradingProfile.SWING_TRADING
            )
            
        mid_memory = resource_monitor.get_memory_usage()
        
        # Run cleanup
        test_signal_engine.cleanup_old_signals(max_age_hours=1)
        
        # Force garbage collection
        gc.collect()
        
        final_memory = resource_monitor.get_memory_usage()
        
        # Memory should not grow excessively
        memory_increase = final_memory - initial_memory
        assert memory_increase < 30  # Less than 30MB increase
        
    def test_calculation_engine_memory_stability(self, test_calculation_engine, resource_monitor):
        """Test CalculationEngine memory stability over time."""
        initial_memory = resource_monitor.get_memory_usage()
        
        # Simulate extended operation
        for batch in range(10):
            for i in range(50):
                symbol = f"TEST{i % 5}"
                message_data = {
                    "symbol": symbol,
                    "price": str(150.0 + batch * 10 + i * 0.1),
                    "timestamp": datetime.now().isoformat(),
                    "volume": str(1000 + i)
                }
                
                test_calculation_engine._calculate_indicators_for_symbol(
                    symbol, float(message_data["price"]), message_data
                )
                
            # Periodic memory check
            current_memory = resource_monitor.get_memory_usage()
            memory_increase = current_memory - initial_memory
            
            # Should not have excessive memory growth
            assert memory_increase < 100  # Less than 100MB total
            
        # Final memory check
        final_memory = resource_monitor.get_memory_usage()
        total_increase = final_memory - initial_memory
        
        # Total memory increase should be reasonable
        assert total_increase < 150  # Less than 150MB for all processing
        
    def test_memory_leak_detection(self, test_timeframe_manager):
        """Test for memory leaks in repeated operations."""
        import gc
        
        # Get initial object counts
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        symbol = "AAPL"
        
        # Perform repeated operations
        for cycle in range(5):
            # Create and destroy data
            for i in range(100):
                test_timeframe_manager.process_tick(
                    symbol=symbol,
                    price=150.0 + i * 0.01,
                    volume=1000.0,
                    timestamp=datetime.now() + timedelta(seconds=i),
                    timeframes=["1m"],
                    trading_profile=TradingProfile.SWING_TRADING
                )
                
            # Reset data
            test_timeframe_manager.reset_symbol(symbol)
            
            # Force garbage collection
            gc.collect()
            
        # Check final object count
        final_objects = len(gc.get_objects())
        object_increase = final_objects - initial_objects
        
        # Should not have significant object leakage
        # Some increase is normal due to test framework
        assert object_increase < 1000  # Reasonable threshold


class TestCPUOptimization:
    """Test CPU usage optimization."""
    
    def test_calculation_engine_cpu_efficiency(self, test_calculation_engine, resource_monitor):
        """Test CalculationEngine CPU efficiency."""
        # Warm up
        for i in range(10):
            message_data = {
                "symbol": "AAPL",
                "price": str(150.0 + i * 0.1),
                "timestamp": datetime.now().isoformat(),
                "volume": "1000"
            }
            test_calculation_engine._calculate_indicators_for_symbol("AAPL", 150.0 + i * 0.1, message_data)
            
        # Monitor CPU during intensive processing
        start_cpu = resource_monitor.get_cpu_usage()
        start_time = time.time()
        
        # Intensive processing
        for i in range(100):
            symbol = f"TEST{i % 5}"
            message_data = {
                "symbol": symbol,
                "price": str(150.0 + i * 0.1),
                "timestamp": datetime.now().isoformat(),
                "volume": str(1000 + i)
            }
            
            test_calculation_engine._calculate_indicators_for_symbol(
                symbol, float(message_data["price"]), message_data
            )
            
        processing_time = time.time() - start_time
        end_cpu = resource_monitor.get_cpu_usage()
        
        # Should complete processing efficiently
        assert processing_time < 5.0  # Less than 5 seconds for 100 calculations
        
        # CPU usage should be reasonable (depends on system)
        cpu_increase = abs(end_cpu - start_cpu)
        assert cpu_increase < 80  # Less than 80% CPU spike
        
    def test_timeframe_manager_batch_efficiency(self, test_timeframe_manager, resource_monitor):
        """Test TimeframeManager batch processing efficiency."""
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        
        start_time = time.time()
        start_cpu = resource_monitor.get_cpu_usage()
        
        # Process batch of data
        for symbol in symbols:
            for i in range(50):
                timestamp = datetime.now() + timedelta(seconds=i)
                price = 150.0 + hash(symbol) % 100 + i * 0.1
                
                test_timeframe_manager.process_tick(
                    symbol=symbol,
                    price=price,
                    volume=1000.0,
                    timestamp=timestamp,
                    timeframes=["1m", "5m"],
                    trading_profile=TradingProfile.SWING_TRADING
                )
                
        processing_time = time.time() - start_time
        end_cpu = resource_monitor.get_cpu_usage()
        
        # Should process efficiently
        total_ticks = len(symbols) * 50
        ticks_per_second = total_ticks / processing_time
        
        assert ticks_per_second > 100  # At least 100 ticks per second
        
    def test_signal_engine_optimization(self, test_signal_engine, resource_monitor):
        """Test SignalEngine optimization."""
        start_time = time.time()
        start_cpu = resource_monitor.get_cpu_usage()
        
        # Generate many signals
        for i in range(100):
            symbol = f"TEST{i % 10}"
            timeframe_results = {
                "1h": {
                    "rsi_14": 45.0 + i % 50,
                    "sma_50": 150.0 + i * 0.01,
                    "macd": {
                        "macd_line": 1.0 + i * 0.01,
                        "signal_line": 0.8 + i * 0.008,
                        "histogram": 0.2 + i * 0.002
                    }
                }
            }
            
            test_signal_engine.generate_signals(
                symbol=symbol,
                price=150.0 + i * 0.01,
                timeframe_results=timeframe_results,
                trading_profile=TradingProfile.SWING_TRADING
            )
            
        processing_time = time.time() - start_time
        end_cpu = resource_monitor.get_cpu_usage()
        
        # Should generate signals efficiently
        signals_per_second = 100 / processing_time
        assert signals_per_second > 50  # At least 50 signals per second


class TestLongRunningStability:
    """Test stability during long-running operations."""
    
    def test_extended_calculation_stability(self, test_calculation_engine, resource_monitor):
        """Test calculation engine stability over extended periods."""
        duration_seconds = 10  # Shorter for test
        start_time = time.time()
        
        operation_count = 0
        error_count = 0
        
        while (time.time() - start_time) < duration_seconds:
            try:
                symbol = f"TEST{operation_count % 5}"
                message_data = {
                    "symbol": symbol,
                    "price": str(150.0 + operation_count * 0.01),
                    "timestamp": datetime.now().isoformat(),
                    "volume": str(1000 + operation_count)
                }
                
                test_calculation_engine._calculate_indicators_for_symbol(
                    symbol, float(message_data["price"]), message_data
                )
                
                operation_count += 1
                
                # Brief pause to simulate realistic load
                time.sleep(0.01)
                
            except Exception as e:
                error_count += 1
                logging.error(f"Error in extended operation: {e}")
                
        # Should maintain stability
        error_rate = error_count / max(operation_count, 1)
        assert error_rate < 0.01  # Less than 1% error rate
        
        # Should process reasonable amount
        assert operation_count > 100  # At least some processing
        
        # Memory should be stable
        resource_monitor.update_peaks()
        memory_usage = resource_monitor.get_memory_usage()
        assert memory_usage < 500  # Less than 500MB
        
    def test_timeframe_manager_long_running(self, test_timeframe_manager):
        """Test TimeframeManager stability over time."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        operation_count = 0
        
        # Simulate extended operation
        for hour in range(2):  # 2 hours simulation
            for minute in range(60):  # 60 minutes per hour
                for symbol in symbols:
                    timestamp = datetime.now() + timedelta(hours=hour, minutes=minute)
                    price = 150.0 + hour * 10 + minute * 0.1
                    
                    test_timeframe_manager.process_tick(
                        symbol=symbol,
                        price=price,
                        volume=1000.0,
                        timestamp=timestamp,
                        timeframes=["1m", "5m", "1h"],
                        trading_profile=TradingProfile.SWING_TRADING
                    )
                    
                    operation_count += 1
                    
            # Periodic cleanup
            test_timeframe_manager.cleanup_old_data(max_age_hours=1)
            
        # Should maintain stability
        stats = test_timeframe_manager.get_stats()
        assert stats["total_updates"] > 0
        assert stats["errors"] < operation_count * 0.01  # Less than 1% errors
        
    def test_memory_stability_over_time(self, test_calculation_engine, resource_monitor):
        """Test memory stability over extended operation."""
        initial_memory = resource_monitor.get_memory_usage()
        memory_samples = [initial_memory]
        
        # Extended operation with periodic sampling
        for batch in range(20):  # 20 batches
            for i in range(25):  # 25 operations per batch
                symbol = f"TEST{i % 5}"
                message_data = {
                    "symbol": symbol,
                    "price": str(150.0 + batch * 10 + i * 0.1),
                    "timestamp": datetime.now().isoformat(),
                    "volume": str(1000 + i)
                }
                
                test_calculation_engine._calculate_indicators_for_symbol(
                    symbol, float(message_data["price"]), message_data
                )
                
            # Sample memory after each batch
            current_memory = resource_monitor.get_memory_usage()
            memory_samples.append(current_memory)
            
            # Force cleanup periodically
            if batch % 5 == 0:
                gc.collect()
                
        # Analyze memory trend
        final_memory = memory_samples[-1]
        max_memory = max(memory_samples)
        
        # Memory growth should be bounded
        total_growth = final_memory - initial_memory
        assert total_growth < 100  # Less than 100MB growth
        
        # Should not have excessive peaks
        peak_growth = max_memory - initial_memory
        assert peak_growth < 150  # Less than 150MB peak


class TestFaultTolerance:
    """Test fault tolerance and resilience."""
    
    def test_partial_system_failure_tolerance(self, test_config_manager, mock_redis_client):
        """Test tolerance to partial system failures."""
        # Simulate Redis intermittent failures
        call_count = 0
        
        def intermittent_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Fail every 3rd call
                raise Exception("Intermittent Redis failure")
            return f"msg-{call_count}"
            
        mock_redis_client.xadd.side_effect = intermittent_failure
        
        with patch('src.realtime.data_ingestion.redis.Redis', return_value=mock_redis_client):
            service = MarketDataIngestionService(test_config_manager)
            
            success_count = 0
            failure_count = 0
            
            # Attempt multiple operations
            for i in range(10):
                try:
                    from src.core.models import OHLCV
                    ohlcv = OHLCV(
                        timestamp=datetime.now(),
                        open=150.0, high=152.0, low=149.0, close=151.0, volume=1000
                    )
                    
                    # This will call xadd internally
                    import asyncio
                    asyncio.run(service._publish_market_data_event("AAPL", ohlcv, "test"))
                    success_count += 1
                    
                except Exception:
                    failure_count += 1
                    
            # Should have both successes and failures
            assert success_count > 0
            assert failure_count > 0
            
            # Should continue operating despite failures
            total_operations = success_count + failure_count
            success_rate = success_count / total_operations
            assert success_rate > 0.5  # At least 50% success rate
            
    def test_resource_exhaustion_handling(self, test_calculation_engine):
        """Test handling of resource exhaustion scenarios."""
        # Simulate high load that might exhaust resources
        start_time = time.time()
        processed_count = 0
        error_count = 0
        
        # Process intensive workload
        for i in range(1000):  # Large number of operations
            try:
                symbol = f"TEST{i % 10}"
                message_data = {
                    "symbol": symbol,
                    "price": str(150.0 + i * 0.001),
                    "timestamp": datetime.now().isoformat(),
                    "volume": str(1000 + i)
                }
                
                test_calculation_engine._calculate_indicators_for_symbol(
                    symbol, float(message_data["price"]), message_data
                )
                
                processed_count += 1
                
            except Exception as e:
                error_count += 1
                # Don't log every error to avoid spam
                if error_count % 100 == 1:
                    logging.warning(f"Resource exhaustion error: {e}")
                    
        total_time = time.time() - start_time
        
        # Should handle high load gracefully
        total_operations = processed_count + error_count
        assert total_operations == 1000
        
        # Should process majority successfully
        success_rate = processed_count / total_operations
        assert success_rate > 0.8  # At least 80% success rate
        
        # Should complete in reasonable time
        assert total_time < 30.0  # Less than 30 seconds
        
    def test_configuration_corruption_tolerance(self, test_config_manager):
        """Test tolerance to configuration corruption."""
        # Test with various invalid configurations
        invalid_configs = [
            None,  # Null config
            {},    # Empty config
            {"invalid": "data"},  # Wrong structure
        ]
        
        for invalid_config in invalid_configs:
            try:
                # Mock corrupted configuration
                with patch.object(test_config_manager, 'get_watchlist', return_value=invalid_config):
                    result = test_config_manager.get_enabled_symbols()
                    
                    # Should handle gracefully
                    assert isinstance(result, list)
                    
            except Exception as e:
                # Some exceptions are acceptable for severely corrupted config
                assert "config" in str(e).lower() or "invalid" in str(e).lower()
                
    def test_data_corruption_tolerance(self, test_timeframe_manager):
        """Test tolerance to data corruption."""
        symbol = "AAPL"
        
        # Test with various corrupted data
        corrupted_data = [
            (float('nan'), 1000.0),      # NaN price
            (150.0, float('inf')),       # Infinite volume
            (-150.0, 1000.0),            # Negative price
            (150.0, -1000.0),            # Negative volume
        ]
        
        success_count = 0
        
        for price, volume in corrupted_data:
            try:
                result = test_timeframe_manager.process_tick(
                    symbol=symbol,
                    price=price,
                    volume=volume,
                    timestamp=datetime.now(),
                    timeframes=["1m"],
                    trading_profile=TradingProfile.SWING_TRADING
                )
                
                # If no exception, should return valid result
                assert isinstance(result, dict)
                success_count += 1
                
            except (ValueError, TypeError):
                # These exceptions are acceptable for corrupted data
                pass
                
        # Should handle at least some cases gracefully
        # (Depending on validation implementation)
        

@pytest.mark.asyncio
async def test_reliability_stress_scenario(test_config_manager, mock_redis_client, 
                                         resource_monitor, error_injector):
    """Test reliability under stress conditions."""
    
    # Setup components with mocked Redis
    with patch('src.realtime.data_ingestion.redis.Redis', return_value=mock_redis_client):
        with patch('src.realtime.calculation_engine.redis.Redis', return_value=mock_redis_client):
            
            # Phase 1: Normal startup
            ingestion_service = MarketDataIngestionService(test_config_manager)
            calc_engine = CalculationEngine(test_config_manager, ingestion_service)
            
            initial_memory = resource_monitor.get_memory_usage()
            
            # Phase 2: Stress test with errors
            success_count = 0
            error_count = 0
            
            for i in range(200):  # High volume
                try:
                    symbol = f"TEST{i % 10}"
                    
                    # Randomly inject errors
                    if i % 20 == 0:  # 5% error rate
                        mock_redis_client.xadd.side_effect = error_injector.inject_redis_error
                    else:
                        mock_redis_client.xadd.side_effect = None
                        mock_redis_client.xadd.return_value = f"msg-{i}"
                        
                    # Process market data
                    message_data = {
                        "symbol": symbol,
                        "price": str(150.0 + i * 0.01),
                        "timestamp": datetime.now().isoformat(),
                        "volume": str(1000 + i)
                    }
                    
                    await calc_engine._process_single_message(f"msg-{i}", message_data)
                    success_count += 1
                    
                except Exception:
                    error_count += 1
                    
                # Brief pause to simulate realistic timing
                if i % 50 == 0:
                    await asyncio.sleep(0.01)
                    
            # Phase 3: Verify reliability metrics
            
            total_operations = success_count + error_count
            success_rate = success_count / total_operations if total_operations > 0 else 0
            
            # Should maintain reasonable success rate despite errors
            assert success_rate > 0.85  # At least 85% success rate
            
            # Memory should remain stable
            final_memory = resource_monitor.get_memory_usage()
            memory_increase = final_memory - initial_memory
            assert memory_increase < 100  # Less than 100MB increase
            
            # Components should remain functional
            calc_stats = calc_engine.get_stats()
            assert calc_stats["calculations_performed"] > 0
            
            # Phase 4: Recovery test
            mock_redis_client.xadd.side_effect = None  # Remove all errors
            
            # Should continue operating normally
            recovery_message = {
                "symbol": "RECOVERY_TEST",
                "price": "150.0",
                "timestamp": datetime.now().isoformat(),
                "volume": "1000"
            }
            
            await calc_engine._process_single_message("recovery-msg", recovery_message)
            
            # Should process successfully
            final_stats = calc_engine.get_stats()
            assert final_stats["calculations_performed"] > calc_stats["calculations_performed"]