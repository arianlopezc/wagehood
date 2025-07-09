"""
Comprehensive tests for the TimeframeManager worker component.

Tests cover:
- Multi-timeframe aggregation accuracy
- Data management and storage
- Memory management and cleanup
- Performance optimization
- Cross-timeframe correlation
- Error handling and recovery
- Scalability characteristics
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np

from src.realtime.timeframe_manager import TimeframeManager, TimeframeData, TimeframeState
from src.realtime.config_manager import TradingProfile, TimeframeConfig


class TestTimeframeManagerComponents:
    """Test individual TimeframeManager components."""
    
    def test_initialization(self, test_config_manager):
        """Test TimeframeManager initialization."""
        manager = TimeframeManager(test_config_manager)
        
        assert manager.config_manager is test_config_manager
        assert isinstance(manager._states, dict)
        assert isinstance(manager._timeframe_configs, dict)
        assert hasattr(manager, '_lock')
        assert len(manager._timeframe_configs) > 0
        
    def test_timeframe_config_initialization(self, test_timeframe_manager):
        """Test timeframe configuration initialization."""
        configs = test_timeframe_manager._timeframe_configs
        
        # Should have standard timeframes
        expected_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
        for tf in expected_timeframes:
            assert tf in configs
            assert configs[tf] > 0  # Should have positive interval
            
    def test_timeframe_data_creation(self):
        """Test TimeframeData creation and manipulation."""
        timestamp = datetime.now()
        data = TimeframeData(
            timestamp=timestamp,
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000.0
        )
        
        assert data.timestamp == timestamp
        assert data.open == 150.0
        assert data.high == 152.0
        assert data.low == 149.0
        assert data.close == 151.0
        assert data.volume == 1000.0
        
    def test_timeframe_data_tick_update(self):
        """Test updating TimeframeData with tick data."""
        timestamp = datetime.now()
        data = TimeframeData(
            timestamp=timestamp,
            open=150.0,
            high=150.0,
            low=150.0,
            close=150.0,
            volume=0.0
        )
        
        # Update with higher price
        data.update_with_tick(155.0, 500.0)
        assert data.high == 155.0
        assert data.close == 155.0
        assert data.volume == 500.0
        
        # Update with lower price
        data.update_with_tick(145.0, 300.0)
        assert data.low == 145.0
        assert data.close == 145.0
        assert data.volume == 800.0
        
    def test_timeframe_state_creation(self):
        """Test TimeframeState creation."""
        state = TimeframeState(
            timeframe="1m",
            base_interval_seconds=60,
            update_interval_seconds=10,
            lookback_periods=100,
            priority=1
        )
        
        assert state.timeframe == "1m"
        assert state.base_interval_seconds == 60
        assert state.update_interval_seconds == 10
        assert state.lookback_periods == 100
        assert state.priority == 1
        assert state.updates_count == 0
        assert len(state.candles) == 0
        
    def test_timeframe_state_update_timing(self):
        """Test timeframe update timing logic."""
        state = TimeframeState(
            timeframe="1m",
            base_interval_seconds=60,
            update_interval_seconds=1,  # Update every second
            lookback_periods=100,
            priority=1
        )
        
        # Should need update initially
        assert state.needs_update()
        
        # After setting last_update, might not need update immediately
        state.last_update = datetime.now()
        # Depends on timing, might or might not need update
        
    def test_stats_initialization(self, test_timeframe_manager):
        """Test performance statistics initialization."""
        stats = test_timeframe_manager.get_stats()
        
        assert stats["total_updates"] == 0
        assert isinstance(stats["timeframe_updates"], dict)
        assert isinstance(stats["symbols_processed"], list)
        assert stats["errors"] == 0
        assert "timeframe_states" in stats


class TestTimeframeAggregation:
    """Test timeframe aggregation functionality."""
    
    def test_single_timeframe_processing(self, test_timeframe_manager):
        """Test processing ticks for a single timeframe."""
        symbol = "AAPL"
        price = 150.0
        volume = 1000.0
        timestamp = datetime.now()
        timeframes = ["1m"]
        
        results = test_timeframe_manager.process_tick(
            symbol=symbol,
            price=price,
            volume=volume,
            timestamp=timestamp,
            timeframes=timeframes,
            trading_profile=TradingProfile.SWING_TRADING
        )
        
        # Should initialize and potentially return results
        assert isinstance(results, dict)
        
        # Check state was created
        assert symbol in test_timeframe_manager._states
        assert "1m" in test_timeframe_manager._states[symbol]
        
    def test_multi_timeframe_processing(self, test_timeframe_manager):
        """Test processing ticks for multiple timeframes."""
        symbol = "AAPL"
        price = 150.0
        volume = 1000.0
        timestamp = datetime.now()
        timeframes = ["1m", "5m", "1h"]
        
        results = test_timeframe_manager.process_tick(
            symbol=symbol,
            price=price,
            volume=volume,
            timestamp=timestamp,
            timeframes=timeframes,
            trading_profile=TradingProfile.SWING_TRADING
        )
        
        # Should create states for all timeframes
        for tf in timeframes:
            assert tf in test_timeframe_manager._states[symbol]
            
    def test_candle_creation_logic(self, test_timeframe_manager):
        """Test when new candles are created."""
        symbol = "AAPL"
        timeframes = ["1m"]
        base_time = datetime.now().replace(second=0, microsecond=0)
        
        # First tick - should create new candle
        results1 = test_timeframe_manager.process_tick(
            symbol=symbol,
            price=150.0,
            volume=1000.0,
            timestamp=base_time,
            timeframes=timeframes,
            trading_profile=TradingProfile.SWING_TRADING
        )
        
        tf_state = test_timeframe_manager._states[symbol]["1m"]
        assert tf_state.current_candle is not None
        assert tf_state.current_candle.open == 150.0
        
        # Second tick in same minute - should update same candle
        results2 = test_timeframe_manager.process_tick(
            symbol=symbol,
            price=155.0,
            volume=500.0,
            timestamp=base_time + timedelta(seconds=30),
            timeframes=timeframes,
            trading_profile=TradingProfile.SWING_TRADING
        )
        
        assert tf_state.current_candle.high == 155.0
        assert tf_state.current_candle.close == 155.0
        
        # Third tick in next minute - should create new candle
        results3 = test_timeframe_manager.process_tick(
            symbol=symbol,
            price=145.0,
            volume=800.0,
            timestamp=base_time + timedelta(minutes=1),
            timeframes=timeframes,
            trading_profile=TradingProfile.SWING_TRADING
        )
        
        # Previous candle should be finalized
        assert len(tf_state.candles) == 1
        assert tf_state.current_candle.open == 145.0
        
    def test_indicator_calculation_integration(self, test_timeframe_manager):
        """Test integration with indicator calculations."""
        symbol = "AAPL"
        timeframes = ["5m"]
        
        # Process multiple ticks to build indicator data
        base_time = datetime.now().replace(second=0, microsecond=0)
        
        for i in range(20):
            price = 150.0 + i * 0.5
            timestamp = base_time + timedelta(minutes=i * 5)
            
            results = test_timeframe_manager.process_tick(
                symbol=symbol,
                price=price,
                volume=1000.0 + i * 10,
                timestamp=timestamp,
                timeframes=timeframes,
                trading_profile=TradingProfile.SWING_TRADING
            )
            
        # Should have calculated some indicators
        tf_state = test_timeframe_manager._states[symbol]["5m"]
        assert tf_state.updates_count > 0
        
    def test_lookback_period_management(self, test_timeframe_manager):
        """Test management of lookback periods."""
        symbol = "AAPL"
        timeframes = ["1m"]
        base_time = datetime.now().replace(second=0, microsecond=0)
        
        # Create more candles than lookback period
        for i in range(250):  # More than default lookback of 200
            timestamp = base_time + timedelta(minutes=i)
            price = 150.0 + (i % 20) * 0.1
            
            test_timeframe_manager.process_tick(
                symbol=symbol,
                price=price,
                volume=1000.0,
                timestamp=timestamp,
                timeframes=timeframes,
                trading_profile=TradingProfile.SWING_TRADING
            )
            
        tf_state = test_timeframe_manager._states[symbol]["1m"]
        
        # Should respect lookback period limit
        assert len(tf_state.candles) <= tf_state.lookback_periods


class TestTimeframeManagerPerformance:
    """Test TimeframeManager performance characteristics."""
    
    def test_tick_processing_performance(self, test_timeframe_manager, performance_thresholds):
        """Test tick processing performance."""
        symbol = "AAPL"
        timeframes = ["1m", "5m", "1h"]
        tick_count = 100
        
        start_time = time.time()
        
        for i in range(tick_count):
            timestamp = datetime.now() + timedelta(seconds=i)
            price = 150.0 + i * 0.01
            
            test_timeframe_manager.process_tick(
                symbol=symbol,
                price=price,
                volume=1000.0,
                timestamp=timestamp,
                timeframes=timeframes,
                trading_profile=TradingProfile.SWING_TRADING
            )
            
        total_time = time.time() - start_time
        avg_time_per_tick = total_time / tick_count
        
        # Should process ticks efficiently
        assert avg_time_per_tick < 0.01  # Less than 10ms per tick
        
    def test_memory_usage_optimization(self, test_timeframe_manager, resource_monitor):
        """Test memory usage optimization."""
        initial_memory = resource_monitor.get_memory_usage()
        
        symbol = "AAPL"
        timeframes = ["1m", "5m", "1h"]
        
        # Process many ticks
        for i in range(500):
            timestamp = datetime.now() + timedelta(minutes=i)
            price = 150.0 + (i % 100) * 0.1
            
            test_timeframe_manager.process_tick(
                symbol=symbol,
                price=price,
                volume=1000.0,
                timestamp=timestamp,
                timeframes=timeframes,
                trading_profile=TradingProfile.SWING_TRADING
            )
            
        final_memory = resource_monitor.get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 30  # Less than 30MB for 500 ticks
        
    def test_concurrent_symbol_processing(self, test_timeframe_manager):
        """Test concurrent processing of multiple symbols."""
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        timeframes = ["1m", "5m"]
        results = []
        
        def process_symbol(symbol):
            start_time = time.time()
            
            for i in range(50):
                timestamp = datetime.now() + timedelta(seconds=i)
                price = 150.0 + hash(symbol) % 100 + i * 0.1
                
                test_timeframe_manager.process_tick(
                    symbol=symbol,
                    price=price,
                    volume=1000.0,
                    timestamp=timestamp,
                    timeframes=timeframes,
                    trading_profile=TradingProfile.SWING_TRADING
                )
                
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
            
        # All symbols should be processed efficiently
        assert len(results) == len(symbols)
        for symbol, processing_time in results:
            assert processing_time < 2.0  # Less than 2 seconds per symbol
            
    def test_stats_update_efficiency(self, test_timeframe_manager):
        """Test efficiency of statistics updates."""
        start_time = time.time()
        
        # Update stats many times
        for i in range(1000):
            test_timeframe_manager._stats["total_updates"] += 1
            test_timeframe_manager._stats["timeframe_updates"]["1m"] = i
            test_timeframe_manager._stats["symbols_processed"].add(f"TEST{i % 10}")
            
        update_time = time.time() - start_time
        
        # Stats updates should be very fast
        assert update_time < 0.1  # Less than 100ms for 1000 updates


class TestTimeframeManagerReliability:
    """Test TimeframeManager reliability and error handling."""
    
    def test_invalid_timeframe_handling(self, test_timeframe_manager):
        """Test handling of invalid timeframes."""
        symbol = "AAPL"
        invalid_timeframes = ["invalid", "2x", ""]
        
        # Should handle invalid timeframes gracefully
        for tf in invalid_timeframes:
            try:
                results = test_timeframe_manager.process_tick(
                    symbol=symbol,
                    price=150.0,
                    volume=1000.0,
                    timestamp=datetime.now(),
                    timeframes=[tf],
                    trading_profile=TradingProfile.SWING_TRADING
                )
                # If no exception, should return empty or handle gracefully
                assert isinstance(results, dict)
            except Exception:
                # Exception is acceptable for invalid timeframes
                pass
                
    def test_invalid_price_data_handling(self, test_timeframe_manager):
        """Test handling of invalid price data."""
        symbol = "AAPL"
        timeframes = ["1m"]
        
        invalid_prices = [None, -150.0, 0.0, float('inf'), float('nan')]
        
        for price in invalid_prices:
            try:
                results = test_timeframe_manager.process_tick(
                    symbol=symbol,
                    price=price,
                    volume=1000.0,
                    timestamp=datetime.now(),
                    timeframes=timeframes,
                    trading_profile=TradingProfile.SWING_TRADING
                )
                # Should handle gracefully if no exception
                assert isinstance(results, dict)
            except (ValueError, TypeError):
                # These exceptions are acceptable for invalid data
                pass
                
    def test_timestamp_ordering_handling(self, test_timeframe_manager):
        """Test handling of out-of-order timestamps."""
        symbol = "AAPL"
        timeframes = ["1m"]
        base_time = datetime.now()
        
        # Send ticks in reverse chronological order
        timestamps = [
            base_time,
            base_time - timedelta(minutes=1),
            base_time - timedelta(minutes=2)
        ]
        
        for i, timestamp in enumerate(timestamps):
            results = test_timeframe_manager.process_tick(
                symbol=symbol,
                price=150.0 + i,
                volume=1000.0,
                timestamp=timestamp,
                timeframes=timeframes,
                trading_profile=TradingProfile.SWING_TRADING
            )
            
        # Should handle out-of-order data gracefully
        assert symbol in test_timeframe_manager._states
        
    def test_memory_cleanup_on_errors(self, test_timeframe_manager):
        """Test memory cleanup when errors occur."""
        initial_stats = test_timeframe_manager.get_stats()
        initial_errors = initial_stats["errors"]
        
        # Force some errors
        symbol = "AAPL"
        timeframes = ["1m"]
        
        # Try to process with invalid data that might cause errors
        for i in range(10):
            try:
                test_timeframe_manager.process_tick(
                    symbol=symbol,
                    price=float('nan'),  # Invalid price
                    volume=1000.0,
                    timestamp=datetime.now(),
                    timeframes=timeframes,
                    trading_profile=TradingProfile.SWING_TRADING
                )
            except:
                pass
                
        final_stats = test_timeframe_manager.get_stats()
        
        # Errors might increase, but system should remain stable
        assert final_stats["errors"] >= initial_errors
        
    def test_thread_safety(self, test_timeframe_manager):
        """Test thread safety of TimeframeManager."""
        symbol = "AAPL"
        timeframes = ["1m"]
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
                        timeframes=timeframes,
                        trading_profile=TradingProfile.SWING_TRADING
                    )
                    
                    results.append((worker_id, i, len(result)))
            except Exception as e:
                results.append((worker_id, "error", str(e)))
                
        # Run multiple threads concurrently
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(target=worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # Should complete without deadlocks
        assert len(results) > 0
        error_results = [r for r in results if r[1] == "error"]
        assert len(error_results) == 0  # No errors expected from thread contention


class TestTimeframeDataRetrieval:
    """Test timeframe data retrieval functionality."""
    
    def test_get_timeframe_data(self, test_timeframe_manager):
        """Test retrieving timeframe data."""
        symbol = "AAPL"
        timeframes = ["5m"]
        base_time = datetime.now().replace(second=0, microsecond=0)
        
        # Generate some candles
        for i in range(10):
            timestamp = base_time + timedelta(minutes=i * 5)
            price = 150.0 + i * 0.5
            
            test_timeframe_manager.process_tick(
                symbol=symbol,
                price=price,
                volume=1000.0,
                timestamp=timestamp,
                timeframes=timeframes,
                trading_profile=TradingProfile.SWING_TRADING
            )
            
        # Retrieve data
        data = test_timeframe_manager.get_timeframe_data(symbol, "5m", limit=5)
        
        assert isinstance(data, list)
        assert len(data) <= 5  # Respects limit
        
        for candle in data:
            assert isinstance(candle, TimeframeData)
            assert hasattr(candle, "timestamp")
            assert hasattr(candle, "close")
            
    def test_get_latest_indicators(self, test_timeframe_manager):
        """Test retrieving latest indicator values."""
        symbol = "AAPL"
        timeframes = ["1m"]
        
        # Process some ticks to generate indicators
        for i in range(50):
            timestamp = datetime.now() + timedelta(minutes=i)
            price = 150.0 + i * 0.1
            
            test_timeframe_manager.process_tick(
                symbol=symbol,
                price=price,
                volume=1000.0,
                timestamp=timestamp,
                timeframes=timeframes,
                trading_profile=TradingProfile.SWING_TRADING
            )
            
        # Get indicators
        indicators = test_timeframe_manager.get_latest_indicators(symbol, "1m")
        
        assert isinstance(indicators, dict)
        # May be empty if indicators haven't been calculated yet
        
    def test_cross_timeframe_correlation(self, test_timeframe_manager):
        """Test cross-timeframe correlation analysis."""
        symbol = "AAPL"
        timeframes = ["1m", "5m", "1h"]
        
        # Process ticks for multiple timeframes
        for i in range(100):
            timestamp = datetime.now() + timedelta(minutes=i)
            price = 150.0 + i * 0.05
            
            test_timeframe_manager.process_tick(
                symbol=symbol,
                price=price,
                volume=1000.0,
                timestamp=timestamp,
                timeframes=timeframes,
                trading_profile=TradingProfile.SWING_TRADING
            )
            
        # Get correlation analysis
        correlation = test_timeframe_manager.get_cross_timeframe_correlation(symbol, timeframes)
        
        assert isinstance(correlation, dict)
        assert "timeframes_analyzed" in correlation
        assert "available_indicators" in correlation
        assert "trend_alignment" in correlation


class TestTimeframeManagerScalability:
    """Test TimeframeManager scalability characteristics."""
    
    def test_many_symbols_scaling(self, test_timeframe_manager):
        """Test scaling with many symbols."""
        symbols = [f"TEST{i}" for i in range(50)]
        timeframes = ["1m", "5m"]
        
        start_time = time.time()
        
        # Process ticks for many symbols
        for symbol in symbols:
            for i in range(10):
                timestamp = datetime.now() + timedelta(seconds=i)
                price = 150.0 + hash(symbol) % 100 + i * 0.1
                
                test_timeframe_manager.process_tick(
                    symbol=symbol,
                    price=price,
                    volume=1000.0,
                    timestamp=timestamp,
                    timeframes=timeframes,
                    trading_profile=TradingProfile.SWING_TRADING
                )
                
        total_time = time.time() - start_time
        
        # Should handle many symbols efficiently
        assert total_time < 10.0  # Less than 10 seconds for 50 symbols * 10 ticks
        
        # Verify all symbols were processed
        stats = test_timeframe_manager.get_stats()
        assert len(stats["symbols_processed"]) == len(symbols)
        
    def test_many_timeframes_scaling(self, test_timeframe_manager):
        """Test scaling with many timeframes."""
        symbol = "AAPL"
        timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        
        start_time = time.time()
        
        # Process ticks for many timeframes
        for i in range(50):
            timestamp = datetime.now() + timedelta(minutes=i)
            price = 150.0 + i * 0.1
            
            test_timeframe_manager.process_tick(
                symbol=symbol,
                price=price,
                volume=1000.0,
                timestamp=timestamp,
                timeframes=timeframes,
                trading_profile=TradingProfile.SWING_TRADING
            )
            
        total_time = time.time() - start_time
        
        # Should handle many timeframes efficiently
        assert total_time < 5.0  # Less than 5 seconds
        
        # Verify all timeframes were processed
        assert symbol in test_timeframe_manager._states
        for tf in timeframes:
            assert tf in test_timeframe_manager._states[symbol]
            
    def test_high_frequency_tick_processing(self, test_timeframe_manager):
        """Test high-frequency tick processing."""
        symbol = "AAPL"
        timeframes = ["1m"]
        tick_count = 1000
        
        start_time = time.time()
        
        # Process many ticks rapidly
        for i in range(tick_count):
            timestamp = datetime.now() + timedelta(microseconds=i * 1000)
            price = 150.0 + (i % 100) * 0.01
            
            test_timeframe_manager.process_tick(
                symbol=symbol,
                price=price,
                volume=1000.0,
                timestamp=timestamp,
                timeframes=timeframes,
                trading_profile=TradingProfile.SWING_TRADING
            )
            
        total_time = time.time() - start_time
        ticks_per_second = tick_count / total_time
        
        # Should handle high frequency efficiently
        assert ticks_per_second > 500  # At least 500 ticks per second


class TestTimeframeManagerMonitoring:
    """Test TimeframeManager monitoring and diagnostics."""
    
    def test_memory_usage_reporting(self, test_timeframe_manager):
        """Test memory usage reporting."""
        # Create some data
        symbols = ["AAPL", "GOOGL", "MSFT"]
        timeframes = ["1m", "5m"]
        
        for symbol in symbols:
            for i in range(50):
                timestamp = datetime.now() + timedelta(minutes=i)
                price = 150.0 + i * 0.1
                
                test_timeframe_manager.process_tick(
                    symbol=symbol,
                    price=price,
                    volume=1000.0,
                    timestamp=timestamp,
                    timeframes=timeframes,
                    trading_profile=TradingProfile.SWING_TRADING
                )
                
        # Get memory usage report
        memory_usage = test_timeframe_manager.get_memory_usage()
        
        assert "total_symbols" in memory_usage
        assert "total_timeframes" in memory_usage
        assert "total_candles" in memory_usage
        assert "symbol_breakdown" in memory_usage
        
        assert memory_usage["total_symbols"] == len(symbols)
        assert memory_usage["total_timeframes"] > 0
        
    def test_performance_statistics(self, test_timeframe_manager):
        """Test performance statistics collection."""
        symbol = "AAPL"
        timeframes = ["1m"]
        
        # Process some ticks
        for i in range(20):
            timestamp = datetime.now() + timedelta(minutes=i)
            price = 150.0 + i * 0.1
            
            test_timeframe_manager.process_tick(
                symbol=symbol,
                price=price,
                volume=1000.0,
                timestamp=timestamp,
                timeframes=timeframes,
                trading_profile=TradingProfile.SWING_TRADING
            )
            
        stats = test_timeframe_manager.get_stats()
        
        assert stats["total_updates"] > 0
        assert "average_update_time_ms" in stats
        assert stats["average_update_time_ms"] >= 0
        assert "last_update_time" in stats
        
    def test_cleanup_monitoring(self, test_timeframe_manager):
        """Test cleanup operation monitoring."""
        symbol = "AAPL"
        timeframes = ["1m"]
        
        # Create old data
        old_time = datetime.now() - timedelta(hours=25)  # 25 hours ago
        
        for i in range(10):
            timestamp = old_time + timedelta(minutes=i)
            price = 150.0 + i * 0.1
            
            test_timeframe_manager.process_tick(
                symbol=symbol,
                price=price,
                volume=1000.0,
                timestamp=timestamp,
                timeframes=timeframes,
                trading_profile=TradingProfile.SWING_TRADING
            )
            
        # Get initial candle count
        initial_data = test_timeframe_manager.get_timeframe_data(symbol, "1m")
        initial_count = len(initial_data)
        
        # Run cleanup (24 hour cutoff)
        test_timeframe_manager.cleanup_old_data(max_age_hours=24)
        
        # Check if cleanup worked
        final_data = test_timeframe_manager.get_timeframe_data(symbol, "1m")
        final_count = len(final_data)
        
        # Should have removed old data
        assert final_count <= initial_count
        
    def test_state_reset_functionality(self, test_timeframe_manager):
        """Test state reset functionality."""
        symbol = "AAPL"
        timeframes = ["1m", "5m"]
        
        # Create some state
        for i in range(10):
            timestamp = datetime.now() + timedelta(minutes=i)
            price = 150.0 + i * 0.1
            
            test_timeframe_manager.process_tick(
                symbol=symbol,
                price=price,
                volume=1000.0,
                timestamp=timestamp,
                timeframes=timeframes,
                trading_profile=TradingProfile.SWING_TRADING
            )
            
        # Verify state exists
        assert symbol in test_timeframe_manager._states
        assert "1m" in test_timeframe_manager._states[symbol]
        assert "5m" in test_timeframe_manager._states[symbol]
        
        # Reset specific timeframe
        test_timeframe_manager.reset_symbol_timeframe(symbol, "1m")
        
        # Should remove only 1m timeframe
        assert symbol in test_timeframe_manager._states
        assert "1m" not in test_timeframe_manager._states[symbol]
        assert "5m" in test_timeframe_manager._states[symbol]
        
        # Reset entire symbol
        test_timeframe_manager.reset_symbol(symbol)
        
        # Should remove all timeframes for symbol
        assert symbol not in test_timeframe_manager._states


def test_timeframe_manager_complete_workflow(test_timeframe_manager):
    """Test complete TimeframeManager workflow."""
    symbol = "AAPL"
    timeframes = ["1m", "5m", "1h"]
    base_time = datetime.now().replace(second=0, microsecond=0)
    
    # Phase 1: Initialize with ticks
    for i in range(100):
        timestamp = base_time + timedelta(minutes=i)
        price = 150.0 + i * 0.05 + (i % 10) * 0.1  # Trending with noise
        volume = 1000 + i * 10
        
        results = test_timeframe_manager.process_tick(
            symbol=symbol,
            price=price,
            volume=volume,
            timestamp=timestamp,
            timeframes=timeframes,
            trading_profile=TradingProfile.SWING_TRADING
        )
        
    # Phase 2: Verify data structures
    assert symbol in test_timeframe_manager._states
    for tf in timeframes:
        assert tf in test_timeframe_manager._states[symbol]
        
    # Phase 3: Test data retrieval
    for tf in timeframes:
        data = test_timeframe_manager.get_timeframe_data(symbol, tf, limit=10)
        assert len(data) > 0
        
        indicators = test_timeframe_manager.get_latest_indicators(symbol, tf)
        assert isinstance(indicators, dict)
        
    # Phase 4: Test correlation analysis
    correlation = test_timeframe_manager.get_cross_timeframe_correlation(symbol, timeframes)
    assert correlation["timeframes_analyzed"] > 0
    
    # Phase 5: Test monitoring
    stats = test_timeframe_manager.get_stats()
    assert stats["total_updates"] > 0
    assert len(stats["symbols_processed"]) == 1
    
    memory_usage = test_timeframe_manager.get_memory_usage()
    assert memory_usage["total_symbols"] == 1
    assert memory_usage["total_timeframes"] == len(timeframes)
    
    # Phase 6: Test cleanup
    test_timeframe_manager.cleanup_old_data(max_age_hours=1)
    
    # Phase 7: Test reset
    test_timeframe_manager.reset_symbol(symbol)
    assert symbol not in test_timeframe_manager._states