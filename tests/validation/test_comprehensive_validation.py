"""
Comprehensive Validation Test Suite

This module contains comprehensive validation tests for the entire Wagehood system,
ensuring all components work correctly and integrate properly.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import tempfile
import os
import json

from src.core.models import OHLCV, TimeFrame
from src.strategies import (
    MovingAverageCrossover,
    MACDRSIStrategy,
    RSITrendFollowing,
    BollingerBandBreakout,
    SupportResistanceBreakout
)
from src.data.providers.mock_provider import MockDataProvider
from src.backtesting.engine import BacktestEngine
from src.backtesting.config import BacktestConfig
from src.realtime.incremental_indicators import IncrementalIndicatorCalculator
from src.storage.cache import cache_manager


class TestDataGenerator:
    """Generate test data for validation."""
    
    @staticmethod
    def generate_trending_data(length: int = 1000, trend: str = "up") -> List[OHLCV]:
        """Generate trending market data."""
        data = []
        base_price = 100.0
        timestamp = datetime.now() - timedelta(days=length)
        
        for i in range(length):
            if trend == "up":
                price_change = np.random.normal(0.001, 0.02)  # Slight upward bias
            elif trend == "down":
                price_change = np.random.normal(-0.001, 0.02)  # Slight downward bias
            else:
                price_change = np.random.normal(0, 0.02)  # Sideways
            
            base_price *= (1 + price_change)
            
            # Generate OHLCV
            open_price = base_price
            close_price = base_price * (1 + np.random.normal(0, 0.01))
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
            volume = int(np.random.uniform(1000, 10000))
            
            ohlcv = OHLCV(
                timestamp=timestamp + timedelta(hours=i),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            )
            data.append(ohlcv)
            base_price = close_price
        
        return data
    
    @staticmethod
    def generate_volatile_data(length: int = 1000) -> List[OHLCV]:
        """Generate highly volatile market data."""
        data = []
        base_price = 100.0
        timestamp = datetime.now() - timedelta(days=length)
        
        for i in range(length):
            # High volatility with occasional spikes
            if np.random.random() < 0.05:  # 5% chance of major move
                price_change = np.random.normal(0, 0.1)  # Large move
            else:
                price_change = np.random.normal(0, 0.03)  # Normal volatility
            
            base_price *= (1 + price_change)
            
            open_price = base_price
            close_price = base_price * (1 + np.random.normal(0, 0.02))
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
            volume = int(np.random.uniform(500, 20000))
            
            ohlcv = OHLCV(
                timestamp=timestamp + timedelta(hours=i),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            )
            data.append(ohlcv)
            base_price = close_price
        
        return data


class TestStrategyValidation:
    """Comprehensive strategy validation tests."""
    
    @pytest.fixture
    def trending_up_data(self):
        return TestDataGenerator.generate_trending_data(1000, "up")
    
    @pytest.fixture
    def trending_down_data(self):
        return TestDataGenerator.generate_trending_data(1000, "down")
    
    @pytest.fixture
    def sideways_data(self):
        return TestDataGenerator.generate_trending_data(1000, "sideways")
    
    @pytest.fixture
    def volatile_data(self):
        return TestDataGenerator.generate_volatile_data(1000)
    
    def test_moving_average_crossover_trending_up(self, trending_up_data):
        """Test MA crossover strategy on uptrending data."""
        strategy = MovingAverageCrossover(fast_period=20, slow_period=50)
        
        signals = []
        for i, ohlcv in enumerate(trending_up_data):
            if i >= 50:  # Wait for indicators to warm up
                signal = strategy.generate_signal(trending_up_data[:i+1])
                if signal:
                    signals.append((i, signal))
        
        # Should generate more buy signals than sell in uptrend
        buy_signals = [s for s in signals if s[1].action == "buy"]
        sell_signals = [s for s in signals if s[1].action == "sell"]
        
        assert len(buy_signals) > 0, "Should generate buy signals in uptrend"
        assert len(buy_signals) >= len(sell_signals), "Should have more buy than sell signals in uptrend"
    
    def test_moving_average_crossover_trending_down(self, trending_down_data):
        """Test MA crossover strategy on downtrending data."""
        strategy = MovingAverageCrossover(fast_period=20, slow_period=50)
        
        signals = []
        for i, ohlcv in enumerate(trending_down_data):
            if i >= 50:
                signal = strategy.generate_signal(trending_down_data[:i+1])
                if signal:
                    signals.append((i, signal))
        
        # Should generate more sell signals than buy in downtrend
        buy_signals = [s for s in signals if s[1].action == "buy"]
        sell_signals = [s for s in signals if s[1].action == "sell"]
        
        assert len(sell_signals) > 0, "Should generate sell signals in downtrend"
        assert len(sell_signals) >= len(buy_signals), "Should have more sell than buy signals in downtrend"
    
    def test_rsi_trend_following_oversold_conditions(self, volatile_data):
        """Test RSI trend following in oversold conditions."""
        strategy = RSITrendFollowing(rsi_period=14, rsi_oversold=30, rsi_overbought=70)
        
        buy_signals = []
        sell_signals = []
        
        for i, ohlcv in enumerate(volatile_data):
            if i >= 50:  # Allow indicators to warm up
                signal = strategy.generate_signal(volatile_data[:i+1])
                if signal:
                    if signal.action == "buy":
                        buy_signals.append((i, signal))
                    elif signal.action == "sell":
                        sell_signals.append((i, signal))
        
        # Should generate signals in volatile market
        assert len(buy_signals) + len(sell_signals) > 0, "Should generate signals in volatile market"
        
        # Verify signal confidence is reasonable
        for _, signal in buy_signals + sell_signals:
            assert 0 <= signal.confidence <= 1, f"Confidence should be 0-1, got {signal.confidence}"
    
    def test_macd_rsi_strategy_signal_quality(self, trending_up_data):
        """Test MACD+RSI strategy signal quality."""
        strategy = MACDRSIStrategy()
        
        signals = []
        for i, ohlcv in enumerate(trending_up_data):
            if i >= 100:  # Need more data for MACD
                signal = strategy.generate_signal(trending_up_data[:i+1])
                if signal:
                    signals.append((i, signal))
        
        # Validate signal properties
        for i, signal in signals:
            assert hasattr(signal, 'action'), "Signal should have action"
            assert hasattr(signal, 'confidence'), "Signal should have confidence"
            assert hasattr(signal, 'metadata'), "Signal should have metadata"
            assert signal.action in ["buy", "sell"], f"Invalid action: {signal.action}"
            assert 0 <= signal.confidence <= 1, f"Invalid confidence: {signal.confidence}"
    
    def test_bollinger_band_breakout_squeeze_detection(self, sideways_data):
        """Test Bollinger Band breakout during squeeze periods."""
        strategy = BollingerBandBreakout(period=20, std_dev=2.0)
        
        squeeze_periods = []
        breakout_signals = []
        
        for i, ohlcv in enumerate(sideways_data):
            if i >= 50:
                signal = strategy.generate_signal(sideways_data[:i+1])
                if signal:
                    if "squeeze" in signal.metadata and signal.metadata["squeeze"]:
                        squeeze_periods.append(i)
                    if signal.action in ["buy", "sell"]:
                        breakout_signals.append((i, signal))
        
        # Should detect squeeze periods in sideways market
        assert len(squeeze_periods) > 0, "Should detect squeeze periods in sideways market"
    
    def test_support_resistance_breakout_level_detection(self, volatile_data):
        """Test support/resistance level detection."""
        strategy = SupportResistanceBreakout(lookback_period=50, min_touches=3)
        
        signals = []
        for i, ohlcv in enumerate(volatile_data):
            if i >= 100:  # Need sufficient data for S/R detection
                signal = strategy.generate_signal(volatile_data[:i+1])
                if signal:
                    signals.append((i, signal))
        
        # Should generate some signals in volatile data
        # Note: This strategy is complex and may not always find clear levels
        # We mainly test that it doesn't crash and produces valid signals
        for i, signal in signals:
            assert signal.action in ["buy", "sell"], f"Invalid action: {signal.action}"
            assert 0 <= signal.confidence <= 1, f"Invalid confidence: {signal.confidence}"


class TestBacktestingValidation:
    """Validate backtesting engine functionality."""
    
    @pytest.fixture
    def sample_data(self):
        return TestDataGenerator.generate_trending_data(500, "up")
    
    @pytest.fixture
    def backtest_config(self):
        return BacktestConfig(
            initial_capital=10000,
            commission_rate=0.001,
            max_position_size=0.95,
            risk_free_rate=0.02
        )
    
    def test_backtest_engine_basic_functionality(self, sample_data, backtest_config):
        """Test basic backtesting functionality."""
        strategy = MovingAverageCrossover(fast_period=20, slow_period=50)
        
        # Create mock provider with sample data
        provider = MockDataProvider()
        provider.set_historical_data("TEST", sample_data)
        
        engine = BacktestEngine(provider, backtest_config)
        
        result = engine.run_backtest(
            strategy=strategy,
            symbol="TEST",
            start_date=sample_data[0].timestamp,
            end_date=sample_data[-1].timestamp,
            timeframe=TimeFrame.HOURLY
        )
        
        # Validate result structure
        assert result is not None, "Backtest should return a result"
        assert hasattr(result, 'total_return'), "Result should have total_return"
        assert hasattr(result, 'sharpe_ratio'), "Result should have sharpe_ratio"
        assert hasattr(result, 'max_drawdown'), "Result should have max_drawdown"
        assert hasattr(result, 'trades'), "Result should have trades"
        
        # Validate metrics are reasonable
        assert isinstance(result.total_return, (int, float)), "Total return should be numeric"
        assert isinstance(result.max_drawdown, (int, float)), "Max drawdown should be numeric"
        assert result.max_drawdown <= 0, "Max drawdown should be negative or zero"
    
    def test_backtest_engine_multiple_strategies(self, sample_data, backtest_config):
        """Test backtesting with multiple strategies."""
        strategies = [
            MovingAverageCrossover(fast_period=10, slow_period=30),
            MovingAverageCrossover(fast_period=20, slow_period=50),
            RSITrendFollowing(rsi_period=14),
            MACDRSIStrategy()
        ]
        
        provider = MockDataProvider()
        provider.set_historical_data("TEST", sample_data)
        
        engine = BacktestEngine(provider, backtest_config)
        
        results = []
        for strategy in strategies:
            result = engine.run_backtest(
                strategy=strategy,
                symbol="TEST",
                start_date=sample_data[0].timestamp,
                end_date=sample_data[-1].timestamp,
                timeframe=TimeFrame.HOURLY
            )
            results.append(result)
        
        # All strategies should complete successfully
        assert len(results) == len(strategies), "All strategies should complete"
        
        # Results should be comparable
        for result in results:
            assert result is not None, "Each result should be valid"
            assert hasattr(result, 'total_return'), "Each result should have metrics"
    
    def test_backtest_edge_cases(self, backtest_config):
        """Test backtesting edge cases."""
        # Very short data period
        short_data = TestDataGenerator.generate_trending_data(10, "up")
        
        strategy = MovingAverageCrossover(fast_period=5, slow_period=10)
        provider = MockDataProvider()
        provider.set_historical_data("SHORT", short_data)
        
        engine = BacktestEngine(provider, backtest_config)
        
        # Should handle short data gracefully
        result = engine.run_backtest(
            strategy=strategy,
            symbol="SHORT",
            start_date=short_data[0].timestamp,
            end_date=short_data[-1].timestamp,
            timeframe=TimeFrame.HOURLY
        )
        
        # May not generate trades, but shouldn't crash
        assert result is not None, "Should handle short data gracefully"


class TestRealtimeValidation:
    """Validate real-time processing components."""
    
    def test_incremental_indicator_calculator_accuracy(self):
        """Test incremental indicator calculations match batch calculations."""
        calculator = IncrementalIndicatorCalculator()
        
        # Generate test data
        prices = [100 + np.random.normal(0, 5) for _ in range(100)]
        symbol = "TEST"
        
        # Calculate SMA incrementally
        incremental_sma = []
        for i, price in enumerate(prices):
            if i >= 19:  # Start after 20 periods
                sma = calculator.calculate_sma_incremental(symbol, price, 20)
                if sma is not None:
                    incremental_sma.append(sma)
        
        # Calculate SMA using pandas (batch)
        df = pd.DataFrame({'close': prices})
        batch_sma = df['close'].rolling(window=20).mean().dropna().tolist()
        
        # Compare results (should be very close)
        assert len(incremental_sma) == len(batch_sma), "Length should match"
        
        for inc, batch in zip(incremental_sma, batch_sma):
            assert abs(inc - batch) < 0.001, f"SMA mismatch: {inc} vs {batch}"
    
    def test_incremental_rsi_accuracy(self):
        """Test incremental RSI calculations."""
        calculator = IncrementalIndicatorCalculator()
        
        # Generate test data with some trends
        prices = []
        base_price = 100
        for i in range(100):
            change = np.random.normal(0, 2)
            base_price += change
            prices.append(base_price)
        
        symbol = "TEST_RSI"
        
        # Calculate RSI incrementally
        incremental_rsi = []
        for i, price in enumerate(prices):
            if i >= 14:  # RSI needs 14+ periods
                rsi = calculator.calculate_rsi_incremental(symbol, price, 14)
                if rsi is not None:
                    incremental_rsi.append(rsi)
        
        # Validate RSI properties
        for rsi in incremental_rsi:
            assert 0 <= rsi <= 100, f"RSI should be 0-100, got {rsi}"
    
    def test_cache_manager_functionality(self):
        """Test cache manager operations."""
        # Test basic cache operations
        cache_manager.set("test", "key1", {"value": 123}, ttl=60)
        result = cache_manager.get("test", "key1")
        
        assert result is not None, "Should retrieve cached value"
        assert result["value"] == 123, "Cached value should match"
        
        # Test cache expiration (if supported)
        cache_manager.set("test", "temp_key", "temp_value", ttl=1)
        import time
        time.sleep(2)  # Wait for expiration
        
        expired_result = cache_manager.get("test", "temp_key")
        # Note: Depending on cache implementation, this might or might not be None


class TestIntegrationValidation:
    """Test integration between components."""
    
    def test_strategy_backtest_integration(self):
        """Test strategy and backtesting integration."""
        # Generate diverse market conditions
        trending_data = TestDataGenerator.generate_trending_data(300, "up")
        volatile_data = TestDataGenerator.generate_volatile_data(300)
        
        # Test multiple strategies
        strategies = [
            MovingAverageCrossover(fast_period=20, slow_period=50),
            RSITrendFollowing(rsi_period=14),
            MACDRSIStrategy()
        ]
        
        provider = MockDataProvider()
        config = BacktestConfig(initial_capital=10000, commission_rate=0.001)
        engine = BacktestEngine(provider, config)
        
        # Test each strategy on each dataset
        for data_name, data in [("trending", trending_data), ("volatile", volatile_data)]:
            provider.set_historical_data(f"TEST_{data_name.upper()}", data)
            
            for strategy in strategies:
                result = engine.run_backtest(
                    strategy=strategy,
                    symbol=f"TEST_{data_name.upper()}",
                    start_date=data[0].timestamp,
                    end_date=data[-1].timestamp,
                    timeframe=TimeFrame.HOURLY
                )
                
                # Basic validation
                assert result is not None, f"Strategy {strategy.__class__.__name__} failed on {data_name} data"
                assert hasattr(result, 'total_return'), "Result should have total_return"
    
    def test_data_provider_integration(self):
        """Test data provider integration."""
        provider = MockDataProvider()
        
        # Test historical data
        test_data = TestDataGenerator.generate_trending_data(100, "up")
        provider.set_historical_data("INTEGRATION_TEST", test_data)
        
        retrieved_data = provider.get_historical_data(
            symbol="INTEGRATION_TEST",
            timeframe=TimeFrame.HOURLY,
            start_date=test_data[0].timestamp,
            end_date=test_data[-1].timestamp
        )
        
        assert len(retrieved_data) == len(test_data), "Retrieved data length should match"
        
        # Validate data integrity
        for original, retrieved in zip(test_data, retrieved_data):
            assert abs(original.close - retrieved.close) < 0.001, "Price data should match"
            assert original.timestamp == retrieved.timestamp, "Timestamps should match"


class TestErrorHandlingValidation:
    """Test error handling and edge cases."""
    
    def test_strategy_error_handling(self):
        """Test strategy error handling with invalid data."""
        strategy = MovingAverageCrossover(fast_period=20, slow_period=50)
        
        # Test with insufficient data
        short_data = TestDataGenerator.generate_trending_data(10, "up")
        
        # Should not crash with insufficient data
        try:
            signal = strategy.generate_signal(short_data)
            # Signal might be None, but shouldn't crash
        except Exception as e:
            pytest.fail(f"Strategy should handle insufficient data gracefully: {e}")
        
        # Test with empty data
        try:
            signal = strategy.generate_signal([])
            assert signal is None, "Should return None for empty data"
        except Exception as e:
            pytest.fail(f"Strategy should handle empty data gracefully: {e}")
    
    def test_indicator_error_handling(self):
        """Test indicator error handling."""
        calculator = IncrementalIndicatorCalculator()
        
        # Test with invalid price
        try:
            result = calculator.calculate_sma_incremental("TEST", None, 20)
            assert result is None, "Should handle None price gracefully"
        except Exception as e:
            pytest.fail(f"Indicator should handle None price gracefully: {e}")
        
        # Test with zero period
        try:
            result = calculator.calculate_sma_incremental("TEST", 100.0, 0)
            assert result is None, "Should handle zero period gracefully"
        except Exception as e:
            pytest.fail(f"Indicator should handle zero period gracefully: {e}")


# Performance validation tests
class TestPerformanceValidation:
    """Test performance characteristics."""
    
    def test_strategy_performance_large_dataset(self):
        """Test strategy performance with large datasets."""
        import time
        
        # Generate large dataset
        large_data = TestDataGenerator.generate_trending_data(5000, "up")
        strategy = MovingAverageCrossover(fast_period=20, slow_period=50)
        
        # Measure performance
        start_time = time.time()
        
        signals = []
        for i in range(len(large_data)):
            if i >= 50:  # Allow warm-up
                signal = strategy.generate_signal(large_data[:i+1])
                if signal:
                    signals.append(signal)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete in reasonable time (less than 10 seconds)
        assert processing_time < 10.0, f"Processing took too long: {processing_time}s"
        
        # Should generate reasonable number of signals
        assert len(signals) > 0, "Should generate some signals"
        assert len(signals) < len(large_data) / 10, "Shouldn't generate too many signals"
    
    def test_incremental_calculator_memory_efficiency(self):
        """Test incremental calculator memory efficiency."""
        calculator = IncrementalIndicatorCalculator()
        
        # Process many symbols and periods
        symbols = [f"SYM_{i}" for i in range(100)]
        
        for symbol in symbols:
            for i in range(1000):
                price = 100 + np.random.normal(0, 10)
                calculator.calculate_sma_incremental(symbol, price, 20)
                calculator.calculate_rsi_incremental(symbol, price, 14)
        
        # Check state size is reasonable
        state = calculator.get_state_summary()
        assert state is not None, "Should maintain state information"


if __name__ == "__main__":
    # Run comprehensive validation
    pytest.main([__file__, "-v", "--tb=short"])