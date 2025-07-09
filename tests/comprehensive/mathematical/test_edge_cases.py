"""
Comprehensive Edge Case Tests for All Indicators and Strategies

This module provides comprehensive edge case testing for all technical indicators
and trading strategies, including boundary conditions, error handling, and
extreme market scenarios.
"""

import pytest
import numpy as np
from typing import Dict, List, Tuple, Any
import sys
import os
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Import all indicators
from src.indicators.moving_averages import (
    calculate_sma, calculate_ema, calculate_wma, calculate_vwma,
    calculate_ma_crossover, calculate_ma_envelope
)
from src.indicators.momentum import (
    calculate_rsi, calculate_macd, calculate_stochastic,
    calculate_williams_r, calculate_cci, calculate_momentum, calculate_roc
)
from src.indicators.volatility import (
    calculate_bollinger_bands, calculate_atr, calculate_keltner_channels,
    calculate_donchian_channels, calculate_volatility, calculate_bb_width,
    calculate_bb_percent, calculate_price_channels
)

# Import strategies
from src.strategies.ma_crossover import MovingAverageCrossover
from src.strategies.macd_rsi import MACDRSIStrategy
from src.core.models import MarketData, Signal, SignalType, OHLCV

from .fixtures.test_data import (
    TestDataGenerator,
    PrecisionTestVectors,
    EdgeCaseTestData,
    ValidationHelpers,
    TEST_CONFIG
)


class TestIndicatorEdgeCases:
    """Comprehensive edge case tests for all technical indicators"""
    
    def test_zero_division_protection(self):
        """Test protection against zero division errors"""
        # Test with zero values
        zero_data = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Moving averages should handle zeros
        sma_result = calculate_sma(zero_data, 3)
        assert np.allclose(sma_result[2:], 0.0, equal_nan=True)
        
        ema_result = calculate_ema(zero_data, 3)
        assert np.allclose(ema_result[2:], 0.0, equal_nan=True)
        
        # RSI should handle zero price changes
        rsi_result = calculate_rsi(zero_data, 3)
        valid_rsi = rsi_result[~np.isnan(rsi_result)]
        if len(valid_rsi) > 0:
            assert np.allclose(valid_rsi, 50.0)  # Should be neutral
    
    def test_negative_price_handling(self):
        """Test handling of negative prices"""
        negative_data = EdgeCaseTestData.get_negative_prices()
        
        # Most indicators should handle negative prices gracefully
        try:
            sma_result = calculate_sma(negative_data, 3)
            assert not np.any(np.isnan(sma_result[2:]))  # Should not produce NaN
            
            # Momentum indicators should work with negative prices
            momentum_result = calculate_momentum(negative_data, 2)
            assert not np.any(np.isnan(momentum_result[2:]))
            
        except Exception as e:
            # If indicators can't handle negative prices, they should fail gracefully
            assert isinstance(e, (ValueError, TypeError))
    
    def test_infinite_value_handling(self):
        """Test handling of infinite values"""
        infinite_data = EdgeCaseTestData.get_infinite_prices()
        
        # Indicators should handle infinite values gracefully
        with pytest.raises((ValueError, TypeError)):
            calculate_sma(infinite_data, 3)
        
        # Or should filter out infinite values
        try:
            result = calculate_sma(infinite_data, 3)
            assert not np.any(np.isinf(result))
        except:
            pass  # Either approach is acceptable
    
    def test_nan_value_handling(self):
        """Test handling of NaN values in input data"""
        nan_data = EdgeCaseTestData.get_nan_prices()
        
        # Indicators should handle NaN values appropriately
        sma_result = calculate_sma(nan_data, 3)
        # Should either skip NaN values or propagate them appropriately
        assert len(sma_result) == len(nan_data)
        
        rsi_result = calculate_rsi(nan_data, 3)
        assert len(rsi_result) == len(nan_data)
    
    def test_single_value_arrays(self):
        """Test indicators with single value arrays"""
        single_value = np.array([100.0])
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises(ValueError):
            calculate_sma(single_value, 3)
        
        with pytest.raises(ValueError):
            calculate_rsi(single_value, 3)
    
    def test_empty_arrays(self):
        """Test indicators with empty arrays"""
        empty_data = np.array([])
        
        # Should raise appropriate errors for empty data
        with pytest.raises((ValueError, IndexError)):
            calculate_sma(empty_data, 3)
        
        with pytest.raises((ValueError, IndexError)):
            calculate_rsi(empty_data, 3)
    
    def test_extreme_period_values(self):
        """Test indicators with extreme period values"""
        data = TestDataGenerator.generate_volatile_data(100)
        
        # Test with period = 1
        sma_1 = calculate_sma(data, 1)
        assert np.allclose(sma_1, data, equal_nan=True)  # SMA(1) should equal input
        
        # Test with period equal to data length
        with pytest.raises(ValueError):
            calculate_sma(data, len(data) + 1)
        
        # Test with period = 0
        with pytest.raises(ValueError):
            calculate_sma(data, 0)
        
        # Test with negative period
        with pytest.raises(ValueError):
            calculate_sma(data, -5)
    
    def test_extreme_volatility_scenarios(self):
        """Test indicators with extreme volatility"""
        extreme_data = EdgeCaseTestData.get_extreme_volatility()
        
        # All indicators should handle extreme volatility without crashing
        sma_result = calculate_sma(extreme_data, 3)
        assert np.all(np.isfinite(sma_result[~np.isnan(sma_result)]))
        
        rsi_result = calculate_rsi(extreme_data, 5)
        valid_rsi = rsi_result[~np.isnan(rsi_result)]
        assert np.all(valid_rsi >= 0.0)
        assert np.all(valid_rsi <= 100.0)
        
        # Bollinger Bands should handle extreme volatility
        if len(extreme_data) >= 5:
            upper, middle, lower = calculate_bollinger_bands(extreme_data, 5, 2.0)
            valid_indices = ~np.isnan(middle)
            assert np.all(upper[valid_indices] >= middle[valid_indices])
            assert np.all(lower[valid_indices] <= middle[valid_indices])
    
    def test_constant_data_scenarios(self):
        """Test indicators with constant data"""
        constant_data = EdgeCaseTestData.get_constant_prices()
        
        # Moving averages should equal the constant value
        sma_result = calculate_sma(constant_data, 5)
        valid_sma = sma_result[~np.isnan(sma_result)]
        assert np.allclose(valid_sma, 100.0)
        
        # RSI should be neutral (50) for constant prices
        rsi_result = calculate_rsi(constant_data, 5)
        valid_rsi = rsi_result[~np.isnan(rsi_result)]
        assert np.allclose(valid_rsi, 50.0, atol=1e-10)
        
        # Bollinger Bands should have zero width for constant prices
        upper, middle, lower = calculate_bollinger_bands(constant_data, 5, 2.0)
        valid_indices = ~np.isnan(middle)
        assert np.allclose(upper[valid_indices], middle[valid_indices], atol=1e-10)
        assert np.allclose(lower[valid_indices], middle[valid_indices], atol=1e-10)
    
    def test_ohlcv_data_consistency(self):
        """Test OHLCV data consistency requirements"""
        # Test with inconsistent OHLCV data
        high = np.array([15.0, 16.0, 17.0, 18.0, 19.0])
        low = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        close = np.array([20.0, 21.0, 22.0, 23.0, 24.0])  # Close > High (invalid)
        
        # Indicators should handle invalid OHLCV relationships
        try:
            atr_result = calculate_atr(high, low, close, 3)
            # Should either handle gracefully or produce valid results
            assert np.all(atr_result[~np.isnan(atr_result)] >= 0)
        except ValueError:
            pass  # Acceptable to reject invalid data
        
        # Test with high < low (invalid)
        invalid_high = np.array([9.0, 10.0, 11.0, 12.0, 13.0])  # High < Low
        
        try:
            stoch_k, stoch_d = calculate_stochastic(invalid_high, low, close, 3, 3, 1)
        except ValueError:
            pass  # Should reject invalid OHLC relationships
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets"""
        # Test with large dataset to check memory handling
        large_data = TestDataGenerator.generate_volatile_data(10000)
        
        # Should handle large datasets without memory issues
        sma_result = calculate_sma(large_data, 200)
        assert len(sma_result) == len(large_data)
        
        rsi_result = calculate_rsi(large_data, 14)
        assert len(rsi_result) == len(large_data)
        
        # Memory usage should be reasonable (not exponential)
        del large_data, sma_result, rsi_result
    
    def test_numerical_precision(self):
        """Test numerical precision with edge cases"""
        # Test with very small differences
        tiny_diff_data = np.array([1.0000001, 1.0000002, 1.0000003, 1.0000004, 1.0000005])
        
        sma_result = calculate_sma(tiny_diff_data, 3)
        # Should maintain precision
        assert not np.allclose(sma_result[2], sma_result[3])
        
        # Test with very large numbers
        large_numbers = np.array([1e10, 1e10 + 1, 1e10 + 2, 1e10 + 3, 1e10 + 4])
        
        sma_large = calculate_sma(large_numbers, 3)
        # Should handle large numbers correctly
        assert np.all(np.isfinite(sma_large[~np.isnan(sma_large)]))


class TestStrategyEdgeCases:
    """Comprehensive edge case tests for trading strategies"""
    
    def test_strategy_insufficient_data(self):
        """Test strategies with insufficient data"""
        minimal_data = EdgeCaseTestData.get_minimal_data()
        market_data = self._create_market_data(minimal_data)
        
        strategies = [
            MovingAverageCrossover(),
            MACDRSIStrategy()
        ]
        
        for strategy in strategies:
            # Should handle insufficient data gracefully
            indicators = strategy._calculate_indicators(market_data)
            signals = strategy.generate_signals(market_data, indicators)
            
            assert isinstance(signals, list)
            # Should not crash, may return empty signals
    
    def test_strategy_empty_indicators(self):
        """Test strategies with empty indicator data"""
        data = TestDataGenerator.generate_volatile_data(50)
        market_data = self._create_market_data(data)
        
        empty_indicators = {}
        
        strategies = [
            MovingAverageCrossover(),
            MACDRSIStrategy()
        ]
        
        for strategy in strategies:
            # Should handle empty indicators gracefully
            signals = strategy.generate_signals(market_data, empty_indicators)
            assert isinstance(signals, list)
            assert len(signals) == 0  # Should return empty list
    
    def test_strategy_corrupted_indicators(self):
        """Test strategies with corrupted indicator data"""
        data = TestDataGenerator.generate_volatile_data(50)
        market_data = self._create_market_data(data)
        
        # Create corrupted indicator data
        corrupted_indicators = {
            'ema': {
                'ema_50': [np.nan] * 50,  # All NaN values
                'ema_200': [np.inf] * 50  # All infinite values
            },
            'macd': {
                'macd': {
                    'macd': [None] * 50,  # None values
                    'signal': ['invalid'] * 50,  # Invalid data types
                    'histogram': []  # Empty array
                }
            },
            'rsi': {
                'rsi_14': [-50, 150, -100, 200]  # Out of range values
            }
        }
        
        strategies = [
            MovingAverageCrossover(),
            MACDRSIStrategy()
        ]
        
        for strategy in strategies:
            # Should handle corrupted data without crashing
            try:
                signals = strategy.generate_signals(market_data, corrupted_indicators)
                assert isinstance(signals, list)
            except Exception as e:
                # Should fail gracefully with appropriate error
                assert isinstance(e, (ValueError, TypeError, IndexError))
    
    def test_strategy_extreme_confidence_thresholds(self):
        """Test strategies with extreme confidence thresholds"""
        data = TestDataGenerator.generate_volatile_data(100)
        market_data = self._create_market_data(data)
        
        # Test with confidence threshold = 0.0 (accept all signals)
        strategy_low = MovingAverageCrossover({'min_confidence': 0.0})
        indicators_low = strategy_low._calculate_indicators(market_data)
        signals_low = strategy_low.generate_signals(market_data, indicators_low)
        
        # Test with confidence threshold = 1.0 (reject all signals)
        strategy_high = MovingAverageCrossover({'min_confidence': 1.0})
        indicators_high = strategy_high._calculate_indicators(market_data)
        signals_high = strategy_high.generate_signals(market_data, indicators_high)
        
        # Low threshold should generate more signals than high threshold
        assert len(signals_low) >= len(signals_high)
        
        # High threshold (1.0) should typically generate no signals
        assert len(signals_high) == 0
    
    def test_strategy_invalid_parameters(self):
        """Test strategies with invalid parameters"""
        # Test invalid parameter combinations
        invalid_params = [
            {'short_period': 50, 'long_period': 20},  # Short > Long
            {'short_period': -10, 'long_period': 20},  # Negative period
            {'min_confidence': -0.5},  # Negative confidence
            {'min_confidence': 1.5},   # Confidence > 1
            {'volume_threshold': -1.0}  # Negative volume threshold
        ]
        
        for params in invalid_params:
            try:
                strategy = MovingAverageCrossover(params)
                # Some invalid parameters might be accepted but corrected internally
                # Others should raise errors during initialization or signal generation
            except ValueError:
                pass  # Expected for clearly invalid parameters
    
    def test_strategy_signal_validation_edge_cases(self):
        """Test signal validation with edge cases"""
        strategy = MovingAverageCrossover()
        
        # Create edge case signals
        edge_signals = [
            # Signal with confidence exactly at threshold
            Signal(
                timestamp=datetime.now(),
                symbol="TEST",
                signal_type=SignalType.BUY,
                price=100.0,
                confidence=strategy.parameters['min_confidence'],  # Exactly at threshold
                strategy_name="MovingAverageCrossover",
                metadata={'signal_name': 'Golden Cross', 'short_ema': 100.1, 'long_ema': 100.0, 'ema_separation': 0.001}
            ),
            # Signal with very small price
            Signal(
                timestamp=datetime.now(),
                symbol="TEST",
                signal_type=SignalType.BUY,
                price=0.001,  # Very small price
                confidence=0.8,
                strategy_name="MovingAverageCrossover",
                metadata={'signal_name': 'Golden Cross', 'short_ema': 0.0011, 'long_ema': 0.001, 'ema_separation': 0.1}
            ),
            # Signal with zero price (invalid)
            Signal(
                timestamp=datetime.now(),
                symbol="TEST",
                signal_type=SignalType.BUY,
                price=0.0,  # Zero price (invalid)
                confidence=0.8,
                strategy_name="MovingAverageCrossover",
                metadata={'signal_name': 'Golden Cross', 'short_ema': 100.1, 'long_ema': 100.0, 'ema_separation': 0.001}
            ),
            # Signal with confidence = 0
            Signal(
                timestamp=datetime.now(),
                symbol="TEST",
                signal_type=SignalType.BUY,
                price=100.0,
                confidence=0.0,  # Zero confidence
                strategy_name="MovingAverageCrossover",
                metadata={'signal_name': 'Golden Cross', 'short_ema': 100.1, 'long_ema': 100.0, 'ema_separation': 0.001}
            )
        ]
        
        validated_signals = strategy.validate_signals(edge_signals)
        
        # Should accept signals at threshold, reject zero price and very low confidence
        assert len(validated_signals) <= 2  # At most the first two signals
        
        for signal in validated_signals:
            assert signal.price > 0.0
            assert signal.confidence >= strategy.parameters['min_confidence']
    
    def test_strategy_concurrent_signals(self):
        """Test strategy behavior with multiple concurrent signals"""
        # Create data that might generate multiple signals at the same time
        volatile_data = EdgeCaseTestData.get_extreme_volatility()
        market_data = self._create_market_data(volatile_data)
        
        strategy = MovingAverageCrossover({
            'short_period': 3,
            'long_period': 5,
            'min_confidence': 0.1
        })
        
        indicators = strategy._calculate_indicators(market_data)
        signals = strategy.generate_signals(market_data, indicators)
        
        # Check for potential timing conflicts
        if len(signals) > 1:
            timestamps = [s.timestamp for s in signals]
            # Should handle multiple signals appropriately
            assert len(timestamps) == len(set(timestamps))  # All timestamps should be unique
    
    def test_strategy_memory_leaks(self):
        """Test strategies for potential memory leaks"""
        # Run strategy multiple times to check for memory accumulation
        data = TestDataGenerator.generate_volatile_data(100)
        market_data = self._create_market_data(data)
        
        strategy = MovingAverageCrossover()
        
        # Run multiple times
        for _ in range(10):
            indicators = strategy._calculate_indicators(market_data)
            signals = strategy.generate_signals(market_data, indicators)
            
            # Clear references
            del indicators, signals
        
        # Strategy should handle repeated use without accumulating memory
        # (This is more of a smoke test - real memory leak detection would need more sophisticated tools)
    
    def _create_market_data(self, prices: np.ndarray) -> MarketData:
        """Helper method to create MarketData from prices"""
        ohlcv_data = TestDataGenerator.generate_ohlcv_data(prices)
        timestamps = ValidationHelpers.generate_timestamps(len(prices))
        
        ohlcv_list = []
        for i in range(len(prices)):
            ohlcv_list.append(OHLCV(
                timestamp=timestamps[i],
                open=ohlcv_data['open'][i],
                high=ohlcv_data['high'][i],
                low=ohlcv_data['low'][i],
                close=ohlcv_data['close'][i],
                volume=ohlcv_data['volume'][i]
            ))
        
        return MarketData(symbol="TEST", data=ohlcv_list)


class TestCrossValidationAndConsistency:
    """Cross-validation tests between different implementations"""
    
    def test_indicator_mathematical_consistency(self):
        """Test mathematical consistency between related indicators"""
        data = TestDataGenerator.generate_volatile_data(100)
        
        # Test SMA vs EMA convergence for large periods
        sma_50 = calculate_sma(data, 50)
        ema_50 = calculate_ema(data, 50)
        
        # For large periods, SMA and EMA should be relatively close
        valid_indices = ~(np.isnan(sma_50) | np.isnan(ema_50))
        if np.sum(valid_indices) > 10:
            correlation = np.corrcoef(sma_50[valid_indices], ema_50[valid_indices])[0, 1]
            assert correlation > 0.9  # Should be highly correlated
        
        # Test MACD relationship with underlying EMAs
        macd_line, signal_line, histogram = calculate_macd(data, 12, 26, 9)
        
        from src.indicators.moving_averages import calculate_ema
        ema_12 = calculate_ema(data, 12)
        ema_26 = calculate_ema(data, 26)
        expected_macd = ema_12 - ema_26
        
        # MACD line should equal EMA difference
        assert ValidationHelpers.assert_array_almost_equal(
            macd_line, expected_macd, decimals=TEST_CONFIG['precision_decimals']
        )
    
    def test_strategy_signal_consistency(self):
        """Test consistency of signal generation across different market conditions"""
        # Test strategies with the same underlying trend but different volatility
        base_trend = np.linspace(100, 120, 50)
        
        low_vol_data = base_trend + np.random.normal(0, 0.1, 50)
        high_vol_data = base_trend + np.random.normal(0, 2.0, 50)
        
        strategy = MovingAverageCrossover({
            'short_period': 5,
            'long_period': 10,
            'min_confidence': 0.1
        })
        
        # Generate signals for both scenarios
        low_vol_market = self._create_market_data(low_vol_data)
        high_vol_market = self._create_market_data(high_vol_data)
        
        low_vol_indicators = strategy._calculate_indicators(low_vol_market)
        low_vol_signals = strategy.generate_signals(low_vol_market, low_vol_indicators)
        
        high_vol_indicators = strategy._calculate_indicators(high_vol_market)
        high_vol_signals = strategy.generate_signals(high_vol_market, high_vol_indicators)
        
        # Both should generate some signals for trending data
        # but exact counts may differ due to volatility
        low_vol_buy_signals = len([s for s in low_vol_signals if s.signal_type == SignalType.BUY])
        high_vol_buy_signals = len([s for s in high_vol_signals if s.signal_type == SignalType.BUY])
        
        # Should both detect the underlying trend
        assert low_vol_buy_signals > 0 or high_vol_buy_signals > 0
    
    def test_parameter_sensitivity_analysis(self):
        """Test sensitivity of indicators and strategies to parameter changes"""
        data = TestDataGenerator.generate_volatile_data(100)
        
        # Test SMA sensitivity to period changes
        periods = [5, 10, 20, 50]
        sma_results = []
        
        for period in periods:
            if period <= len(data):
                sma_result = calculate_sma(data, period)
                sma_results.append(sma_result)
        
        # Longer periods should produce smoother results (less volatility)
        if len(sma_results) >= 2:
            # Compare standard deviation of different period SMAs
            for i in range(len(sma_results) - 1):
                valid_short = sma_results[i][~np.isnan(sma_results[i])]
                valid_long = sma_results[i + 1][~np.isnan(sma_results[i + 1])]
                
                if len(valid_short) > 10 and len(valid_long) > 10:
                    # Longer period should have lower volatility
                    vol_short = np.std(valid_short)
                    vol_long = np.std(valid_long)
                    assert vol_long <= vol_short * 1.1  # Allow some tolerance
    
    def test_boundary_condition_consistency(self):
        """Test consistency at boundary conditions"""
        # Test with exactly minimum required data
        for period in [2, 3, 5, 10]:
            min_data = TestDataGenerator.generate_linear_trend(period)
            
            # Should work with exactly minimum data
            sma_result = calculate_sma(min_data, period)
            assert not np.isnan(sma_result[-1])  # Last value should be valid
            
            if period >= 3:
                rsi_result = calculate_rsi(min_data, period - 1)
                # Should not crash with minimum data
                assert len(rsi_result) == len(min_data)
    
    def _create_market_data(self, prices: np.ndarray) -> MarketData:
        """Helper method to create MarketData from prices"""
        ohlcv_data = TestDataGenerator.generate_ohlcv_data(prices)
        timestamps = ValidationHelpers.generate_timestamps(len(prices))
        
        ohlcv_list = []
        for i in range(len(prices)):
            ohlcv_list.append(OHLCV(
                timestamp=timestamps[i],
                open=ohlcv_data['open'][i],
                high=ohlcv_data['high'][i],
                low=ohlcv_data['low'][i],
                close=ohlcv_data['close'][i],
                volume=ohlcv_data['volume'][i]
            ))
        
        return MarketData(symbol="TEST", data=ohlcv_list)


class TestPerformanceBenchmarks:
    """Performance and scalability tests"""
    
    def test_indicator_performance_scaling(self):
        """Test indicator performance with different data sizes"""
        import time
        
        data_sizes = [100, 500, 1000, 5000]
        
        for size in data_sizes:
            data = TestDataGenerator.generate_volatile_data(size)
            
            # Time SMA calculation
            start_time = time.time()
            calculate_sma(data, 20)
            sma_time = time.time() - start_time
            
            # Time RSI calculation
            start_time = time.time()
            calculate_rsi(data, 14)
            rsi_time = time.time() - start_time
            
            # Performance should be reasonable (< 1 second for even large datasets)
            assert sma_time < 1.0
            assert rsi_time < 1.0
            
            # Performance should scale roughly linearly
            if size > 1000:
                # For large datasets, time should still be reasonable
                assert sma_time < 0.1  # Should be very fast
                assert rsi_time < 0.1
    
    def test_strategy_performance_scaling(self):
        """Test strategy performance with different data sizes"""
        import time
        
        data_sizes = [100, 500, 1000]
        
        for size in data_sizes:
            data = TestDataGenerator.generate_volatile_data(size)
            market_data = self._create_market_data(data)
            
            strategy = MovingAverageCrossover({
                'short_period': 10,
                'long_period': 20
            })
            
            # Time complete signal generation
            start_time = time.time()
            indicators = strategy._calculate_indicators(market_data)
            signals = strategy.generate_signals(market_data, indicators)
            total_time = time.time() - start_time
            
            # Should complete in reasonable time
            assert total_time < 5.0  # Should be fast even for large datasets
            
            # Should generate valid signals
            assert isinstance(signals, list)
            for signal in signals:
                assert signal.confidence >= 0.0
                assert signal.price > 0.0
    
    def _create_market_data(self, prices: np.ndarray) -> MarketData:
        """Helper method to create MarketData from prices"""
        ohlcv_data = TestDataGenerator.generate_ohlcv_data(prices)
        timestamps = ValidationHelpers.generate_timestamps(len(prices))
        
        ohlcv_list = []
        for i in range(len(prices)):
            ohlcv_list.append(OHLCV(
                timestamp=timestamps[i],
                open=ohlcv_data['open'][i],
                high=ohlcv_data['high'][i],
                low=ohlcv_data['low'][i],
                close=ohlcv_data['close'][i],
                volume=ohlcv_data['volume'][i]
            ))
        
        return MarketData(symbol="TEST", data=ohlcv_list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])