"""
Unit tests for technical indicators.

Tests all indicator calculations with known values and edge cases.
"""

import pytest
import numpy as np
import math
from typing import List

from src.indicators.moving_averages import (
    calculate_sma, calculate_ema, calculate_wma, calculate_vwma,
    calculate_ma_crossover, calculate_ma_envelope
)
from src.indicators.momentum import (
    calculate_rsi, calculate_macd, calculate_stochastic,
    calculate_williams_r, calculate_cci, calculate_momentum, calculate_roc
)
from src.indicators.volatility import (
    calculate_bollinger_bands, calculate_atr, calculate_volatility,
    calculate_keltner_channels
)
from src.indicators.levels import (
    calculate_support_resistance, calculate_pivot_points, calculate_fibonacci_levels
)
from src.indicators.calculator import IndicatorCalculator


class TestMovingAverages:
    """Test moving average indicators."""
    
    def test_sma_basic_calculation(self):
        """Test basic SMA calculation."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        period = 3
        
        result = calculate_sma(data, period)
        
        # First two values should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        
        # Check calculated values
        assert result[2] == 2.0  # (1+2+3)/3
        assert result[3] == 3.0  # (2+3+4)/3
        assert result[4] == 4.0  # (3+4+5)/3
        assert result[9] == 9.0  # (8+9+10)/3
    
    def test_sma_single_period(self):
        """Test SMA with period of 1."""
        data = [1, 2, 3, 4, 5]
        period = 1
        
        result = calculate_sma(data, period)
        
        # Should equal original data
        np.testing.assert_array_equal(result, data)
    
    def test_sma_invalid_period(self):
        """Test SMA with invalid period."""
        data = [1, 2, 3, 4, 5]
        
        with pytest.raises(ValueError, match="Period must be at least 1"):
            calculate_sma(data, 0)
    
    def test_sma_insufficient_data(self):
        """Test SMA with insufficient data."""
        data = [1, 2]
        period = 5
        
        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_sma(data, period)
    
    def test_ema_basic_calculation(self):
        """Test basic EMA calculation."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        period = 3
        
        result = calculate_ema(data, period)
        
        # First two values should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        
        # Third value should be SMA
        assert result[2] == 2.0  # (1+2+3)/3
        
        # Check EMA calculation (alpha = 2/(3+1) = 0.5)
        alpha = 2.0 / (period + 1)
        expected_3 = alpha * 4 + (1 - alpha) * 2.0
        assert abs(result[3] - expected_3) < 1e-10
    
    def test_ema_custom_alpha(self):
        """Test EMA with custom alpha."""
        data = [1, 2, 3, 4, 5]
        period = 3
        alpha = 0.3
        
        result = calculate_ema(data, period, alpha)
        
        # Should use custom alpha
        assert not np.isnan(result[2])
    
    def test_ema_invalid_alpha(self):
        """Test EMA with invalid alpha."""
        data = [1, 2, 3, 4, 5]
        period = 3
        
        with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
            calculate_ema(data, period, alpha=1.5)
        
        with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
            calculate_ema(data, period, alpha=0.0)
    
    def test_wma_basic_calculation(self):
        """Test WMA calculation."""
        data = [1, 2, 3, 4, 5]
        period = 3
        
        result = calculate_wma(data, period)
        
        # First two values should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        
        # WMA = (1*1 + 2*2 + 3*3) / (1+2+3) = 14/6 = 2.333...
        expected = (1*1 + 2*2 + 3*3) / (1+2+3)
        assert abs(result[2] - expected) < 1e-10
    
    def test_vwma_basic_calculation(self):
        """Test VWMA calculation."""
        data = [10, 20, 30, 40, 50]
        volume = [100, 200, 300, 400, 500]
        period = 3
        
        result = calculate_vwma(data, volume, period)
        
        # First two values should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        
        # VWMA = (10*100 + 20*200 + 30*300) / (100+200+300)
        expected = (10*100 + 20*200 + 30*300) / (100+200+300)
        assert abs(result[2] - expected) < 1e-10
    
    def test_vwma_zero_volume(self):
        """Test VWMA with zero volume."""
        data = [10, 20, 30]
        volume = [0, 0, 0]
        period = 3
        
        result = calculate_vwma(data, volume, period)
        
        # Should return NaN when volume is zero
        assert np.isnan(result[2])
    
    def test_vwma_mismatched_lengths(self):
        """Test VWMA with mismatched data lengths."""
        data = [10, 20, 30]
        volume = [100, 200]
        period = 2
        
        with pytest.raises(ValueError, match="same length"):
            calculate_vwma(data, volume, period)
    
    def test_ma_crossover_basic(self):
        """Test MA crossover calculation."""
        fast_ma = np.array([1, 2, 3, 2, 1, 2, 3, 4])
        slow_ma = np.array([2, 2, 2, 2, 2, 2, 2, 2])
        
        result = calculate_ma_crossover(fast_ma, slow_ma)
        
        # Check crossover points
        assert result[2] == 1   # Fast crosses above slow
        assert result[4] == -1  # Fast crosses below slow
        assert result[6] == 1   # Fast crosses above slow again
    
    def test_ma_crossover_mismatched_lengths(self):
        """Test MA crossover with mismatched lengths."""
        fast_ma = np.array([1, 2, 3])
        slow_ma = np.array([1, 2])
        
        with pytest.raises(ValueError, match="same length"):
            calculate_ma_crossover(fast_ma, slow_ma)
    
    def test_ma_envelope(self):
        """Test MA envelope calculation."""
        data = [100, 102, 104, 106, 108]
        period = 3
        envelope_pct = 0.05  # 5%
        
        upper, middle, lower = calculate_ma_envelope(data, period, envelope_pct)
        
        # Middle should be SMA
        expected_sma = calculate_sma(data, period)
        np.testing.assert_array_equal(middle, expected_sma)
        
        # Check envelope calculations
        for i in range(len(middle)):
            if not np.isnan(middle[i]):
                assert upper[i] == middle[i] * 1.05
                assert lower[i] == middle[i] * 0.95
    
    def test_ma_envelope_negative_percentage(self):
        """Test MA envelope with negative percentage."""
        data = [100, 102, 104]
        period = 2
        
        with pytest.raises(ValueError, match="non-negative"):
            calculate_ma_envelope(data, period, -0.05)


class TestMomentumIndicators:
    """Test momentum indicators."""
    
    def test_rsi_basic_calculation(self):
        """Test basic RSI calculation."""
        # Known data with expected RSI
        data = [44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 46.08, 45.89,
                46.03, 46.83, 47.69, 46.49, 46.26, 47.09, 47.37, 47.20, 46.21, 46.25]
        period = 14
        
        result = calculate_rsi(data, period)
        
        # First 14 values should be NaN
        for i in range(period):
            assert np.isnan(result[i])
        
        # RSI should be between 0 and 100
        for i in range(period, len(result)):
            assert 0 <= result[i] <= 100
        
        # Check approximate value for known data
        assert 45 <= result[-1] <= 55  # Approximate range
    
    def test_rsi_extreme_values(self):
        """Test RSI with extreme values."""
        # All increasing values should give RSI near 100
        data = list(range(1, 31))
        period = 14
        
        result = calculate_rsi(data, period)
        
        # Later values should be high
        assert result[-1] > 80
        
        # All decreasing values should give RSI near 0
        data = list(range(30, 0, -1))
        result = calculate_rsi(data, period)
        
        # Later values should be low
        assert result[-1] < 20
    
    def test_rsi_insufficient_data(self):
        """Test RSI with insufficient data."""
        data = [1, 2, 3]
        period = 14
        
        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_rsi(data, period)
    
    def test_macd_basic_calculation(self):
        """Test basic MACD calculation."""
        # Create trending data
        data = [100 + i * 0.5 + np.sin(i * 0.1) for i in range(50)]
        fast = 12
        slow = 26
        signal = 9
        
        macd_line, signal_line, histogram = calculate_macd(data, fast, slow, signal)
        
        # Check array lengths
        assert len(macd_line) == len(data)
        assert len(signal_line) == len(data)
        assert len(histogram) == len(data)
        
        # MACD line should start with NaN values
        for i in range(slow - 1):
            assert np.isnan(macd_line[i])
        
        # Histogram = MACD - Signal
        for i in range(len(data)):
            if not (np.isnan(macd_line[i]) or np.isnan(signal_line[i])):
                assert abs(histogram[i] - (macd_line[i] - signal_line[i])) < 1e-10
    
    def test_macd_invalid_periods(self):
        """Test MACD with invalid periods."""
        data = list(range(1, 31))
        
        with pytest.raises(ValueError, match="Fast period must be less than slow"):
            calculate_macd(data, 26, 12, 9)
        
        with pytest.raises(ValueError, match="All periods must be at least 1"):
            calculate_macd(data, 0, 26, 9)
    
    def test_stochastic_basic_calculation(self):
        """Test basic Stochastic calculation."""
        # Create sample data
        high = [105, 110, 108, 107, 112, 115, 113, 111, 109, 114, 116, 118, 120, 119, 117]
        low = [95, 98, 96, 97, 100, 103, 101, 99, 98, 102, 104, 106, 108, 107, 105]
        close = [100, 105, 102, 104, 108, 110, 107, 105, 103, 109, 112, 115, 118, 115, 113]
        
        k_period = 14
        d_period = 3
        
        k_percent, d_percent = calculate_stochastic(high, low, close, k_period, d_period)
        
        # Check array lengths
        assert len(k_percent) == len(close)
        assert len(d_percent) == len(close)
        
        # %K values should be between 0 and 100
        for i in range(k_period - 1, len(k_percent)):
            if not np.isnan(k_percent[i]):
                assert 0 <= k_percent[i] <= 100
    
    def test_stochastic_mismatched_lengths(self):
        """Test Stochastic with mismatched array lengths."""
        high = [105, 110, 108]
        low = [95, 98]
        close = [100, 105, 102]
        
        with pytest.raises(ValueError, match="same length"):
            calculate_stochastic(high, low, close)
    
    def test_williams_r_calculation(self):
        """Test Williams %R calculation."""
        high = [105, 110, 108, 107, 112]
        low = [95, 98, 96, 97, 100]
        close = [100, 105, 102, 104, 108]
        period = 4
        
        result = calculate_williams_r(high, low, close, period)
        
        # Williams %R should be between -100 and 0
        for i in range(period - 1, len(result)):
            if not np.isnan(result[i]):
                assert -100 <= result[i] <= 0
    
    def test_cci_calculation(self):
        """Test CCI calculation."""
        high = [105, 110, 108, 107, 112, 115, 113, 111, 109, 114,
                116, 118, 120, 119, 117, 122, 124, 123, 121, 125]
        low = [95, 98, 96, 97, 100, 103, 101, 99, 98, 102,
               104, 106, 108, 107, 105, 110, 112, 111, 109, 113]
        close = [100, 105, 102, 104, 108, 110, 107, 105, 103, 109,
                 112, 115, 118, 115, 113, 119, 121, 118, 116, 122]
        period = 20
        
        result = calculate_cci(high, low, close, period)
        
        # First 19 values should be NaN
        for i in range(period - 1):
            assert np.isnan(result[i])
        
        # CCI values are typically between -200 and +200, but can exceed
        for i in range(period - 1, len(result)):
            if not np.isnan(result[i]):
                assert -500 <= result[i] <= 500  # Reasonable bounds
    
    def test_momentum_calculation(self):
        """Test momentum calculation."""
        data = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]
        period = 4
        
        result = calculate_momentum(data, period)
        
        # First 4 values should be NaN
        for i in range(period):
            assert np.isnan(result[i])
        
        # Check momentum calculation
        for i in range(period, len(data)):
            expected = data[i] - data[i - period]
            assert abs(result[i] - expected) < 1e-10
    
    def test_roc_calculation(self):
        """Test ROC calculation."""
        data = [100, 102, 104, 106, 108]
        period = 2
        
        result = calculate_roc(data, period)
        
        # First 2 values should be NaN
        for i in range(period):
            assert np.isnan(result[i])
        
        # ROC = ((current - previous) / previous) * 100
        for i in range(period, len(data)):
            expected = ((data[i] - data[i - period]) / data[i - period]) * 100.0
            assert abs(result[i] - expected) < 1e-10
    
    def test_roc_zero_division(self):
        """Test ROC with zero division."""
        data = [0, 100, 200]
        period = 2
        
        result = calculate_roc(data, period)
        
        # Should handle zero division gracefully
        assert np.isnan(result[2])


class TestVolatilityIndicators:
    """Test volatility indicators."""
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation."""
        data = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
        period = 5
        std_dev = 2.0
        
        upper, middle, lower = calculate_bollinger_bands(data, period, std_dev)
        
        # Check array lengths
        assert len(upper) == len(data)
        assert len(middle) == len(data)
        assert len(lower) == len(data)
        
        # First 4 values should be NaN
        for i in range(period - 1):
            assert np.isnan(upper[i])
            assert np.isnan(middle[i])
            assert np.isnan(lower[i])
        
        # Middle band should be SMA
        expected_sma = calculate_sma(data, period)
        np.testing.assert_array_equal(middle, expected_sma)
        
        # Check band relationships
        for i in range(period - 1, len(data)):
            if not np.isnan(middle[i]):
                assert upper[i] > middle[i] > lower[i]
    
    def test_atr_calculation(self):
        """Test ATR calculation."""
        high = [105, 110, 108, 107, 112, 115, 113, 111, 109, 114]
        low = [95, 98, 96, 97, 100, 103, 101, 99, 98, 102]
        close = [100, 105, 102, 104, 108, 110, 107, 105, 103, 109]
        period = 5
        
        result = calculate_atr(high, low, close, period)
        
        # ATR should be positive
        for i in range(period, len(result)):
            if not np.isnan(result[i]):
                assert result[i] > 0
    
    def test_volatility_calculation(self):
        """Test volatility calculation."""
        data = [100, 102, 98, 105, 99, 106, 101, 108, 103, 110]
        period = 5
        
        result = calculate_volatility(data, period)
        
        # Volatility should be non-negative
        for i in range(period - 1, len(result)):
            if not np.isnan(result[i]):
                assert result[i] >= 0
    
    def test_keltner_channels_calculation(self):
        """Test Keltner Channels calculation."""
        high = [105, 110, 108, 107, 112, 115, 113, 111, 109, 114]
        low = [95, 98, 96, 97, 100, 103, 101, 99, 98, 102]
        close = [100, 105, 102, 104, 108, 110, 107, 105, 103, 109]
        period = 5
        multiplier = 2.0
        
        upper, middle, lower = calculate_keltner_channels(high, low, close, period, multiplier)
        
        # Check channel relationships
        for i in range(period, len(close)):
            if not (np.isnan(upper[i]) or np.isnan(middle[i]) or np.isnan(lower[i])):
                assert upper[i] > middle[i] > lower[i]


class TestLevelsIndicators:
    """Test support/resistance and level indicators."""
    
    def test_support_resistance_calculation(self):
        """Test support and resistance calculation."""
        high = [105, 110, 108, 107, 112, 115, 113, 111, 109, 114, 116, 118, 120, 119, 117]
        low = [95, 98, 96, 97, 100, 103, 101, 99, 98, 102, 104, 106, 108, 107, 105]
        close = [100, 105, 102, 104, 108, 110, 107, 105, 103, 109, 112, 115, 118, 115, 113]
        
        result = calculate_support_resistance(high, low, close)
        
        # Should return support and resistance levels
        assert 'support_levels' in result
        assert 'resistance_levels' in result
        assert isinstance(result['support_levels'], list)
        assert isinstance(result['resistance_levels'], list)
    
    def test_pivot_points_calculation(self):
        """Test pivot points calculation."""
        high = 105.0
        low = 95.0
        close = 100.0
        
        result = calculate_pivot_points(high, low, close)
        
        # Should contain pivot point and support/resistance levels
        assert 'pivot' in result
        assert 'r1' in result
        assert 'r2' in result
        assert 's1' in result
        assert 's2' in result
        
        # Pivot point should be average of HLC
        expected_pivot = (high + low + close) / 3
        assert abs(result['pivot'] - expected_pivot) < 1e-10
    
    def test_fibonacci_levels_calculation(self):
        """Test Fibonacci retracement levels."""
        high = 120.0
        low = 80.0
        
        result = calculate_fibonacci_levels(high, low)
        
        # Should contain standard Fibonacci levels
        assert 'level_0' in result
        assert 'level_236' in result
        assert 'level_382' in result
        assert 'level_500' in result
        assert 'level_618' in result
        assert 'level_1000' in result
        
        # Check level calculations
        range_val = high - low
        assert result['level_0'] == high
        assert result['level_1000'] == low
        assert abs(result['level_500'] - (high - 0.5 * range_val)) < 1e-10


class TestIndicatorCalculator:
    """Test the IndicatorCalculator class."""
    
    def test_calculator_initialization(self):
        """Test calculator initialization."""
        calc = IndicatorCalculator()
        assert calc is not None
    
    def test_calculator_with_config(self):
        """Test calculator with configuration."""
        config = {
            'sma_periods': [20, 50],
            'ema_periods': [12, 26],
            'rsi_period': 14
        }
        
        calc = IndicatorCalculator(config)
        assert calc.config == config
    
    def test_calculator_sma(self):
        """Test calculator SMA method."""
        calc = IndicatorCalculator()
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        period = 3
        
        result = calc.calculate_sma(data, period)
        expected = calculate_sma(data, period)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_calculator_multiple_indicators(self):
        """Test calculator with multiple indicators."""
        calc = IndicatorCalculator()
        
        # Sample data
        high = [105, 110, 108, 107, 112, 115, 113, 111, 109, 114]
        low = [95, 98, 96, 97, 100, 103, 101, 99, 98, 102]
        close = [100, 105, 102, 104, 108, 110, 107, 105, 103, 109]
        volume = [1000, 1200, 800, 1500, 2000, 1800, 1100, 900, 1300, 1600]
        
        # Calculate multiple indicators
        sma = calc.calculate_sma(close, 5)
        ema = calc.calculate_ema(close, 5)
        rsi = calc.calculate_rsi(close, 5)
        
        # All should have same length as input
        assert len(sma) == len(close)
        assert len(ema) == len(close)
        assert len(rsi) == len(close)


class TestIndicatorEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_data(self):
        """Test indicators with empty data."""
        empty_data = []
        
        with pytest.raises(ValueError):
            calculate_sma(empty_data, 5)
        
        with pytest.raises(ValueError):
            calculate_ema(empty_data, 5)
        
        with pytest.raises(ValueError):
            calculate_rsi(empty_data, 5)
    
    def test_single_data_point(self):
        """Test indicators with single data point."""
        single_data = [100]
        
        with pytest.raises(ValueError):
            calculate_sma(single_data, 5)
        
        result = calculate_sma(single_data, 1)
        assert result[0] == 100
    
    def test_nan_data(self):
        """Test indicators with NaN data."""
        nan_data = [1, 2, np.nan, 4, 5]
        
        # Most indicators should handle NaN gracefully
        result = calculate_sma(nan_data, 3)
        
        # Result should contain NaN where appropriate
        assert np.isnan(result[2])
    
    def test_infinite_data(self):
        """Test indicators with infinite data."""
        inf_data = [1, 2, np.inf, 4, 5]
        
        # Should handle infinite values gracefully
        result = calculate_sma(inf_data, 3)
        
        # May contain inf where appropriate
        assert np.isinf(result[4])
    
    def test_very_large_numbers(self):
        """Test indicators with very large numbers."""
        large_data = [1e10, 1.1e10, 1.2e10, 1.3e10, 1.4e10]
        
        result = calculate_sma(large_data, 3)
        
        # Should handle large numbers without overflow
        assert not np.isnan(result[2])
        assert not np.isinf(result[2])
    
    def test_very_small_numbers(self):
        """Test indicators with very small numbers."""
        small_data = [1e-10, 1.1e-10, 1.2e-10, 1.3e-10, 1.4e-10]
        
        result = calculate_sma(small_data, 3)
        
        # Should handle small numbers without underflow
        assert result[2] > 0
    
    def test_negative_numbers(self):
        """Test indicators with negative numbers."""
        negative_data = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
        
        result = calculate_sma(negative_data, 3)
        
        # Should handle negative numbers correctly
        assert result[2] == -4.0  # (-5 + -4 + -3) / 3
        assert result[5] == -1.0  # (-1 + 0 + 1) / 3
    
    def test_constant_data(self):
        """Test indicators with constant data."""
        constant_data = [100] * 10
        
        sma_result = calculate_sma(constant_data, 3)
        ema_result = calculate_ema(constant_data, 3)
        rsi_result = calculate_rsi(constant_data, 5)
        
        # SMA and EMA should equal the constant
        for i in range(2, len(sma_result)):
            assert sma_result[i] == 100
            assert ema_result[i] == 100
        
        # RSI should be 50 for constant data (no price change)
        for i in range(5, len(rsi_result)):
            assert abs(rsi_result[i] - 50.0) < 1e-6


class TestIndicatorPerformance:
    """Test indicator performance and benchmarks."""
    
    def test_sma_performance(self, execution_timer):
        """Test SMA calculation performance."""
        # Large dataset
        data = list(range(10000))
        period = 50
        
        execution_timer.start()
        result = calculate_sma(data, period)
        execution_timer.stop()
        
        # Should complete quickly
        assert execution_timer.get_elapsed_time() < 1.0
        assert len(result) == len(data)
    
    def test_ema_performance(self, execution_timer):
        """Test EMA calculation performance."""
        data = list(range(10000))
        period = 50
        
        execution_timer.start()
        result = calculate_ema(data, period)
        execution_timer.stop()
        
        assert execution_timer.get_elapsed_time() < 1.0
        assert len(result) == len(data)
    
    def test_rsi_performance(self, execution_timer):
        """Test RSI calculation performance."""
        data = [100 + i * 0.1 + np.sin(i * 0.01) for i in range(5000)]
        period = 14
        
        execution_timer.start()
        result = calculate_rsi(data, period)
        execution_timer.stop()
        
        assert execution_timer.get_elapsed_time() < 2.0
        assert len(result) == len(data)
    
    def test_macd_performance(self, execution_timer):
        """Test MACD calculation performance."""
        data = [100 + i * 0.1 + np.sin(i * 0.01) for i in range(5000)]
        
        execution_timer.start()
        macd_line, signal_line, histogram = calculate_macd(data, 12, 26, 9)
        execution_timer.stop()
        
        assert execution_timer.get_elapsed_time() < 2.0
        assert len(macd_line) == len(data)


# Property-based tests using hypothesis
try:
    from hypothesis import given, strategies as st, assume
    
    class TestIndicatorProperties:
        """Property-based tests for indicators."""
        
        @given(st.lists(st.floats(min_value=1.0, max_value=1000.0), min_size=10, max_size=100))
        def test_sma_properties(self, data):
            """Test SMA properties."""
            assume(len(data) >= 3)
            period = min(3, len(data))
            
            result = calculate_sma(data, period)
            
            # Length should match input
            assert len(result) == len(data)
            
            # First period-1 values should be NaN
            for i in range(period - 1):
                assert np.isnan(result[i])
        
        @given(st.lists(st.floats(min_value=1.0, max_value=1000.0), min_size=15, max_size=100))
        def test_rsi_properties(self, data):
            """Test RSI properties."""
            assume(len(data) >= 15)
            period = 14
            
            result = calculate_rsi(data, period)
            
            # RSI should be between 0 and 100
            for i in range(period, len(result)):
                if not np.isnan(result[i]):
                    assert 0 <= result[i] <= 100

except ImportError:
    # Hypothesis not available, skip property-based tests
    pass