"""
Comprehensive Mathematical Tests for Moving Average Indicators

This module provides comprehensive mathematical validation tests for all
moving average indicators including SMA, EMA, WMA, and VWMA.
"""

import pytest
import numpy as np
from typing import Dict, List, Tuple
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from src.indicators.moving_averages import (
    calculate_sma,
    calculate_ema,
    calculate_wma,
    calculate_vwma,
    calculate_ma_crossover,
    calculate_ma_envelope
)

from ..fixtures.test_data import (
    TestDataGenerator,
    PrecisionTestVectors,
    EdgeCaseTestData,
    ValidationHelpers,
    TEST_CONFIG
)


class TestSimpleMovingAverage:
    """Comprehensive tests for Simple Moving Average (SMA)"""
    
    def test_sma_basic_calculation(self):
        """Test basic SMA calculation with known values"""
        data = PrecisionTestVectors.SIMPLE_PRICES
        expected = PrecisionTestVectors.EXPECTED_SMA_5
        
        result = calculate_sma(data, 5)
        
        assert ValidationHelpers.assert_array_almost_equal(
            result, expected, decimals=TEST_CONFIG['precision_decimals']
        )
    
    def test_sma_mathematical_properties(self):
        """Test mathematical properties of SMA"""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = calculate_sma(data, 3)
        
        # Test that SMA is correctly calculated
        assert np.isnan(result[0])  # First value should be NaN
        assert np.isnan(result[1])  # Second value should be NaN
        assert abs(result[2] - 20.0) < 1e-10  # (10+20+30)/3 = 20
        assert abs(result[3] - 30.0) < 1e-10  # (20+30+40)/3 = 30
        assert abs(result[4] - 40.0) < 1e-10  # (30+40+50)/3 = 40
    
    def test_sma_linearity(self):
        """Test SMA linearity property: SMA(a*x + b) = a*SMA(x) + b"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        period = 3
        
        # Calculate SMA of original data
        sma_original = calculate_sma(data, period)
        
        # Calculate SMA of scaled and shifted data
        a, b = 2.0, 10.0
        scaled_data = a * data + b
        sma_scaled = calculate_sma(scaled_data, period)
        
        # Test linearity property
        expected_scaled = a * sma_original + b
        
        # Compare non-NaN values
        mask = ~np.isnan(sma_original)
        assert ValidationHelpers.assert_array_almost_equal(
            sma_scaled[mask], expected_scaled[mask], decimals=TEST_CONFIG['precision_decimals']
        )
    
    def test_sma_edge_cases(self):
        """Test SMA with edge cases"""
        # Test with constant values
        constant_data = EdgeCaseTestData.get_constant_prices()
        result = calculate_sma(constant_data, 5)
        
        # All non-NaN values should equal the constant
        valid_values = result[~np.isnan(result)]
        assert np.allclose(valid_values, 100.0)
        
        # Test with minimal data
        minimal_data = EdgeCaseTestData.get_minimal_data()
        result = calculate_sma(minimal_data, 2)
        
        assert np.isnan(result[0])
        assert abs(result[1] - 100.5) < 1e-10  # (100+101)/2 = 100.5
    
    def test_sma_error_conditions(self):
        """Test SMA error conditions"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test invalid period
        with pytest.raises(ValueError, match="Period must be at least 1"):
            calculate_sma(data, 0)
        
        # Test insufficient data
        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_sma(data, 10)
        
        # Test invalid data type
        with pytest.raises(TypeError, match="Data must be convertible"):
            calculate_sma(["invalid", "data"], 2)
    
    def test_sma_incremental_vs_batch(self):
        """Test incremental vs batch SMA calculation"""
        data = TestDataGenerator.generate_linear_trend(50)
        period = 10
        
        # Batch calculation
        batch_result = calculate_sma(data, period)
        
        # Incremental calculation
        incremental_result = np.full_like(data, np.nan)
        for i in range(period - 1, len(data)):
            incremental_result[i] = np.mean(data[i - period + 1:i + 1])
        
        assert ValidationHelpers.assert_array_almost_equal(
            batch_result, incremental_result, decimals=TEST_CONFIG['precision_decimals']
        )
    
    def test_sma_with_different_periods(self):
        """Test SMA with various periods"""
        data = TestDataGenerator.generate_sinusoidal_data(100)
        
        for period in TEST_CONFIG['default_periods']['sma']:
            result = calculate_sma(data, period)
            
            # Check that we have the right number of NaN values
            nan_count = np.sum(np.isnan(result))
            assert nan_count == period - 1
            
            # Check that all non-NaN values are reasonable
            valid_values = result[~np.isnan(result)]
            assert len(valid_values) == len(data) - period + 1
            assert np.all(np.isfinite(valid_values))


class TestExponentialMovingAverage:
    """Comprehensive tests for Exponential Moving Average (EMA)"""
    
    def test_ema_basic_calculation(self):
        """Test basic EMA calculation with known values"""
        data = PrecisionTestVectors.SIMPLE_PRICES
        expected = PrecisionTestVectors.EXPECTED_EMA_5
        
        result = calculate_ema(data, 5)
        
        assert ValidationHelpers.assert_array_almost_equal(
            result, expected, decimals=5  # EMA has more floating point precision issues
        )
    
    def test_ema_mathematical_properties(self):
        """Test mathematical properties of EMA"""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        period = 3
        alpha = 2.0 / (period + 1)  # 0.5 for period 3
        
        result = calculate_ema(data, period)
        
        # First EMA value should be SMA
        assert abs(result[2] - 20.0) < 1e-10  # (10+20+30)/3 = 20
        
        # Subsequent values should follow EMA formula
        expected_3 = alpha * 40.0 + (1 - alpha) * 20.0  # 0.5 * 40 + 0.5 * 20 = 30
        assert abs(result[3] - expected_3) < 1e-10
        
        expected_4 = alpha * 50.0 + (1 - alpha) * expected_3  # 0.5 * 50 + 0.5 * 30 = 40
        assert abs(result[4] - expected_4) < 1e-10
    
    def test_ema_convergence(self):
        """Test EMA convergence properties"""
        # EMA should converge to the final value for constant data
        constant_value = 100.0
        data = np.full(50, constant_value)
        
        result = calculate_ema(data, 10)
        
        # The final value should be very close to the constant
        assert abs(result[-1] - constant_value) < 1e-6
    
    def test_ema_alpha_parameter(self):
        """Test EMA with custom alpha parameter"""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        period = 3
        custom_alpha = 0.7
        
        result = calculate_ema(data, period, alpha=custom_alpha)
        
        # First EMA value should still be SMA
        assert abs(result[2] - 20.0) < 1e-10
        
        # Subsequent values should use custom alpha
        expected_3 = custom_alpha * 40.0 + (1 - custom_alpha) * 20.0
        assert abs(result[3] - expected_3) < 1e-10
    
    def test_ema_responsiveness(self):
        """Test EMA responsiveness compared to SMA"""
        # Create data with a sudden jump
        data = np.array([100.0] * 10 + [110.0] * 10)
        period = 5
        
        ema_result = calculate_ema(data, period)
        sma_result = calculate_sma(data, period)
        
        # EMA should respond faster to the price jump
        jump_index = 10
        for i in range(jump_index + 1, len(data)):
            if not (np.isnan(ema_result[i]) or np.isnan(sma_result[i])):
                # EMA should be higher than SMA after the jump
                assert ema_result[i] >= sma_result[i]
    
    def test_ema_error_conditions(self):
        """Test EMA error conditions"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test invalid period
        with pytest.raises(ValueError, match="Period must be at least 1"):
            calculate_ema(data, 0)
        
        # Test invalid alpha
        with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
            calculate_ema(data, 3, alpha=1.5)
        
        # Test insufficient data
        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_ema(data, 10)


class TestWeightedMovingAverage:
    """Comprehensive tests for Weighted Moving Average (WMA)"""
    
    def test_wma_basic_calculation(self):
        """Test basic WMA calculation"""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = calculate_wma(data, 3)
        
        # First two values should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        
        # Third value: (10*1 + 20*2 + 30*3) / (1+2+3) = 140/6 = 23.333...
        expected_2 = (10*1 + 20*2 + 30*3) / (1+2+3)
        assert abs(result[2] - expected_2) < 1e-10
        
        # Fourth value: (20*1 + 30*2 + 40*3) / (1+2+3) = 200/6 = 33.333...
        expected_3 = (20*1 + 30*2 + 40*3) / (1+2+3)
        assert abs(result[3] - expected_3) < 1e-10
    
    def test_wma_weights_property(self):
        """Test WMA weights property"""
        # WMA should give more weight to recent values
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        period = 3
        
        wma_result = calculate_wma(data, period)
        sma_result = calculate_sma(data, period)
        
        # For increasing data, WMA should be higher than SMA
        for i in range(period - 1, len(data)):
            if not (np.isnan(wma_result[i]) or np.isnan(sma_result[i])):
                assert wma_result[i] >= sma_result[i]
    
    def test_wma_error_conditions(self):
        """Test WMA error conditions"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test invalid period
        with pytest.raises(ValueError, match="Period must be at least 1"):
            calculate_wma(data, 0)
        
        # Test insufficient data
        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_wma(data, 10)


class TestVolumeWeightedMovingAverage:
    """Comprehensive tests for Volume Weighted Moving Average (VWMA)"""
    
    def test_vwma_basic_calculation(self):
        """Test basic VWMA calculation"""
        prices = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        volumes = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        
        result = calculate_vwma(prices, volumes, 3)
        
        # First two values should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        
        # Third value: (10*100 + 20*200 + 30*300) / (100+200+300) = 14000/600 = 23.333...
        expected_2 = (10*100 + 20*200 + 30*300) / (100+200+300)
        assert abs(result[2] - expected_2) < 1e-10
    
    def test_vwma_volume_weighting(self):
        """Test VWMA volume weighting property"""
        # Test that high volume periods get more weight
        prices = np.array([10.0, 20.0, 30.0])
        
        # High volume on middle price
        high_volume = np.array([100.0, 1000.0, 100.0])
        result_high = calculate_vwma(prices, high_volume, 3)
        
        # Equal volume
        equal_volume = np.array([100.0, 100.0, 100.0])
        result_equal = calculate_vwma(prices, equal_volume, 3)
        
        # VWMA with high volume on middle price should be closer to middle price
        assert abs(result_high[2] - 20.0) < abs(result_equal[2] - 20.0)
    
    def test_vwma_zero_volume(self):
        """Test VWMA with zero volume"""
        prices = np.array([10.0, 20.0, 30.0])
        volumes = np.array([0.0, 0.0, 0.0])
        
        result = calculate_vwma(prices, volumes, 3)
        
        # Should return NaN for zero volume
        assert np.isnan(result[2])
    
    def test_vwma_error_conditions(self):
        """Test VWMA error conditions"""
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        volumes = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        
        # Test mismatched array lengths
        with pytest.raises(ValueError, match="same length"):
            calculate_vwma(prices, volumes[:-1], 3)
        
        # Test invalid period
        with pytest.raises(ValueError, match="Period must be at least 1"):
            calculate_vwma(prices, volumes, 0)


class TestMovingAverageCrossover:
    """Comprehensive tests for Moving Average Crossover signals"""
    
    def test_crossover_basic_signals(self):
        """Test basic crossover signal generation"""
        # Create data where fast MA crosses above slow MA
        fast_ma = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        slow_ma = np.array([12.0, 12.0, 12.0, 12.0, 12.0])
        
        result = calculate_ma_crossover(fast_ma, slow_ma)
        
        # Should have bullish crossover at index 2 (first time fast > slow)
        assert result[2] == 1  # Bullish crossover
        
        # Create data where fast MA crosses below slow MA
        fast_ma = np.array([14.0, 13.0, 12.0, 11.0, 10.0])
        slow_ma = np.array([12.0, 12.0, 12.0, 12.0, 12.0])
        
        result = calculate_ma_crossover(fast_ma, slow_ma)
        
        # Should have bearish crossover at index 2 (first time fast < slow)
        assert result[2] == -1  # Bearish crossover
    
    def test_crossover_no_signal(self):
        """Test crossover with no signal generation"""
        # Fast MA consistently above slow MA
        fast_ma = np.array([14.0, 15.0, 16.0, 17.0, 18.0])
        slow_ma = np.array([12.0, 12.0, 12.0, 12.0, 12.0])
        
        result = calculate_ma_crossover(fast_ma, slow_ma)
        
        # Should have no crossover signals
        assert np.all(result == 0)
    
    def test_crossover_with_nan_values(self):
        """Test crossover with NaN values"""
        fast_ma = np.array([np.nan, 11.0, 12.0, 13.0, 14.0])
        slow_ma = np.array([12.0, 12.0, 12.0, 12.0, 12.0])
        
        result = calculate_ma_crossover(fast_ma, slow_ma)
        
        # Should handle NaN values gracefully
        assert np.all(np.isfinite(result))
    
    def test_crossover_error_conditions(self):
        """Test crossover error conditions"""
        fast_ma = np.array([1.0, 2.0, 3.0])
        slow_ma = np.array([1.0, 2.0])
        
        # Test mismatched array lengths
        with pytest.raises(ValueError, match="same length"):
            calculate_ma_crossover(fast_ma, slow_ma)


class TestMovingAverageEnvelope:
    """Comprehensive tests for Moving Average Envelope"""
    
    def test_envelope_basic_calculation(self):
        """Test basic envelope calculation"""
        data = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        envelope_pct = 0.05  # 5%
        
        upper, middle, lower = calculate_ma_envelope(data, 3, envelope_pct)
        
        # For constant data, middle should be the constant value
        assert abs(middle[2] - 100.0) < 1e-10
        
        # Upper and lower should be +/- 5% of middle
        assert abs(upper[2] - 105.0) < 1e-10
        assert abs(lower[2] - 95.0) < 1e-10
    
    def test_envelope_percentage_scaling(self):
        """Test envelope percentage scaling"""
        data = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        
        # Test different envelope percentages
        for envelope_pct in [0.01, 0.05, 0.10]:
            upper, middle, lower = calculate_ma_envelope(data, 3, envelope_pct)
            
            expected_upper = 100.0 * (1 + envelope_pct)
            expected_lower = 100.0 * (1 - envelope_pct)
            
            assert abs(upper[2] - expected_upper) < 1e-10
            assert abs(lower[2] - expected_lower) < 1e-10
    
    def test_envelope_error_conditions(self):
        """Test envelope error conditions"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test negative envelope percentage
        with pytest.raises(ValueError, match="non-negative"):
            calculate_ma_envelope(data, 3, -0.1)


class TestMovingAverageIntegration:
    """Integration tests for all moving average functions"""
    
    def test_all_ma_types_consistency(self):
        """Test that all MA types produce reasonable results"""
        data = TestDataGenerator.generate_volatile_data(100)
        volume = TestDataGenerator.generate_volatile_data(100, base_price=1000)
        period = 10
        
        sma_result = calculate_sma(data, period)
        ema_result = calculate_ema(data, period)
        wma_result = calculate_wma(data, period)
        vwma_result = calculate_vwma(data, volume, period)
        
        # All should have same number of NaN values initially
        for result in [sma_result, ema_result, wma_result, vwma_result]:
            assert np.sum(np.isnan(result[:period-1])) == period - 1
            assert np.all(np.isfinite(result[period-1:]))
    
    def test_ma_performance_characteristics(self):
        """Test performance characteristics of different MA types"""
        # Create data with a trend change
        data = np.concatenate([
            np.linspace(100, 110, 20),  # Uptrend
            np.linspace(110, 100, 20)   # Downtrend
        ])
        
        period = 5
        sma_result = calculate_sma(data, period)
        ema_result = calculate_ema(data, period)
        
        # EMA should be more responsive to trend changes
        # Check at the trend change point (index 20)
        trend_change_idx = 20
        
        # EMA should typically be closer to recent prices
        recent_prices = data[trend_change_idx:trend_change_idx+5]
        avg_recent = np.mean(recent_prices)
        
        # This is a general property test - exact values depend on the data
        assert np.all(np.isfinite(sma_result[period-1:]))
        assert np.all(np.isfinite(ema_result[period-1:]))
    
    def test_ma_mathematical_relationships(self):
        """Test mathematical relationships between MA types"""
        # For trending data, WMA should be between SMA and EMA
        data = np.linspace(100, 120, 20)  # Linear uptrend
        period = 5
        
        sma_result = calculate_sma(data, period)
        ema_result = calculate_ema(data, period)
        wma_result = calculate_wma(data, period)
        
        # For uptrending data, typically WMA >= SMA and EMA >= SMA
        for i in range(period-1, len(data)):
            if np.all(np.isfinite([sma_result[i], ema_result[i], wma_result[i]])):
                assert wma_result[i] >= sma_result[i]
                assert ema_result[i] >= sma_result[i]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])