"""
Comprehensive Mathematical Tests for Momentum Indicators

This module provides comprehensive mathematical validation tests for all
momentum indicators including RSI, MACD, Stochastic, Williams %R, and CCI.
"""

import pytest
import numpy as np
from typing import Dict, List, Tuple
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from src.indicators.momentum import (
    calculate_rsi,
    calculate_macd,
    calculate_stochastic,
    calculate_williams_r,
    calculate_cci,
    calculate_momentum,
    calculate_roc
)

from ..fixtures.test_data import (
    TestDataGenerator,
    PrecisionTestVectors,
    EdgeCaseTestData,
    ValidationHelpers,
    TEST_CONFIG
)


class TestRelativeStrengthIndex:
    """Comprehensive tests for Relative Strength Index (RSI)"""
    
    def test_rsi_basic_calculation(self):
        """Test basic RSI calculation with known values"""
        data, expected = PrecisionTestVectors.get_rsi_test_data()
        
        result = calculate_rsi(data, 14)
        
        # Compare only the non-NaN values
        mask = ~np.isnan(expected)
        assert ValidationHelpers.assert_array_almost_equal(
            result[mask], expected[mask], decimals=2  # RSI typically accurate to 2 decimal places
        )
    
    def test_rsi_mathematical_properties(self):
        """Test mathematical properties of RSI"""
        # RSI should be between 0 and 100
        data = TestDataGenerator.generate_volatile_data(100)
        result = calculate_rsi(data, 14)
        
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values >= 0.0)
        assert np.all(valid_values <= 100.0)
    
    def test_rsi_extreme_values(self):
        """Test RSI with extreme price movements"""
        # All gains, no losses
        data = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
                        20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0])
        
        result = calculate_rsi(data, 14)
        
        # RSI should approach 100 for continuous gains
        assert result[-1] > 90.0
        
        # All losses, no gains
        data = np.array([30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0,
                        20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0])
        
        result = calculate_rsi(data, 14)
        
        # RSI should approach 0 for continuous losses
        assert result[-1] < 10.0
    
    def test_rsi_constant_prices(self):
        """Test RSI with constant prices"""
        data = EdgeCaseTestData.get_constant_prices()
        result = calculate_rsi(data, 14)
        
        # RSI should be 50 for constant prices (no gains or losses)
        valid_values = result[~np.isnan(result)]
        assert np.allclose(valid_values, 50.0)
    
    def test_rsi_overbought_oversold_levels(self):
        """Test RSI overbought/oversold detection"""
        # Create oscillating data
        data = TestDataGenerator.generate_sinusoidal_data(100, amplitude=20, period=10)
        result = calculate_rsi(data, 14)
        
        # RSI should oscillate and occasionally reach overbought/oversold levels
        valid_values = result[~np.isnan(result)]
        assert np.max(valid_values) > 70.0  # Should reach overbought
        assert np.min(valid_values) < 30.0  # Should reach oversold
    
    def test_rsi_wilder_smoothing(self):
        """Test RSI Wilder's smoothing method"""
        data = np.array([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 46.08, 45.89,
                        46.03, 46.83, 46.69, 46.45, 46.59, 46.3, 46.28, 46.28, 46.00])
        
        result = calculate_rsi(data, 14)
        
        # Test that the smoothing is working correctly
        # First RSI value should be calculated from simple averages
        assert not np.isnan(result[14])
        
        # Subsequent values should use Wilder's smoothing
        if len(result) > 15:
            assert not np.isnan(result[15])
    
    def test_rsi_error_conditions(self):
        """Test RSI error conditions"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test invalid period
        with pytest.raises(ValueError, match="Period must be at least 1"):
            calculate_rsi(data, 0)
        
        # Test insufficient data
        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_rsi(data, 15)  # Need period + 1 points
    
    def test_rsi_different_periods(self):
        """Test RSI with different periods"""
        data = TestDataGenerator.generate_volatile_data(100)
        
        for period in [7, 14, 21, 28]:
            result = calculate_rsi(data, period)
            
            # Check that we have the right number of NaN values
            nan_count = np.sum(np.isnan(result))
            assert nan_count == period
            
            # Check that all non-NaN values are in valid range
            valid_values = result[~np.isnan(result)]
            assert np.all(valid_values >= 0.0)
            assert np.all(valid_values <= 100.0)


class TestMACD:
    """Comprehensive tests for MACD indicator"""
    
    def test_macd_basic_calculation(self):
        """Test basic MACD calculation"""
        data = TestDataGenerator.generate_linear_trend(100)
        
        macd_line, signal_line, histogram = calculate_macd(data, 12, 26, 9)
        
        # Check that arrays have the same length
        assert len(macd_line) == len(signal_line) == len(histogram)
        
        # Check that histogram = macd_line - signal_line
        diff = macd_line - signal_line
        assert ValidationHelpers.assert_array_almost_equal(
            histogram, diff, decimals=TEST_CONFIG['precision_decimals']
        )
    
    def test_macd_mathematical_properties(self):
        """Test mathematical properties of MACD"""
        # For uptrending data, MACD should eventually turn positive
        data = np.linspace(100, 150, 60)  # Strong uptrend
        
        macd_line, signal_line, histogram = calculate_macd(data, 12, 26, 9)
        
        # MACD line should be positive for strong uptrend
        valid_macd = macd_line[~np.isnan(macd_line)]
        assert np.any(valid_macd > 0)
        
        # For downtrending data, MACD should eventually turn negative
        data = np.linspace(150, 100, 60)  # Strong downtrend
        
        macd_line, signal_line, histogram = calculate_macd(data, 12, 26, 9)
        
        # MACD line should be negative for strong downtrend
        valid_macd = macd_line[~np.isnan(macd_line)]
        assert np.any(valid_macd < 0)
    
    def test_macd_crossover_signals(self):
        """Test MACD crossover signals"""
        # Create data that should produce crossover
        data = np.concatenate([
            np.linspace(100, 110, 30),  # Uptrend
            np.linspace(110, 100, 30)   # Downtrend
        ])
        
        macd_line, signal_line, histogram = calculate_macd(data, 12, 26, 9)
        
        # Look for crossovers in histogram (changes in sign)
        valid_histogram = histogram[~np.isnan(histogram)]
        
        if len(valid_histogram) > 1:
            # Check that histogram changes sign (indicating crossovers)
            signs = np.sign(valid_histogram)
            sign_changes = np.diff(signs)
            crossovers = np.sum(sign_changes != 0)
            
            # Should have at least one crossover for this data
            assert crossovers > 0
    
    def test_macd_ema_relationship(self):
        """Test MACD relationship with underlying EMAs"""
        data = TestDataGenerator.generate_volatile_data(100)
        
        # Calculate MACD
        macd_line, signal_line, histogram = calculate_macd(data, 12, 26, 9)
        
        # Calculate EMAs directly
        from src.indicators.moving_averages import calculate_ema
        fast_ema = calculate_ema(data, 12)
        slow_ema = calculate_ema(data, 26)
        
        # MACD line should equal fast EMA - slow EMA
        expected_macd = fast_ema - slow_ema
        
        assert ValidationHelpers.assert_array_almost_equal(
            macd_line, expected_macd, decimals=TEST_CONFIG['precision_decimals']
        )
    
    def test_macd_error_conditions(self):
        """Test MACD error conditions"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test fast >= slow
        with pytest.raises(ValueError, match="Fast period must be less than slow period"):
            calculate_macd(data, 26, 12, 9)
        
        # Test invalid periods
        with pytest.raises(ValueError, match="All periods must be at least 1"):
            calculate_macd(data, 0, 26, 9)
        
        # Test insufficient data
        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_macd(data, 12, 30, 9)
    
    def test_macd_different_parameters(self):
        """Test MACD with different parameters"""
        data = TestDataGenerator.generate_volatile_data(100)
        
        # Test different parameter combinations
        params = [(5, 20, 5), (8, 21, 5), (12, 26, 9), (19, 39, 9)]
        
        for fast, slow, signal in params:
            macd_line, signal_line, histogram = calculate_macd(data, fast, slow, signal)
            
            # Check that results are finite where valid
            valid_mask = ~np.isnan(macd_line)
            assert np.all(np.isfinite(macd_line[valid_mask]))
            assert np.all(np.isfinite(signal_line[valid_mask]))
            assert np.all(np.isfinite(histogram[valid_mask]))


class TestStochasticOscillator:
    """Comprehensive tests for Stochastic Oscillator"""
    
    def test_stochastic_basic_calculation(self):
        """Test basic Stochastic calculation"""
        # Create simple OHLC data
        high = np.array([15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 19.0, 18.0, 17.0, 16.0])
        low = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 14.0, 13.0, 12.0, 11.0])
        close = np.array([12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 16.0, 15.0, 14.0, 13.0])
        
        k_percent, d_percent = calculate_stochastic(high, low, close, 5, 3, 1)
        
        # %K should be between 0 and 100
        valid_k = k_percent[~np.isnan(k_percent)]
        assert np.all(valid_k >= 0.0)
        assert np.all(valid_k <= 100.0)
        
        # %D should be between 0 and 100
        valid_d = d_percent[~np.isnan(d_percent)]
        assert np.all(valid_d >= 0.0)
        assert np.all(valid_d <= 100.0)
    
    def test_stochastic_mathematical_properties(self):
        """Test mathematical properties of Stochastic"""
        # When close equals high, %K should be 100
        high = np.array([15.0, 16.0, 17.0, 18.0, 19.0])
        low = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        close = high.copy()  # Close equals high
        
        k_percent, d_percent = calculate_stochastic(high, low, close, 3, 3, 1)
        
        # %K should be 100 when close = high
        valid_k = k_percent[~np.isnan(k_percent)]
        assert np.allclose(valid_k, 100.0)
        
        # When close equals low, %K should be 0
        close = low.copy()  # Close equals low
        k_percent, d_percent = calculate_stochastic(high, low, close, 3, 3, 1)
        
        valid_k = k_percent[~np.isnan(k_percent)]
        assert np.allclose(valid_k, 0.0)
    
    def test_stochastic_zero_range(self):
        """Test Stochastic with zero range (high = low)"""
        high = np.array([15.0, 15.0, 15.0, 15.0, 15.0])
        low = np.array([15.0, 15.0, 15.0, 15.0, 15.0])
        close = np.array([15.0, 15.0, 15.0, 15.0, 15.0])
        
        k_percent, d_percent = calculate_stochastic(high, low, close, 3, 3, 1)
        
        # Should return 50% for zero range to avoid division by zero
        valid_k = k_percent[~np.isnan(k_percent)]
        assert np.allclose(valid_k, 50.0)
    
    def test_stochastic_smoothing(self):
        """Test Stochastic smoothing parameters"""
        data = TestDataGenerator.generate_ohlcv_data(
            TestDataGenerator.generate_volatile_data(50)
        )
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Test different smoothing parameters
        k_raw, d_raw = calculate_stochastic(high, low, close, 14, 3, 1)  # No smoothing
        k_smooth, d_smooth = calculate_stochastic(high, low, close, 14, 3, 3)  # Smoothing
        
        # Smoothed %K should be less volatile
        valid_raw = k_raw[~np.isnan(k_raw)]
        valid_smooth = k_smooth[~np.isnan(k_smooth)]
        
        if len(valid_raw) > 10 and len(valid_smooth) > 10:
            # Calculate volatility (standard deviation)
            vol_raw = np.std(valid_raw)
            vol_smooth = np.std(valid_smooth)
            
            # Smoothed version should be less volatile
            assert vol_smooth <= vol_raw
    
    def test_stochastic_error_conditions(self):
        """Test Stochastic error conditions"""
        high = np.array([15.0, 16.0, 17.0])
        low = np.array([10.0, 11.0, 12.0])
        close = np.array([12.0, 13.0, 14.0])
        
        # Test invalid periods
        with pytest.raises(ValueError, match="All periods must be at least 1"):
            calculate_stochastic(high, low, close, 0, 3, 1)
        
        # Test mismatched array lengths
        with pytest.raises(ValueError, match="same length"):
            calculate_stochastic(high, low[:-1], close, 3, 3, 1)
        
        # Test insufficient data
        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_stochastic(high, low, close, 5, 3, 1)


class TestWilliamsR:
    """Comprehensive tests for Williams %R"""
    
    def test_williams_r_basic_calculation(self):
        """Test basic Williams %R calculation"""
        high = np.array([15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 19.0, 18.0, 17.0, 16.0])
        low = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 14.0, 13.0, 12.0, 11.0])
        close = np.array([12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 16.0, 15.0, 14.0, 13.0])
        
        result = calculate_williams_r(high, low, close, 5)
        
        # Williams %R should be between -100 and 0
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values >= -100.0)
        assert np.all(valid_values <= 0.0)
    
    def test_williams_r_mathematical_properties(self):
        """Test mathematical properties of Williams %R"""
        # When close equals high, Williams %R should be 0
        high = np.array([15.0, 16.0, 17.0, 18.0, 19.0])
        low = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        close = high.copy()  # Close equals high
        
        result = calculate_williams_r(high, low, close, 3)
        
        # Williams %R should be 0 when close = high
        valid_values = result[~np.isnan(result)]
        assert np.allclose(valid_values, 0.0)
        
        # When close equals low, Williams %R should be -100
        close = low.copy()  # Close equals low
        result = calculate_williams_r(high, low, close, 3)
        
        valid_values = result[~np.isnan(result)]
        assert np.allclose(valid_values, -100.0)
    
    def test_williams_r_zero_range(self):
        """Test Williams %R with zero range"""
        high = np.array([15.0, 15.0, 15.0, 15.0, 15.0])
        low = np.array([15.0, 15.0, 15.0, 15.0, 15.0])
        close = np.array([15.0, 15.0, 15.0, 15.0, 15.0])
        
        result = calculate_williams_r(high, low, close, 3)
        
        # Should return -50% for zero range
        valid_values = result[~np.isnan(result)]
        assert np.allclose(valid_values, -50.0)
    
    def test_williams_r_error_conditions(self):
        """Test Williams %R error conditions"""
        high = np.array([15.0, 16.0, 17.0])
        low = np.array([10.0, 11.0, 12.0])
        close = np.array([12.0, 13.0, 14.0])
        
        # Test invalid period
        with pytest.raises(ValueError, match="Period must be at least 1"):
            calculate_williams_r(high, low, close, 0)
        
        # Test mismatched array lengths
        with pytest.raises(ValueError, match="same length"):
            calculate_williams_r(high, low[:-1], close, 3)


class TestCCI:
    """Comprehensive tests for Commodity Channel Index"""
    
    def test_cci_basic_calculation(self):
        """Test basic CCI calculation"""
        data = TestDataGenerator.generate_ohlcv_data(
            TestDataGenerator.generate_volatile_data(50)
        )
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        result = calculate_cci(high, low, close, 20)
        
        # CCI can theoretically go beyond +/-100, but should be reasonable
        valid_values = result[~np.isnan(result)]
        assert np.all(np.isfinite(valid_values))
        
        # Most CCI values should be between -200 and +200 for normal data
        assert np.all(valid_values >= -500.0)  # Very loose bounds
        assert np.all(valid_values <= 500.0)
    
    def test_cci_mathematical_properties(self):
        """Test mathematical properties of CCI"""
        # For constant prices, CCI should be 0
        high = np.full(25, 15.0)
        low = np.full(25, 10.0)
        close = np.full(25, 12.5)
        
        result = calculate_cci(high, low, close, 20)
        
        # CCI should be 0 for constant typical price
        valid_values = result[~np.isnan(result)]
        assert np.allclose(valid_values, 0.0, atol=1e-10)
    
    def test_cci_zero_mean_deviation(self):
        """Test CCI with zero mean deviation"""
        # Create data where typical price equals its moving average
        high = np.array([15.0] * 25)
        low = np.array([10.0] * 25)
        close = np.array([12.5] * 25)
        
        result = calculate_cci(high, low, close, 20)
        
        # Should return 0 when mean deviation is 0
        valid_values = result[~np.isnan(result)]
        assert np.allclose(valid_values, 0.0)
    
    def test_cci_error_conditions(self):
        """Test CCI error conditions"""
        high = np.array([15.0, 16.0, 17.0])
        low = np.array([10.0, 11.0, 12.0])
        close = np.array([12.0, 13.0, 14.0])
        
        # Test invalid period
        with pytest.raises(ValueError, match="Period must be at least 1"):
            calculate_cci(high, low, close, 0)
        
        # Test mismatched array lengths
        with pytest.raises(ValueError, match="same length"):
            calculate_cci(high, low[:-1], close, 3)


class TestMomentumAndROC:
    """Comprehensive tests for Momentum and Rate of Change"""
    
    def test_momentum_basic_calculation(self):
        """Test basic momentum calculation"""
        data = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
        result = calculate_momentum(data, 3)
        
        # Momentum should be current price - price n periods ago
        expected = np.array([np.nan, np.nan, np.nan, 6.0, 6.0, 6.0])  # 16-10, 18-12, 20-14
        
        assert ValidationHelpers.assert_array_almost_equal(
            result, expected, decimals=TEST_CONFIG['precision_decimals']
        )
    
    def test_roc_basic_calculation(self):
        """Test basic ROC calculation"""
        data = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
        result = calculate_roc(data, 3)
        
        # ROC should be ((current - previous) / previous) * 100
        expected = np.array([np.nan, np.nan, np.nan, 60.0, 50.0, 42.857143])
        # 16-10)/10*100=60, (18-12)/12*100=50, (20-14)/14*100≈42.86
        
        assert ValidationHelpers.assert_array_almost_equal(
            result, expected, decimals=5
        )
    
    def test_momentum_roc_relationship(self):
        """Test relationship between momentum and ROC"""
        data = TestDataGenerator.generate_linear_trend(20)
        period = 5
        
        momentum_result = calculate_momentum(data, period)
        roc_result = calculate_roc(data, period)
        
        # For positive momentum, ROC should also be positive
        valid_momentum = momentum_result[~np.isnan(momentum_result)]
        valid_roc = roc_result[~np.isnan(roc_result)]
        
        # Check that signs are consistent
        if len(valid_momentum) > 0 and len(valid_roc) > 0:
            momentum_signs = np.sign(valid_momentum)
            roc_signs = np.sign(valid_roc)
            
            # Most values should have the same sign
            same_sign_count = np.sum(momentum_signs == roc_signs)
            assert same_sign_count >= len(valid_momentum) * 0.8  # At least 80% same sign
    
    def test_momentum_roc_error_conditions(self):
        """Test momentum and ROC error conditions"""
        data = np.array([1.0, 2.0, 3.0])
        
        # Test invalid period
        with pytest.raises(ValueError, match="Period must be at least 1"):
            calculate_momentum(data, 0)
        
        with pytest.raises(ValueError, match="Period must be at least 1"):
            calculate_roc(data, 0)
        
        # Test insufficient data
        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_momentum(data, 5)
        
        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_roc(data, 5)
    
    def test_roc_zero_division(self):
        """Test ROC with zero values"""
        data = np.array([10.0, 0.0, 5.0, 10.0])
        result = calculate_roc(data, 2)
        
        # Should handle zero division gracefully
        assert np.isnan(result[2])  # (5-0)/0 should be NaN


class TestMomentumIntegration:
    """Integration tests for all momentum indicators"""
    
    def test_all_momentum_indicators_consistency(self):
        """Test that all momentum indicators produce reasonable results"""
        data = TestDataGenerator.generate_ohlcv_data(
            TestDataGenerator.generate_volatile_data(100)
        )
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate all momentum indicators
        rsi = calculate_rsi(close, 14)
        macd_line, signal_line, histogram = calculate_macd(close, 12, 26, 9)
        k_percent, d_percent = calculate_stochastic(high, low, close, 14, 3, 3)
        williams_r = calculate_williams_r(high, low, close, 14)
        cci = calculate_cci(high, low, close, 20)
        momentum = calculate_momentum(close, 10)
        roc = calculate_roc(close, 10)
        
        # All should produce finite values where valid
        indicators = [rsi, macd_line, signal_line, histogram, k_percent, d_percent, 
                     williams_r, cci, momentum, roc]
        
        for indicator in indicators:
            valid_values = indicator[~np.isnan(indicator)]
            assert np.all(np.isfinite(valid_values))
    
    def test_momentum_indicator_ranges(self):
        """Test that momentum indicators stay within expected ranges"""
        data = TestDataGenerator.generate_ohlcv_data(
            TestDataGenerator.generate_volatile_data(100)
        )
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Test RSI range
        rsi = calculate_rsi(close, 14)
        valid_rsi = rsi[~np.isnan(rsi)]
        assert np.all(valid_rsi >= 0.0)
        assert np.all(valid_rsi <= 100.0)
        
        # Test Stochastic range
        k_percent, d_percent = calculate_stochastic(high, low, close, 14, 3, 3)
        valid_k = k_percent[~np.isnan(k_percent)]
        valid_d = d_percent[~np.isnan(d_percent)]
        assert np.all(valid_k >= 0.0)
        assert np.all(valid_k <= 100.0)
        assert np.all(valid_d >= 0.0)
        assert np.all(valid_d <= 100.0)
        
        # Test Williams %R range
        williams_r = calculate_williams_r(high, low, close, 14)
        valid_williams = williams_r[~np.isnan(williams_r)]
        assert np.all(valid_williams >= -100.0)
        assert np.all(valid_williams <= 0.0)
    
    def test_momentum_indicator_correlations(self):
        """Test correlations between momentum indicators"""
        # Create trending data
        data = TestDataGenerator.generate_linear_trend(100, slope=0.5)
        ohlcv = TestDataGenerator.generate_ohlcv_data(data)
        
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close']
        
        # Calculate momentum indicators
        rsi = calculate_rsi(close, 14)
        momentum = calculate_momentum(close, 10)
        roc = calculate_roc(close, 10)
        
        # For trending data, momentum indicators should generally agree
        # (all positive for uptrend, all negative for downtrend)
        
        # Get overlapping valid values
        valid_indices = ~(np.isnan(rsi) | np.isnan(momentum) | np.isnan(roc))
        
        if np.sum(valid_indices) > 10:
            valid_rsi = rsi[valid_indices]
            valid_momentum = momentum[valid_indices]
            valid_roc = roc[valid_indices]
            
            # For uptrending data, RSI should be generally above 50
            assert np.mean(valid_rsi) > 50.0
            
            # Momentum and ROC should be mostly positive
            assert np.mean(valid_momentum) > 0.0
            assert np.mean(valid_roc) > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])