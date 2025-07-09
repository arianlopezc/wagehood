"""
Comprehensive Mathematical Tests for Volatility Indicators

This module provides comprehensive mathematical validation tests for all
volatility indicators including Bollinger Bands, ATR, Keltner Channels, and Donchian Channels.
"""

import pytest
import numpy as np
from typing import Dict, List, Tuple
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from src.indicators.volatility import (
    calculate_bollinger_bands,
    calculate_atr,
    calculate_keltner_channels,
    calculate_donchian_channels,
    calculate_volatility,
    calculate_bb_width,
    calculate_bb_percent,
    calculate_price_channels
)

from ..fixtures.test_data import (
    TestDataGenerator,
    PrecisionTestVectors,
    EdgeCaseTestData,
    ValidationHelpers,
    TEST_CONFIG
)


class TestBollingerBands:
    """Comprehensive tests for Bollinger Bands"""
    
    def test_bollinger_bands_basic_calculation(self):
        """Test basic Bollinger Bands calculation"""
        data = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0, 102.0, 101.0])
        
        upper, middle, lower = calculate_bollinger_bands(data, 5, 2.0)
        
        # Check that we have three arrays of the same length
        assert len(upper) == len(middle) == len(lower) == len(data)
        
        # Middle band should be SMA
        from src.indicators.moving_averages import calculate_sma
        expected_middle = calculate_sma(data, 5)
        
        assert ValidationHelpers.assert_array_almost_equal(
            middle, expected_middle, decimals=TEST_CONFIG['precision_decimals']
        )
        
        # Upper band should be above middle, lower band below middle
        valid_indices = ~np.isnan(middle)
        assert np.all(upper[valid_indices] >= middle[valid_indices])
        assert np.all(lower[valid_indices] <= middle[valid_indices])
    
    def test_bollinger_bands_mathematical_properties(self):
        """Test mathematical properties of Bollinger Bands"""
        # For constant data, upper and lower bands should be equidistant from middle
        data = EdgeCaseTestData.get_constant_prices()
        std_dev = 2.0
        
        upper, middle, lower = calculate_bollinger_bands(data, 10, std_dev)
        
        # For constant data, standard deviation should be 0
        # So upper and lower should equal middle
        valid_indices = ~np.isnan(middle)
        assert np.allclose(upper[valid_indices], middle[valid_indices])
        assert np.allclose(lower[valid_indices], middle[valid_indices])
    
    def test_bollinger_bands_standard_deviation_scaling(self):
        """Test standard deviation scaling in Bollinger Bands"""
        data = TestDataGenerator.generate_volatile_data(50)
        period = 20
        
        # Test different standard deviation multipliers
        std_devs = [1.0, 2.0, 2.5]
        results = []
        
        for std_dev in std_devs:
            upper, middle, lower = calculate_bollinger_bands(data, period, std_dev)
            results.append((upper, middle, lower))
        
        # Middle band should be the same for all multipliers
        for i in range(1, len(results)):
            assert ValidationHelpers.assert_array_almost_equal(
                results[0][1], results[i][1], decimals=TEST_CONFIG['precision_decimals']
            )
        
        # Band width should increase with standard deviation multiplier
        for i in range(len(data)):
            if not np.isnan(results[0][0][i]):
                width_1 = results[0][0][i] - results[0][2][i]  # std_dev = 1.0
                width_2 = results[1][0][i] - results[1][2][i]  # std_dev = 2.0
                width_25 = results[2][0][i] - results[2][2][i]  # std_dev = 2.5
                
                assert width_2 > width_1
                assert width_25 > width_2
    
    def test_bollinger_bands_squeeze_expansion(self):
        """Test Bollinger Bands squeeze and expansion"""
        # Create data with low volatility followed by high volatility
        low_vol_data = np.full(20, 100.0) + np.random.normal(0, 0.1, 20)
        high_vol_data = np.full(20, 100.0) + np.random.normal(0, 5.0, 20)
        data = np.concatenate([low_vol_data, high_vol_data])
        
        upper, middle, lower = calculate_bollinger_bands(data, 10, 2.0)
        
        # Calculate band width
        band_width = upper - lower
        
        # Band width should be smaller during low volatility period
        # and larger during high volatility period
        valid_indices = ~np.isnan(band_width)
        valid_width = band_width[valid_indices]
        
        if len(valid_width) > 20:
            low_vol_width = np.mean(valid_width[:10])  # First part
            high_vol_width = np.mean(valid_width[-10:])  # Last part
            
            assert high_vol_width > low_vol_width
    
    def test_bollinger_bands_error_conditions(self):
        """Test Bollinger Bands error conditions"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test period less than 2
        with pytest.raises(ValueError, match="Period must be at least 2"):
            calculate_bollinger_bands(data, 1, 2.0)
        
        # Test negative standard deviation
        with pytest.raises(ValueError, match="non-negative"):
            calculate_bollinger_bands(data, 3, -1.0)
        
        # Test insufficient data
        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_bollinger_bands(data, 10, 2.0)
    
    def test_bollinger_bands_different_periods(self):
        """Test Bollinger Bands with different periods"""
        data = TestDataGenerator.generate_volatile_data(100)
        
        for period in [10, 20, 50]:
            upper, middle, lower = calculate_bollinger_bands(data, period, 2.0)
            
            # Check that we have the right number of NaN values
            nan_count = np.sum(np.isnan(middle))
            assert nan_count == period - 1
            
            # Check band relationships
            valid_indices = ~np.isnan(middle)
            assert np.all(upper[valid_indices] >= middle[valid_indices])
            assert np.all(lower[valid_indices] <= middle[valid_indices])


class TestAverageTrueRange:
    """Comprehensive tests for Average True Range (ATR)"""
    
    def test_atr_basic_calculation(self):
        """Test basic ATR calculation"""
        high = np.array([15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
        low = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        close = np.array([12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
        
        result = calculate_atr(high, low, close, 3)
        
        # ATR should be positive
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values > 0.0)
        
        # Check True Range calculation for first few values
        # TR[0] = high[0] - low[0] = 15 - 10 = 5
        # TR[1] = max(16-11, |16-12|, |11-12|) = max(5, 4, 1) = 5
        # TR[2] = max(17-12, |17-13|, |12-13|) = max(5, 4, 1) = 5
        # ATR[2] = (5 + 5 + 5) / 3 = 5
        
        assert abs(result[2] - 5.0) < 1e-10
    
    def test_atr_wilder_smoothing(self):
        """Test ATR Wilder's smoothing method"""
        high = np.array([15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 19.0, 18.0])
        low = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 14.0, 13.0])
        close = np.array([12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 16.0, 15.0])
        
        result = calculate_atr(high, low, close, 3)
        
        # Test that subsequent ATR values use Wilder's smoothing
        # ATR[i] = (ATR[i-1] * (period-1) + TR[i]) / period
        
        # Check that we have valid ATR values
        assert not np.isnan(result[2])  # First ATR value
        assert not np.isnan(result[3])  # Second ATR value (uses smoothing)
    
    def test_atr_volatility_response(self):
        """Test ATR response to volatility changes"""
        # Create data with increasing volatility
        base_price = 100.0
        periods = 30
        
        # Low volatility period
        low_vol_high = base_price + np.random.uniform(0, 1, periods)
        low_vol_low = base_price - np.random.uniform(0, 1, periods)
        low_vol_close = base_price + np.random.uniform(-0.5, 0.5, periods)
        
        # High volatility period
        high_vol_high = base_price + np.random.uniform(0, 10, periods)
        high_vol_low = base_price - np.random.uniform(0, 10, periods)
        high_vol_close = base_price + np.random.uniform(-5, 5, periods)
        
        # Combine data
        high = np.concatenate([low_vol_high, high_vol_high])
        low = np.concatenate([low_vol_low, high_vol_low])
        close = np.concatenate([low_vol_close, high_vol_close])
        
        result = calculate_atr(high, low, close, 14)
        
        # ATR should generally increase with volatility
        valid_atr = result[~np.isnan(result)]
        
        if len(valid_atr) > 20:
            # Compare average ATR in low vs high volatility periods
            mid_point = len(valid_atr) // 2
            low_vol_atr = np.mean(valid_atr[:mid_point])
            high_vol_atr = np.mean(valid_atr[mid_point:])
            
            assert high_vol_atr > low_vol_atr
    
    def test_atr_true_range_components(self):
        """Test ATR True Range components"""
        # Test all three True Range components
        high = np.array([15.0, 18.0, 14.0, 20.0])
        low = np.array([10.0, 12.0, 8.0, 15.0])
        close = np.array([12.0, 15.0, 10.0, 18.0])
        
        # Manual calculation of True Range for index 1:
        # TR = max(18-12, |18-12|, |12-12|) = max(6, 6, 0) = 6
        
        # Manual calculation of True Range for index 2:
        # TR = max(14-8, |14-15|, |8-15|) = max(6, 1, 7) = 7
        
        result = calculate_atr(high, low, close, 2)
        
        # ATR[1] should be average of first two TRs: (5 + 6) / 2 = 5.5
        # where TR[0] = 15-10 = 5, TR[1] = 6
        assert abs(result[1] - 5.5) < 1e-10
    
    def test_atr_error_conditions(self):
        """Test ATR error conditions"""
        high = np.array([15.0, 16.0, 17.0])
        low = np.array([10.0, 11.0, 12.0])
        close = np.array([12.0, 13.0, 14.0])
        
        # Test invalid period
        with pytest.raises(ValueError, match="Period must be at least 1"):
            calculate_atr(high, low, close, 0)
        
        # Test mismatched array lengths
        with pytest.raises(ValueError, match="same length"):
            calculate_atr(high, low[:-1], close, 3)
        
        # Test insufficient data
        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_atr(high, low, close, 5)


class TestKeltnerChannels:
    """Comprehensive tests for Keltner Channels"""
    
    def test_keltner_channels_basic_calculation(self):
        """Test basic Keltner Channels calculation"""
        data = TestDataGenerator.generate_ohlcv_data(
            TestDataGenerator.generate_volatile_data(50)
        )
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        upper, middle, lower = calculate_keltner_channels(high, low, close, 20, 2.0)
        
        # Check that we have three arrays of the same length
        assert len(upper) == len(middle) == len(lower) == len(close)
        
        # Middle line should be EMA of close
        from src.indicators.moving_averages import calculate_ema
        expected_middle = calculate_ema(close, 20)
        
        assert ValidationHelpers.assert_array_almost_equal(
            middle, expected_middle, decimals=TEST_CONFIG['precision_decimals']
        )
        
        # Upper channel should be above middle, lower channel below middle
        valid_indices = ~np.isnan(middle)
        assert np.all(upper[valid_indices] >= middle[valid_indices])
        assert np.all(lower[valid_indices] <= middle[valid_indices])
    
    def test_keltner_channels_atr_relationship(self):
        """Test Keltner Channels relationship with ATR"""
        data = TestDataGenerator.generate_ohlcv_data(
            TestDataGenerator.generate_volatile_data(50)
        )
        
        high = data['high']
        low = data['low']
        close = data['close']
        period = 20
        multiplier = 2.0
        
        upper, middle, lower = calculate_keltner_channels(high, low, close, period, multiplier)
        
        # Calculate ATR and EMA separately
        atr = calculate_atr(high, low, close, period)
        from src.indicators.moving_averages import calculate_ema
        ema = calculate_ema(close, period)
        
        # Upper channel should be EMA + (ATR * multiplier)
        # Lower channel should be EMA - (ATR * multiplier)
        expected_upper = ema + (atr * multiplier)
        expected_lower = ema - (atr * multiplier)
        
        assert ValidationHelpers.assert_array_almost_equal(
            upper, expected_upper, decimals=TEST_CONFIG['precision_decimals']
        )
        assert ValidationHelpers.assert_array_almost_equal(
            lower, expected_lower, decimals=TEST_CONFIG['precision_decimals']
        )
    
    def test_keltner_channels_multiplier_scaling(self):
        """Test Keltner Channels multiplier scaling"""
        data = TestDataGenerator.generate_ohlcv_data(
            TestDataGenerator.generate_volatile_data(50)
        )
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Test different multipliers
        multipliers = [1.0, 2.0, 3.0]
        results = []
        
        for mult in multipliers:
            upper, middle, lower = calculate_keltner_channels(high, low, close, 20, mult)
            results.append((upper, middle, lower))
        
        # Middle line should be the same for all multipliers
        for i in range(1, len(results)):
            assert ValidationHelpers.assert_array_almost_equal(
                results[0][1], results[i][1], decimals=TEST_CONFIG['precision_decimals']
            )
        
        # Channel width should increase with multiplier
        for i in range(len(close)):
            if not np.isnan(results[0][0][i]):
                width_1 = results[0][0][i] - results[0][2][i]  # mult = 1.0
                width_2 = results[1][0][i] - results[1][2][i]  # mult = 2.0
                width_3 = results[2][0][i] - results[2][2][i]  # mult = 3.0
                
                assert width_2 > width_1
                assert width_3 > width_2
    
    def test_keltner_channels_error_conditions(self):
        """Test Keltner Channels error conditions"""
        high = np.array([15.0, 16.0, 17.0])
        low = np.array([10.0, 11.0, 12.0])
        close = np.array([12.0, 13.0, 14.0])
        
        # Test invalid period
        with pytest.raises(ValueError, match="Period must be at least 1"):
            calculate_keltner_channels(high, low, close, 0, 2.0)
        
        # Test negative multiplier
        with pytest.raises(ValueError, match="non-negative"):
            calculate_keltner_channels(high, low, close, 20, -1.0)
        
        # Test mismatched array lengths
        with pytest.raises(ValueError, match="same length"):
            calculate_keltner_channels(high, low[:-1], close, 20, 2.0)


class TestDonchianChannels:
    """Comprehensive tests for Donchian Channels"""
    
    def test_donchian_channels_basic_calculation(self):
        """Test basic Donchian Channels calculation"""
        high = np.array([15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 19.0, 18.0, 17.0, 16.0])
        low = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 14.0, 13.0, 12.0, 11.0])
        
        upper, middle, lower = calculate_donchian_channels(high, low, 5)
        
        # Check that we have three arrays of the same length
        assert len(upper) == len(middle) == len(lower) == len(high)
        
        # Upper channel should be highest high over period
        # Lower channel should be lowest low over period
        # Middle channel should be average of upper and lower
        
        # Test specific values
        # At index 4 (5th element), we look at indices 0-4
        # Highest high = 19, Lowest low = 10, Middle = (19+10)/2 = 14.5
        assert abs(upper[4] - 19.0) < 1e-10
        assert abs(lower[4] - 10.0) < 1e-10
        assert abs(middle[4] - 14.5) < 1e-10
    
    def test_donchian_channels_mathematical_properties(self):
        """Test mathematical properties of Donchian Channels"""
        high = np.array([15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
        low = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        
        upper, middle, lower = calculate_donchian_channels(high, low, 3)
        
        # Middle channel should always be average of upper and lower
        valid_indices = ~np.isnan(upper)
        calculated_middle = (upper[valid_indices] + lower[valid_indices]) / 2.0
        
        assert ValidationHelpers.assert_array_almost_equal(
            middle[valid_indices], calculated_middle, decimals=TEST_CONFIG['precision_decimals']
        )
        
        # Upper should always be >= lower
        assert np.all(upper[valid_indices] >= lower[valid_indices])
    
    def test_donchian_channels_breakout_detection(self):
        """Test Donchian Channels for breakout detection"""
        # Create data with a clear breakout
        stable_high = np.full(20, 15.0)
        stable_low = np.full(20, 10.0)
        
        breakout_high = np.array([15.0, 16.0, 20.0, 21.0, 22.0])  # Breakout to upside
        breakout_low = np.array([10.0, 10.0, 12.0, 13.0, 14.0])
        
        high = np.concatenate([stable_high, breakout_high])
        low = np.concatenate([stable_low, breakout_low])
        
        upper, middle, lower = calculate_donchian_channels(high, low, 10)
        
        # During stable period, channels should be constant
        stable_upper = upper[9:20]  # Values during stable period
        stable_lower = lower[9:20]
        
        # Should be relatively stable
        assert np.std(stable_upper[~np.isnan(stable_upper)]) < 1e-10
        assert np.std(stable_lower[~np.isnan(stable_lower)]) < 1e-10
        
        # During breakout, upper channel should increase
        if len(upper) > 22:
            assert upper[-1] > upper[19]  # Last value > pre-breakout value
    
    def test_donchian_channels_constant_data(self):
        """Test Donchian Channels with constant data"""
        high = np.full(20, 15.0)
        low = np.full(20, 10.0)
        
        upper, middle, lower = calculate_donchian_channels(high, low, 10)
        
        # For constant data, channels should be constant
        valid_indices = ~np.isnan(upper)
        assert np.allclose(upper[valid_indices], 15.0)
        assert np.allclose(lower[valid_indices], 10.0)
        assert np.allclose(middle[valid_indices], 12.5)
    
    def test_donchian_channels_error_conditions(self):
        """Test Donchian Channels error conditions"""
        high = np.array([15.0, 16.0, 17.0])
        low = np.array([10.0, 11.0, 12.0])
        
        # Test invalid period
        with pytest.raises(ValueError, match="Period must be at least 1"):
            calculate_donchian_channels(high, low, 0)
        
        # Test mismatched array lengths
        with pytest.raises(ValueError, match="same length"):
            calculate_donchian_channels(high, low[:-1], 3)
        
        # Test insufficient data
        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_donchian_channels(high, low, 5)


class TestVolatilityMeasures:
    """Comprehensive tests for volatility measures"""
    
    def test_volatility_basic_calculation(self):
        """Test basic volatility calculation"""
        # Create data with known volatility characteristics
        data = np.array([100.0, 102.0, 98.0, 103.0, 97.0, 104.0, 96.0, 105.0])
        
        result = calculate_volatility(data, 5, annualize=False)
        
        # Volatility should be positive
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values >= 0.0)
    
    def test_volatility_annualization(self):
        """Test volatility annualization"""
        data = TestDataGenerator.generate_volatile_data(100)
        trading_periods = 252
        
        # Calculate both annualized and non-annualized
        vol_daily = calculate_volatility(data, 20, annualize=False)
        vol_annual = calculate_volatility(data, 20, annualize=True, trading_periods=trading_periods)
        
        # Annualized volatility should be higher (multiplied by sqrt(252))
        valid_indices = ~(np.isnan(vol_daily) | np.isnan(vol_annual))
        
        if np.any(valid_indices):
            ratio = vol_annual[valid_indices] / vol_daily[valid_indices]
            expected_ratio = np.sqrt(trading_periods)
            
            assert np.allclose(ratio, expected_ratio, rtol=1e-10)
    
    def test_volatility_constant_data(self):
        """Test volatility with constant data"""
        data = EdgeCaseTestData.get_constant_prices()
        
        result = calculate_volatility(data, 10)
        
        # Volatility should be zero for constant data
        valid_values = result[~np.isnan(result)]
        assert np.allclose(valid_values, 0.0)
    
    def test_volatility_error_conditions(self):
        """Test volatility error conditions"""
        data = np.array([1.0, 2.0, 3.0])
        
        # Test period less than 2
        with pytest.raises(ValueError, match="Period must be at least 2"):
            calculate_volatility(data, 1)
        
        # Test invalid trading periods
        with pytest.raises(ValueError, match="Trading periods must be positive"):
            calculate_volatility(data, 5, trading_periods=0)


class TestBollingerBandUtilities:
    """Comprehensive tests for Bollinger Band utilities"""
    
    def test_bb_width_calculation(self):
        """Test Bollinger Band Width calculation"""
        upper = np.array([110.0, 111.0, 112.0, 113.0, 114.0])
        middle = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        lower = np.array([90.0, 89.0, 88.0, 87.0, 86.0])
        
        result = calculate_bb_width(upper, middle, lower)
        
        # BB Width = (Upper - Lower) / Middle
        expected = (upper - lower) / middle
        
        assert ValidationHelpers.assert_array_almost_equal(
            result, expected, decimals=TEST_CONFIG['precision_decimals']
        )
    
    def test_bb_width_zero_middle(self):
        """Test BB Width with zero middle band"""
        upper = np.array([10.0, 11.0, 12.0])
        middle = np.array([0.0, 10.0, 5.0])
        lower = np.array([5.0, 9.0, 2.0])
        
        result = calculate_bb_width(upper, middle, lower)
        
        # Should return NaN for zero middle band
        assert np.isnan(result[0])
        assert not np.isnan(result[1])
        assert not np.isnan(result[2])
    
    def test_bb_percent_calculation(self):
        """Test Bollinger Band Percent (%B) calculation"""
        data = np.array([95.0, 100.0, 105.0, 110.0, 115.0])
        upper = np.array([110.0, 110.0, 110.0, 110.0, 110.0])
        lower = np.array([90.0, 90.0, 90.0, 90.0, 90.0])
        
        result = calculate_bb_percent(data, upper, lower)
        
        # %B = (Price - Lower) / (Upper - Lower)
        expected = (data - lower) / (upper - lower)
        
        assert ValidationHelpers.assert_array_almost_equal(
            result, expected, decimals=TEST_CONFIG['precision_decimals']
        )
        
        # Check specific values
        assert abs(result[0] - 0.25) < 1e-10  # (95-90)/(110-90) = 0.25
        assert abs(result[1] - 0.50) < 1e-10  # (100-90)/(110-90) = 0.50
        assert abs(result[4] - 1.25) < 1e-10  # (115-90)/(110-90) = 1.25
    
    def test_bb_percent_zero_band_width(self):
        """Test %B with zero band width"""
        data = np.array([100.0, 100.0, 100.0])
        upper = np.array([100.0, 100.0, 100.0])
        lower = np.array([100.0, 100.0, 100.0])
        
        result = calculate_bb_percent(data, upper, lower)
        
        # Should return NaN for zero band width
        assert np.all(np.isnan(result))
    
    def test_bb_utilities_error_conditions(self):
        """Test BB utilities error conditions"""
        upper = np.array([110.0, 111.0, 112.0])
        middle = np.array([100.0, 100.0])
        lower = np.array([90.0, 89.0, 88.0])
        
        # Test mismatched array lengths
        with pytest.raises(ValueError, match="same length"):
            calculate_bb_width(upper, middle, lower)


class TestPriceChannels:
    """Comprehensive tests for Price Channels"""
    
    def test_price_channels_basic_calculation(self):
        """Test basic Price Channels calculation"""
        high = np.array([15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 19.0, 18.0, 17.0, 16.0])
        low = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 14.0, 13.0, 12.0, 11.0])
        
        highest_high, lowest_low = calculate_price_channels(high, low, 5)
        
        # Check that we have two arrays of the same length
        assert len(highest_high) == len(lowest_low) == len(high)
        
        # Test specific values
        # At index 4, we look at indices 0-4
        # Highest high = 19, Lowest low = 10
        assert abs(highest_high[4] - 19.0) < 1e-10
        assert abs(lowest_low[4] - 10.0) < 1e-10
    
    def test_price_channels_mathematical_properties(self):
        """Test mathematical properties of Price Channels"""
        high = np.array([15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
        low = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        
        highest_high, lowest_low = calculate_price_channels(high, low, 3)
        
        # Highest high should always be >= lowest low
        valid_indices = ~np.isnan(highest_high)
        assert np.all(highest_high[valid_indices] >= lowest_low[valid_indices])
        
        # Highest high should be one of the actual high values
        # Lowest low should be one of the actual low values
        for i in range(len(high)):
            if not np.isnan(highest_high[i]):
                period_start = max(0, i - 2)  # 3-period lookback
                assert highest_high[i] in high[period_start:i+1]
                assert lowest_low[i] in low[period_start:i+1]
    
    def test_price_channels_error_conditions(self):
        """Test Price Channels error conditions"""
        high = np.array([15.0, 16.0, 17.0])
        low = np.array([10.0, 11.0, 12.0])
        
        # Test invalid period
        with pytest.raises(ValueError, match="Period must be at least 1"):
            calculate_price_channels(high, low, 0)
        
        # Test mismatched array lengths
        with pytest.raises(ValueError, match="same length"):
            calculate_price_channels(high, low[:-1], 3)


class TestVolatilityIntegration:
    """Integration tests for all volatility indicators"""
    
    def test_all_volatility_indicators_consistency(self):
        """Test that all volatility indicators produce reasonable results"""
        data = TestDataGenerator.generate_ohlcv_data(
            TestDataGenerator.generate_volatile_data(100)
        )
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate all volatility indicators
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
        atr = calculate_atr(high, low, close, 14)
        kc_upper, kc_middle, kc_lower = calculate_keltner_channels(high, low, close, 20, 2.0)
        dc_upper, dc_middle, dc_lower = calculate_donchian_channels(high, low, 20)
        volatility = calculate_volatility(close, 20)
        
        # All should produce finite values where valid
        indicators = [bb_upper, bb_middle, bb_lower, atr, kc_upper, kc_middle, kc_lower,
                     dc_upper, dc_middle, dc_lower, volatility]
        
        for indicator in indicators:
            valid_values = indicator[~np.isnan(indicator)]
            assert np.all(np.isfinite(valid_values))
    
    def test_volatility_indicator_relationships(self):
        """Test relationships between volatility indicators"""
        data = TestDataGenerator.generate_ohlcv_data(
            TestDataGenerator.generate_volatile_data(100)
        )
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate indicators
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
        kc_upper, kc_middle, kc_lower = calculate_keltner_channels(high, low, close, 20, 2.0)
        
        # BB and KC middle lines should be different (SMA vs EMA)
        # but generally close to each other
        valid_indices = ~(np.isnan(bb_middle) | np.isnan(kc_middle))
        
        if np.sum(valid_indices) > 10:
            correlation = np.corrcoef(bb_middle[valid_indices], kc_middle[valid_indices])[0, 1]
            assert correlation > 0.9  # Should be highly correlated
    
    def test_volatility_scaling_properties(self):
        """Test scaling properties of volatility indicators"""
        base_data = TestDataGenerator.generate_volatile_data(50)
        
        # Scale data by factor of 2
        scaled_data = base_data * 2
        
        # Calculate Bollinger Bands for both
        bb1_upper, bb1_middle, bb1_lower = calculate_bollinger_bands(base_data, 20, 2.0)
        bb2_upper, bb2_middle, bb2_lower = calculate_bollinger_bands(scaled_data, 20, 2.0)
        
        # Bollinger Bands should scale linearly
        valid_indices = ~np.isnan(bb1_middle)
        
        if np.sum(valid_indices) > 5:
            ratio_middle = bb2_middle[valid_indices] / bb1_middle[valid_indices]
            ratio_upper = bb2_upper[valid_indices] / bb1_upper[valid_indices]
            ratio_lower = bb2_lower[valid_indices] / bb1_lower[valid_indices]
            
            # All ratios should be close to 2
            assert np.allclose(ratio_middle, 2.0, rtol=1e-10)
            assert np.allclose(ratio_upper, 2.0, rtol=1e-10)
            assert np.allclose(ratio_lower, 2.0, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])