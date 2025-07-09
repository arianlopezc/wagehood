"""
Incremental vs Batch Calculation Validation Tests

This module provides comprehensive tests to validate that incremental calculations
produce the same results as batch calculations for all technical indicators.
This is crucial for real-time trading systems.
"""

import pytest
import numpy as np
from typing import Dict, List, Tuple, Any
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.indicators.moving_averages import (
    calculate_sma, calculate_ema, calculate_wma, calculate_vwma
)
from src.indicators.momentum import (
    calculate_rsi, calculate_macd, calculate_stochastic,
    calculate_williams_r, calculate_cci, calculate_momentum, calculate_roc
)
from src.indicators.volatility import (
    calculate_bollinger_bands, calculate_atr, calculate_keltner_channels,
    calculate_donchian_channels, calculate_volatility
)

from .fixtures.test_data import (
    TestDataGenerator,
    PrecisionTestVectors,
    EdgeCaseTestData,
    ValidationHelpers,
    TEST_CONFIG
)


class IncrementalCalculator:
    """Helper class for performing incremental calculations"""
    
    @staticmethod
    def incremental_sma(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate SMA incrementally"""
        result = np.full(len(data), np.nan)
        
        for i in range(period - 1, len(data)):
            window = data[i - period + 1:i + 1]
            result[i] = np.mean(window)
        
        return result
    
    @staticmethod
    def incremental_ema(data: np.ndarray, period: int, alpha: float = None) -> np.ndarray:
        """Calculate EMA incrementally"""
        if alpha is None:
            alpha = 2.0 / (period + 1)
        
        result = np.full(len(data), np.nan)
        
        if len(data) >= period:
            # Start with SMA for first value
            result[period - 1] = np.mean(data[:period])
            
            # Calculate incrementally
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        
        return result
    
    @staticmethod
    def incremental_rsi(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI incrementally using Wilder's smoothing"""
        result = np.full(len(data), np.nan)
        
        if len(data) <= period:
            return result
        
        # Calculate price changes
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Initial average gain and loss
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # Calculate first RSI
        if avg_loss == 0:
            result[period] = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            result[period] = 100.0 - (100.0 / (1 + rs))
        
        # Calculate subsequent RSI values incrementally
        for i in range(period + 1, len(data)):
            gain_idx = i - 1
            loss_idx = i - 1
            
            # Wilder's smoothing
            avg_gain = (avg_gain * (period - 1) + gains[gain_idx]) / period
            avg_loss = (avg_loss * (period - 1) + losses[loss_idx]) / period
            
            if avg_loss == 0:
                result[i] = 100.0 if avg_gain > 0 else 50.0
            else:
                rs = avg_gain / avg_loss
                result[i] = 100.0 - (100.0 / (1 + rs))
        
        return result
    
    @staticmethod
    def incremental_bollinger_bands(data: np.ndarray, period: int, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands incrementally"""
        upper = np.full(len(data), np.nan)
        middle = np.full(len(data), np.nan)
        lower = np.full(len(data), np.nan)
        
        for i in range(period - 1, len(data)):
            window = data[i - period + 1:i + 1]
            sma = np.mean(window)
            std = np.std(window, ddof=0)
            
            middle[i] = sma
            upper[i] = sma + (std * std_dev)
            lower[i] = sma - (std * std_dev)
        
        return upper, middle, lower
    
    @staticmethod
    def incremental_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate ATR incrementally"""
        result = np.full(len(close), np.nan)
        
        # Calculate True Range incrementally
        tr = np.full(len(close), np.nan)
        tr[0] = high[0] - low[0]
        
        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)
        
        # Calculate ATR using Wilder's smoothing
        if len(tr) >= period:
            result[period - 1] = np.mean(tr[:period])
            
            for i in range(period, len(close)):
                result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
        
        return result
    
    @staticmethod
    def incremental_macd(data: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD incrementally"""
        # Calculate EMAs incrementally
        fast_ema = IncrementalCalculator.incremental_ema(data, fast)
        slow_ema = IncrementalCalculator.incremental_ema(data, slow)
        
        # MACD line
        macd_line = fast_ema - slow_ema
        
        # Signal line (EMA of MACD line)
        signal_line = np.full(len(data), np.nan)
        
        # Find first valid MACD value
        valid_start = slow - 1
        if len(data) > valid_start:
            valid_macd = macd_line[valid_start:]
            valid_macd_clean = valid_macd[~np.isnan(valid_macd)]
            
            if len(valid_macd_clean) >= signal:
                signal_ema = IncrementalCalculator.incremental_ema(valid_macd_clean, signal)
                signal_line[valid_start:valid_start + len(signal_ema)] = signal_ema
        
        # Histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram


class TestMovingAverageIncremental:
    """Test incremental vs batch calculations for moving averages"""
    
    def test_sma_incremental_vs_batch(self):
        """Test SMA incremental vs batch calculation"""
        data = TestDataGenerator.generate_volatile_data(100)
        period = 20
        
        # Batch calculation
        batch_result = calculate_sma(data, period)
        
        # Incremental calculation
        incremental_result = IncrementalCalculator.incremental_sma(data, period)
        
        # Should be identical
        assert ValidationHelpers.assert_array_almost_equal(
            batch_result, incremental_result, decimals=TEST_CONFIG['precision_decimals']
        )
    
    def test_ema_incremental_vs_batch(self):
        """Test EMA incremental vs batch calculation"""
        data = TestDataGenerator.generate_volatile_data(100)
        period = 20
        
        # Batch calculation
        batch_result = calculate_ema(data, period)
        
        # Incremental calculation
        incremental_result = IncrementalCalculator.incremental_ema(data, period)
        
        # Should be very close (allowing for small floating point differences)
        assert ValidationHelpers.assert_array_almost_equal(
            batch_result, incremental_result, decimals=10
        )
    
    def test_wma_incremental_vs_batch(self):
        """Test WMA incremental vs batch calculation"""
        data = TestDataGenerator.generate_volatile_data(50)
        period = 10
        
        # Batch calculation
        batch_result = calculate_wma(data, period)
        
        # Manual incremental calculation
        incremental_result = np.full(len(data), np.nan)
        weights = np.arange(1, period + 1)
        weights_sum = np.sum(weights)
        
        for i in range(period - 1, len(data)):
            window = data[i - period + 1:i + 1]
            incremental_result[i] = np.sum(window * weights) / weights_sum
        
        # Should be identical
        assert ValidationHelpers.assert_array_almost_equal(
            batch_result, incremental_result, decimals=TEST_CONFIG['precision_decimals']
        )
    
    def test_vwma_incremental_vs_batch(self):
        """Test VWMA incremental vs batch calculation"""
        data = TestDataGenerator.generate_volatile_data(50)
        volume = TestDataGenerator.generate_volatile_data(50, base_price=1000)
        period = 10
        
        # Batch calculation
        batch_result = calculate_vwma(data, volume, period)
        
        # Manual incremental calculation
        incremental_result = np.full(len(data), np.nan)
        
        for i in range(period - 1, len(data)):
            price_window = data[i - period + 1:i + 1]
            volume_window = volume[i - period + 1:i + 1]
            
            if np.sum(volume_window) == 0:
                incremental_result[i] = np.nan
            else:
                incremental_result[i] = np.sum(price_window * volume_window) / np.sum(volume_window)
        
        # Should be identical
        assert ValidationHelpers.assert_array_almost_equal(
            batch_result, incremental_result, decimals=TEST_CONFIG['precision_decimals']
        )
    
    def test_ema_real_time_simulation(self):
        """Test EMA in real-time simulation mode"""
        data = TestDataGenerator.generate_linear_trend(50)
        period = 10
        
        # Batch calculation for reference
        batch_result = calculate_ema(data, period)
        
        # Simulate real-time calculation
        realtime_result = np.full(len(data), np.nan)
        alpha = 2.0 / (period + 1)
        
        # Process data point by point
        for i in range(len(data)):
            if i < period - 1:
                continue
            elif i == period - 1:
                # First EMA value is SMA
                realtime_result[i] = np.mean(data[:i + 1])
            else:
                # Subsequent values use EMA formula
                realtime_result[i] = alpha * data[i] + (1 - alpha) * realtime_result[i - 1]
        
        # Should match batch calculation
        assert ValidationHelpers.assert_array_almost_equal(
            batch_result, realtime_result, decimals=10
        )


class TestMomentumIndicatorIncremental:
    """Test incremental vs batch calculations for momentum indicators"""
    
    def test_rsi_incremental_vs_batch(self):
        """Test RSI incremental vs batch calculation"""
        data = TestDataGenerator.generate_volatile_data(100)
        period = 14
        
        # Batch calculation
        batch_result = calculate_rsi(data, period)
        
        # Incremental calculation
        incremental_result = IncrementalCalculator.incremental_rsi(data, period)
        
        # Should be very close (RSI has some floating point accumulation)
        assert ValidationHelpers.assert_array_almost_equal(
            batch_result, incremental_result, decimals=8
        )
    
    def test_macd_incremental_vs_batch(self):
        """Test MACD incremental vs batch calculation"""
        data = TestDataGenerator.generate_volatile_data(100)
        fast, slow, signal = 12, 26, 9
        
        # Batch calculation
        batch_macd, batch_signal, batch_histogram = calculate_macd(data, fast, slow, signal)
        
        # Incremental calculation
        inc_macd, inc_signal, inc_histogram = IncrementalCalculator.incremental_macd(data, fast, slow, signal)
        
        # Should be very close
        assert ValidationHelpers.assert_array_almost_equal(
            batch_macd, inc_macd, decimals=8
        )
        assert ValidationHelpers.assert_array_almost_equal(
            batch_signal, inc_signal, decimals=8
        )
        assert ValidationHelpers.assert_array_almost_equal(
            batch_histogram, inc_histogram, decimals=8
        )
    
    def test_momentum_incremental_vs_batch(self):
        """Test momentum incremental vs batch calculation"""
        data = TestDataGenerator.generate_volatile_data(50)
        period = 10
        
        # Batch calculation
        batch_result = calculate_momentum(data, period)
        
        # Manual incremental calculation
        incremental_result = np.full(len(data), np.nan)
        
        for i in range(period, len(data)):
            incremental_result[i] = data[i] - data[i - period]
        
        # Should be identical
        assert ValidationHelpers.assert_array_almost_equal(
            batch_result, incremental_result, decimals=TEST_CONFIG['precision_decimals']
        )
    
    def test_roc_incremental_vs_batch(self):
        """Test Rate of Change incremental vs batch calculation"""
        data = TestDataGenerator.generate_volatile_data(50)
        period = 10
        
        # Batch calculation
        batch_result = calculate_roc(data, period)
        
        # Manual incremental calculation
        incremental_result = np.full(len(data), np.nan)
        
        for i in range(period, len(data)):
            if data[i - period] != 0:
                incremental_result[i] = ((data[i] - data[i - period]) / data[i - period]) * 100.0
            else:
                incremental_result[i] = np.nan
        
        # Should be identical
        assert ValidationHelpers.assert_array_almost_equal(
            batch_result, incremental_result, decimals=TEST_CONFIG['precision_decimals']
        )
    
    def test_stochastic_incremental_vs_batch(self):
        """Test Stochastic oscillator incremental vs batch calculation"""
        data = TestDataGenerator.generate_ohlcv_data(
            TestDataGenerator.generate_volatile_data(50)
        )
        
        high = data['high']
        low = data['low']
        close = data['close']
        k_period = 14
        
        # Batch calculation
        batch_k, batch_d = calculate_stochastic(high, low, close, k_period, 3, 1)
        
        # Manual incremental calculation for %K
        incremental_k = np.full(len(close), np.nan)
        
        for i in range(k_period - 1, len(close)):
            period_high = np.max(high[i - k_period + 1:i + 1])
            period_low = np.min(low[i - k_period + 1:i + 1])
            
            if period_high == period_low:
                incremental_k[i] = 50.0
            else:
                incremental_k[i] = ((close[i] - period_low) / (period_high - period_low)) * 100.0
        
        # Should be very close for %K
        assert ValidationHelpers.assert_array_almost_equal(
            batch_k, incremental_k, decimals=TEST_CONFIG['precision_decimals']
        )
    
    def test_williams_r_incremental_vs_batch(self):
        """Test Williams %R incremental vs batch calculation"""
        data = TestDataGenerator.generate_ohlcv_data(
            TestDataGenerator.generate_volatile_data(50)
        )
        
        high = data['high']
        low = data['low']
        close = data['close']
        period = 14
        
        # Batch calculation
        batch_result = calculate_williams_r(high, low, close, period)
        
        # Manual incremental calculation
        incremental_result = np.full(len(close), np.nan)
        
        for i in range(period - 1, len(close)):
            period_high = np.max(high[i - period + 1:i + 1])
            period_low = np.min(low[i - period + 1:i + 1])
            
            if period_high == period_low:
                incremental_result[i] = -50.0
            else:
                incremental_result[i] = ((period_high - close[i]) / (period_high - period_low)) * -100.0
        
        # Should be identical
        assert ValidationHelpers.assert_array_almost_equal(
            batch_result, incremental_result, decimals=TEST_CONFIG['precision_decimals']
        )


class TestVolatilityIndicatorIncremental:
    """Test incremental vs batch calculations for volatility indicators"""
    
    def test_bollinger_bands_incremental_vs_batch(self):
        """Test Bollinger Bands incremental vs batch calculation"""
        data = TestDataGenerator.generate_volatile_data(100)
        period = 20
        std_dev = 2.0
        
        # Batch calculation
        batch_upper, batch_middle, batch_lower = calculate_bollinger_bands(data, period, std_dev)
        
        # Incremental calculation
        inc_upper, inc_middle, inc_lower = IncrementalCalculator.incremental_bollinger_bands(data, period, std_dev)
        
        # Should be identical
        assert ValidationHelpers.assert_array_almost_equal(
            batch_upper, inc_upper, decimals=TEST_CONFIG['precision_decimals']
        )
        assert ValidationHelpers.assert_array_almost_equal(
            batch_middle, inc_middle, decimals=TEST_CONFIG['precision_decimals']
        )
        assert ValidationHelpers.assert_array_almost_equal(
            batch_lower, inc_lower, decimals=TEST_CONFIG['precision_decimals']
        )
    
    def test_atr_incremental_vs_batch(self):
        """Test ATR incremental vs batch calculation"""
        data = TestDataGenerator.generate_ohlcv_data(
            TestDataGenerator.generate_volatile_data(100)
        )
        
        high = data['high']
        low = data['low']
        close = data['close']
        period = 14
        
        # Batch calculation
        batch_result = calculate_atr(high, low, close, period)
        
        # Incremental calculation
        incremental_result = IncrementalCalculator.incremental_atr(high, low, close, period)
        
        # Should be very close (ATR uses Wilder's smoothing)
        assert ValidationHelpers.assert_array_almost_equal(
            batch_result, incremental_result, decimals=8
        )
    
    def test_donchian_channels_incremental_vs_batch(self):
        """Test Donchian Channels incremental vs batch calculation"""
        data = TestDataGenerator.generate_ohlcv_data(
            TestDataGenerator.generate_volatile_data(50)
        )
        
        high = data['high']
        low = data['low']
        period = 20
        
        # Batch calculation
        batch_upper, batch_middle, batch_lower = calculate_donchian_channels(high, low, period)
        
        # Manual incremental calculation
        inc_upper = np.full(len(high), np.nan)
        inc_lower = np.full(len(high), np.nan)
        inc_middle = np.full(len(high), np.nan)
        
        for i in range(period - 1, len(high)):
            inc_upper[i] = np.max(high[i - period + 1:i + 1])
            inc_lower[i] = np.min(low[i - period + 1:i + 1])
            inc_middle[i] = (inc_upper[i] + inc_lower[i]) / 2.0
        
        # Should be identical
        assert ValidationHelpers.assert_array_almost_equal(
            batch_upper, inc_upper, decimals=TEST_CONFIG['precision_decimals']
        )
        assert ValidationHelpers.assert_array_almost_equal(
            batch_lower, inc_lower, decimals=TEST_CONFIG['precision_decimals']
        )
        assert ValidationHelpers.assert_array_almost_equal(
            batch_middle, inc_middle, decimals=TEST_CONFIG['precision_decimals']
        )
    
    def test_volatility_incremental_vs_batch(self):
        """Test historical volatility incremental vs batch calculation"""
        data = TestDataGenerator.generate_volatile_data(100)
        period = 20
        
        # Batch calculation
        batch_result = calculate_volatility(data, period, annualize=False)
        
        # Manual incremental calculation
        incremental_result = np.full(len(data), np.nan)
        
        # Calculate log returns
        log_returns = np.log(data[1:] / data[:-1])
        
        for i in range(period, len(data)):
            period_returns = log_returns[i - period:i]
            incremental_result[i] = np.std(period_returns, ddof=1)
        
        # Should be very close
        assert ValidationHelpers.assert_array_almost_equal(
            batch_result, incremental_result, decimals=10
        )


class TestRealTimeSimulation:
    """Test real-time calculation scenarios"""
    
    def test_streaming_ema_calculation(self):
        """Test EMA calculation in streaming mode"""
        # Simulate receiving data points one by one
        full_data = TestDataGenerator.generate_volatile_data(100)
        period = 20
        alpha = 2.0 / (period + 1)
        
        # Calculate EMA as data streams in
        streaming_ema = []
        ema_value = None
        
        for i, price in enumerate(full_data):
            if i < period - 1:
                streaming_ema.append(np.nan)
            elif i == period - 1:
                # First EMA is SMA
                ema_value = np.mean(full_data[:i + 1])
                streaming_ema.append(ema_value)
            else:
                # Update EMA incrementally
                ema_value = alpha * price + (1 - alpha) * ema_value
                streaming_ema.append(ema_value)
        
        # Compare with batch calculation
        batch_ema = calculate_ema(full_data, period)
        
        assert ValidationHelpers.assert_array_almost_equal(
            np.array(streaming_ema), batch_ema, decimals=10
        )
    
    def test_streaming_rsi_calculation(self):
        """Test RSI calculation in streaming mode"""
        full_data = TestDataGenerator.generate_volatile_data(50)
        period = 14
        
        # Simulate streaming RSI calculation
        streaming_rsi = []
        avg_gain = None
        avg_loss = None
        
        for i, price in enumerate(full_data):
            if i == 0:
                streaming_rsi.append(np.nan)
                continue
            
            # Calculate price change
            change = price - full_data[i - 1]
            gain = change if change > 0 else 0
            loss = -change if change < 0 else 0
            
            if i < period:
                streaming_rsi.append(np.nan)
                continue
            elif i == period:
                # Initialize averages
                changes = np.diff(full_data[:i + 1])
                gains = np.where(changes > 0, changes, 0)
                losses = np.where(changes < 0, -changes, 0)
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
            else:
                # Update averages using Wilder's smoothing
                avg_gain = (avg_gain * (period - 1) + gain) / period
                avg_loss = (avg_loss * (period - 1) + loss) / period
            
            # Calculate RSI
            if avg_loss == 0:
                rsi = 100.0 if avg_gain > 0 else 50.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1 + rs))
            
            streaming_rsi.append(rsi)
        
        # Compare with batch calculation
        batch_rsi = calculate_rsi(full_data, period)
        
        assert ValidationHelpers.assert_array_almost_equal(
            np.array(streaming_rsi), batch_rsi, decimals=8
        )
    
    def test_partial_data_updates(self):
        """Test indicator updates with partial data"""
        base_data = TestDataGenerator.generate_linear_trend(50)
        
        # Calculate indicators on initial data
        initial_sma = calculate_sma(base_data, 10)
        initial_ema = calculate_ema(base_data, 10)
        
        # Add new data points
        new_points = TestDataGenerator.generate_linear_trend(10, start_price=base_data[-1], slope=0.5)
        extended_data = np.concatenate([base_data, new_points])
        
        # Calculate on extended data
        extended_sma = calculate_sma(extended_data, 10)
        extended_ema = calculate_ema(extended_data, 10)
        
        # Initial portion should match
        assert ValidationHelpers.assert_array_almost_equal(
            initial_sma, extended_sma[:len(initial_sma)], decimals=TEST_CONFIG['precision_decimals']
        )
        assert ValidationHelpers.assert_array_almost_equal(
            initial_ema, extended_ema[:len(initial_ema)], decimals=10
        )
    
    def test_indicator_state_consistency(self):
        """Test that indicator calculations are stateless"""
        data = TestDataGenerator.generate_volatile_data(100)
        
        # Calculate multiple times
        sma1 = calculate_sma(data, 20)
        sma2 = calculate_sma(data, 20)
        sma3 = calculate_sma(data, 20)
        
        # Should be identical every time
        assert np.array_equal(sma1, sma2, equal_nan=True)
        assert np.array_equal(sma2, sma3, equal_nan=True)
        
        # Same for more complex indicators
        rsi1 = calculate_rsi(data, 14)
        rsi2 = calculate_rsi(data, 14)
        
        assert ValidationHelpers.assert_array_almost_equal(
            rsi1, rsi2, decimals=TEST_CONFIG['precision_decimals']
        )


class TestCalculationAccuracy:
    """Test calculation accuracy across different scenarios"""
    
    def test_precision_with_known_values(self):
        """Test precision with known mathematical values"""
        # Use simple data with known SMA values
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
        # Calculate SMA(3)
        sma_result = calculate_sma(data, 3)
        
        # Known values: [NaN, NaN, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        expected = np.array([np.nan, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        
        assert ValidationHelpers.assert_array_almost_equal(
            sma_result, expected, decimals=TEST_CONFIG['precision_decimals']
        )
        
        # Calculate incremental version
        incremental = IncrementalCalculator.incremental_sma(data, 3)
        
        assert ValidationHelpers.assert_array_almost_equal(
            sma_result, incremental, decimals=TEST_CONFIG['precision_decimals']
        )
    
    def test_numerical_stability(self):
        """Test numerical stability with challenging data"""
        # Test with very large numbers
        large_data = np.array([1e10, 1e10 + 1, 1e10 + 2, 1e10 + 3, 1e10 + 4])
        
        batch_sma = calculate_sma(large_data, 3)
        incremental_sma = IncrementalCalculator.incremental_sma(large_data, 3)
        
        # Should be numerically stable
        assert ValidationHelpers.assert_array_almost_equal(
            batch_sma, incremental_sma, decimals=6  # Allow some tolerance for large numbers
        )
        
        # Test with very small differences
        small_diff_data = np.array([1.0000001, 1.0000002, 1.0000003, 1.0000004, 1.0000005])
        
        batch_sma = calculate_sma(small_diff_data, 3)
        incremental_sma = IncrementalCalculator.incremental_sma(small_diff_data, 3)
        
        assert ValidationHelpers.assert_array_almost_equal(
            batch_sma, incremental_sma, decimals=TEST_CONFIG['precision_decimals']
        )
    
    def test_floating_point_accumulation(self):
        """Test for floating point accumulation errors"""
        # Generate data that might cause accumulation errors
        data = np.array([0.1] * 100)  # Repeating decimal
        
        # Calculate long-period SMA
        batch_sma = calculate_sma(data, 50)
        incremental_sma = IncrementalCalculator.incremental_sma(data, 50)
        
        # Should be very close despite potential floating point issues
        assert ValidationHelpers.assert_array_almost_equal(
            batch_sma, incremental_sma, decimals=10
        )
        
        # All valid values should be very close to 0.1
        valid_values = batch_sma[~np.isnan(batch_sma)]
        assert np.allclose(valid_values, 0.1, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])