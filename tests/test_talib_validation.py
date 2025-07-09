"""
TA-Lib Validation Tests

Tests to validate that TA-Lib implementations produce equivalent results to custom implementations.
"""

import pytest
import numpy as np
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock

from src.indicators.talib_wrapper import calculate_rsi, calculate_macd
from src.indicators.momentum import calculate_rsi as custom_rsi, calculate_macd as custom_macd


class TestTalibValidation:
    """Test TA-Lib wrapper functions against custom implementations."""
    
    def test_rsi_comparison(self):
        """Test that TA-Lib RSI matches custom RSI implementation."""
        # Create test data
        test_data = np.array([
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 46.08, 45.89, 46.03,
            46.83, 47.69, 46.49, 46.26, 47.09, 46.66, 46.80, 46.05, 46.22, 45.94,
            46.03, 46.83, 46.99, 47.05, 47.24, 47.69, 47.76, 47.77, 47.72, 47.62
        ])
        
        # Calculate RSI using both implementations
        talib_rsi = calculate_rsi(test_data, 14)
        custom_rsi_result = custom_rsi(test_data, 14)
        
        # Find valid indices (non-NaN values)
        valid_mask = ~np.isnan(talib_rsi) & ~np.isnan(custom_rsi_result)
        
        # Compare valid values with tolerance
        np.testing.assert_allclose(
            talib_rsi[valid_mask], 
            custom_rsi_result[valid_mask], 
            rtol=0.01, 
            atol=0.1,
            err_msg="TA-Lib RSI should match custom RSI implementation"
        )
    
    def test_macd_comparison(self):
        """Test that TA-Lib MACD matches custom MACD implementation."""
        # Create test data
        test_data = np.array([
            12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9,
            13.0, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9,
            14.0, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9,
            15.0, 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9
        ])
        
        # Calculate MACD using both implementations
        talib_macd, talib_signal, talib_hist = calculate_macd(test_data, 12, 26, 9)
        custom_macd_result, custom_signal_result, custom_hist_result = custom_macd(test_data, 12, 26, 9)
        
        # Find valid indices (non-NaN values)
        valid_mask = (~np.isnan(talib_macd) & ~np.isnan(custom_macd_result) & 
                     ~np.isnan(talib_signal) & ~np.isnan(custom_signal_result))
        
        # Compare MACD line
        np.testing.assert_allclose(
            talib_macd[valid_mask], 
            custom_macd_result[valid_mask], 
            rtol=0.01, 
            atol=0.01,
            err_msg="TA-Lib MACD line should match custom MACD implementation"
        )
        
        # Compare signal line
        np.testing.assert_allclose(
            talib_signal[valid_mask], 
            custom_signal_result[valid_mask], 
            rtol=0.01, 
            atol=0.01,
            err_msg="TA-Lib signal line should match custom signal implementation"
        )
        
        # Compare histogram
        np.testing.assert_allclose(
            talib_hist[valid_mask], 
            custom_hist_result[valid_mask], 
            rtol=0.01, 
            atol=0.01,
            err_msg="TA-Lib histogram should match custom histogram implementation"
        )
    
    def test_rsi_edge_cases(self):
        """Test RSI edge cases for both implementations."""
        # Test with all increasing prices
        increasing_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        talib_rsi = calculate_rsi(increasing_data, 14)
        custom_rsi_result = custom_rsi(increasing_data, 14)
        
        # Both should handle this case without error
        assert not np.all(np.isnan(talib_rsi))
        assert not np.all(np.isnan(custom_rsi_result))
        
        # Test with all decreasing prices
        decreasing_data = np.array([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        talib_rsi = calculate_rsi(decreasing_data, 14)
        custom_rsi_result = custom_rsi(decreasing_data, 14)
        
        # Both should handle this case without error
        assert not np.all(np.isnan(talib_rsi))
        assert not np.all(np.isnan(custom_rsi_result))
        
        # Test with constant prices
        constant_data = np.array([10] * 20)
        talib_rsi = calculate_rsi(constant_data, 14)
        custom_rsi_result = custom_rsi(constant_data, 14)
        
        # Both should handle this case without error
        assert not np.all(np.isnan(talib_rsi))
        assert not np.all(np.isnan(custom_rsi_result))
    
    def test_rsi_strategy_compatibility(self):
        """Test that RSI Trend strategy works with TA-Lib."""
        from src.strategies.rsi_trend import RSITrendFollowing
        
        # Create mock market data
        test_data = {
            'data': [
                {'timestamp': f'2023-01-{i:02d}T10:00:00Z', 'open': 100 + i, 'high': 102 + i, 
                 'low': 98 + i, 'close': 100 + i, 'volume': 1000} 
                for i in range(1, 31)
            ],
            'symbol': 'TEST',
            'last_updated': '2023-01-30T10:00:00Z'
        }
        
        # Create strategy instance
        strategy = RSITrendFollowing()
        
        # Test that strategy can calculate indicators
        indicators = strategy._calculate_indicators(test_data)
        
        # Verify RSI is calculated
        assert 'rsi' in indicators
        assert 'rsi_14' in indicators['rsi']
        assert 'rsi_21' in indicators['rsi']
        
        # Verify RSI values are reasonable
        rsi_14 = indicators['rsi']['rsi_14']
        assert isinstance(rsi_14, np.ndarray)
        assert len(rsi_14) == 30
        
        # Check that some values are not NaN
        valid_values = ~np.isnan(rsi_14)
        assert np.sum(valid_values) > 0
        
        # Check RSI range
        valid_rsi = rsi_14[valid_values]
        assert np.all(valid_rsi >= 0)
        assert np.all(valid_rsi <= 100)
    
    def test_signal_generation_compatibility(self):
        """Test that signal generation works with TA-Lib implementation."""
        from src.strategies.rsi_trend import RSITrendFollowing
        
        # Create realistic test data that should generate signals
        close_prices = [
            100, 102, 104, 106, 108, 110, 112, 114, 116, 118,  # Uptrend
            116, 114, 112, 110, 108, 106, 104, 102, 100, 98,   # Downtrend
            100, 102, 104, 106, 108, 110, 112, 114, 116, 118   # Uptrend again
        ]
        
        test_data = {
            'data': [
                {'timestamp': f'2023-01-{i:02d}T10:00:00Z', 'open': price, 'high': price + 2, 
                 'low': price - 2, 'close': price, 'volume': 1000} 
                for i, price in enumerate(close_prices, 1)
            ],
            'symbol': 'TEST',
            'last_updated': '2023-01-30T10:00:00Z'
        }
        
        # Mock data.to_arrays() method
        class MockData:
            def __init__(self, data):
                self.data = data
                
            def to_arrays(self):
                return {
                    'timestamp': [item['timestamp'] for item in self.data['data']],
                    'close': [item['close'] for item in self.data['data']],
                    'open': [item['open'] for item in self.data['data']],
                    'high': [item['high'] for item in self.data['data']],
                    'low': [item['low'] for item in self.data['data']],
                    'volume': [item['volume'] for item in self.data['data']]
                }
            
            def get(self, key, default=None):
                return self.data.get(key, default)
        
        # Create strategy instance
        strategy = RSITrendFollowing()
        
        # Create mock data object
        mock_data = MockData(test_data)
        
        # Calculate indicators
        indicators = strategy._calculate_indicators(test_data)
        
        # Test signal generation
        signals = strategy.generate_signals(mock_data, indicators)
        
        # Verify signals are generated (should be list, may be empty)
        assert isinstance(signals, list)
        
        # If signals are generated, verify they have correct structure
        for signal in signals:
            assert 'timestamp' in signal
            assert 'symbol' in signal
            assert 'signal_type' in signal
            assert 'price' in signal
            assert 'confidence' in signal
            assert 'strategy_name' in signal
            assert 'metadata' in signal
            
            # Verify signal type
            assert signal['signal_type'] in ['BUY', 'SELL']
            
            # Verify confidence range
            assert 0 <= signal['confidence'] <= 1
            
            # Verify strategy name
            assert signal['strategy_name'] == 'RSITrendFollowing'
    
    def test_performance_comparison(self):
        """Test performance comparison between custom and TA-Lib implementations."""
        import time
        
        # Create larger test dataset
        test_data = np.random.rand(1000) * 100
        
        # Time custom RSI
        start_time = time.time()
        custom_result = custom_rsi(test_data, 14)
        custom_time = time.time() - start_time
        
        # Time TA-Lib RSI
        start_time = time.time()
        talib_result = calculate_rsi(test_data, 14)
        talib_time = time.time() - start_time
        
        # Log performance results
        print(f"Custom RSI time: {custom_time:.4f}s")
        print(f"TA-Lib RSI time: {talib_time:.4f}s")
        print(f"TA-Lib speedup: {custom_time / talib_time:.2f}x")
        
        # Verify results are still equivalent
        valid_mask = ~np.isnan(talib_result) & ~np.isnan(custom_result)
        np.testing.assert_allclose(
            talib_result[valid_mask], 
            custom_result[valid_mask], 
            rtol=0.01, 
            atol=0.1
        )
        
        # Assert TA-Lib is at least as fast (allowing for some variance)
        assert talib_time <= custom_time * 1.5, "TA-Lib should be competitive in performance"