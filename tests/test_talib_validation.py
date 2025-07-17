"""
TA-Lib Functionality Tests

Tests to validate that TA-Lib implementations work correctly and produce expected results.
"""

import pytest
import numpy as np
from typing import List, Dict, Any

from src.indicators.talib_wrapper import calculate_rsi, calculate_macd


class TestTalibValidation:
    """Test TA-Lib wrapper functions for correctness and expected behavior."""
    
    def test_rsi_functionality(self):
        """Test that TA-Lib RSI produces valid results."""
        # Create test data
        test_data = np.array([
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 46.08, 45.89, 46.03,
            46.83, 47.69, 46.49, 46.26, 47.09, 46.66, 46.80, 46.05, 46.22, 45.94,
            46.03, 46.83, 46.99, 47.05, 47.24, 47.69, 47.76, 47.77, 47.72, 47.62
        ])
        
        # Calculate RSI using TA-Lib
        rsi_result = calculate_rsi(test_data, 14)
        
        # Validate RSI properties
        assert isinstance(rsi_result, np.ndarray)
        assert len(rsi_result) == len(test_data)
        
        # Check valid RSI values are in range [0, 100]
        valid_values = rsi_result[~np.isnan(rsi_result)]
        assert len(valid_values) > 0, "Should have some valid RSI values"
        assert np.all(valid_values >= 0), "RSI values should be >= 0"
        assert np.all(valid_values <= 100), "RSI values should be <= 100"
        
        # First few values should be NaN (warming up period)
        assert np.isnan(rsi_result[0]), "First RSI value should be NaN"
    
    def test_macd_functionality(self):
        """Test that TA-Lib MACD produces valid results."""
        # Create test data with trending pattern
        test_data = np.array([
            12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9,
            13.0, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9,
            14.0, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9,
            15.0, 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9
        ])
        
        # Calculate MACD using TA-Lib
        macd_line, signal_line, histogram = calculate_macd(test_data, 12, 26, 9)
        
        # Validate MACD output structure
        assert isinstance(macd_line, np.ndarray)
        assert isinstance(signal_line, np.ndarray)
        assert isinstance(histogram, np.ndarray)
        assert len(macd_line) == len(test_data)
        assert len(signal_line) == len(test_data)
        assert len(histogram) == len(test_data)
        
        # Check that histogram equals macd_line - signal_line (where both are valid)
        valid_mask = ~np.isnan(macd_line) & ~np.isnan(signal_line) & ~np.isnan(histogram)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 0:
            for i in valid_indices:
                expected_hist = macd_line[i] - signal_line[i]
                assert abs(histogram[i] - expected_hist) < 1e-10, f"Histogram should equal MACD - Signal at index {i}"
    
    def test_rsi_edge_cases(self):
        """Test RSI edge cases with TA-Lib implementation."""
        # Test with all increasing prices
        increasing_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        rsi_result = calculate_rsi(increasing_data, 14)
        
        # Should handle this case without error and produce valid values
        assert not np.all(np.isnan(rsi_result))
        valid_values = rsi_result[~np.isnan(rsi_result)]
        if len(valid_values) > 0:
            # Should tend toward high RSI for increasing prices
            assert np.max(valid_values) > 50, "Increasing prices should produce high RSI"
        
        # Test with all decreasing prices
        decreasing_data = np.array([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        rsi_result = calculate_rsi(decreasing_data, 14)
        
        # Should handle this case without error
        assert not np.all(np.isnan(rsi_result))
        valid_values = rsi_result[~np.isnan(rsi_result)]
        if len(valid_values) > 0:
            # Should tend toward low RSI for decreasing prices
            assert np.min(valid_values) < 50, "Decreasing prices should produce low RSI"
        
        # Test with constant prices
        constant_data = np.array([10] * 20)
        rsi_result = calculate_rsi(constant_data, 14)
        
        # Should handle this case without error
        assert not np.all(np.isnan(rsi_result))
        valid_values = rsi_result[~np.isnan(rsi_result)]
        if len(valid_values) > 0:
            # TA-Lib produces RSI of 0 for constant prices (no gains or losses)
            # This is the correct mathematical behavior
            assert np.allclose(valid_values, 0, atol=1), "Constant prices should produce RSI of 0 (no movement)"
    
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
    
    def test_talib_performance(self):
        """Test TA-Lib RSI performance with larger dataset."""
        import time
        
        # Create larger test dataset
        test_data = np.random.rand(1000) * 100
        
        # Time TA-Lib RSI
        start_time = time.time()
        talib_result = calculate_rsi(test_data, 14)
        talib_time = time.time() - start_time
        
        # Log performance results
        print(f"TA-Lib RSI time: {talib_time:.4f}s")
        
        # Verify results are valid
        valid_values = talib_result[~np.isnan(talib_result)]
        assert len(valid_values) > 0, "Should have some valid RSI values"
        assert np.all(valid_values >= 0), "RSI values should be >= 0"
        assert np.all(valid_values <= 100), "RSI values should be <= 100"
        
        # Performance should be reasonable (less than 1 second for 1000 points)
        assert talib_time < 1.0, "TA-Lib RSI should be fast"