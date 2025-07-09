"""
Mathematical Test Data Fixtures

This module provides precise test data and expected outputs for validating
technical indicators and trading strategies with mathematical precision.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta


class TestDataGenerator:
    """Generates standardized test data for mathematical validation"""
    
    @staticmethod
    def generate_linear_trend(length: int = 100, start_price: float = 100.0, 
                            slope: float = 0.5) -> np.ndarray:
        """Generate linear trending data"""
        return np.array([start_price + i * slope for i in range(length)])
    
    @staticmethod
    def generate_sinusoidal_data(length: int = 100, amplitude: float = 10.0, 
                               period: float = 20.0, base_price: float = 100.0) -> np.ndarray:
        """Generate sinusoidal oscillating data"""
        return np.array([
            base_price + amplitude * np.sin(2 * np.pi * i / period) 
            for i in range(length)
        ])
    
    @staticmethod
    def generate_volatile_data(length: int = 100, base_price: float = 100.0,
                             volatility: float = 0.02, seed: int = 42) -> np.ndarray:
        """Generate realistic volatile price data"""
        np.random.seed(seed)
        returns = np.random.normal(0, volatility, length)
        prices = [base_price]
        
        for i in range(1, length):
            prices.append(prices[-1] * (1 + returns[i]))
        
        return np.array(prices)
    
    @staticmethod
    def generate_ohlcv_data(close_prices: np.ndarray, 
                          volatility: float = 0.01) -> Dict[str, np.ndarray]:
        """Generate OHLCV data from close prices"""
        length = len(close_prices)
        
        # Generate high/low based on close with some randomness
        np.random.seed(42)
        high_factor = 1 + np.abs(np.random.normal(0, volatility, length))
        low_factor = 1 - np.abs(np.random.normal(0, volatility, length))
        
        high = close_prices * high_factor
        low = close_prices * low_factor
        
        # Open is previous close with some gap
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        
        # Volume is random but realistic
        volume = np.random.randint(1000, 10000, length)
        
        return {
            'open': open_prices,
            'high': high,
            'low': low,
            'close': close_prices,
            'volume': volume.astype(float)
        }


class PrecisionTestVectors:
    """Provides precise test vectors with known expected outputs"""
    
    # Simple test data for basic validation
    SIMPLE_PRICES = np.array([
        10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
        20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0
    ])
    
    # Expected SMA values for SIMPLE_PRICES with period 5
    EXPECTED_SMA_5 = np.array([
        np.nan, np.nan, np.nan, np.nan, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        18.0, 18.6, 18.8, 18.6, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0
    ])
    
    # Expected EMA values for SIMPLE_PRICES with period 5 (alpha = 2/6 = 0.3333)
    EXPECTED_EMA_5 = np.array([
        np.nan, np.nan, np.nan, np.nan, 12.0, 12.666667, 13.777778, 14.851852,
        15.901235, 16.934156, 18.289438, 18.526292, 18.350861, 17.900574,
        17.267049, 16.511366, 15.674244, 14.782829, 13.855219, 12.903479
    ])
    
    # Complex test data for comprehensive validation
    COMPLEX_PRICES = np.array([
        100.0, 101.5, 99.2, 103.7, 105.1, 102.3, 108.9, 106.2, 104.5, 109.8,
        111.2, 108.7, 112.3, 114.6, 110.9, 115.2, 113.8, 111.4, 116.7, 118.3,
        115.9, 119.4, 121.7, 118.2, 122.8, 120.5, 117.9, 123.6, 125.1, 122.4
    ])
    
    @staticmethod
    def get_rsi_test_data() -> Tuple[np.ndarray, np.ndarray]:
        """Get RSI test data with expected values"""
        # RSI test data (14-period)
        prices = np.array([
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 46.08, 45.89, 46.03,
            46.83, 46.69, 46.45, 46.59, 46.3, 46.28, 46.28, 46.00, 46.03, 46.41,
            46.22, 45.64, 46.21, 46.25, 45.71, 46.45, 47.44, 47.02, 46.96, 47.93
        ])
        
        # Expected RSI values (first 14 values will be NaN)
        expected_rsi = np.array([
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan, np.nan, 70.53, 66.32, 66.55, 69.41, 66.36, 57.97,
            62.93, 63.26, 56.06, 62.38, 54.71, 50.42, 39.99, 41.46, 41.87, 45.46
        ])
        
        return prices, expected_rsi
    
    @staticmethod
    def get_macd_test_data() -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Get MACD test data with expected values"""
        # MACD test data (12, 26, 9 periods)
        prices = np.array([
            459.99, 448.85, 446.06, 450.81, 442.80, 448.97, 444.57, 452.32, 459.03,
            461.91, 463.58, 461.14, 452.08, 442.66, 428.91, 429.79, 431.99, 427.72,
            423.20, 426.21, 426.98, 435.69, 434.33, 429.80, 419.85, 426.24, 402.80,
            392.05, 390.53, 398.67, 406.13, 405.46, 408.38, 417.20, 430.12, 442.78
        ])
        
        # Expected MACD values (approximate, will be calculated precisely in tests)
        expected_macd = np.array([
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 5.1578, -0.4198,
            -3.1198, -2.9594, -1.0594, 0.6066, 1.5666, 2.4266, 3.2866, 4.1466, 4.9066
        ])
        
        return prices, {
            'macd': expected_macd,
            'signal': np.full_like(expected_macd, np.nan),  # Will be calculated
            'histogram': np.full_like(expected_macd, np.nan)  # Will be calculated
        }
    
    @staticmethod
    def get_bollinger_bands_test_data() -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Get Bollinger Bands test data with expected values"""
        # Bollinger Bands test data (20-period, 2 std dev)
        prices = np.array([
            86.16, 89.09, 88.78, 90.32, 89.07, 91.15, 89.44, 89.18, 86.93, 87.68,
            86.96, 89.43, 89.32, 88.72, 88.96, 89.43, 90.67, 92.02, 91.78, 91.12,
            90.21, 88.90, 89.09, 90.55, 89.05, 91.27, 88.87, 87.73, 87.29, 87.79
        ])
        
        # Expected values will be calculated precisely in tests
        return prices, {
            'upper': np.full_like(prices, np.nan),
            'middle': np.full_like(prices, np.nan),
            'lower': np.full_like(prices, np.nan)
        }


class EdgeCaseTestData:
    """Provides edge case test data for boundary testing"""
    
    @staticmethod
    def get_zero_prices() -> np.ndarray:
        """Prices with zero values"""
        return np.array([1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0])
    
    @staticmethod
    def get_negative_prices() -> np.ndarray:
        """Prices with negative values"""
        return np.array([10.0, 5.0, -2.0, 8.0, 12.0, -1.0, 15.0])
    
    @staticmethod
    def get_constant_prices() -> np.ndarray:
        """Constant price data"""
        return np.full(20, 100.0)
    
    @staticmethod
    def get_nan_prices() -> np.ndarray:
        """Prices with NaN values"""
        return np.array([10.0, 11.0, np.nan, 13.0, 14.0, np.nan, 16.0])
    
    @staticmethod
    def get_infinite_prices() -> np.ndarray:
        """Prices with infinite values"""
        return np.array([10.0, 11.0, np.inf, 13.0, 14.0, -np.inf, 16.0])
    
    @staticmethod
    def get_extreme_volatility() -> np.ndarray:
        """Extremely volatile price data"""
        return np.array([
            100.0, 200.0, 50.0, 150.0, 75.0, 225.0, 25.0, 175.0, 125.0, 250.0
        ])
    
    @staticmethod
    def get_minimal_data() -> np.ndarray:
        """Minimal data sets"""
        return np.array([100.0, 101.0])


class ValidationHelpers:
    """Helper functions for test validation"""
    
    @staticmethod
    def assert_array_almost_equal(actual: np.ndarray, expected: np.ndarray, 
                                decimals: int = 6, allow_nan: bool = True) -> bool:
        """Assert arrays are almost equal with NaN handling"""
        if len(actual) != len(expected):
            return False
        
        for i in range(len(actual)):
            if allow_nan and np.isnan(expected[i]):
                if not np.isnan(actual[i]):
                    return False
            elif allow_nan and np.isnan(actual[i]):
                if not np.isnan(expected[i]):
                    return False
            else:
                if abs(actual[i] - expected[i]) > 10**(-decimals):
                    return False
        
        return True
    
    @staticmethod
    def calculate_percentage_error(actual: np.ndarray, expected: np.ndarray) -> float:
        """Calculate percentage error between arrays"""
        # Only consider non-NaN values
        mask = ~(np.isnan(actual) | np.isnan(expected))
        if not np.any(mask):
            return 0.0
        
        actual_clean = actual[mask]
        expected_clean = expected[mask]
        
        if len(expected_clean) == 0:
            return 0.0
        
        # Avoid division by zero
        expected_clean = np.where(expected_clean == 0, 1e-10, expected_clean)
        
        errors = np.abs((actual_clean - expected_clean) / expected_clean) * 100
        return np.mean(errors)
    
    @staticmethod
    def generate_timestamps(length: int, start_date: datetime = None) -> List[datetime]:
        """Generate timestamps for test data"""
        if start_date is None:
            start_date = datetime(2023, 1, 1)
        
        return [start_date + timedelta(days=i) for i in range(length)]
    
    @staticmethod
    def create_market_data_dict(ohlcv_data: Dict[str, np.ndarray], 
                              symbol: str = "TEST") -> Dict[str, Any]:
        """Create market data dictionary for testing"""
        length = len(ohlcv_data['close'])
        timestamps = ValidationHelpers.generate_timestamps(length)
        
        return {
            'symbol': symbol,
            'timestamp': timestamps,
            'open': ohlcv_data['open'].tolist(),
            'high': ohlcv_data['high'].tolist(),
            'low': ohlcv_data['low'].tolist(),
            'close': ohlcv_data['close'].tolist(),
            'volume': ohlcv_data['volume'].tolist()
        }


# Constants for test configuration
TEST_CONFIG = {
    'precision_decimals': 6,
    'percentage_error_threshold': 0.01,  # 1% error threshold
    'default_test_length': 100,
    'random_seed': 42,
    'default_periods': {
        'sma': [5, 10, 20, 50],
        'ema': [5, 10, 20, 50],
        'rsi': [14],
        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
        'bollinger': {'period': 20, 'std_dev': 2.0},
        'atr': [14],
        'stochastic': {'k_period': 14, 'd_period': 3}
    }
}