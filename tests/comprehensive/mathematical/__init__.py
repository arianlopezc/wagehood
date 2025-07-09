"""
Mathematical validation tests for the Wagehood trading system.

This module contains tests that validate the mathematical correctness of
trading strategies, technical indicators, and calculation engines.
"""

# Mathematical validation constants
MATHEMATICAL_TOLERANCE = 1e-8
PERCENTAGE_TOLERANCE = 0.01  # 1% tolerance for percentage calculations
INDICATOR_TOLERANCE = 1e-6  # Stricter tolerance for indicators

# Test data parameters
TEST_DATA_POINTS = 1000
TEST_PERIODS = [14, 20, 50, 200]  # Common indicator periods
TEST_SYMBOLS = ["SPY", "QQQ", "AAPL"]  # Test symbols

# Expected ranges for indicators
INDICATOR_RANGES = {
    "RSI": (0, 100),
    "MACD": (-10, 10),  # Approximate range
    "BB_UPPER": (0, float("inf")),
    "BB_LOWER": (0, float("inf")),
    "MA": (0, float("inf")),
    "STOCH": (0, 100),
}

# Performance thresholds for mathematical tests
PERFORMANCE_THRESHOLDS = {
    "max_execution_time": 5.0,  # seconds
    "max_memory_mb": 100,  # MB
    "max_cpu_percent": 80,  # %
}
