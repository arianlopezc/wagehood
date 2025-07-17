"""
Technical Indicators Module

This module provides a comprehensive set of technical indicators for trading analysis,
including moving averages, support/resistance levels, and TA-Lib wrapper functions.
"""

from .moving_averages import calculate_sma, calculate_ema, calculate_wma, calculate_vwma
from .levels import (
    calculate_support_resistance,
    calculate_pivot_points,
    calculate_fibonacci_retracements,
)
from .talib_wrapper import (
    calculate_rsi,
    calculate_macd,
    calculate_bb,
)

# Simple IndicatorCalculator class for backward compatibility
class IndicatorCalculator:
    """Basic indicator calculator for backward compatibility."""
    
    def __init__(self, enable_cache: bool = True, cache_ttl: int = 3600):
        # No caching - always use fresh data
        pass

__all__ = [
    # Calculator class
    "IndicatorCalculator",
    # Moving averages
    "calculate_sma",
    "calculate_ema",
    "calculate_wma",
    "calculate_vwma",
    # TA-Lib indicators
    "calculate_rsi",
    "calculate_macd", 
    "calculate_bb",
    # Support/Resistance levels
    "calculate_support_resistance",
    "calculate_pivot_points",
    "calculate_fibonacci_retracements",
]
