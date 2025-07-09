"""
Data providers module for the trading system.

This module provides interfaces and implementations for different
data sources including real-time feeds, historical data, and mock data.
"""

from .base import DataProvider
from .mock_provider import MockProvider

try:
    # from .alpaca_provider import AlpacaProvider  # Import when needed
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# Export providers based on availability
if ALPACA_AVAILABLE:
    __all__ = ['DataProvider', 'MockProvider', 'AlpacaProvider']
else:
    __all__ = ['DataProvider', 'MockProvider']