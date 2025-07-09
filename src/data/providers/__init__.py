"""
Data providers module for the trading system.

This module provides interfaces and implementations for different
data sources including real-time feeds, historical data, and mock data.
"""

from .base import DataProvider
from .alpaca_provider import AlpacaProvider

__all__ = ['DataProvider', 'AlpacaProvider']