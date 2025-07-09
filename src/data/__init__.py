"""
Data layer module for the trading system.

This module provides:
- DataProvider: Abstract interface for data sources
- AlpacaProvider: Alpaca Markets data provider
"""

from .providers.base import DataProvider
from .providers.alpaca_provider import AlpacaProvider

__all__ = [
    'DataProvider',
    'AlpacaProvider'
]