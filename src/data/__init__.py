"""
Data layer module for the trading system.

This module provides:
- DataStore: In-memory storage for OHLCV data
- MockDataGenerator: Generates realistic market data for testing
- DataProvider: Abstract interface for data sources
- MockProvider: Mock implementation for testing

Example usage:
    from data import DataStore, MockDataGenerator
    
    store = DataStore()
    generator = MockDataGenerator()
    
    # Generate some test data
    data = generator.generate_trending_data(periods=100, trend_strength=0.02)
    store.store_ohlcv("AAPL", TimeFrame.DAILY, data)
"""

from .store import DataStore
from .mock_generator import MockDataGenerator
from .providers.base import DataProvider
from .providers.mock_provider import MockProvider

__all__ = [
    'DataStore',
    'MockDataGenerator', 
    'DataProvider',
    'MockProvider'
]