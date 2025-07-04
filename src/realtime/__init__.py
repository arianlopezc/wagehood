"""
Real-time market data processing module.

This module provides real-time market data ingestion, processing, and calculation
services using Redis Streams and Celery for high-performance financial analysis.
"""

from .config_manager import ConfigManager
from .data_ingestion import MarketDataIngestionService
from .calculation_engine import CalculationEngine
from .stream_processor import StreamProcessor

__all__ = [
    'ConfigManager',
    'MarketDataIngestionService', 
    'CalculationEngine',
    'StreamProcessor'
]