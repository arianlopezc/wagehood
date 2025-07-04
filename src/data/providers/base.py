"""
Abstract base class for data providers.

This module defines the interface that all data providers must implement
to ensure consistent access to market data across different sources.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict, Any

from ...core.models import OHLCV, TimeFrame, MarketData


class DataProvider(ABC):
    """
    Abstract base class for market data providers.
    
    This class defines the interface that all data providers must implement
    to ensure consistent access to market data from different sources such as
    APIs, databases, files, or mock data generators.
    
    Implementations should handle:
    - Connection management
    - Rate limiting
    - Error handling and retries
    - Data validation
    - Caching strategies
    """
    
    def __init__(self, provider_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data provider.
        
        Args:
            provider_name: Name of the data provider
            config: Optional configuration dictionary
        """
        self.provider_name = provider_name
        self.config = config or {}
        self._connected = False
        self._last_error = None
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the data source.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close connection to the data source.
        """
        pass
    
    @abstractmethod
    async def get_historical_data(self, symbol: str, timeframe: TimeFrame,
                                 start_date: datetime, end_date: datetime,
                                 limit: Optional[int] = None) -> List[OHLCV]:
        """
        Retrieve historical OHLCV data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'EURUSD')
            timeframe: Data timeframe
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            limit: Maximum number of records to return
            
        Returns:
            List of OHLCV data points
            
        Raises:
            ConnectionError: If not connected to data source
            ValueError: If parameters are invalid
            DataProviderError: If data retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_latest_data(self, symbol: str, timeframe: TimeFrame,
                            periods: int = 1) -> List[OHLCV]:
        """
        Get the latest N periods of data for a symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            periods: Number of latest periods to retrieve
            
        Returns:
            List of latest OHLCV data points
            
        Raises:
            ConnectionError: If not connected to data source
            ValueError: If parameters are invalid
            DataProviderError: If data retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_symbols(self) -> List[str]:
        """
        Get list of available symbols from the data provider.
        
        Returns:
            List of available trading symbols
            
        Raises:
            ConnectionError: If not connected to data source
            DataProviderError: If symbol retrieval fails
        """
        pass
    
    @abstractmethod
    async def get_market_data(self, symbol: str, timeframe: TimeFrame,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> MarketData:
        """
        Get complete market data object for a symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_date: Start date for data (optional)
            end_date: End date for data (optional)
            
        Returns:
            MarketData object with OHLCV data and metadata
            
        Raises:
            ConnectionError: If not connected to data source
            ValueError: If parameters are invalid
            DataProviderError: If data retrieval fails
        """
        pass
    
    def is_connected(self) -> bool:
        """
        Check if provider is connected to data source.
        
        Returns:
            True if connected, False otherwise
        """
        return self._connected
    
    def get_last_error(self) -> Optional[str]:
        """
        Get the last error message.
        
        Returns:
            Last error message or None if no errors
        """
        return self._last_error
    
    def _set_error(self, error_message: str) -> None:
        """
        Set the last error message.
        
        Args:
            error_message: Error message to store
        """
        self._last_error = error_message
    
    def _clear_error(self) -> None:
        """Clear the last error message."""
        self._last_error = None
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the data provider.
        
        Returns:
            Dictionary with provider information
        """
        return {
            'name': self.provider_name,
            'connected': self._connected,
            'last_error': self._last_error,
            'config': self.config
        }
    
    @abstractmethod
    def get_supported_timeframes(self) -> List[TimeFrame]:
        """
        Get list of supported timeframes.
        
        Returns:
            List of supported TimeFrame values
        """
        pass
    
    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """
        Get list of supported symbols.
        
        Returns:
            List of supported trading symbols
        """
        pass
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol is supported by this provider.
        
        Args:
            symbol: Trading symbol to validate
            
        Returns:
            True if symbol is supported, False otherwise
        """
        try:
            supported_symbols = self.get_supported_symbols()
            return symbol.upper() in [s.upper() for s in supported_symbols]
        except Exception:
            return False
    
    def validate_timeframe(self, timeframe: TimeFrame) -> bool:
        """
        Validate if a timeframe is supported by this provider.
        
        Args:
            timeframe: TimeFrame to validate
            
        Returns:
            True if timeframe is supported, False otherwise
        """
        try:
            supported_timeframes = self.get_supported_timeframes()
            return timeframe in supported_timeframes
        except Exception:
            return False
    
    def validate_date_range(self, start_date: datetime, end_date: datetime) -> bool:
        """
        Validate if a date range is reasonable.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            True if date range is valid, False otherwise
        """
        if start_date >= end_date:
            return False
        
        # Check if date range is not too large (e.g., more than 10 years)
        max_days = 365 * 10  # 10 years
        if (end_date - start_date).days > max_days:
            return False
        
        # Check if end date is not in the future
        if end_date > datetime.now():
            return False
        
        return True
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    def __str__(self) -> str:
        """String representation of the provider."""
        return f"{self.provider_name}(connected={self._connected})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the provider."""
        return f"{self.__class__.__name__}(name='{self.provider_name}', connected={self._connected})"


class DataProviderError(Exception):
    """Base exception for data provider errors."""
    pass


class ConnectionError(DataProviderError):
    """Exception raised when connection to data source fails."""
    pass


class DataRetrievalError(DataProviderError):
    """Exception raised when data retrieval fails."""
    pass


class ValidationError(DataProviderError):
    """Exception raised when data validation fails."""
    pass