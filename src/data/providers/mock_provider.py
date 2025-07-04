"""
Mock data provider implementation for testing and development.

This module provides a MockProvider class that generates realistic
market data for testing strategies and backtesting without requiring
external data sources.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from .base import DataProvider, DataProviderError
from ..mock_generator import MockDataGenerator
from ...core.models import OHLCV, TimeFrame, MarketData


class MockProvider(DataProvider):
    """
    Mock data provider for testing and development.
    
    This provider generates realistic market data using the MockDataGenerator
    and provides it through the standard DataProvider interface. It's useful
    for testing strategies, backtesting, and development without requiring
    external data sources.
    
    Features:
    - Generates realistic OHLCV data with proper relationships
    - Supports all standard timeframes
    - Configurable market scenarios
    - Consistent data generation with optional seeding
    - Simulates realistic delays and error conditions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the mock data provider.
        
        Args:
            config: Configuration dictionary with options:
                - seed: Random seed for reproducible data
                - symbols: List of symbols to support
                - delay_ms: Simulated delay in milliseconds
                - error_rate: Probability of simulated errors (0.0 to 1.0)
                - default_periods: Default number of periods to generate
                - market_scenario: Default market scenario to use
        """
        super().__init__("MockProvider", config)
        
        # Configuration
        self.seed = self.config.get('seed', None)
        self.delay_ms = self.config.get('delay_ms', 0)
        self.error_rate = self.config.get('error_rate', 0.0)
        self.default_periods = self.config.get('default_periods', 1000)
        self.market_scenario = self.config.get('market_scenario', 'normal')
        
        # Supported symbols
        self.symbols = self.config.get('symbols', [
            'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX',
            'SPY', 'QQQ', 'VTI', 'IWM',
            'BTC', 'ETH', 'LTC', 'XRP',
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD',
            'GOLD', 'SILVER', 'OIL', 'NATGAS'
        ])
        
        # All timeframes supported
        self.timeframes = list(TimeFrame)
        
        # Initialize generator
        self.generator = MockDataGenerator(seed=self.seed)
        
        # Cache for generated data
        self._data_cache: Dict[str, Dict[TimeFrame, List[OHLCV]]] = {}
        
        # Market scenarios
        self.scenarios = self.generator.create_market_scenarios()
    
    async def connect(self) -> bool:
        """
        Establish connection to the mock data source.
        
        Returns:
            True if connection successful
        """
        try:
            await self._simulate_delay()
            await self._simulate_error("Connection failed")
            
            self._connected = True
            self._clear_error()
            return True
            
        except Exception as e:
            self._set_error(str(e))
            return False
    
    async def disconnect(self) -> None:
        """
        Close connection to the mock data source.
        """
        await self._simulate_delay()
        self._connected = False
        self._clear_error()
    
    async def get_historical_data(self, symbol: str, timeframe: TimeFrame,
                                 start_date: datetime, end_date: datetime,
                                 limit: Optional[int] = None) -> List[OHLCV]:
        """
        Retrieve historical OHLCV data for a symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            limit: Maximum number of records to return
            
        Returns:
            List of OHLCV data points
            
        Raises:
            ConnectionError: If not connected
            ValueError: If parameters are invalid
            DataProviderError: If data generation fails
        """
        if not self._connected:
            raise DataProviderError("Not connected to data source")
        
        await self._simulate_delay()
        await self._simulate_error("Failed to retrieve historical data")
        
        # Validate parameters
        if not self.validate_symbol(symbol):
            raise ValueError(f"Unsupported symbol: {symbol}")
        
        if not self.validate_timeframe(timeframe):
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        if not self.validate_date_range(start_date, end_date):
            raise ValueError("Invalid date range")
        
        # Calculate number of periods needed
        periods = self._calculate_periods(start_date, end_date, timeframe)
        
        if limit is not None:
            periods = min(periods, limit)
        
        # Generate data
        cache_key = f"{symbol}_{timeframe.value}_{start_date.isoformat()}_{end_date.isoformat()}"
        
        if cache_key not in self._data_cache:
            # Generate realistic data based on market scenario
            if self.market_scenario in self.scenarios:
                scenario = self.scenarios[self.market_scenario]
                data = self.generator.generate_realistic_data(
                    symbol=symbol,
                    periods=periods,
                    start_date=start_date,
                    timeframe=timeframe
                )
            else:
                # Default generation
                data = self.generator.generate_trending_data(
                    periods=periods,
                    start_date=start_date,
                    timeframe=timeframe
                )
            
            # Add volume profile
            data = self.generator.add_volume_profile(data, 'normal')
            
            # Cache the data
            if symbol not in self._data_cache:
                self._data_cache[symbol] = {}
            self._data_cache[symbol][timeframe] = data
        
        data = self._data_cache[symbol][timeframe]
        
        # Filter by date range
        filtered_data = [
            d for d in data
            if start_date <= d.timestamp <= end_date
        ]
        
        return filtered_data[:limit] if limit else filtered_data
    
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
        """
        if not self._connected:
            raise DataProviderError("Not connected to data source")
        
        await self._simulate_delay()
        await self._simulate_error("Failed to retrieve latest data")
        
        # Generate data ending at current time
        end_date = datetime.now()
        start_date = end_date - self._get_timeframe_delta(timeframe) * periods * 2
        
        data = await self.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            limit=periods
        )
        
        return data[-periods:] if len(data) >= periods else data
    
    async def get_symbols(self) -> List[str]:
        """
        Get list of available symbols from the mock provider.
        
        Returns:
            List of available trading symbols
        """
        if not self._connected:
            raise DataProviderError("Not connected to data source")
        
        await self._simulate_delay()
        await self._simulate_error("Failed to retrieve symbols")
        
        return self.symbols.copy()
    
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
        """
        if not self._connected:
            raise DataProviderError("Not connected to data source")
        
        # Use default date range if not specified
        if end_date is None:
            end_date = datetime.now()
        
        if start_date is None:
            start_date = end_date - timedelta(days=365)  # Default to 1 year
        
        # Get OHLCV data
        ohlcv_data = await self.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Create MarketData object
        market_data = MarketData(
            symbol=symbol,
            timeframe=timeframe,
            data=ohlcv_data,
            indicators={},  # Empty indicators - can be populated by analysis modules
            last_updated=datetime.now()
        )
        
        return market_data
    
    def get_supported_timeframes(self) -> List[TimeFrame]:
        """
        Get list of supported timeframes.
        
        Returns:
            List of supported TimeFrame values
        """
        return self.timeframes.copy()
    
    def get_supported_symbols(self) -> List[str]:
        """
        Get list of supported symbols.
        
        Returns:
            List of supported trading symbols
        """
        return self.symbols.copy()
    
    def set_market_scenario(self, scenario_name: str) -> None:
        """
        Set the market scenario for data generation.
        
        Args:
            scenario_name: Name of the market scenario
            
        Raises:
            ValueError: If scenario name is not supported
        """
        if scenario_name not in self.scenarios:
            available_scenarios = list(self.scenarios.keys())
            raise ValueError(f"Unsupported scenario: {scenario_name}. Available: {available_scenarios}")
        
        self.market_scenario = scenario_name
        
        # Clear cache to force regeneration with new scenario
        self._data_cache.clear()
    
    def get_available_scenarios(self) -> List[str]:
        """
        Get list of available market scenarios.
        
        Returns:
            List of scenario names
        """
        return list(self.scenarios.keys())
    
    def get_scenario_info(self, scenario_name: str) -> Dict[str, Any]:
        """
        Get information about a market scenario.
        
        Args:
            scenario_name: Name of the scenario
            
        Returns:
            Dictionary with scenario information
            
        Raises:
            ValueError: If scenario name is not supported
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unsupported scenario: {scenario_name}")
        
        scenario = self.scenarios[scenario_name]
        return {
            'name': scenario.name,
            'periods': scenario.periods,
            'trend_strength': scenario.trend_strength,
            'volatility': scenario.volatility,
            'volume_profile': scenario.volume_profile,
            'noise_level': scenario.noise_level,
            'description': scenario.description
        }
    
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._data_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_symbols = len(self._data_cache)
        total_timeframes = sum(len(tf_data) for tf_data in self._data_cache.values())
        total_data_points = sum(
            len(data) for symbol_data in self._data_cache.values()
            for data in symbol_data.values()
        )
        
        return {
            'cached_symbols': total_symbols,
            'cached_timeframes': total_timeframes,
            'total_data_points': total_data_points,
            'cache_size_mb': total_data_points * 0.0001  # Rough estimate
        }
    
    async def _simulate_delay(self) -> None:
        """Simulate network delay if configured."""
        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000)
    
    async def _simulate_error(self, error_message: str) -> None:
        """Simulate random errors if configured."""
        if self.error_rate > 0:
            import random
            if random.random() < self.error_rate:
                raise DataProviderError(error_message)
    
    def _calculate_periods(self, start_date: datetime, end_date: datetime, 
                          timeframe: TimeFrame) -> int:
        """Calculate number of periods between two dates for a given timeframe."""
        delta = end_date - start_date
        timeframe_delta = self._get_timeframe_delta(timeframe)
        
        return max(1, int(delta.total_seconds() / timeframe_delta.total_seconds()))
    
    def _get_timeframe_delta(self, timeframe: TimeFrame) -> timedelta:
        """Get timedelta for a given timeframe."""
        if timeframe == TimeFrame.MINUTE_1:
            return timedelta(minutes=1)
        elif timeframe == TimeFrame.MINUTE_5:
            return timedelta(minutes=5)
        elif timeframe == TimeFrame.MINUTE_15:
            return timedelta(minutes=15)
        elif timeframe == TimeFrame.MINUTE_30:
            return timedelta(minutes=30)
        elif timeframe == TimeFrame.HOUR_1:
            return timedelta(hours=1)
        elif timeframe == TimeFrame.HOUR_4:
            return timedelta(hours=4)
        elif timeframe == TimeFrame.DAILY:
            return timedelta(days=1)
        elif timeframe == TimeFrame.WEEKLY:
            return timedelta(weeks=1)
        elif timeframe == TimeFrame.MONTHLY:
            return timedelta(days=30)  # Approximate
        else:
            return timedelta(days=1)  # Default
    
    def __str__(self) -> str:
        """String representation of the mock provider."""
        return f"MockProvider(scenario={self.market_scenario}, symbols={len(self.symbols)})"