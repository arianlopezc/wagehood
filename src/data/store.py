"""
In-memory data store for OHLCV data.

This module provides the DataStore class for storing and retrieving
market data in memory with fast lookups and efficient filtering.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging

# Optional numpy import for performance
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from ..core.models import OHLCV, TimeFrame, MarketData
from ..storage.cache import cache_manager, cached

logger = logging.getLogger(__name__)


class DataStore:
    """
    High-performance in-memory storage for OHLCV market data.
    
    This class provides fast storage and retrieval of time-series market data
    with support for multiple symbols and timeframes. Data is stored in memory
    for maximum performance during backtesting and analysis.
    
    Attributes:
        _data: Internal storage dictionary organized by symbol and timeframe
        _symbols: Set of all symbols currently stored
        _last_updated: Dictionary tracking last update time for each symbol/timeframe
    """
    
    def __init__(self, enable_cache: bool = True, cache_ttl: int = 3600):
        """Initialize empty data store."""
        # Structure: {symbol: {timeframe: List[OHLCV]}}
        self._data: Dict[str, Dict[TimeFrame, List[OHLCV]]] = defaultdict(lambda: defaultdict(list))
        self._symbols: set = set()
        self._last_updated: Dict[Tuple[str, TimeFrame], datetime] = {}
        self._enable_cache = enable_cache
        self._cache_ttl = cache_ttl
        
        # Cache TTL settings for different data types
        self._cache_ttl_settings = {
            'market_data': 3600,  # 1 hour for market data
            'latest_data': 300,   # 5 minutes for latest data
            'data_range': 3600,   # 1 hour for data range
            'statistics': 1800,   # 30 minutes for statistics
        }
    
    def store_ohlcv(self, symbol: str, timeframe: TimeFrame, data: List[OHLCV]) -> None:
        """
        Store OHLCV data for a symbol and timeframe.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'EURUSD')
            timeframe: Data timeframe
            data: List of OHLCV data points
            
        Raises:
            ValueError: If data is empty or contains invalid entries
        """
        if not data:
            raise ValueError("Data cannot be empty")
        
        if not isinstance(data, list):
            raise ValueError("Data must be a list of OHLCV objects")
        
        # Validate all entries are OHLCV objects
        for entry in data:
            if not isinstance(entry, OHLCV):
                raise ValueError("All data entries must be OHLCV objects")
        
        # Sort data by timestamp to ensure chronological order
        sorted_data = sorted(data, key=lambda x: x.timestamp)
        
        # Store the data
        self._data[symbol][timeframe] = sorted_data
        self._symbols.add(symbol)
        self._last_updated[(symbol, timeframe)] = datetime.now()
        
        # Invalidate cache entries for this symbol and timeframe
        if self._enable_cache:
            self._invalidate_cache(symbol, timeframe)
    
    def get_ohlcv(self, symbol: str, timeframe: TimeFrame, 
                  start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None) -> List[OHLCV]:
        """
        Retrieve OHLCV data for a symbol and timeframe within date range.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_date: Start date for filtering (inclusive). If None, returns from beginning
            end_date: End date for filtering (inclusive). If None, returns until end
            
        Returns:
            List of OHLCV data points within the specified date range
            
        Raises:
            KeyError: If symbol or timeframe not found in store
        """
        # Try to get from cache first
        if self._enable_cache and start_date is None and end_date is None:
            cache_key = cache_manager.cache_key_hash(symbol, timeframe.value)
            cached_data = cache_manager.get("market_data", cache_key)
            if cached_data is not None:
                return cached_data
        
        if symbol not in self._data:
            raise KeyError(f"Symbol '{symbol}' not found in data store")
        
        if timeframe not in self._data[symbol]:
            raise KeyError(f"Timeframe '{timeframe.value}' not found for symbol '{symbol}'")
        
        data = self._data[symbol][timeframe]
        
        # Apply date filtering if specified
        if start_date is not None or end_date is not None:
            filtered_data = []
            for entry in data:
                if start_date is not None and entry.timestamp < start_date:
                    continue
                if end_date is not None and entry.timestamp > end_date:
                    continue
                filtered_data.append(entry)
            return filtered_data
        
        result = data.copy()  # Return a copy to prevent external modification
        
        # Cache the result if no date filtering was applied
        if self._enable_cache and start_date is None and end_date is None:
            cache_key = cache_manager.cache_key_hash(symbol, timeframe.value)
            cache_manager.set("market_data", cache_key, result, self._cache_ttl_settings['market_data'])
        
        return result
    
    def get_latest_data(self, symbol: str, timeframe: TimeFrame, periods: int) -> List[OHLCV]:
        """
        Get the latest N periods of data for a symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            periods: Number of latest periods to retrieve
            
        Returns:
            List of latest OHLCV data points (up to periods count)
            
        Raises:
            KeyError: If symbol or timeframe not found in store
            ValueError: If periods is not positive
        """
        if periods <= 0:
            raise ValueError("Periods must be positive")
        
        # Try to get from cache first
        if self._enable_cache:
            cache_key = cache_manager.cache_key_hash(symbol, timeframe.value, periods, "latest")
            cached_data = cache_manager.get("market_data", cache_key)
            if cached_data is not None:
                return cached_data
        
        data = self.get_ohlcv(symbol, timeframe)
        
        # Return the last N periods
        result = data[-periods:] if len(data) >= periods else data
        
        # Cache the result
        if self._enable_cache:
            cache_key = cache_manager.cache_key_hash(symbol, timeframe.value, periods, "latest")
            cache_manager.set("market_data", cache_key, result, self._cache_ttl_settings['latest_data'])
        
        return result
    
    def list_symbols(self) -> List[str]:
        """
        Get list of all symbols stored in the data store.
        
        Returns:
            Sorted list of symbol names
        """
        return sorted(list(self._symbols))
    
    def list_timeframes(self, symbol: str) -> List[TimeFrame]:
        """
        Get list of available timeframes for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of available timeframes for the symbol
            
        Raises:
            KeyError: If symbol not found in store
        """
        if symbol not in self._data:
            raise KeyError(f"Symbol '{symbol}' not found in data store")
        
        return list(self._data[symbol].keys())
    
    def get_data_range(self, symbol: str, timeframe: TimeFrame) -> Tuple[datetime, datetime]:
        """
        Get the date range of available data for a symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            Tuple of (start_date, end_date)
            
        Raises:
            KeyError: If symbol or timeframe not found in store
        """
        # Try to get from cache first
        if self._enable_cache:
            cache_key = cache_manager.cache_key_hash(symbol, timeframe.value, "data_range")
            cached_range = cache_manager.get("market_data", cache_key)
            if cached_range is not None:
                return cached_range
        
        data = self.get_ohlcv(symbol, timeframe)
        
        if not data:
            raise KeyError(f"No data found for {symbol} {timeframe.value}")
        
        result = (data[0].timestamp, data[-1].timestamp)
        
        # Cache the result
        if self._enable_cache:
            cache_key = cache_manager.cache_key_hash(symbol, timeframe.value, "data_range")
            cache_manager.set("market_data", cache_key, result, self._cache_ttl_settings['data_range'])
        
        return result
    
    def get_data_count(self, symbol: str, timeframe: TimeFrame) -> int:
        """
        Get the number of data points for a symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            Number of data points
            
        Raises:
            KeyError: If symbol or timeframe not found in store
        """
        if symbol not in self._data:
            return 0
        
        if timeframe not in self._data[symbol]:
            return 0
        
        return len(self._data[symbol][timeframe])
    
    def clear_data(self, symbol: Optional[str] = None, timeframe: Optional[TimeFrame] = None) -> None:
        """
        Clear data from the store.
        
        Args:
            symbol: If specified, only clear data for this symbol. If None, clear all symbols
            timeframe: If specified, only clear data for this timeframe. If None, clear all timeframes
        """
        if symbol is None:
            # Clear all data
            self._data.clear()
            self._symbols.clear()
            self._last_updated.clear()
            # Clear all cache entries
            if self._enable_cache:
                cache_manager.clear_namespace("market_data")
        elif timeframe is None:
            # Clear all timeframes for the symbol
            if symbol in self._data:
                # Remove all last_updated entries for this symbol
                keys_to_remove = [k for k in self._last_updated.keys() if k[0] == symbol]
                for key in keys_to_remove:
                    del self._last_updated[key]
                
                del self._data[symbol]
                self._symbols.discard(symbol)
                # Invalidate cache for this symbol
                if self._enable_cache:
                    self._invalidate_cache(symbol)
        else:
            # Clear specific symbol and timeframe
            if symbol in self._data and timeframe in self._data[symbol]:
                del self._data[symbol][timeframe]
                self._last_updated.pop((symbol, timeframe), None)
                
                # If no timeframes left for this symbol, remove the symbol
                if not self._data[symbol]:
                    del self._data[symbol]
                    self._symbols.discard(symbol)
                
                # Invalidate cache for this symbol and timeframe
                if self._enable_cache:
                    self._invalidate_cache(symbol, timeframe)
    
    def get_market_data(self, symbol: str, timeframe: TimeFrame) -> MarketData:
        """
        Get MarketData object for a symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            
        Returns:
            MarketData object with OHLCV data and metadata
            
        Raises:
            KeyError: If symbol or timeframe not found in store
        """
        data = self.get_ohlcv(symbol, timeframe)
        last_updated = self._last_updated.get((symbol, timeframe), datetime.now())
        
        return MarketData(
            symbol=symbol,
            timeframe=timeframe,
            data=data,
            indicators={},  # Empty indicators dict - can be populated by analysis modules
            last_updated=last_updated
        )
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics including symbol count, 
            total data points, and memory usage estimates
        """
        # Try to get from cache first
        if self._enable_cache:
            cache_key = cache_manager.cache_key_hash("statistics", str(len(self._symbols)))
            cached_stats = cache_manager.get("market_data", cache_key)
            if cached_stats is not None:
                return cached_stats
        
        total_points = 0
        symbol_stats = {}
        
        for symbol in self._symbols:
            symbol_points = 0
            timeframes = {}
            
            for timeframe in self._data[symbol]:
                count = len(self._data[symbol][timeframe])
                timeframes[timeframe.value] = count
                symbol_points += count
            
            symbol_stats[symbol] = {
                'total_points': symbol_points,
                'timeframes': timeframes
            }
            total_points += symbol_points
        
        result = {
            'total_symbols': len(self._symbols),
            'total_data_points': total_points,
            'symbol_breakdown': symbol_stats,
            'estimated_memory_mb': total_points * 0.0001  # Rough estimate
        }
        
        # Cache the result
        if self._enable_cache:
            cache_key = cache_manager.cache_key_hash("statistics", str(len(self._symbols)))
            cache_manager.set("market_data", cache_key, result, self._cache_ttl_settings['statistics'])
        
        return result
    
    def __len__(self) -> int:
        """Return total number of data points across all symbols and timeframes."""
        return sum(
            len(timeframe_data) 
            for symbol_data in self._data.values() 
            for timeframe_data in symbol_data.values()
        )
    
    def __contains__(self, item) -> bool:
        """Check if symbol or (symbol, timeframe) tuple exists in store."""
        if isinstance(item, str):
            return item in self._symbols
        elif isinstance(item, tuple) and len(item) == 2:
            symbol, timeframe = item
            return symbol in self._data and timeframe in self._data[symbol]
        return False
    
    def __repr__(self) -> str:
        """String representation of the data store."""
        return f"DataStore(symbols={len(self._symbols)}, total_points={len(self)})"
    
    def _invalidate_cache(self, symbol: str, timeframe: Optional[TimeFrame] = None) -> None:
        """
        Invalidate cache entries for a symbol and optionally specific timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Optional timeframe to invalidate specific entries
        """
        if not self._enable_cache:
            return
        
        try:
            # Keys to invalidate
            keys_to_delete = []
            
            if timeframe is not None:
                # Invalidate specific symbol-timeframe entries
                keys_to_delete.extend([
                    cache_manager.cache_key_hash(symbol, timeframe.value),
                    cache_manager.cache_key_hash(symbol, timeframe.value, "data_range"),
                ])
                
                # Invalidate latest data entries for this symbol-timeframe
                for periods in [10, 20, 50, 100, 200, 500]:  # Common periods
                    keys_to_delete.append(
                        cache_manager.cache_key_hash(symbol, timeframe.value, periods, "latest")
                    )
            else:
                # Invalidate all entries for this symbol
                for tf in self._data.get(symbol, {}):
                    keys_to_delete.extend([
                        cache_manager.cache_key_hash(symbol, tf.value),
                        cache_manager.cache_key_hash(symbol, tf.value, "data_range"),
                    ])
                    
                    # Invalidate latest data entries
                    for periods in [10, 20, 50, 100, 200, 500]:  # Common periods
                        keys_to_delete.append(
                            cache_manager.cache_key_hash(symbol, tf.value, periods, "latest")
                        )
            
            # Also invalidate statistics cache
            keys_to_delete.append(cache_manager.cache_key_hash("statistics", str(len(self._symbols))))
            
            # Delete all keys
            for key in keys_to_delete:
                cache_manager.delete("market_data", key)
                
            logger.debug(f"Invalidated {len(keys_to_delete)} cache entries for {symbol}")
            
        except Exception as e:
            logger.warning(f"Failed to invalidate cache for {symbol}: {e}")
    
    def get_cache_stats(self) -> Dict[str, any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics if caching is enabled
        """
        if not self._enable_cache:
            return {"caching_enabled": False}
        
        stats = cache_manager.get_stats()
        stats["caching_enabled"] = True
        stats["cache_ttl_settings"] = self._cache_ttl_settings
        
        return stats