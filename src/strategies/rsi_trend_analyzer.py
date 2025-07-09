"""
RSI Trend Signal Analyzer - Optimized Version

This module provides optimized functionality to analyze market data using the RSI Trend strategy
and generate trading signals by fetching data from the Alpaca provider.
Performance improvements include better caching, vectorized operations, and early exit conditions.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import asyncio

from ..data.providers.alpaca_provider import AlpacaProvider
from ..indicators.momentum import calculate_rsi
from .rsi_trend import RSITrendFollowing

logger = logging.getLogger(__name__)


class RSITrendAnalyzer:
    """
    RSI Trend signal analyzer that fetches data from Alpaca and generates trading signals.
    
    This class orchestrates the complete process of:
    1. Fetching historical price data from Alpaca
    2. Calculating RSI indicators
    3. Running RSI Trend strategy analysis
    4. Returning validated trading signals
    """

    def __init__(self, alpaca_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RSI Trend Analyzer with optimized configuration.
        
        Args:
            alpaca_config: Configuration for Alpaca provider (optional)
        """
        self.alpaca_provider = AlpacaProvider(alpaca_config)
        self.rsi_strategy = RSITrendFollowing()
        self.logger = logging.getLogger(__name__)
        
        # Performance optimization: cache frequently used values
        self._last_validation_result = None
        self._validation_cache_time = None
        
    async def analyze_symbol(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d",
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze a symbol for RSI trend signals.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'MSFT')
            start_date: Start date for analysis
            end_date: End date for analysis
            timeframe: Data timeframe ('1h' or '1d')
            strategy_params: Optional strategy parameters override
            
        Returns:
            List of trading signals found, empty list if no signals or errors
        """
        try:
            # Validate inputs with caching for performance
            validation_result = self._validate_inputs_cached(symbol, start_date, end_date, timeframe)
            if not validation_result["valid"]:
                self.logger.error(f"Input validation failed for {symbol}: {validation_result['error']}")
                return []
            
            # Initialize strategy with custom parameters if provided
            if strategy_params:
                self.rsi_strategy = RSITrendFollowing(strategy_params)
            else:
                # Reset cache for new analysis
                self.rsi_strategy.reset_cache()
            
            # Connect to Alpaca
            if not await self._ensure_connection():
                self.logger.error(f"Failed to connect to Alpaca for {symbol}")
                return []
            
            # Fetch market data
            market_data = await self._fetch_market_data(symbol, start_date, end_date, timeframe)
            if not market_data:
                self.logger.error(f"No market data available for {symbol} from {start_date} to {end_date}")
                return []
            
            # Calculate indicators
            indicators = await self._calculate_indicators(market_data)
            if not indicators:
                self.logger.error(f"Failed to calculate indicators for {symbol}")
                return []
            
            # Generate signals
            signals = await self._generate_signals(market_data, indicators)
            
            self.logger.info(f"Generated {len(signals)} RSI trend signals for {symbol}")
            return signals
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {str(e)}")
            return []
        finally:
            # Ensure cleanup
            try:
                await self.alpaca_provider.disconnect()
            except Exception:
                pass  # Ignore disconnection errors
    
    def _validate_inputs_cached(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> Dict[str, Any]:
        """
        Validate inputs with caching for repeated validations.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
            
        Returns:
            Dictionary with validation result
        """
        # Create cache key
        cache_key = f"{symbol}_{start_date}_{end_date}_{timeframe}"
        current_time = datetime.now()
        
        # Check cache validity (5 minutes)
        if (self._validation_cache_time and 
            (current_time - self._validation_cache_time).seconds < 300 and
            hasattr(self, '_last_cache_key') and
            self._last_cache_key == cache_key):
            return self._last_validation_result
        
        # Perform validation
        result = self._validate_inputs(symbol, start_date, end_date, timeframe)
        
        # Cache result
        self._last_validation_result = result
        self._validation_cache_time = current_time
        self._last_cache_key = cache_key
        
        return result
    
    def _validate_inputs(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> Dict[str, Any]:
        """
        Validate all input parameters.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
            
        Returns:
            Dictionary with validation result
        """
        # Validate symbol
        if not symbol or not isinstance(symbol, str):
            return {"valid": False, "error": "Symbol must be a non-empty string"}
        
        if not symbol.isalpha() or len(symbol) < 1 or len(symbol) > 10:
            return {"valid": False, "error": "Symbol must be 1-10 alphabetic characters"}
        
        # Validate dates
        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
            return {"valid": False, "error": "Start and end dates must be datetime objects"}
        
        if start_date >= end_date:
            return {"valid": False, "error": "Start date must be before end date"}
        
        # Validate timeframe
        if timeframe not in ["1h", "1d"]:
            return {"valid": False, "error": "Timeframe must be '1h' or '1d'"}
        
        # Validate date range for timeframe
        date_range = end_date - start_date
        now = datetime.now()
        
        if timeframe == "1h":
            # For hourly data, limit to 30 days max and must be recent
            if date_range.days > 30:
                return {"valid": False, "error": "Hourly data limited to 30 days maximum"}
            if (now - end_date).days > 30:
                return {"valid": False, "error": "Hourly data must be within last 30 days"}
        
        elif timeframe == "1d":
            # For daily data, limit to 2 years max
            if date_range.days > 730:
                return {"valid": False, "error": "Daily data limited to 2 years maximum"}
            if (now - end_date).days > 1825:  # 5 years
                return {"valid": False, "error": "Daily data must be within last 5 years"}
        
        # Validate minimum data requirements
        min_periods = self.rsi_strategy.parameters.get("trend_confirmation_periods", 10)
        rsi_period = self.rsi_strategy.parameters.get("rsi_period", 14)
        min_required_periods = max(min_periods, rsi_period) + 5  # Buffer
        
        if timeframe == "1h":
            min_days = max(3, min_required_periods // 24 + 1)
        else:  # 1d
            min_days = max(min_required_periods, 20)
        
        if date_range.days < min_days:
            return {
                "valid": False,
                "error": f"Date range too small. Minimum {min_days} days required for {timeframe} timeframe"
            }
        
        return {"valid": True, "error": None}
    
    async def _ensure_connection(self) -> bool:
        """
        Ensure connection to Alpaca provider.
        
        Returns:
            True if connected, False otherwise
        """
        try:
            if not self.alpaca_provider.is_connected():
                return await self.alpaca_provider.connect()
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca: {str(e)}")
            return False
    
    async def _fetch_market_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch market data from Alpaca provider.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
            
        Returns:
            Market data dictionary or None if failed
        """
        try:
            # Fetch historical data
            ohlcv_data = await self.alpaca_provider.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if not ohlcv_data:
                self.logger.warning(f"No OHLCV data returned for {symbol}")
                return None
            
            # Validate minimum data requirements
            min_periods = self.rsi_strategy.parameters.get("trend_confirmation_periods", 10)
            rsi_period = self.rsi_strategy.parameters.get("rsi_period", 14)
            min_required = max(min_periods, rsi_period) + 5
            
            if len(ohlcv_data) < min_required:
                self.logger.warning(
                    f"Insufficient data for {symbol}: got {len(ohlcv_data)}, need {min_required}"
                )
                return None
            
            # Create market data object
            market_data = MarketDataWrapper(symbol, timeframe, ohlcv_data)
            
            self.logger.info(f"Fetched {len(ohlcv_data)} data points for {symbol}")
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol}: {str(e)}")
            return None
    
    async def _calculate_indicators(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Calculate required indicators for RSI trend strategy.
        
        Args:
            market_data: Market data object
            
        Returns:
            Dictionary of calculated indicators or None if failed
        """
        try:
            arrays = market_data.to_arrays()
            close_prices = arrays["close"]
            
            # Calculate RSI indicators
            rsi_period = self.rsi_strategy.parameters.get("rsi_period", 14)
            rsi_main_period = self.rsi_strategy.parameters.get("rsi_main_period", 21)
            
            # Calculate RSI for both periods
            rsi_14 = calculate_rsi(close_prices, rsi_period)
            rsi_21 = calculate_rsi(close_prices, rsi_main_period) if rsi_main_period != rsi_period else rsi_14
            
            indicators = {
                "rsi": {
                    f"rsi_{rsi_period}": rsi_14.tolist() if hasattr(rsi_14, 'tolist') else rsi_14,
                    f"rsi_{rsi_main_period}": rsi_21.tolist() if hasattr(rsi_21, 'tolist') else rsi_21
                }
            }
            
            self.logger.debug(f"Calculated RSI indicators for {market_data.get('symbol', 'unknown')}")
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return None
    
    async def _generate_signals(
        self,
        market_data: Dict[str, Any],
        indicators: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate trading signals using RSI trend strategy.
        
        Args:
            market_data: Market data object
            indicators: Calculated indicators
            
        Returns:
            List of trading signals
        """
        try:
            signals = self.rsi_strategy.generate_signals(market_data, indicators)
            
            # Additional validation and filtering
            valid_signals = []
            for signal in signals:
                if self._validate_signal(signal):
                    valid_signals.append(signal)
                else:
                    self.logger.debug(f"Filtered invalid signal: {signal}")
            
            return valid_signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return []
    
    def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Validate a trading signal.
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["timestamp", "symbol", "signal_type", "price", "confidence", "strategy_name"]
        
        # Check required fields
        for field in required_fields:
            if field not in signal:
                return False
        
        # Validate signal type
        if signal["signal_type"] not in ["BUY", "SELL"]:
            return False
        
        # Validate confidence
        confidence = signal.get("confidence", 0)
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            return False
        
        # Validate price
        price = signal.get("price", 0)
        if not isinstance(price, (int, float)) or price <= 0:
            return False
        
        return True
    
    def get_supported_symbols(self) -> List[str]:
        """
        Get list of supported symbols from environment.
        
        Returns:
            List of supported symbols
        """
        return self.alpaca_provider.get_supported_symbols()
    
    def get_supported_timeframes(self) -> List[str]:
        """
        Get list of supported timeframes.
        
        Returns:
            List of supported timeframes
        """
        return self.alpaca_provider.get_supported_timeframes()


class MarketDataWrapper:
    """
    Wrapper class for market data to provide consistent interface.
    """
    
    def __init__(self, symbol: str, timeframe: str, ohlcv_data: List[Dict[str, Any]]):
        """
        Initialize market data wrapper.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            ohlcv_data: List of OHLCV data dictionaries
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.ohlcv_data = ohlcv_data
    
    def to_arrays(self) -> Dict[str, List]:
        """
        Convert OHLCV data to arrays format.
        
        Returns:
            Dictionary with arrays of timestamp, open, high, low, close, volume
        """
        arrays = {
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": []
        }
        
        for data_point in self.ohlcv_data:
            arrays["timestamp"].append(data_point["timestamp"])
            arrays["open"].append(data_point["open"])
            arrays["high"].append(data_point["high"])
            arrays["low"].append(data_point["low"])
            arrays["close"].append(data_point["close"])
            arrays["volume"].append(data_point["volume"])
        
        return arrays
    
    def get(self, key: str, default=None):
        """
        Get attribute value.
        
        Args:
            key: Attribute key
            default: Default value if not found
            
        Returns:
            Attribute value or default
        """
        if key == "symbol":
            return self.symbol
        elif key == "timeframe":
            return self.timeframe
        else:
            return default
    
    def __len__(self):
        """Return number of data points."""
        return len(self.ohlcv_data)