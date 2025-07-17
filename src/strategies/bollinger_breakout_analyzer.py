"""
Bollinger Breakout Signal Analyzer

This module provides functionality to analyze market data using the Bollinger Breakout strategy
and generate trading signals by fetching data from the Alpaca provider.
"""

import logging
import os
from datetime import datetime, timedelta
from ..utils.timezone_utils import utc_now
from typing import List, Dict, Any, Optional, Tuple
import asyncio

from ..data.providers.alpaca_provider import AlpacaProviderSingleton
from ..indicators.talib_wrapper import calculate_bb as calculate_bollinger_bands
from .bollinger_breakout import BollingerBandBreakout

logger = logging.getLogger(__name__)


class BollingerBreakoutAnalyzer:
    """
    Bollinger Breakout signal analyzer that fetches data from Alpaca and generates trading signals.

    This class orchestrates the complete process of:
    1. Fetching historical price data from Alpaca
    2. Calculating Bollinger Band indicators using TA-Lib
    3. Running Bollinger Breakout strategy analysis
    4. Returning validated trading signals
    """

    def __init__(self, alpaca_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Bollinger Breakout Analyzer.

        Args:
            alpaca_config: Configuration for Alpaca provider (optional)
        """
        self.alpaca_singleton = AlpacaProviderSingleton()
        self.alpaca_config = alpaca_config
        self.alpaca_provider = None  # Will be set via singleton
        self.bollinger_strategy = BollingerBandBreakout()
        self.logger = logging.getLogger(__name__)

    async def analyze_symbol(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d",
        strategy_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Analyze a symbol for Bollinger Breakout signals.

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
            # Validate inputs
            validation_result = self._validate_inputs(
                symbol, start_date, end_date, timeframe
            )
            if not validation_result["valid"]:
                self.logger.error(
                    f"Input validation failed for {symbol}: {validation_result['error']}"
                )
                return []

            # Initialize strategy with custom parameters if provided
            if strategy_params:
                self.bollinger_strategy = BollingerBandBreakout(strategy_params)
            else:
                # Reset strategy for new analysis
                self.bollinger_strategy = BollingerBandBreakout()

            # Connect to Alpaca
            if not await self._ensure_connection():
                self.logger.error(f"Failed to connect to Alpaca for {symbol}")
                return []

            # Fetch market data
            market_data = await self._fetch_market_data(
                symbol, start_date, end_date, timeframe
            )
            if not market_data:
                self.logger.error(
                    f"No market data available for {symbol} from {start_date} to {end_date}"
                )
                return []

            # Calculate indicators
            indicators = await self._calculate_indicators(market_data)
            if not indicators:
                self.logger.error(f"Failed to calculate indicators for {symbol}")
                return []

            # Generate signals
            signals = await self._generate_signals(market_data, indicators)

            # Sort signals by timestamp in descending order (newest first)
            signals.sort(key=lambda s: s["timestamp"], reverse=True)

            self.logger.info(
                f"Generated {len(signals)} Bollinger Breakout signals for {symbol}"
            )
            return signals

        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {str(e)}")
            return []
        finally:
            # Don't disconnect shared provider - just clear reference
            self.alpaca_provider = None

    def _validate_inputs(
        self, symbol: str, start_date: datetime, end_date: datetime, timeframe: str
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
            return {
                "valid": False,
                "error": "Symbol must be 1-10 alphabetic characters",
            }

        # Validate dates
        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
            return {
                "valid": False,
                "error": "Start and end dates must be datetime objects",
            }

        if start_date >= end_date:
            return {"valid": False, "error": "Start date must be before end date"}

        # Validate timeframe
        if timeframe not in ["1h", "1d"]:
            return {"valid": False, "error": "Timeframe must be '1h' or '1d'"}

        # Validate date range for timeframe
        date_range = end_date - start_date
        now = utc_now()

        # Handle timezone-aware/naive datetime comparison
        if end_date.tzinfo is None and now.tzinfo is not None:
            # Convert timezone-naive end_date to timezone-aware for comparison
            import pytz

            end_date_for_comparison = pytz.utc.localize(end_date)
        elif end_date.tzinfo is not None and now.tzinfo is None:
            # Convert timezone-aware now to timezone-naive for comparison
            now = now.replace(tzinfo=None)
            end_date_for_comparison = end_date
        else:
            # Both have same timezone status
            end_date_for_comparison = end_date

        if timeframe == "1h":
            # For hourly data, limit to 30 days max and must be recent
            if date_range.days > 30:
                return {
                    "valid": False,
                    "error": "Hourly data limited to 30 days maximum",
                }
            if (now - end_date_for_comparison).days > 30:
                return {
                    "valid": False,
                    "error": "Hourly data must be within last 30 days",
                }

        elif timeframe == "1d":
            # For daily data, limit to 2 years max
            if date_range.days > 730:
                return {
                    "valid": False,
                    "error": "Daily data limited to 2 years maximum",
                }
            if (now - end_date_for_comparison).days > 1825:  # 5 years
                return {
                    "valid": False,
                    "error": "Daily data must be within last 5 years",
                }

        # Validate minimum data requirements using centralized calculator
        from ..utils.data_requirements import DataRequirementsCalculator

        validation_result = DataRequirementsCalculator.validate_date_range(
            start_date=start_date,
            end_date=end_date,
            strategy_type="bollinger",
            strategy_parameters=self.bollinger_strategy.parameters,
            timeframe=timeframe,
        )

        if not validation_result["valid"]:
            return validation_result

        return {"valid": True, "error": None}

    async def _ensure_connection(self) -> bool:
        """
        Ensure connection to Alpaca provider.

        Returns:
            True if connected, False otherwise
        """
        try:
            self.alpaca_provider = await self.alpaca_singleton.get_provider(
                self.alpaca_config
            )
            return self.alpaca_provider.is_connected()
        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca: {str(e)}")
            return False

    async def _fetch_market_data(
        self, symbol: str, start_date: datetime, end_date: datetime, timeframe: str
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
                end_date=end_date,
            )

            if not ohlcv_data:
                self.logger.warning(f"No OHLCV data returned for {symbol}")
                return None

            # Validate minimum data requirements using centralized calculator
            from ..utils.data_requirements import DataRequirementsCalculator

            validation_result = DataRequirementsCalculator.validate_data_sufficiency(
                data_points=len(ohlcv_data),
                strategy_type="bollinger",
                strategy_parameters=self.bollinger_strategy.parameters,
            )

            if not validation_result["valid"]:
                self.logger.warning(
                    f"Insufficient data for {symbol}: {validation_result['error']}"
                )
                return None

            # Create market data object
            market_data = MarketDataWrapper(symbol, timeframe, ohlcv_data)

            self.logger.info(f"Fetched {len(ohlcv_data)} data points for {symbol}")
            return market_data

        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol}: {str(e)}")
            return None

    async def _calculate_indicators(
        self, market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate required indicators for Bollinger Breakout strategy.

        Args:
            market_data: Market data object

        Returns:
            Dictionary of calculated indicators or None if failed
        """
        try:
            arrays = market_data.to_arrays()
            close_prices = arrays["close"]

            # Calculate Bollinger Bands indicators
            bb_period = self.bollinger_strategy.parameters.get("bb_period", 20)
            bb_std = self.bollinger_strategy.parameters.get("bb_std", 2.0)

            # Calculate Bollinger Bands using TA-Lib
            upper_band, middle_band, lower_band = calculate_bollinger_bands(
                close_prices, bb_period, bb_std
            )

            indicators = {
                "bollinger": {
                    f"bollinger_{bb_period}": {
                        "upper": (
                            upper_band.tolist()
                            if hasattr(upper_band, "tolist")
                            else upper_band
                        ),
                        "middle": (
                            middle_band.tolist()
                            if hasattr(middle_band, "tolist")
                            else middle_band
                        ),
                        "lower": (
                            lower_band.tolist()
                            if hasattr(lower_band, "tolist")
                            else lower_band
                        ),
                    }
                }
            }

            self.logger.debug(
                f"Calculated Bollinger Bands for {market_data.get('symbol', 'unknown')}"
            )
            return indicators

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return None

    async def _generate_signals(
        self, market_data: Dict[str, Any], indicators: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate trading signals using Bollinger Breakout strategy.

        Args:
            market_data: Market data object
            indicators: Calculated indicators

        Returns:
            List of trading signals
        """
        try:
            signals = self.bollinger_strategy.generate_signals(market_data, indicators)

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
        required_fields = [
            "timestamp",
            "symbol",
            "signal_type",
            "price",
            "confidence",
            "strategy_name",
        ]

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
        if self.alpaca_provider is None:
            # For get_supported_symbols, we can create a provider instance directly
            # since it only reads from environment variables, no connection needed
            try:
                from ..data.providers.alpaca_provider import AlpacaProvider

                # Create provider directly without connection
                self.alpaca_provider = AlpacaProvider(self.alpaca_config)

                # If we need connection-based operations later, we can try to connect
                # but for symbols from env vars, we don't need it
            except Exception as e:
                # If we can't create provider at all, return empty list
                self.logger.warning(f"Could not create AlpacaProvider: {e}")
                return []
        return self.alpaca_provider.get_supported_symbols()

    def get_supported_timeframes(self) -> List[str]:
        """
        Get list of supported timeframes.

        Returns:
            List of supported timeframes
        """
        if self.alpaca_provider is None:
            # For get_supported_timeframes, we can create a provider instance directly
            # since it only reads from class constants, no connection needed
            try:
                from ..data.providers.alpaca_provider import AlpacaProvider

                # Create provider directly without connection
                self.alpaca_provider = AlpacaProvider(self.alpaca_config)

                # If we need connection-based operations later, we can try to connect
                # but for timeframes from class constants, we don't need it
            except Exception as e:
                # If we can't create provider at all, return default timeframes
                self.logger.warning(f"Could not create AlpacaProvider: {e}")
                return ["1h", "1d"]
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
            "volume": [],
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
