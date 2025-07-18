"""
MACD+RSI Combined Signal Analyzer

This module provides functionality to analyze market data using the MACD+RSI combined strategy
and generate trading signals by fetching data from the Alpaca provider.
"""

import logging
import os
from datetime import datetime, timedelta
from ..utils.timezone_utils import utc_now
from typing import List, Dict, Any, Optional, Tuple
import asyncio

from ..data.providers.alpaca_provider import AlpacaProvider, AlpacaProviderSingleton
from ..data.providers.base import DataRetrievalError
from ..indicators.talib_wrapper import calculate_macd, calculate_rsi
from .macd_rsi import MACDRSIStrategy

logger = logging.getLogger(__name__)


class MACDRSIAnalyzer:
    """
    MACD+RSI signal analyzer that fetches data from Alpaca and generates trading signals.

    This class orchestrates the complete process of:
    1. Fetching historical price data from Alpaca
    2. Calculating MACD and RSI indicators using TA-Lib
    3. Running MACD+RSI strategy analysis
    4. Returning validated trading signals
    """

    def __init__(self, alpaca_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MACD+RSI Analyzer.

        Args:
            alpaca_config: Configuration for Alpaca provider (optional)
        """
        self.alpaca_config = alpaca_config
        self.alpaca_provider = None  # Will be set via singleton
        self.alpaca_singleton = AlpacaProviderSingleton()
        self.macd_rsi_strategy = MACDRSIStrategy()
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
        Analyze a symbol for MACD+RSI signals.

        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'MSFT')
            start_date: Start date for analysis
            end_date: End date for analysis
            timeframe: Data timeframe ('1h' or '1d')
            strategy_params: Optional strategy parameters override

        Returns:
            List of trading signals found
        Raises:
            Exception: If any step fails
        """
        self.logger.info(
            f"Starting analysis for {symbol} from {start_date} to {end_date}"
        )

        # Validate inputs
        validation_result = self._validate_inputs(
            symbol, start_date, end_date, timeframe
        )
        if not validation_result["valid"]:
            error_msg = (
                f"Input validation failed for {symbol}: {validation_result['error']}"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Initialize strategy with custom parameters if provided
        if strategy_params:
            self.macd_rsi_strategy = MACDRSIStrategy(strategy_params)
        else:
            # Reset strategy for new analysis
            self.macd_rsi_strategy = MACDRSIStrategy()

        # Connect to Alpaca
        await self._ensure_connection()

        # Fetch market data
        market_data = await self._fetch_market_data(
            symbol, start_date, end_date, timeframe
        )
        if not market_data:
            error_msg = (
                f"No market data available for {symbol} from {start_date} to {end_date}"
            )
            self.logger.error(error_msg)
            raise DataRetrievalError(error_msg)

        # Calculate indicators
        indicators = await self._calculate_indicators(market_data)
        if not indicators:
            error_msg = f"Failed to calculate indicators for {symbol}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Generate signals
        signals = await self._generate_signals(market_data, indicators)

        self.logger.info(f"Generated {len(signals)} MACD+RSI signals for {symbol}")

        return signals

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

        # Allow alphanumeric and '/' for crypto symbols (e.g., BTC/USD)
        if len(symbol) < 1 or len(symbol) > 10:
            return {
                "valid": False,
                "error": "Symbol must be 1-10 characters",
            }
        
        # Check if it's a valid stock symbol (alphabetic) or crypto symbol (contains '/')
        is_crypto = '/' in symbol
        if not is_crypto and not symbol.isalpha():
            return {
                "valid": False,
                "error": "Stock symbols must be alphabetic characters only",
            }
        
        if is_crypto:
            # Validate crypto symbol format (e.g., BTC/USD)
            parts = symbol.split('/')
            if len(parts) != 2 or not all(part.isalpha() for part in parts):
                return {
                    "valid": False,
                    "error": "Crypto symbols must be in format BASE/QUOTE (e.g., BTC/USD)",
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
            strategy_type="macd_rsi",
            strategy_parameters=self.macd_rsi_strategy.parameters,
            timeframe=timeframe,
        )

        if not validation_result["valid"]:
            return validation_result

        return {"valid": True, "error": None}

    async def _ensure_connection(self) -> bool:
        """
        Ensure connection to Alpaca provider using singleton pattern.

        Returns:
            True if connected
        Raises:
            Exception: If connection fails
        """
        self.logger.info("Ensuring Alpaca connection...")
        self.alpaca_provider = await self.alpaca_singleton.get_provider(
            self.alpaca_config
        )

        if not self.alpaca_provider.is_connected():
            raise ConnectionError("AlpacaProvider is not connected after get_provider")

        return True

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
            # Fetch historical data with retry logic
            ohlcv_data = None
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    ohlcv_data = await self.alpaca_provider.get_historical_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    if ohlcv_data and len(ohlcv_data) > 0:
                        break  # Success
                    else:
                        self.logger.warning(
                            f"Attempt {attempt + 1}: Empty data for {symbol}, retrying..."
                        )
                        if attempt < max_retries - 1:
                            import asyncio

                            await asyncio.sleep(
                                0.5 * (attempt + 1)
                            )  # Progressive delay
                except Exception as e:
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed for {symbol}: {e}"
                    )
                    if attempt < max_retries - 1:
                        import asyncio

                        await asyncio.sleep(0.5 * (attempt + 1))
                    else:
                        raise

            if not ohlcv_data:
                self.logger.warning(f"No OHLCV data returned for {symbol}")
                return None

            # Validate minimum data requirements using centralized calculator
            from ..utils.data_requirements import DataRequirementsCalculator

            validation_result = DataRequirementsCalculator.validate_data_sufficiency(
                data_points=len(ohlcv_data),
                strategy_type="macd_rsi",
                strategy_parameters=self.macd_rsi_strategy.parameters,
            )

            if not validation_result["valid"]:
                self.logger.warning(
                    f"Insufficient data for {symbol}: {validation_result['error']}"
                )
                return None

            self.logger.debug(
                f"Data validation passed for {symbol}: {len(ohlcv_data)} >= {validation_result['minimum_required']}"
            )

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
        Calculate required indicators for MACD+RSI strategy.

        Args:
            market_data: Market data object

        Returns:
            Dictionary of calculated indicators or None if failed
        """
        try:
            arrays = market_data.to_arrays()
            close_prices = arrays["close"]

            # Calculate MACD indicators
            macd_fast = self.macd_rsi_strategy.parameters.get("macd_fast", 12)
            macd_slow = self.macd_rsi_strategy.parameters.get("macd_slow", 26)
            macd_signal = self.macd_rsi_strategy.parameters.get("macd_signal", 9)

            # Calculate RSI indicators
            rsi_period = self.macd_rsi_strategy.parameters.get("rsi_period", 14)

            # Calculate MACD using TA-Lib
            macd_line, signal_line, histogram = calculate_macd(
                close_prices, macd_fast, macd_slow, macd_signal
            )

            # Calculate RSI using TA-Lib
            rsi_values = calculate_rsi(close_prices, rsi_period)

            indicators = {
                "macd": {
                    "macd": {
                        "macd": (
                            macd_line.tolist()
                            if hasattr(macd_line, "tolist")
                            else macd_line
                        ),
                        "signal": (
                            signal_line.tolist()
                            if hasattr(signal_line, "tolist")
                            else signal_line
                        ),
                        "histogram": (
                            histogram.tolist()
                            if hasattr(histogram, "tolist")
                            else histogram
                        ),
                    }
                },
                "rsi": {
                    f"rsi_{rsi_period}": (
                        rsi_values.tolist()
                        if hasattr(rsi_values, "tolist")
                        else rsi_values
                    )
                },
            }

            self.logger.debug(
                f"Calculated MACD+RSI indicators for {market_data.get('symbol', 'unknown')}"
            )
            return indicators

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return None

    async def _generate_signals(
        self, market_data: Dict[str, Any], indicators: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate trading signals using MACD+RSI strategy.

        Args:
            market_data: Market data object
            indicators: Calculated indicators

        Returns:
            List of trading signals
        """
        try:
            signals = self.macd_rsi_strategy.generate_signals(market_data, indicators)

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
