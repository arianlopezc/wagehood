"""
Alpaca Markets Data Provider

This module implements the Alpaca Markets data provider using alpaca-trade-api.
Supports historical data retrieval and real-time streaming for stock market data.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
from threading import Thread
from ...utils.timezone_utils import utc_now

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.common import URL
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    # Define dummy classes to avoid NameError
    class tradeapi:
        REST = None
        Stream = None
        TimeFrame = None
        TimeFrameUnit = None
    
    class URL:
        def __init__(self, url):
            pass

from .base import DataProvider, ConnectionError, DataRetrievalError

logger = logging.getLogger(__name__)


class AlpacaProviderSingleton:
    """Singleton manager for AlpacaProvider to prevent multiple concurrent connections."""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._provider = None
            self._connection_lock = None
            self._initialized = True
    
    async def get_provider(self, config: Optional[Dict[str, Any]] = None) -> 'AlpacaProvider':
        """Get or create the shared AlpacaProvider instance with connection safety."""
        if self._connection_lock is None:
            import asyncio
            self._connection_lock = asyncio.Lock()
        
        async with self._connection_lock:
            if self._provider is None:
                logger.info("Creating new AlpacaProvider instance")
                self._provider = AlpacaProvider(config)
                # FAIL-FAST: connect() will raise exception if it fails
                await self._provider.connect()
                logger.info("AlpacaProvider connected successfully")
            elif not self._provider.is_connected():
                logger.warning("AlpacaProvider disconnected, reconnecting...")
                # FAIL-FAST: Let connection errors propagate
                await self._provider.connect()
                logger.info("AlpacaProvider reconnected successfully")
        
        return self._provider


class AlpacaProvider(DataProvider):
    """
    Alpaca Markets data provider implementation using alpaca-trade-api.
    
    Provides access to:
    - Historical OHLCV data with 1-hour and 1-day timeframes
    - Real-time price streaming
    """
    
    # Supported timeframes mapping
    TIMEFRAME_MAPPING = {
        '1h': tradeapi.TimeFrame.Hour,
        '1d': tradeapi.TimeFrame.Day
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Alpaca data provider.
        
        Args:
            config: Configuration dictionary with options:
                - api_key: Alpaca API key (or use ALPACA_API_KEY env var)
                - secret_key: Alpaca secret key (or use ALPACA_SECRET_KEY env var)
        """
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-trade-api is required for AlpacaProvider. Install with: pip install alpaca-trade-api")
        
        super().__init__("Alpaca", config)
        
        # Get credentials from config or environment
        self.api_key = self.config.get('api_key') or os.getenv('ALPACA_API_KEY')
        self.secret_key = self.config.get('secret_key') or os.getenv('ALPACA_SECRET_KEY')
        
        # Validate credentials
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API credentials required. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "environment variables or pass api_key and secret_key in config."
            )
        
        # Initialize clients
        self.rest_client = None
        self.stream_client = None
        
        # Streaming state
        self._stream_active = False
        self._stream_thread = None
        self._stream_handlers = {}
        
        logger.info("Initialized AlpacaProvider (live API)")
    
    async def connect(self) -> bool:
        """
        Establish connection to Alpaca data sources.
        
        Returns:
            True if connection successful
        Raises:
            ConnectionError: If connection fails
        """
        # FAIL-FAST: Let exceptions propagate up
        self._clear_error()
        
        # Initialize clients (always use live API)
        base_url = URL('https://api.alpaca.markets')
        
        logger.info(f"Connecting to Alpaca API at {base_url}")
        logger.info(f"Using API key: {self.api_key[:8]}...")
        
        self.rest_client = tradeapi.REST(
            key_id=self.api_key,
            secret_key=self.secret_key,
            base_url=base_url
        )
        
        self.stream_client = tradeapi.Stream(
            key_id=self.api_key,
            secret_key=self.secret_key,
            base_url=base_url
        )
        
        # Test connection - will raise exception if fails
        await self._test_connection()
        
        self._connected = True
        logger.info("Successfully connected to Alpaca Markets")
        return True
    
    async def disconnect(self) -> None:
        """Close connection to Alpaca data sources."""
        try:
            if self._stream_active:
                await self.stop_streaming()
            
            self._connected = False
            logger.info("Disconnected from Alpaca Markets")
            
        except Exception as e:
            error_msg = f"Error during disconnect: {str(e)}"
            logger.error(error_msg)
            self._set_error(error_msg)
    
    async def _test_connection(self) -> None:
        """Test connection by making a simple API request."""
        # FAIL-FAST: No try-catch - let exceptions propagate
        logger.info("Testing Alpaca API connection...")
        account = self.rest_client.get_account()
        
        if not account:
            raise ConnectionError("Connection test failed: No account data returned")
        
        if hasattr(account, 'status'):
            logger.info(f"Connection test successful - Account status: {account.status}")
        else:
            logger.info("Connection test successful - Account data received")
    
    def _get_alpaca_timeframe(self, timeframe: str):
        """Convert timeframe string to Alpaca TimeFrame object."""
        if timeframe not in self.TIMEFRAME_MAPPING:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {list(self.TIMEFRAME_MAPPING.keys())}")
        return self.TIMEFRAME_MAPPING[timeframe]
    
    def _convert_bar_to_ohlcv(self, bar, timestamp=None) -> Dict[str, Any]:
        """Convert a single bar to OHLCV dictionary."""
        return {
            'timestamp': timestamp or bar.timestamp,
            'open': float(bar.open),
            'high': float(bar.high),
            'low': float(bar.low),
            'close': float(bar.close),
            'volume': float(bar.volume) if bar.volume is not None else 0.0
        }
    
    def _convert_bars_to_dict(self, bars_response, symbol: str) -> List[Dict[str, Any]]:
        """Convert Alpaca bars response to list of OHLCV dictionaries."""
        ohlcv_data = []
        
        try:
            # Handle DataFrame response format (alpaca-trade-api format)
            if hasattr(bars_response, 'df') and not bars_response.df.empty:
                df = bars_response.df
                
                # Convert each row to OHLCV dictionary
                for timestamp, row in df.iterrows():
                    ohlcv = {
                        'timestamp': timestamp,
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if row['volume'] is not None else 0.0
                    }
                    ohlcv_data.append(ohlcv)
            
            # Handle list of bars format (alpaca-trade-api)
            elif hasattr(bars_response, '__iter__'):
                for bar in bars_response:
                    if hasattr(bar, 'timestamp'):
                        ohlcv_data.append(self._convert_bar_to_ohlcv(bar))
            
            # Handle single bar
            elif hasattr(bars_response, 'timestamp'):
                ohlcv_data.append(self._convert_bar_to_ohlcv(bars_response))
            
        except Exception as e:
            logger.error(f"Error converting bars response: {e}")
            raise DataRetrievalError(f"Failed to convert bars response: {str(e)}")
        
        return ohlcv_data
    
    async def get_historical_data(self, symbol: str, timeframe: str,
                                 start_date: datetime, end_date: datetime,
                                 limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve historical OHLCV data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'MSFT')
            timeframe: Data timeframe ('1h' or '1d')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            limit: Maximum number of records to return (default: 1000)
            
        Returns:
            List of OHLCV data dictionaries
        """
        if not self._connected or not self.rest_client:
            raise ConnectionError("Not connected to Alpaca Markets")
        
        # FAIL-FAST: No try-catch - let exceptions propagate
        logger.info(f"Requesting historical data for {symbol} ({timeframe}) from {start_date} to {end_date}")
        
        # Convert timeframe string to appropriate call
        # Format dates for Alpaca API (YYYY-MM-DD format)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Log the exact API call parameters
        logger.info(f"Alpaca API call: symbol={symbol}, timeframe={timeframe}, start={start_str}, end={end_str}, limit={limit or 1000}")
        
        # Check if this is a crypto symbol (contains "/" like BTC/USD)
        is_crypto = '/' in symbol
        
        if timeframe == '1d':
            if is_crypto:
                bars = self.rest_client.get_crypto_bars(
                    symbol,
                    tradeapi.TimeFrame.Day,
                    start=start_str,
                    end=end_str,
                    limit=limit or 1000
                )
            else:
                bars = self.rest_client.get_bars(
                    symbol,
                    tradeapi.TimeFrame.Day,
                    start=start_str,
                    end=end_str,
                    limit=limit or 1000
                )
        elif timeframe == '1h':
            if is_crypto:
                bars = self.rest_client.get_crypto_bars(
                    symbol,
                    tradeapi.TimeFrame.Hour,
                    start=start_str,
                    end=end_str,
                    limit=limit or 1000
                )
            else:
                bars = self.rest_client.get_bars(
                    symbol,
                    tradeapi.TimeFrame.Hour,
                    start=start_str,
                    end=end_str,
                    limit=limit or 1000
                )
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # Convert to our format
        ohlcv_data = self._convert_bars_to_dict(bars, symbol)
        
        if not ohlcv_data:
            raise DataRetrievalError(f"No data returned from Alpaca API for {symbol}")
        
        logger.info(f"Retrieved {len(ohlcv_data)} historical bars for {symbol}")
        return ohlcv_data
    
    async def get_latest_data(self, symbol: str, timeframe: str,
                            periods: int = 1) -> List[Dict[str, Any]]:
        """
        Get the latest N periods of data for a symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe ('1h' or '1d')
            periods: Number of latest periods to retrieve
            
        Returns:
            List of latest OHLCV data dictionaries
        """
        # Calculate start date with buffer to ensure we get enough data
        now = utc_now()
        
        if timeframe == '1h':
            buffer_hours = max(periods * 3, 72)  # At least 3 days
            start_date = now - timedelta(hours=buffer_hours)
        elif timeframe == '1d':
            buffer_days = max(periods * 3, 14)  # At least 2 weeks
            start_date = now - timedelta(days=buffer_days)
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # Get historical data and return latest periods
        data = await self.get_historical_data(symbol, timeframe, start_date, now)
        return data[-periods:] if len(data) >= periods else data
    
    async def start_streaming(self, symbol: str, on_bar: Optional[Callable] = None) -> None:
        """
        Start real-time data streaming for a symbol.
        
        Args:
            symbol: Symbol to stream
            on_bar: Callback function for bar data
        """
        if not self._connected or not self.stream_client:
            raise ConnectionError("Not connected to Alpaca Markets")
        
        if self._stream_active:
            logger.warning("Stream already active")
            return
        
        try:
            # Store handler
            if on_bar:
                self._stream_handlers[symbol] = on_bar
            
            # Set up bar subscription
            async def bar_handler(bar):
                try:
                    bar_data = {
                        'timestamp': bar.timestamp,
                        'symbol': bar.symbol,
                        'open': float(bar.open),
                        'high': float(bar.high),
                        'low': float(bar.low),
                        'close': float(bar.close),
                        'volume': float(bar.volume) if bar.volume is not None else 0.0
                    }
                    
                    # Call user handler
                    if symbol in self._stream_handlers:
                        await self._stream_handlers[symbol](bar_data)
                        
                except Exception as e:
                    logger.error(f"Error in bar handler: {e}")
            
            # Subscribe to bars and start streaming
            self.stream_client.subscribe_bars(bar_handler, symbol)
            
            def run_stream():
                try:
                    self.stream_client.run()
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    self._stream_active = False
            
            self._stream_thread = Thread(target=run_stream, daemon=True)
            self._stream_thread.start()
            self._stream_active = True
            
            logger.info(f"Started streaming for {symbol}")
            
        except Exception as e:
            error_msg = f"Error starting stream for {symbol}: {str(e)}"
            logger.error(error_msg)
            self._set_error(error_msg)
            raise ConnectionError(error_msg)
    
    async def stop_streaming(self) -> None:
        """Stop all real-time data streaming."""
        try:
            if self.stream_client and self._stream_active:
                await self.stream_client.stop_ws()
                self._stream_active = False
                
                # Wait for thread to finish
                if self._stream_thread and self._stream_thread.is_alive():
                    self._stream_thread.join(timeout=5.0)
                
                self._stream_handlers.clear()
                logger.info("Stopped data streaming")
            
        except Exception as e:
            error_msg = f"Error stopping stream: {str(e)}"
            logger.error(error_msg)
            self._set_error(error_msg)
    
    def is_streaming(self) -> bool:
        """Check if streaming is active."""
        return self._stream_active
    
    def get_supported_timeframes(self) -> List[str]:
        """Get list of supported timeframes."""
        return list(self.TIMEFRAME_MAPPING.keys())
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols from environment."""
        symbols_str = os.getenv('SUPPORTED_SYMBOLS', '')
        if symbols_str:
            return [s.strip() for s in symbols_str.split(',') if s.strip()]
        return []
    
    async def get_symbols(self) -> List[str]:
        """Get list of available symbols."""
        return self.get_supported_symbols()
    
    async def get_market_data(self, symbol: str, timeframe: str,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get complete market data object for a symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe ('1h' or '1d')
            start_date: Start date for data (optional)
            end_date: End date for data (optional)
            
        Returns:
            Dictionary with OHLCV data and metadata
        """
        # Use default date range if not provided
        if end_date is None:
            end_date = utc_now()
        
        if start_date is None:
            start_date = end_date - timedelta(days=60)  # Increased for better analysis quality
        
        # Get historical data
        ohlcv_data = await self.get_historical_data(symbol, timeframe, start_date, end_date)
        
        # Create market data dictionary
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'data': ohlcv_data,
            'last_updated': utc_now(),
            'start_date': start_date,
            'end_date': end_date
        }