"""
Alpaca Markets Data Provider

This module implements the Alpaca Markets data provider for real-time and historical
market data using the official alpaca-py SDK. Supports both IEX (free) and SIP (paid)
data feeds with WebSocket streaming capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Callable
import os

try:
    from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
    from alpaca.data.live import StockDataStream, CryptoDataStream
    from alpaca.data.requests import (
        StockBarsRequest, StockQuotesRequest, StockTradesRequest,
        CryptoBarsRequest, CryptoQuoteRequest, CryptoTradesRequest
    )
    from alpaca.data.timeframe import TimeFrame as AlpacaTimeFrame, TimeFrameUnit
    from alpaca.common.exceptions import APIError
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    # Define dummy classes to avoid NameError
    class AlpacaTimeFrame:
        pass
    class TimeFrameUnit:
        pass
    class APIError(Exception):
        pass

from src.data.providers.base import DataProvider, DataProviderError, ConnectionError, DataRetrievalError
from src.core.models import OHLCV, TimeFrame, MarketData

logger = logging.getLogger(__name__)


class AlpacaProvider(DataProvider):
    """
    Alpaca Markets data provider implementation.
    
    Provides access to real-time and historical market data from Alpaca Markets
    using their official Python SDK. Supports both free IEX data and paid SIP data.
    
    Features:
    - Historical data retrieval (stocks and crypto)
    - Real-time WebSocket streaming
    - Paper trading and live trading support
    - Automatic rate limiting and error handling
    - Circuit breaker pattern for resilience
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Alpaca data provider.
        
        Args:
            config: Configuration dictionary with options:
                - api_key: Alpaca API key (or use ALPACA_API_KEY env var)
                - secret_key: Alpaca secret key (or use ALPACA_SECRET_KEY env var)
                - paper: Use paper trading environment (default: True)
                - feed: Data feed type ('iex' or 'sip', default: 'iex')
                - crypto_feed: Crypto data feed ('us' default)
                - max_retries: Maximum retry attempts (default: 3)
                - retry_delay: Delay between retries in seconds (default: 1.0)
        """
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-py is required for AlpacaProvider. Install with: pip install alpaca-py")
        
        super().__init__("Alpaca", config)
        
        # Configuration
        self.api_key = self.config.get('api_key') or os.getenv('ALPACA_API_KEY')
        self.secret_key = self.config.get('secret_key') or os.getenv('ALPACA_SECRET_KEY')
        self.paper = self.config.get('paper', True)
        self.feed = self.config.get('feed', 'iex')  # 'iex' or 'sip'
        self.crypto_feed = self.config.get('crypto_feed', 'us')
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)
        
        # Validate credentials
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API credentials required. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "environment variables or pass api_key and secret_key in config."
            )
        
        # Initialize clients
        self.stock_client = None
        self.crypto_client = None
        self.stock_stream = None
        self.crypto_stream = None
        
        # Connection state
        self._stream_handlers = {}
        self._stream_connected = False
        
        logger.info(f"Initialized AlpacaProvider (paper={self.paper}, feed={self.feed})")
    
    async def connect(self) -> bool:
        """
        Establish connection to Alpaca data sources.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self._clear_error()
            
            # Initialize historical data clients
            # Note: Historical clients don't require authentication for crypto
            if self.api_key and self.secret_key:
                self.stock_client = StockHistoricalDataClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key
                )
            else:
                self.stock_client = StockHistoricalDataClient()
            
            self.crypto_client = CryptoHistoricalDataClient()
            
            # Initialize streaming clients (require authentication)
            if self.api_key and self.secret_key:
                self.stock_stream = StockDataStream(
                    api_key=self.api_key,
                    secret_key=self.secret_key,
                    feed=self.feed
                )
                
                self.crypto_stream = CryptoDataStream(
                    api_key=self.api_key,
                    secret_key=self.secret_key,
                    feed=self.crypto_feed
                )
            
            # Test connection with a simple request
            await self._test_connection()
            
            self._connected = True
            logger.info("Successfully connected to Alpaca Markets")
            return True
            
        except Exception as e:
            error_msg = f"Failed to connect to Alpaca: {str(e)}"
            self._set_error(error_msg)
            logger.error(error_msg)
            return False
    
    async def disconnect(self) -> None:
        """Close connection to Alpaca data sources."""
        try:
            # Stop streaming clients
            if self.stock_stream and self._stream_connected:
                try:
                    await self.stock_stream.stop_ws()
                except Exception as e:
                    logger.warning(f"Error stopping stock stream: {e}")
            
            if self.crypto_stream and self._stream_connected:
                try:
                    await self.crypto_stream.stop_ws()
                except Exception as e:
                    logger.warning(f"Error stopping crypto stream: {e}")
            
            self._stream_connected = False
            self._connected = False
            self._stream_handlers.clear()
            
            logger.info("Disconnected from Alpaca Markets")
            
        except Exception as e:
            error_msg = f"Error during disconnect: {str(e)}"
            logger.error(error_msg)
            self._set_error(error_msg)
    
    async def _test_connection(self) -> None:
        """Test connection by making a simple API request."""
        try:
            # Test with a simple crypto data request (no auth required)
            request = CryptoBarsRequest(
                symbol_or_symbols=["BTC/USD"],
                timeframe=AlpacaTimeFrame.Day,
                start=datetime.now() - timedelta(days=2),
                end=datetime.now() - timedelta(days=1)
            )
            
            bars = self.crypto_client.get_crypto_bars(request)
            if hasattr(bars, 'df') and not bars.df.empty:
                logger.debug("Connection test successful")
            else:
                logger.warning("Connection test returned empty data")
                
        except Exception as e:
            raise ConnectionError(f"Connection test failed: {str(e)}")
    
    def _convert_timeframe(self, timeframe: TimeFrame) -> AlpacaTimeFrame:
        """
        Convert internal TimeFrame to Alpaca TimeFrame.
        
        Args:
            timeframe: Internal TimeFrame enum
            
        Returns:
            Alpaca TimeFrame object
        """
        mapping = {
            TimeFrame.MINUTE_1: AlpacaTimeFrame.Minute,
            TimeFrame.MINUTE_5: AlpacaTimeFrame(5, TimeFrameUnit.Minute),
            TimeFrame.MINUTE_15: AlpacaTimeFrame(15, TimeFrameUnit.Minute),
            TimeFrame.MINUTE_30: AlpacaTimeFrame(30, TimeFrameUnit.Minute),
            TimeFrame.HOUR_1: AlpacaTimeFrame.Hour,
            TimeFrame.HOUR_4: AlpacaTimeFrame(4, TimeFrameUnit.Hour),
            TimeFrame.DAILY: AlpacaTimeFrame.Day,
            TimeFrame.WEEKLY: AlpacaTimeFrame.Week,
            TimeFrame.MONTHLY: AlpacaTimeFrame.Month,
        }
        
        if timeframe not in mapping:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        return mapping[timeframe]
    
    def _alpaca_bar_to_ohlcv(self, bar, symbol: str) -> OHLCV:
        """
        Convert Alpaca bar data to OHLCV object.
        
        Args:
            bar: Alpaca bar data
            symbol: Trading symbol
            
        Returns:
            OHLCV object
        """
        return OHLCV(
            timestamp=bar.timestamp,
            open=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            volume=float(bar.volume) if bar.volume is not None else 0.0
        )
    
    async def get_historical_data(self, symbol: str, timeframe: TimeFrame,
                                 start_date: datetime, end_date: datetime,
                                 limit: Optional[int] = None) -> List[OHLCV]:
        """
        Retrieve historical OHLCV data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'BTC/USD')
            timeframe: Data timeframe
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            limit: Maximum number of records to return
            
        Returns:
            List of OHLCV data points
        """
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca Markets")
        
        try:
            alpaca_timeframe = self._convert_timeframe(timeframe)
            
            # Determine if this is a crypto symbol
            is_crypto = '/' in symbol
            
            if is_crypto:
                # Use crypto client
                request = CryptoBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=alpaca_timeframe,
                    start=start_date,
                    end=end_date,
                    limit=limit
                )
                
                bars = self.crypto_client.get_crypto_bars(request)
                
            else:
                # Use stock client
                request = StockBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=alpaca_timeframe,
                    start=start_date,
                    end=end_date,
                    limit=limit
                )
                
                bars = self.stock_client.get_stock_bars(request)
            
            # Convert to OHLCV objects
            ohlcv_data = []
            
            # Handle DataFrame response format
            if hasattr(bars, 'df') and not bars.df.empty:
                # Data is in DataFrame format
                df = bars.df
                
                # Filter for the specific symbol if multiple symbols in response
                if 'symbol' in df.index.names:
                    symbol_df = df.loc[symbol] if symbol in df.index.get_level_values('symbol') else df
                else:
                    symbol_df = df
                
                # Convert each row to OHLCV
                for timestamp, row in symbol_df.iterrows():
                    # Handle multi-index timestamp
                    if isinstance(timestamp, tuple):
                        timestamp = timestamp[-1]  # Get the actual timestamp
                    
                    ohlcv = OHLCV(
                        timestamp=timestamp,
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=float(row['volume']) if row['volume'] is not None else 0.0
                    )
                    ohlcv_data.append(ohlcv)
            
            # Legacy format handling (in case the API returns the old format)
            elif hasattr(bars, '__getitem__') and symbol in bars:
                for bar in bars[symbol]:
                    ohlcv_data.append(self._alpaca_bar_to_ohlcv(bar, symbol))
            
            logger.debug(f"Retrieved {len(ohlcv_data)} historical bars for {symbol}")
            return ohlcv_data
            
        except APIError as e:
            error_msg = f"Alpaca API error retrieving data for {symbol}: {str(e)}"
            self._set_error(error_msg)
            logger.error(error_msg)
            raise DataRetrievalError(error_msg)
        
        except Exception as e:
            error_msg = f"Error retrieving historical data for {symbol}: {str(e)}"
            self._set_error(error_msg)
            logger.error(error_msg)
            raise DataRetrievalError(error_msg)
    
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
        # Calculate start date based on timeframe and periods
        now = datetime.now()
        
        # Calculate timeframe duration in minutes
        timeframe_minutes = {
            TimeFrame.MINUTE_1: 1,
            TimeFrame.MINUTE_5: 5,
            TimeFrame.MINUTE_15: 15,
            TimeFrame.MINUTE_30: 30,
            TimeFrame.HOUR_1: 60,
            TimeFrame.HOUR_4: 240,
            TimeFrame.DAILY: 1440,
            TimeFrame.WEEKLY: 10080,
            TimeFrame.MONTHLY: 43200,
        }.get(timeframe, 60)
        
        # Add buffer to ensure we get enough data
        buffer_periods = max(10, periods * 2)
        start_date = now - timedelta(minutes=timeframe_minutes * buffer_periods)
        
        data = await self.get_historical_data(symbol, timeframe, start_date, now)
        
        # Return the latest N periods
        return data[-periods:] if len(data) >= periods else data
    
    async def get_symbols(self) -> List[str]:
        """
        Get list of available symbols from Alpaca.
        
        Returns:
            List of available trading symbols
        """
        # Note: Alpaca doesn't provide a simple "list all symbols" endpoint
        # In practice, you would maintain a list of symbols you're interested in
        # For now, return a default list of popular symbols
        return [
            # Popular stocks
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO',
            
            # Popular crypto
            'BTC/USD', 'ETH/USD', 'LTC/USD', 'BCH/USD', 'DOGE/USD'
        ]
    
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
        # Use default date range if not provided
        if end_date is None:
            end_date = datetime.now()
        
        if start_date is None:
            # Default to 100 periods of data
            start_date = end_date - timedelta(days=100)
        
        # Get historical data
        ohlcv_data = await self.get_historical_data(symbol, timeframe, start_date, end_date)
        
        # Create MarketData object
        market_data = MarketData(
            symbol=symbol,
            timeframe=timeframe,
            data=ohlcv_data,
            indicators={},  # Will be populated by indicator calculators
            last_updated=datetime.now()
        )
        
        return market_data
    
    def get_supported_timeframes(self) -> List[TimeFrame]:
        """Get list of supported timeframes."""
        return [
            TimeFrame.MINUTE_1,
            TimeFrame.MINUTE_5,
            TimeFrame.MINUTE_15,
            TimeFrame.MINUTE_30,
            TimeFrame.HOUR_1,
            TimeFrame.HOUR_4,
            TimeFrame.DAILY,
            TimeFrame.WEEKLY,
            TimeFrame.MONTHLY
        ]
    
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols."""
        # This would typically query Alpaca's assets endpoint
        # For now, return the same list as get_symbols
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO',
            'BTC/USD', 'ETH/USD', 'LTC/USD', 'BCH/USD', 'DOGE/USD'
        ]
    
    # Real-time streaming methods
    
    async def start_streaming(self, symbols: List[str], 
                            on_bar: Optional[Callable] = None,
                            on_trade: Optional[Callable] = None,
                            on_quote: Optional[Callable] = None) -> None:
        """
        Start real-time data streaming for specified symbols.
        
        Args:
            symbols: List of symbols to stream
            on_bar: Callback for bar/candle data
            on_trade: Callback for trade data
            on_quote: Callback for quote data
        """
        if not self.stock_stream:
            raise ConnectionError("Streaming requires authenticated connection")
        
        try:
            # Separate stock and crypto symbols
            stock_symbols = [s for s in symbols if '/' not in s]
            crypto_symbols = [s for s in symbols if '/' in s]
            
            # Set up stock streaming
            if stock_symbols:
                if on_bar:
                    self.stock_stream.subscribe_bars(self._wrap_bar_handler(on_bar), *stock_symbols)
                
                if on_trade:
                    self.stock_stream.subscribe_trades(self._wrap_trade_handler(on_trade), *stock_symbols)
                
                if on_quote:
                    self.stock_stream.subscribe_quotes(self._wrap_quote_handler(on_quote), *stock_symbols)
            
            # Set up crypto streaming
            if crypto_symbols and self.crypto_stream:
                if on_bar:
                    self.crypto_stream.subscribe_bars(self._wrap_bar_handler(on_bar), *crypto_symbols)
                
                if on_trade:
                    self.crypto_stream.subscribe_trades(self._wrap_trade_handler(on_trade), *crypto_symbols)
                
                if on_quote:
                    self.crypto_stream.subscribe_quotes(self._wrap_quote_handler(on_quote), *crypto_symbols)
            
            # Start streaming
            if stock_symbols:
                asyncio.create_task(self.stock_stream.run())
            
            if crypto_symbols and self.crypto_stream:
                asyncio.create_task(self.crypto_stream.run())
            
            self._stream_connected = True
            logger.info(f"Started streaming for {len(symbols)} symbols")
            
        except Exception as e:
            error_msg = f"Error starting stream: {str(e)}"
            self._set_error(error_msg)
            logger.error(error_msg)
            raise ConnectionError(error_msg)
    
    def _wrap_bar_handler(self, handler: Callable) -> Callable:
        """Wrap bar handler to convert Alpaca format to internal format."""
        async def wrapped_handler(bar):
            try:
                ohlcv = self._alpaca_bar_to_ohlcv(bar, bar.symbol)
                await handler(ohlcv)
            except Exception as e:
                logger.error(f"Error in bar handler: {e}")
        
        return wrapped_handler
    
    def _wrap_trade_handler(self, handler: Callable) -> Callable:
        """Wrap trade handler to convert Alpaca format to internal format."""
        async def wrapped_handler(trade):
            try:
                trade_data = {
                    'symbol': trade.symbol,
                    'price': float(trade.price),
                    'size': float(trade.size),
                    'timestamp': trade.timestamp,
                    'exchange': getattr(trade, 'exchange', None)
                }
                await handler(trade_data)
            except Exception as e:
                logger.error(f"Error in trade handler: {e}")
        
        return wrapped_handler
    
    def _wrap_quote_handler(self, handler: Callable) -> Callable:
        """Wrap quote handler to convert Alpaca format to internal format."""
        async def wrapped_handler(quote):
            try:
                quote_data = {
                    'symbol': quote.symbol,
                    'bid_price': float(quote.bid_price) if hasattr(quote, 'bid_price') else None,
                    'ask_price': float(quote.ask_price) if hasattr(quote, 'ask_price') else None,
                    'bid_size': float(quote.bid_size) if hasattr(quote, 'bid_size') else None,
                    'ask_size': float(quote.ask_size) if hasattr(quote, 'ask_size') else None,
                    'timestamp': quote.timestamp
                }
                await handler(quote_data)
            except Exception as e:
                logger.error(f"Error in quote handler: {e}")
        
        return wrapped_handler
    
    async def stop_streaming(self) -> None:
        """Stop all real-time data streaming."""
        try:
            if self.stock_stream and self._stream_connected:
                await self.stock_stream.stop_ws()
            
            if self.crypto_stream and self._stream_connected:
                await self.crypto_stream.stop_ws()
            
            self._stream_connected = False
            logger.info("Stopped data streaming")
            
        except Exception as e:
            error_msg = f"Error stopping stream: {str(e)}"
            logger.error(error_msg)
            self._set_error(error_msg)
    
    def is_streaming(self) -> bool:
        """Check if streaming is active."""
        return self._stream_connected
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get detailed provider information."""
        info = super().get_provider_info()
        info.update({
            'feed': self.feed,
            'crypto_feed': self.crypto_feed,
            'paper_trading': self.paper,
            'streaming_active': self._stream_connected,
            'supported_assets': ['stocks', 'crypto'],
            'real_time_data': True,
            'historical_data': True
        })
        return info