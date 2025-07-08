"""
Alpaca Real-time Data Ingestion Service

This module provides real-time market data ingestion from Alpaca Markets
WebSocket streams, publishing events to Redis Streams for consumption by
the calculation engine and other services.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from src.data.providers.alpaca_provider import AlpacaProvider
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

from src.core.models import OHLCV, TimeFrame
from src.realtime.config_manager import ConfigManager, AssetConfig

logger = logging.getLogger(__name__)


@dataclass
class AlpacaMarketEvent:
    """Market data event from Alpaca for Redis Streams."""
    event_id: str
    symbol: str
    timestamp: datetime
    event_type: str  # 'bar', 'trade', 'quote'
    data: Dict[str, Any]
    source: str = "alpaca"


class CircuitBreakerState:
    """Circuit breaker state management."""
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

    def can_execute(self) -> bool:
        """Check if operation can be executed."""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
                logger.info("Circuit breaker moved to half-open state")
                return True
            return False
        
        # half-open state
        return True


class AlpacaIngestionService:
    """
    Real-time data ingestion service for Alpaca Markets.
    
    This service connects to Alpaca's WebSocket streams and publishes
    market data events to Redis Streams for consumption by other services.
    
    Features:
    - WebSocket connection management with auto-reconnection
    - Circuit breaker pattern for resilience
    - Rate limiting and backpressure handling
    - Multiple data types (bars, trades, quotes)
    - Configurable symbol subscriptions
    """
    
    def __init__(self, config_manager: ConfigManager, redis_config: Optional[Dict[str, Any]] = None):
        """
        Initialize Alpaca ingestion service.
        
        Args:
            config_manager: Configuration manager for assets and settings
            redis_config: Redis connection configuration
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is required for AlpacaIngestionService")
        
        if not ALPACA_AVAILABLE:
            raise ImportError("AlpacaProvider is required for AlpacaIngestionService")
        
        self.config_manager = config_manager
        self.redis_config = redis_config or {}
        
        # Redis connection
        self.redis_client = None
        self.stream_names = {
            'bars': 'alpaca:market:bars',
            'trades': 'alpaca:market:trades',
            'quotes': 'alpaca:market:quotes'
        }
        
        # Alpaca provider
        self.alpaca_provider = None
        
        # State management
        self.running = False
        self.connected = False
        self.subscribed_symbols: Set[str] = set()
        
        # Circuit breaker for resilience
        self.circuit_breaker = CircuitBreakerState()
        
        # Performance metrics
        self.metrics = {
            'events_processed': 0,
            'events_per_second': 0,
            'last_event_time': None,
            'connection_attempts': 0,
            'successful_connections': 0,
            'stream_errors': 0
        }
        
        # Rate limiting
        self.max_events_per_second = 1000
        self.event_times = []
        
        logger.info("Initialized AlpacaIngestionService")
    
    async def start(self) -> None:
        """Start the Alpaca ingestion service."""
        try:
            logger.info("Starting Alpaca ingestion service...")
            
            # Initialize Redis connection
            await self._initialize_redis()
            
            # Initialize Alpaca provider
            await self._initialize_alpaca()
            
            # Start data ingestion
            await self._start_data_ingestion()
            
            self.running = True
            logger.info("Alpaca ingestion service started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Alpaca ingestion service: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the Alpaca ingestion service."""
        logger.info("Stopping Alpaca ingestion service...")
        
        self.running = False
        
        try:
            # Stop Alpaca streaming
            if self.alpaca_provider and self.alpaca_provider.is_streaming():
                await self.alpaca_provider.stop_streaming()
            
            # Disconnect from Alpaca
            if self.alpaca_provider and self.alpaca_provider.is_connected():
                await self.alpaca_provider.disconnect()
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.aclose()
            
            self.connected = False
            logger.info("Alpaca ingestion service stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Alpaca ingestion service: {e}")
    
    async def _initialize_redis(self) -> None:
        """Initialize Redis connection and create streams."""
        try:
            # Create Redis connection
            redis_host = self.redis_config.get('host', 'localhost')
            redis_port = self.redis_config.get('port', 6379)
            redis_db = self.redis_config.get('db', 0)
            
            self.redis_client = redis.asyncio.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            
            # Create streams if they don't exist
            for stream_name in self.stream_names.values():
                try:
                    # Create stream with initial dummy message
                    await self.redis_client.xadd(
                        stream_name,
                        {'_init': 'true', 'timestamp': datetime.now().isoformat()},
                        maxlen=10000  # Keep last 10k events
                    )
                    logger.debug(f"Created Redis stream: {stream_name}")
                except redis.ResponseError:
                    # Stream already exists
                    pass
            
            logger.info("Redis connection established")
            
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Redis: {e}")
    
    async def _initialize_alpaca(self) -> None:
        """Initialize Alpaca provider and establish connection."""
        try:
            # Get Alpaca configuration
            alpaca_config = {
                'paper': True,  # Start with paper trading
                'feed': 'iex',  # Start with free IEX feed
                'max_retries': 3,
                'retry_delay': 1.0
            }
            
            # Create Alpaca provider
            self.alpaca_provider = AlpacaProvider(alpaca_config)
            
            # Connect to Alpaca
            self.metrics['connection_attempts'] += 1
            
            if not self.circuit_breaker.can_execute():
                raise ConnectionError("Circuit breaker is open")
            
            success = await self.alpaca_provider.connect()
            
            if success:
                self.connected = True
                self.circuit_breaker.record_success()
                self.metrics['successful_connections'] += 1
                logger.info("Connected to Alpaca Markets")
            else:
                self.circuit_breaker.record_failure()
                raise ConnectionError("Failed to connect to Alpaca Markets")
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise ConnectionError(f"Failed to initialize Alpaca: {e}")
    
    async def _start_data_ingestion(self) -> None:
        """Start real-time data ingestion from Alpaca."""
        try:
            # Get symbols to subscribe to
            symbols = await self._get_subscription_symbols()
            
            if not symbols:
                logger.warning("No symbols configured for subscription")
                return
            
            # Start WebSocket streaming
            await self.alpaca_provider.start_streaming(
                symbols=symbols,
                on_bar=self._handle_bar_data,
                on_trade=self._handle_trade_data,
                on_quote=self._handle_quote_data
            )
            
            self.subscribed_symbols = set(symbols)
            logger.info(f"Started streaming for {len(symbols)} symbols: {symbols}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to start data ingestion: {e}")
    
    async def _get_subscription_symbols(self) -> List[str]:
        """Get list of symbols to subscribe to from configuration."""
        try:
            # Get enabled assets from config manager
            asset_configs = self.config_manager.get_watchlist()
            
            symbols = []
            for asset_config in asset_configs:
                if asset_config.enabled:
                    symbols.append(asset_config.symbol)
            
            # If no configured symbols, get from environment or use SPY as minimal default
            if not symbols:
                import os
                default_symbols_str = os.environ.get('DEFAULT_SYMBOLS', 'SPY')
                symbols = [s.strip() for s in default_symbols_str.split(',') if s.strip()]
                logger.warning(f"No symbols configured in watchlist, using environment default: {symbols}")
                logger.warning("Configure symbols using WATCHLIST_SYMBOLS environment variable")
            
            return symbols
            
        except Exception as e:
            logger.error(f"Error getting subscription symbols: {e}")
            return []
    
    async def _handle_bar_data(self, ohlcv: OHLCV) -> None:
        """Handle incoming bar/candle data."""
        try:
            if not await self._rate_limit_check():
                return
            
            event = AlpacaMarketEvent(
                event_id=f"bar_{ohlcv.symbol}_{int(ohlcv.timestamp.timestamp())}",
                symbol=ohlcv.symbol,
                timestamp=ohlcv.timestamp,
                event_type='bar',
                data={
                    'open': ohlcv.open,
                    'high': ohlcv.high,
                    'low': ohlcv.low,
                    'close': ohlcv.close,
                    'volume': ohlcv.volume
                }
            )
            
            await self._publish_event(self.stream_names['bars'], event)
            await self._update_metrics()
            
        except Exception as e:
            logger.error(f"Error handling bar data: {e}")
            self.metrics['stream_errors'] += 1
    
    async def _handle_trade_data(self, trade_data: Dict[str, Any]) -> None:
        """Handle incoming trade data."""
        try:
            if not await self._rate_limit_check():
                return
            
            event = AlpacaMarketEvent(
                event_id=f"trade_{trade_data['symbol']}_{int(time.time() * 1000000)}",
                symbol=trade_data['symbol'],
                timestamp=trade_data['timestamp'],
                event_type='trade',
                data={
                    'price': trade_data['price'],
                    'size': trade_data['size'],
                    'exchange': trade_data.get('exchange', 'unknown')
                }
            )
            
            await self._publish_event(self.stream_names['trades'], event)
            await self._update_metrics()
            
        except Exception as e:
            logger.error(f"Error handling trade data: {e}")
            self.metrics['stream_errors'] += 1
    
    async def _handle_quote_data(self, quote_data: Dict[str, Any]) -> None:
        """Handle incoming quote data."""
        try:
            if not await self._rate_limit_check():
                return
            
            event = AlpacaMarketEvent(
                event_id=f"quote_{quote_data['symbol']}_{int(time.time() * 1000000)}",
                symbol=quote_data['symbol'],
                timestamp=quote_data['timestamp'],
                event_type='quote',
                data={
                    'bid_price': quote_data.get('bid_price'),
                    'ask_price': quote_data.get('ask_price'),
                    'bid_size': quote_data.get('bid_size'),
                    'ask_size': quote_data.get('ask_size')
                }
            )
            
            await self._publish_event(self.stream_names['quotes'], event)
            await self._update_metrics()
            
        except Exception as e:
            logger.error(f"Error handling quote data: {e}")
            self.metrics['stream_errors'] += 1
    
    async def _publish_event(self, stream_name: str, event: AlpacaMarketEvent) -> None:
        """Publish event to Redis Stream."""
        try:
            # Convert event to dictionary
            event_data = {
                'event_id': event.event_id,
                'symbol': event.symbol,
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type,
                'source': event.source,
                'data': json.dumps(event.data)
            }
            
            # Publish to Redis Stream
            message_id = await self.redis_client.xadd(
                stream_name,
                event_data,
                maxlen=10000  # Keep last 10k events
            )
            
            logger.debug(f"Published event {event.event_id} to {stream_name}: {message_id}")
            
        except Exception as e:
            logger.error(f"Error publishing event to Redis: {e}")
            raise
    
    async def _rate_limit_check(self) -> bool:
        """Check if we're within rate limits."""
        current_time = time.time()
        
        # Remove events older than 1 second
        self.event_times = [t for t in self.event_times if current_time - t < 1.0]
        
        # Check if we're at the limit
        if len(self.event_times) >= self.max_events_per_second:
            logger.warning("Rate limit exceeded, dropping event")
            return False
        
        # Add current event time
        self.event_times.append(current_time)
        return True
    
    async def _update_metrics(self) -> None:
        """Update performance metrics."""
        self.metrics['events_processed'] += 1
        self.metrics['last_event_time'] = datetime.now()
        
        # Calculate events per second
        current_time = time.time()
        recent_events = [t for t in self.event_times if current_time - t < 1.0]
        self.metrics['events_per_second'] = len(recent_events)
    
    async def add_symbol(self, symbol: str) -> None:
        """Add a symbol to the subscription list."""
        if symbol not in self.subscribed_symbols:
            logger.info(f"Adding symbol to subscription: {symbol}")
            # Note: In practice, this would require restarting the stream
            # or using Alpaca's dynamic subscription features
            self.subscribed_symbols.add(symbol)
    
    async def remove_symbol(self, symbol: str) -> None:
        """Remove a symbol from the subscription list."""
        if symbol in self.subscribed_symbols:
            logger.info(f"Removing symbol from subscription: {symbol}")
            self.subscribed_symbols.discard(symbol)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            **self.metrics,
            'subscribed_symbols': list(self.subscribed_symbols),
            'running': self.running,
            'connected': self.connected,
            'circuit_breaker_state': self.circuit_breaker.state,
            'redis_streams': self.stream_names
        }
    
    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        return (
            self.running and
            self.connected and
            self.alpaca_provider and
            self.alpaca_provider.is_connected() and
            self.circuit_breaker.state != "open"
        )
    
    async def reconnect(self) -> None:
        """Attempt to reconnect to Alpaca and restart streaming."""
        logger.info("Attempting to reconnect to Alpaca...")
        
        try:
            # Stop current streaming
            if self.alpaca_provider and self.alpaca_provider.is_streaming():
                await self.alpaca_provider.stop_streaming()
            
            # Disconnect
            if self.alpaca_provider and self.alpaca_provider.is_connected():
                await self.alpaca_provider.disconnect()
            
            # Wait a bit before reconnecting
            await asyncio.sleep(5)
            
            # Reconnect
            await self._initialize_alpaca()
            await self._start_data_ingestion()
            
            logger.info("Reconnection successful")
            
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            self.circuit_breaker.record_failure()
            raise


# Singleton instance for global access
_alpaca_ingestion_service: Optional[AlpacaIngestionService] = None


async def get_alpaca_ingestion_service(config_manager: ConfigManager, 
                                     redis_config: Optional[Dict[str, Any]] = None) -> AlpacaIngestionService:
    """
    Get or create the Alpaca ingestion service singleton.
    
    Args:
        config_manager: Configuration manager instance
        redis_config: Redis configuration
        
    Returns:
        AlpacaIngestionService instance
    """
    global _alpaca_ingestion_service
    
    if _alpaca_ingestion_service is None:
        _alpaca_ingestion_service = AlpacaIngestionService(config_manager, redis_config)
    
    return _alpaca_ingestion_service