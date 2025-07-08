"""
Market Data Ingestion Service

This module provides real-time market data ingestion using Redis Streams
for high-performance event-driven data processing. It fetches market data
from various providers and publishes to Redis Streams for consumption by
calculation engines.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import threading

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from src.storage.cache import cache_manager
from src.core.models import OHLCV
from src.realtime.config_manager import ConfigManager

# Check if alpaca-py is available (defer AlpacaProvider import)
try:
    import alpaca.data.historical

    ALPACA_PROVIDER_AVAILABLE = True
except ImportError:
    ALPACA_PROVIDER_AVAILABLE = False

logger = logging.getLogger(__name__)


class MinimalAlpacaProvider:
    """Minimal Alpaca provider implementation for data ingestion."""

    def __init__(self, config):
        """Initialize minimal Alpaca provider - PRODUCTION MODE."""
        self.name = "Alpaca"
        self.config = config
        self._connected = False
        self.last_error = None

        # Validate required credentials
        if not config.get("api_key") or not config.get("secret_key"):
            raise ValueError(
                "Alpaca API credentials are MANDATORY for production operation. "
                "Provide api_key and secret_key in config."
            )

        # Initialize clients
        self.stock_client = None

        logger.info(
            f"Initializing Alpaca provider (paper={config.get('paper', True)}, "
            f"feed={config.get('feed', 'iex')})"
        )

    async def connect(self):
        """Connect to Alpaca - REQUIRED for production."""
        try:
            from alpaca.data.historical import StockHistoricalDataClient

            logger.info("Connecting to Alpaca Markets...")
            self.stock_client = StockHistoricalDataClient(
                api_key=self.config["api_key"], secret_key=self.config["secret_key"]
            )

            # Test connection - MANDATORY validation
            await self._test_connection()
            self._connected = True
            logger.info("✅ Successfully connected to Alpaca Markets")
            return True

        except Exception as e:
            self.last_error = str(e)
            logger.error(f"❌ CRITICAL: Failed to connect to Alpaca Markets: {e}")
            raise RuntimeError(
                f"Cannot establish connection to Alpaca Markets. "
                f"Check credentials and network connectivity: {e}"
            )

    async def disconnect(self):
        """Disconnect from Alpaca."""
        self._connected = False

    async def _test_connection(self):
        """Test connection with a simple request."""
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame as AlpacaTimeFrame
            from datetime import timedelta
            import pytz

            # Test with a simple stock request - use older data for IEX feed
            # IEX data has delays, so use data from at least 15 minutes ago
            utc = pytz.UTC
            end_date = datetime.now(utc) - timedelta(days=2)  # 2 days old for IEX
            start_date = end_date - timedelta(days=5)  # 5 days of data

            request = StockBarsRequest(
                symbol_or_symbols=["AAPL"],
                timeframe=AlpacaTimeFrame.Day,
                start=start_date,
                end=end_date,
            )
            self.stock_client.get_stock_bars(request)
            return True

        except Exception as e:
            raise Exception(f"Connection test failed: {str(e)}")

    def get_latest_data(self, symbol):
        """Get latest data for symbol."""
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame as AlpacaTimeFrame
            from datetime import timedelta

            if not self._connected:
                return None

            # Get recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)

            request = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=AlpacaTimeFrame.Day,
                start=start_date,
                end=end_date,
            )

            bars = self.stock_client.get_stock_bars(request)

            # Convert to OHLCV format
            if hasattr(bars, "df") and not bars.df.empty:
                df = bars.df

                # Get the latest row
                latest_row = df.iloc[-1]
                latest_timestamp = df.index[-1]

                # Handle multi-index timestamp
                if isinstance(latest_timestamp, tuple):
                    latest_timestamp = latest_timestamp[-1]

                ohlcv = OHLCV(
                    timestamp=latest_timestamp,
                    open=float(latest_row["open"]),
                    high=float(latest_row["high"]),
                    low=float(latest_row["low"]),
                    close=float(latest_row["close"]),
                    volume=(
                        float(latest_row["volume"])
                        if latest_row["volume"] is not None
                        else 0.0
                    ),
                )
                return ohlcv

            return None

        except Exception as e:
            logger.error(f"Error getting latest data for {symbol}: {e}")
            return None


@dataclass
class MarketDataEvent:
    """Market data event for Redis Streams."""

    event_id: str
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    source: str
    metadata: Dict[str, Any]


@dataclass
class StreamConfig:
    """Configuration for Redis Streams."""

    stream_name: str
    max_len: int
    consumer_group: str
    consumer_name: str


class CircuitBreaker:
    """Circuit breaker for external data provider calls."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time < self.recovery_timeout:
                    raise Exception("Circuit breaker is OPEN")
                else:
                    self.state = "HALF_OPEN"

            try:
                result = func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.reset()
                return result
            except Exception as e:
                self._record_failure()
                raise e

    def _record_failure(self):
        """Record a failure and potentially open the circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )

    def reset(self):
        """Reset the circuit breaker."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"


class MarketDataIngestionService:
    """
    High-performance market data ingestion service using Redis Streams.

    This service fetches real-time market data from configured providers
    and publishes events to Redis Streams for consumption by calculation
    engines and other downstream services.
    """

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the market data ingestion service.

        Args:
            config_manager: Configuration manager instance
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis package required for streaming. Install with: pip install redis"
            )

        self.config_manager = config_manager
        self._redis_client = None
        self._running = False
        self._tasks = []
        self._circuit_breakers = {}

        # Stream configuration
        self.streams = {
            "market_data": StreamConfig(
                stream_name="market_data_stream",
                max_len=10000,
                consumer_group="data_processors",
                consumer_name="ingestion_service",
            ),
            "calculation_events": StreamConfig(
                stream_name="calculation_events_stream",
                max_len=5000,
                consumer_group="calculation_consumers",
                consumer_name="calc_engine",
            ),
            "alerts": StreamConfig(
                stream_name="alert_stream",
                max_len=1000,
                consumer_group="alert_processors",
                consumer_name="alert_service",
            ),
        }

        # Data providers - PRODUCTION MODE: Alpaca only
        self._providers = {}

        # Initialize Alpaca provider - REQUIRED for production
        if not ALPACA_PROVIDER_AVAILABLE:
            raise ImportError(
                "alpaca-py is required for production operation. "
                "Install with: pip install alpaca-py"
            )

        try:
            # Create Alpaca provider - MANDATORY for production
            alpaca_provider = self._create_alpaca_provider()
            if alpaca_provider:
                self._providers["alpaca"] = alpaca_provider
                logger.info("Alpaca provider initialized successfully")
            else:
                raise ValueError(
                    "Alpaca credentials are REQUIRED for production operation. "
                    "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
                )

        except Exception as e:
            logger.error(f"CRITICAL: Failed to initialize Alpaca provider: {e}")
            raise RuntimeError(
                f"Cannot start production service without valid Alpaca credentials: {e}"
            )

        # Performance tracking
        self._stats = {
            "events_published": 0,
            "errors": 0,
            "last_publish_time": None,
            "provider_calls": {},
            "circuit_breaker_trips": 0,
        }

        self._initialize_redis()
        self._initialize_streams()

    def _create_alpaca_provider(self):
        """Create Alpaca provider - REQUIRED for production."""
        try:
            import os

            # Get configuration - MANDATORY
            api_key = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_SECRET_KEY")

            if not api_key or not secret_key:
                raise ValueError(
                    "ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables are REQUIRED. "
                    "Cannot operate in production without valid Alpaca credentials."
                )

            config = {
                "api_key": api_key,
                "secret_key": secret_key,
                "paper": os.getenv("ALPACA_PAPER_TRADING", "true").lower() == "true",
                "feed": os.getenv("ALPACA_DATA_FEED", "iex"),
                "max_retries": int(os.getenv("ALPACA_MAX_RETRIES", "3")),
                "retry_delay": float(os.getenv("ALPACA_RETRY_DELAY", "1.0")),
            }

            # Create and validate Alpaca provider
            provider = MinimalAlpacaProvider(config)
            logger.info(
                f"Created Alpaca provider with feed: {config['feed']}, paper: {config['paper']}"
            )
            return provider

        except Exception as e:
            logger.error(f"CRITICAL: Cannot create Alpaca provider: {e}")
            raise

    def _initialize_redis(self):
        """Initialize Redis connection for streams."""
        try:
            from src.core.constants import (
                REDIS_HOST,
                REDIS_PORT,
                REDIS_DB,
                REDIS_PASSWORD,
            )

            self._redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=False,  # We'll handle encoding ourselves
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                max_connections=20,
            )

            # Test connection
            self._redis_client.ping()
            logger.info("Redis connection initialized for streaming")

        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {e}")
            raise

    def _initialize_streams(self):
        """Initialize Redis Streams and consumer groups."""
        try:
            for stream_config in self.streams.values():
                try:
                    # Create consumer group (idempotent operation)
                    self._redis_client.xgroup_create(
                        stream_config.stream_name,
                        stream_config.consumer_group,
                        id="0",
                        mkstream=True,
                    )
                    logger.info(f"Initialized stream: {stream_config.stream_name}")
                except redis.exceptions.ResponseError as e:
                    if "BUSYGROUP" in str(e):
                        # Consumer group already exists
                        logger.debug(
                            f"Consumer group already exists for {stream_config.stream_name}"
                        )
                    else:
                        raise e

        except Exception as e:
            logger.error(f"Failed to initialize streams: {e}")
            raise

    def add_provider(self, name: str, provider):
        """
        Add a data provider.

        Args:
            name: Provider name
            provider: Provider instance with get_latest_data method
        """
        self._providers[name] = provider
        self._circuit_breakers[name] = CircuitBreaker()
        logger.info(f"Added data provider: {name}")

    async def start(self):
        """Start the market data ingestion service."""
        if self._running:
            logger.warning("Service is already running")
            return

        self._running = True
        logger.info("Starting market data ingestion service")

        try:
            # Connect all providers - MANDATORY for production
            connected_providers = 0
            for provider_name, provider in self._providers.items():
                try:
                    if hasattr(provider, "connect"):
                        await provider.connect()  # This will raise exception if fails
                        connected_providers += 1
                        logger.info(f"✅ Connected to {provider_name} provider")
                    else:
                        raise RuntimeError(
                            f"Provider {provider_name} does not support connection"
                        )
                except Exception as e:
                    logger.error(
                        f"❌ CRITICAL: Failed to connect to {provider_name} provider: {e}"
                    )
                    raise RuntimeError(
                        f"Cannot start service without {provider_name} connectivity: {e}"
                    )

            if connected_providers == 0:
                raise RuntimeError(
                    "No data providers connected - cannot start production service"
                )

            # Get system configuration
            system_config = self.config_manager.get_system_config()
            if not system_config:
                raise Exception("System configuration not found")

            # Start ingestion tasks for each enabled symbol
            enabled_symbols = self.config_manager.get_enabled_symbols()

            if not enabled_symbols:
                logger.warning("No enabled symbols found in watchlist")
                return

            logger.info(
                f"Starting ingestion for {len(enabled_symbols)} symbols: {enabled_symbols}"
            )

            # Create tasks for each symbol
            for symbol in enabled_symbols:
                task = asyncio.create_task(
                    self._ingest_symbol_data(
                        symbol, system_config.data_update_interval_seconds
                    )
                )
                self._tasks.append(task)

            # Start monitoring task
            monitor_task = asyncio.create_task(self._monitor_performance())
            self._tasks.append(monitor_task)

            # Wait for all tasks
            await asyncio.gather(*self._tasks)

        except Exception as e:
            logger.error(f"Error in ingestion service: {e}")
            raise
        finally:
            self._running = False

    async def stop(self):
        """Stop the market data ingestion service."""
        if not self._running:
            return

        logger.info("Stopping market data ingestion service")
        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        logger.info("Market data ingestion service stopped")

    async def _ingest_symbol_data(self, symbol: str, interval_seconds: int):
        """
        Continuously ingest data for a specific symbol.

        Args:
            symbol: Trading symbol
            interval_seconds: Update interval in seconds
        """
        logger.info(
            f"Started data ingestion for {symbol} (interval: {interval_seconds}s)"
        )

        while self._running:
            try:
                # Get asset configuration
                watchlist = self.config_manager.get_watchlist()
                asset_config = next((a for a in watchlist if a.symbol == symbol), None)

                if not asset_config or not asset_config.enabled:
                    logger.warning(
                        f"Symbol {symbol} not found or disabled in watchlist"
                    )
                    await asyncio.sleep(interval_seconds)
                    continue

                # Get data provider
                provider = self._providers.get(asset_config.data_provider)
                if not provider:
                    logger.error(
                        f"Data provider '{asset_config.data_provider}' not found for {symbol}"
                    )
                    await asyncio.sleep(interval_seconds)
                    continue

                # Fetch latest data with circuit breaker protection
                circuit_breaker = self._circuit_breakers.get(asset_config.data_provider)
                if not circuit_breaker:
                    circuit_breaker = CircuitBreaker()
                    self._circuit_breakers[asset_config.data_provider] = circuit_breaker

                try:
                    latest_data = await self._fetch_data_with_circuit_breaker(
                        circuit_breaker, provider, symbol
                    )

                    if latest_data:
                        await self._publish_market_data_event(
                            symbol, latest_data, asset_config.data_provider
                        )
                        self._stats["events_published"] += 1
                        self._stats["last_publish_time"] = datetime.now()

                        # Update provider call stats
                        provider_name = asset_config.data_provider
                        if provider_name not in self._stats["provider_calls"]:
                            self._stats["provider_calls"][provider_name] = 0
                        self._stats["provider_calls"][provider_name] += 1

                except Exception as e:
                    if "Circuit breaker is OPEN" in str(e):
                        self._stats["circuit_breaker_trips"] += 1
                        logger.warning(
                            f"Circuit breaker open for provider {asset_config.data_provider}"
                        )
                    else:
                        self._stats["errors"] += 1
                        logger.error(f"Error fetching data for {symbol}: {e}")

                # Wait for next interval
                await asyncio.sleep(interval_seconds)

            except asyncio.CancelledError:
                logger.info(f"Data ingestion cancelled for {symbol}")
                break
            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"Unexpected error in data ingestion for {symbol}: {e}")
                await asyncio.sleep(interval_seconds)

    async def _fetch_data_with_circuit_breaker(
        self, circuit_breaker: CircuitBreaker, provider, symbol: str
    ) -> Optional[OHLCV]:
        """
        Fetch data with circuit breaker protection.

        Args:
            circuit_breaker: Circuit breaker instance
            provider: Data provider instance
            symbol: Trading symbol

        Returns:
            Latest OHLCV data or None if failed
        """
        try:
            # Run in thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = loop.run_in_executor(
                    executor,
                    lambda: circuit_breaker.call(provider.get_latest_data, symbol),
                )
                return await future
        except Exception as e:
            logger.debug(f"Circuit breaker call failed for {symbol}: {e}")
            return None

    async def _publish_market_data_event(self, symbol: str, data: OHLCV, source: str):
        """
        Publish market data event to Redis Stream.

        Args:
            symbol: Trading symbol
            data: OHLCV data
            source: Data source name
        """
        try:
            # Create event
            event = MarketDataEvent(
                event_id=f"{symbol}_{int(time.time() * 1000)}",
                symbol=symbol,
                timestamp=data.timestamp,
                price=data.close,  # Use close as current price
                volume=data.volume,
                open_price=data.open,
                high_price=data.high,
                low_price=data.low,
                close_price=data.close,
                source=source,
                metadata={"timeframe": "1m"},  # Default to 1-minute data
            )

            # Convert to Redis Stream format
            event_data = {
                "event_id": event.event_id,
                "symbol": event.symbol,
                "timestamp": event.timestamp.isoformat(),
                "price": str(event.price),
                "volume": str(event.volume),
                "open": str(event.open_price),
                "high": str(event.high_price),
                "low": str(event.low_price),
                "close": str(event.close_price),
                "source": event.source,
                "metadata": json.dumps(event.metadata),
            }

            # Publish to Redis Stream
            stream_config = self.streams["market_data"]
            message_id = self._redis_client.xadd(
                stream_config.stream_name,
                event_data,
                maxlen=stream_config.max_len,
                approximate=True,
            )

            logger.debug(f"Published market data event for {symbol}: {message_id}")

            # Store latest symbol values in Redis
            await self._store_symbol_values(symbol, data, source)

            # Also cache latest data for quick access
            cache_key = f"latest_price_{symbol}"
            cache_manager.set("market_data", cache_key, asdict(event), ttl=60)

        except Exception as e:
            logger.error(f"Failed to publish market data event for {symbol}: {e}")
            raise

    async def _store_symbol_values(self, symbol: str, data: OHLCV, source: str):
        """
        Store latest symbol values in Redis for quick access.

        Args:
            symbol: Trading symbol
            data: OHLCV data
            source: Data source name
        """
        try:
            # Check if symbol storage is enabled
            if not self._is_symbol_storage_enabled():
                return

            current_time = datetime.now()

            # Store latest values as JSON (more memory efficient than hash)
            latest_key = f"symbol:latest:{symbol}"
            latest_data = {
                "timestamp": data.timestamp.isoformat(),
                "open": data.open,
                "high": data.high,
                "low": data.low,
                "close": data.close,
                "volume": data.volume,
                "source": source,
                "updated_at": current_time.isoformat(),
            }

            # Atomic symbol storage using Redis transaction
            # This prevents race conditions when multiple instances update the same symbol
            history_key = f"symbol:history:{symbol}"
            history_value = json.dumps(
                {"price": data.close, "volume": data.volume, "source": source}
            )
            timestamp_score = int(data.timestamp.timestamp())
            max_history_size = self._get_symbol_history_max_size()
            history_ttl = self._get_symbol_history_ttl()

            # Use Redis pipeline with MULTI/EXEC for atomic operations
            pipe = self._redis_client.pipeline()
            pipe.multi()

            # Store latest values as JSON
            pipe.set(latest_key, json.dumps(latest_data))

            # Add to price history sorted set
            pipe.zadd(history_key, {history_value: timestamp_score})

            # Trim history to max size if configured
            if max_history_size > 0:
                pipe.zremrangebyrank(history_key, 0, -(max_history_size + 1))

            # Set TTL for history if configured
            if history_ttl > 0:
                pipe.expire(history_key, history_ttl)

            # Execute all operations atomically
            pipe.execute()

            # Update symbol metadata
            await self._update_symbol_metadata(symbol, current_time)

            logger.debug(f"Stored symbol values for {symbol} in Redis")

        except Exception as e:
            logger.error(f"Failed to store symbol values for {symbol}: {e}")
            # Don't raise - symbol storage is optional

    async def _update_symbol_metadata(self, symbol: str, current_time: datetime):
        """Update symbol metadata in Redis using atomic operation."""
        try:
            meta_key = f"symbol:meta:{symbol}"
            current_time_iso = current_time.isoformat()

            # Use Lua script for atomic read-modify-write operation
            # This prevents race conditions when multiple instances update the same symbol
            lua_script = """
            local key = KEYS[1]
            local current_time = ARGV[1]
            
            local existing = redis.call('GET', key)
            if not existing then
                -- First time seeing this symbol
                local meta = {
                    first_seen = current_time,
                    last_updated = current_time,
                    total_updates = 1,
                    active = true
                }
                local json_meta = cjson.encode(meta)
                redis.call('SET', key, json_meta)
                return json_meta
            else
                -- Update existing metadata
                local success, meta = pcall(cjson.decode, existing)
                if not success then
                    -- Corrupted data, recreate
                    local meta = {
                        first_seen = current_time,
                        last_updated = current_time,
                        total_updates = 1,
                        active = true
                    }
                    local json_meta = cjson.encode(meta)
                    redis.call('SET', key, json_meta)
                    return json_meta
                else
                    -- Update existing metadata
                    meta.last_updated = current_time
                    meta.active = true
                    meta.total_updates = (meta.total_updates or 0) + 1
                    local json_meta = cjson.encode(meta)
                    redis.call('SET', key, json_meta)
                    return json_meta
                end
            end
            """

            # Execute atomic metadata update
            self._redis_client.eval(lua_script, 1, meta_key, current_time_iso)

        except Exception as e:
            logger.error(f"Failed to update symbol metadata for {symbol}: {e}")
            # Fallback to non-atomic update if Lua script fails
            try:
                meta_data = {
                    "first_seen": current_time.isoformat(),
                    "last_updated": current_time.isoformat(),
                    "total_updates": 1,
                    "active": True,
                }
                self._redis_client.set(meta_key, json.dumps(meta_data))
            except Exception as fallback_error:
                logger.error(
                    f"Fallback metadata update also failed for {symbol}: {fallback_error}"
                )

    def _is_symbol_storage_enabled(self) -> bool:
        """Check if symbol storage is enabled in configuration."""
        import os

        return os.getenv("REDIS_STORE_SYMBOL_VALUES", "true").lower() == "true"

    def _get_symbol_history_max_size(self) -> int:
        """Get maximum size for symbol history storage."""
        import os

        try:
            return int(os.getenv("REDIS_SYMBOL_HISTORY_MAX_SIZE", "1000"))
        except (ValueError, TypeError):
            return 1000

    def _get_symbol_history_ttl(self) -> int:
        """Get TTL for symbol history storage in seconds."""
        import os

        try:
            return int(os.getenv("REDIS_SYMBOL_HISTORY_TTL", "86400"))  # 24 hours
        except (ValueError, TypeError):
            return 86400

    def get_latest_symbol_value(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest stored value for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with latest symbol data or None if not found
        """
        try:
            latest_key = f"symbol:latest:{symbol}"
            data = self._redis_client.get(latest_key)

            if not data:
                return None

            # Parse JSON data
            if isinstance(data, bytes):
                data = data.decode("utf-8")

            return json.loads(data)

        except Exception as e:
            logger.error(f"Failed to get latest symbol value for {symbol}: {e}")
            return None

    def get_symbol_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get price history for a symbol.

        Args:
            symbol: Trading symbol
            limit: Maximum number of historical entries to return

        Returns:
            List of historical price data
        """
        try:
            history_key = f"symbol:history:{symbol}"

            # Get latest entries (highest scores first)
            raw_data = self._redis_client.zrevrange(
                history_key, 0, limit - 1, withscores=True
            )

            history = []
            for value, timestamp in raw_data:
                if isinstance(value, bytes):
                    value = value.decode("utf-8")

                try:
                    data = json.loads(value)
                    data["timestamp"] = datetime.fromtimestamp(timestamp).isoformat()
                    history.append(data)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse history entry for {symbol}: {e}")
                    continue

            return history

        except Exception as e:
            logger.error(f"Failed to get symbol history for {symbol}: {e}")
            return []

    def get_symbol_metadata(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with symbol metadata or None if not found
        """
        try:
            meta_key = f"symbol:meta:{symbol}"
            data = self._redis_client.get(meta_key)

            if not data:
                return None

            # Parse JSON data
            if isinstance(data, bytes):
                data = data.decode("utf-8")

            return json.loads(data)

        except Exception as e:
            logger.error(f"Failed to get symbol metadata for {symbol}: {e}")
            return None

    def get_all_stored_symbols(self) -> List[str]:
        """
        Get list of all symbols that have stored data.

        Returns:
            List of symbol names
        """
        try:
            # Look for all latest symbol keys
            pattern = "symbol:latest:*"
            keys = self._redis_client.keys(pattern)

            # Extract symbol names from keys
            symbols = []
            for key in keys:
                if isinstance(key, bytes):
                    key = key.decode("utf-8")
                # Extract symbol from key pattern "symbol:latest:SYMBOL"
                symbol = key.split(":")[-1]
                symbols.append(symbol)

            return sorted(symbols)

        except Exception as e:
            logger.error(f"Failed to get stored symbols list: {e}")
            return []

    async def _monitor_performance(self):
        """Monitor service performance and log statistics."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Report every minute

                if self._stats["events_published"] > 0:
                    logger.info(
                        f"Ingestion Stats - Events: {self._stats['events_published']}, "
                        f"Errors: {self._stats['errors']}, "
                        f"CB Trips: {self._stats['circuit_breaker_trips']}, "
                        f"Provider Calls: {self._stats['provider_calls']}"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get service statistics.

        Returns:
            Dictionary with performance statistics
        """
        return {
            **self._stats,
            "running": self._running,
            "active_tasks": len(self._tasks),
            "enabled_symbols": self.config_manager.get_enabled_symbols(),
            "stream_info": self._get_stream_info(),
        }

    def _get_stream_info(self) -> Dict[str, Any]:
        """Get Redis Stream information."""
        try:
            stream_info = {}
            for name, config in self.streams.items():
                try:
                    info = self._redis_client.xinfo_stream(config.stream_name)
                    stream_info[name] = {
                        "length": info.get("length", 0),
                        "consumer_groups": info.get("groups", 0),
                        "last_entry_id": info.get("last-generated-id", ""),
                    }
                except Exception as e:
                    stream_info[name] = {"error": str(e)}
            return stream_info
        except Exception as e:
            return {"error": str(e)}

    async def publish_calculation_event(
        self, symbol: str, calculation_results: Dict[str, Any]
    ):
        """
        Publish calculation results to the calculation events stream.

        Args:
            symbol: Trading symbol
            calculation_results: Dictionary with calculation results
        """
        try:
            event_data = {
                "event_id": f"calc_{symbol}_{int(time.time() * 1000)}",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "results": json.dumps(calculation_results),
            }

            stream_config = self.streams["calculation_events"]
            message_id = self._redis_client.xadd(
                stream_config.stream_name,
                event_data,
                maxlen=stream_config.max_len,
                approximate=True,
            )

            logger.debug(f"Published calculation event for {symbol}: {message_id}")

        except Exception as e:
            logger.error(f"Failed to publish calculation event for {symbol}: {e}")

    async def publish_alert(
        self,
        alert_type: str,
        symbol: str,
        message: str,
        metadata: Dict[str, Any] = None,
    ):
        """
        Publish alert to the alert stream.

        Args:
            alert_type: Type of alert (signal, error, warning)
            symbol: Trading symbol
            message: Alert message
            metadata: Additional metadata
        """
        try:
            event_data = {
                "event_id": f"alert_{alert_type}_{int(time.time() * 1000)}",
                "type": alert_type,
                "symbol": symbol,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "metadata": json.dumps(metadata or {}),
            }

            stream_config = self.streams["alerts"]
            message_id = self._redis_client.xadd(
                stream_config.stream_name,
                event_data,
                maxlen=stream_config.max_len,
                approximate=True,
            )

            logger.info(f"Published alert [{alert_type}] for {symbol}: {message}")

        except Exception as e:
            logger.error(f"Failed to publish alert: {e}")

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest price for a symbol from cache.

        Args:
            symbol: Trading symbol

        Returns:
            Latest price or None if not available
        """
        try:
            cache_key = f"latest_price_{symbol}"
            cached_data = cache_manager.get("market_data", cache_key)
            if cached_data:
                return cached_data.get("price")
            return None
        except Exception as e:
            logger.error(f"Failed to get latest price for {symbol}: {e}")
            return None

    def cleanup(self):
        """Cleanup resources."""
        try:
            if self._redis_client:
                self._redis_client.close()
                logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Factory function for easy instantiation
def create_ingestion_service(
    config_manager: ConfigManager = None,
) -> MarketDataIngestionService:
    """
    Create a market data ingestion service instance.

    Args:
        config_manager: Optional config manager (creates new one if None)

    Returns:
        MarketDataIngestionService instance
    """
    if config_manager is None:
        config_manager = ConfigManager()

    return MarketDataIngestionService(config_manager)
