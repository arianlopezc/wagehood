"""
Real-time Calculation Engine

This module provides the main calculation engine that processes market data
events from Redis Streams and performs incremental technical indicator
calculations and strategy signal generation.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from src.storage.cache import cache_manager
from src.realtime.config_manager import ConfigManager, IndicatorConfig, StrategyConfig
from src.realtime.incremental_indicators import IncrementalIndicatorCalculator
from src.realtime.data_ingestion import MarketDataIngestionService

logger = logging.getLogger(__name__)


class CalculationEngine:
    """
    Real-time calculation engine for technical indicators and trading strategies.
    
    This engine processes market data events from Redis Streams and performs
    incremental calculations to maintain up-to-date indicator values and
    trading signals.
    """
    
    def __init__(self, config_manager: ConfigManager, ingestion_service: MarketDataIngestionService):
        """
        Initialize the calculation engine.
        
        Args:
            config_manager: Configuration manager instance
            ingestion_service: Market data ingestion service
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis package required for streaming. Install with: pip install redis")
        
        self.config_manager = config_manager
        self.ingestion_service = ingestion_service
        self.indicator_calculator = IncrementalIndicatorCalculator()
        
        self._redis_client = None
        self._running = False
        self._tasks = []
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Performance tracking with thread-safe counters
        self._stats_lock = threading.Lock()
        self._stats = {
            "calculations_performed": 0,
            "signals_generated": 0,
            "errors": 0,
            "last_calculation_time": None,
            "average_calculation_time_ms": 0.0,
            "symbols_processed": set(),
            "indicators_calculated": {},
            "strategies_processed": {}
        }
        
        # Batch processing buffer
        self._message_buffer: List[Tuple[str, Dict]] = []
        self._buffer_lock = threading.Lock()
        self._last_buffer_flush = time.time()
        
        # Consumer tracking
        self._last_message_ids = {}
        
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection for stream consumption."""
        try:
            from ..core.constants import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD
            
            self._redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=True,  # We want string responses for stream data
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                max_connections=20
            )
            
            # Test connection
            self._redis_client.ping()
            logger.info("Redis connection initialized for calculation engine")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {e}")
            raise
    
    async def start(self):
        """Start the calculation engine."""
        if self._running:
            logger.warning("Calculation engine is already running")
            return
        
        self._running = True
        logger.info("Starting calculation engine")
        
        try:
            # Get system configuration
            system_config = self.config_manager.get_system_config()
            if not system_config:
                raise Exception("System configuration not found")
            
            # Start consumer tasks
            consumer_task = asyncio.create_task(self._consume_market_data_stream())
            self._tasks.append(consumer_task)
            
            # Start monitoring task
            monitor_task = asyncio.create_task(self._monitor_performance())
            self._tasks.append(monitor_task)
            
            # Start cache maintenance task
            cache_task = asyncio.create_task(self._maintain_cache())
            self._tasks.append(cache_task)
            
            # Wait for all tasks
            await asyncio.gather(*self._tasks)
            
        except Exception as e:
            logger.error(f"Error in calculation engine: {e}")
            raise
        finally:
            self._running = False
            self._executor.shutdown(wait=True)
    
    async def stop(self):
        """Stop the calculation engine."""
        if not self._running:
            return
        
        logger.info("Stopping calculation engine")
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        logger.info("Calculation engine stopped")
    
    async def _consume_market_data_stream(self):
        """Consume market data events from Redis Stream."""
        stream_name = "market_data_stream"
        consumer_group = "calculation_workers"
        consumer_name = f"calc_engine_{int(time.time())}"
        
        logger.info(f"Starting stream consumer: {consumer_name}")
        
        try:
            # Ensure consumer group exists
            try:
                self._redis_client.xgroup_create(stream_name, consumer_group, id='0', mkstream=True)
            except redis.exceptions.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise e
            
            # Adaptive batch size based on load
            batch_size = 10
            max_batch_size = 100
            min_batch_size = 5
            
            while self._running:
                try:
                    # Read from stream with adaptive batching
                    messages = self._redis_client.xreadgroup(
                        consumer_group,
                        consumer_name,
                        {stream_name: '>'},
                        count=batch_size,
                        block=1000  # 1 second timeout
                    )
                    
                    if messages:
                        processing_start = time.time()
                        await self._process_market_data_messages(messages[0][1])
                        processing_time = time.time() - processing_start
                        
                        # Adjust batch size based on processing time
                        if processing_time < 0.5 and batch_size < max_batch_size:
                            batch_size = min(batch_size + 5, max_batch_size)
                        elif processing_time > 2.0 and batch_size > min_batch_size:
                            batch_size = max(batch_size - 5, min_batch_size)
                    
                except redis.exceptions.ConnectionError as e:
                    logger.error(f"Redis connection error in stream consumer: {e}")
                    await asyncio.sleep(5)  # Wait before retrying
                except Exception as e:
                    logger.error(f"Error in stream consumer: {e}")
                    self._increment_error_count()
                    await asyncio.sleep(1)
        
        except asyncio.CancelledError:
            logger.info("Stream consumer cancelled")
        except Exception as e:
            logger.error(f"Fatal error in stream consumer: {e}")
    
    async def _process_market_data_messages(self, messages: List[Tuple[str, Dict]]):
        """
        Process a batch of market data messages.
        
        Args:
            messages: List of (message_id, message_data) tuples
        """
        try:
            # Group messages by symbol for more efficient processing
            symbol_messages: Dict[str, List[Tuple[str, Dict]]] = {}
            
            for message_id, message_data in messages:
                symbol = message_data.get('symbol', '')
                if symbol:
                    if symbol not in symbol_messages:
                        symbol_messages[symbol] = []
                    symbol_messages[symbol].append((message_id, message_data))
            
            # Process each symbol's messages together
            tasks = []
            for symbol, symbol_msgs in symbol_messages.items():
                task = asyncio.create_task(self._process_symbol_batch(symbol, symbol_msgs))
                tasks.append(task)
            
            # Process all symbols concurrently
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error processing message batch: {e}")
    
    async def _process_single_message(self, message_id: str, message_data: Dict):
        """
        Process a single market data message.
        
        Args:
            message_id: Redis Stream message ID
            message_data: Message data dictionary
        """
        start_time = time.time()
        
        try:
            # Extract market data
            symbol = message_data.get('symbol')
            price = float(message_data.get('price', 0))
            timestamp_str = message_data.get('timestamp')
            
            if not symbol or not price:
                logger.warning(f"Invalid message data: {message_data}")
                return
            
            # Track symbol
            self._stats["symbols_processed"].add(symbol)
            
            # Process calculations in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._executor,
                self._calculate_indicators_for_symbol,
                symbol,
                price,
                message_data
            )
            
            # Update performance stats
            calculation_time = (time.time() - start_time) * 1000  # Convert to ms
            self._update_performance_stats(calculation_time)
            
            # Acknowledge message
            self._redis_client.xack("market_data_stream", "calculation_workers", message_id)
            
        except Exception as e:
            logger.error(f"Error processing message {message_id}: {e}")
            self._stats["errors"] += 1
    
    def _calculate_indicators_for_symbol(self, symbol: str, price: float, message_data: Dict):
        """
        Calculate all enabled indicators for a symbol.
        
        Args:
            symbol: Trading symbol
            price: Current price
            message_data: Additional market data
        """
        try:
            # Get enabled indicators (cached)
            enabled_indicators = self._get_enabled_indicators_cached()
            
            calculation_results = {}
            
            # Batch calculate indicators for better performance
            indicator_batches = self._group_indicators_by_type(enabled_indicators)
            
            for indicator_type, configs in indicator_batches.items():
                try:
                    # Calculate all indicators of the same type together
                    batch_results = self._calculate_indicator_batch(
                        indicator_type, configs, symbol, price, message_data
                    )
                    calculation_results.update(batch_results)
                    
                    # Update stats in batch
                    self._update_indicator_stats(batch_results.keys())
                
                except Exception as e:
                    logger.error(f"Error calculating {indicator_type} indicators for {symbol}: {e}")
                    self._increment_error_count()
            
            # Generate trading signals
            signals = self._generate_trading_signals(symbol, calculation_results)
            if signals:
                calculation_results["signals"] = signals
            
            # Store results in cache
            self._store_calculation_results(symbol, calculation_results)
            
            # Publish calculation event asynchronously
            asyncio.create_task(
                self.ingestion_service.publish_calculation_event(symbol, calculation_results)
            )
            
            self._increment_calculation_count()
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            self._increment_error_count()
    
    def _calculate_single_indicator(self, indicator_config: IndicatorConfig, 
                                   symbol: str, price: float, message_data: Dict) -> Optional[Any]:
        """
        Calculate a single indicator.
        
        Args:
            indicator_config: Indicator configuration
            symbol: Trading symbol
            price: Current price
            message_data: Additional market data
            
        Returns:
            Calculation result or None if failed
        """
        try:
            params = indicator_config.parameters
            
            if indicator_config.name.startswith("sma"):
                period = params.get("period", 50)
                return self.indicator_calculator.calculate_sma_incremental(symbol, price, period)
            
            elif indicator_config.name.startswith("ema"):
                period = params.get("period", 50)
                return self.indicator_calculator.calculate_ema_incremental(symbol, price, period)
            
            elif indicator_config.name.startswith("rsi"):
                period = params.get("period", 14)
                return self.indicator_calculator.calculate_rsi_incremental(symbol, price, period)
            
            elif indicator_config.name == "macd":
                fast = params.get("fast", 12)
                slow = params.get("slow", 26)
                signal = params.get("signal", 9)
                result = self.indicator_calculator.calculate_macd_incremental(symbol, price, fast, slow, signal)
                if result:
                    return {
                        "macd_line": result[0],
                        "signal_line": result[1],
                        "histogram": result[2]
                    }
            
            elif indicator_config.name == "bollinger_bands":
                period = params.get("period", 20)
                std_dev = params.get("std_dev", 2.0)
                result = self.indicator_calculator.calculate_bollinger_bands_incremental(
                    symbol, price, period, std_dev
                )
                if result:
                    return {
                        "upper_band": result[0],
                        "middle_band": result[1],
                        "lower_band": result[2]
                    }
            
            else:
                logger.warning(f"Unknown indicator: {indicator_config.name}")
                return None
            
        except Exception as e:
            logger.error(f"Error in single indicator calculation: {e}")
            return None
    
    def _generate_trading_signals(self, symbol: str, calculation_results: Dict) -> Dict[str, Any]:
        """
        Generate trading signals based on calculation results.
        
        Args:
            symbol: Trading symbol
            calculation_results: Current calculation results
            
        Returns:
            Dictionary with trading signals
        """
        try:
            signals = {}
            enabled_strategies = self.config_manager.get_enabled_strategies()
            
            for strategy_config in enabled_strategies:
                try:
                    signal = self._calculate_strategy_signal(
                        strategy_config, symbol, calculation_results
                    )
                    if signal is not None:
                        signals[strategy_config.name] = signal
                        
                        # Update strategy stats
                        if strategy_config.name not in self._stats["strategies_processed"]:
                            self._stats["strategies_processed"][strategy_config.name] = 0
                        self._stats["strategies_processed"][strategy_config.name] += 1
                
                except Exception as e:
                    logger.error(f"Error calculating strategy {strategy_config.name} for {symbol}: {e}")
                    self._stats["errors"] += 1
            
            if signals:
                self._stats["signals_generated"] += 1
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals for {symbol}: {e}")
            return {}
    
    def _calculate_strategy_signal(self, strategy_config: StrategyConfig, 
                                  symbol: str, calculation_results: Dict) -> Optional[Dict[str, Any]]:
        """
        Calculate trading signal for a specific strategy.
        
        Args:
            strategy_config: Strategy configuration
            symbol: Trading symbol
            calculation_results: Current calculation results
            
        Returns:
            Strategy signal or None if not applicable
        """
        try:
            # Check if required indicators are available
            for required_indicator in strategy_config.required_indicators:
                if required_indicator not in calculation_results:
                    return None  # Missing required indicator
            
            params = strategy_config.parameters
            
            if strategy_config.name == "macd_rsi_strategy":
                return self._calculate_macd_rsi_signal(calculation_results, params)
            
            elif strategy_config.name == "ma_crossover_strategy":
                return self._calculate_ma_crossover_signal(calculation_results, params)
            
            elif strategy_config.name == "rsi_trend_strategy":
                return self._calculate_rsi_trend_signal(calculation_results, params)
            
            elif strategy_config.name == "bollinger_breakout_strategy":
                return self._calculate_bollinger_breakout_signal(calculation_results, params)
            
            elif strategy_config.name == "sr_breakout_strategy":
                return self._calculate_sr_breakout_signal(calculation_results, params)
            
            else:
                logger.warning(f"Unknown strategy: {strategy_config.name}")
                return None
            
        except Exception as e:
            logger.error(f"Error calculating strategy signal: {e}")
            return None
    
    def _calculate_macd_rsi_signal(self, results: Dict, params: Dict) -> Optional[Dict[str, Any]]:
        """Calculate MACD+RSI strategy signal."""
        try:
            macd_data = results.get("macd")
            rsi_value = results.get("rsi_14")
            
            if not macd_data or rsi_value is None:
                return None
            
            rsi_overbought = params.get("rsi_overbought", 70)
            rsi_oversold = params.get("rsi_oversold", 30)
            
            # MACD bullish/bearish
            macd_bullish = macd_data["macd_line"] > macd_data["signal_line"]
            macd_bearish = macd_data["macd_line"] < macd_data["signal_line"]
            
            # Generate signals
            signal = None
            confidence = 0.0
            
            if macd_bullish and rsi_value < rsi_oversold:
                signal = "buy"
                confidence = 0.8
            elif macd_bearish and rsi_value > rsi_overbought:
                signal = "sell"
                confidence = 0.8
            elif macd_bullish and 30 <= rsi_value <= 70:
                signal = "weak_buy"
                confidence = 0.5
            elif macd_bearish and 30 <= rsi_value <= 70:
                signal = "weak_sell"
                confidence = 0.5
            
            return {
                "signal": signal,
                "confidence": confidence,
                "metadata": {
                    "macd_bullish": macd_bullish,
                    "rsi_value": rsi_value,
                    "macd_histogram": macd_data["histogram"]
                }
            }
            
        except Exception as e:
            logger.error(f"Error in MACD+RSI signal calculation: {e}")
            return None
    
    def _calculate_ma_crossover_signal(self, results: Dict, params: Dict) -> Optional[Dict[str, Any]]:
        """Calculate Moving Average Crossover strategy signal."""
        try:
            fast_period = params.get("fast_period", 50)
            slow_period = params.get("slow_period", 200)
            
            fast_ma = results.get(f"sma_{fast_period}")
            slow_ma = results.get(f"sma_{slow_period}")
            
            if fast_ma is None or slow_ma is None:
                return None
            
            # Determine signal
            signal = None
            confidence = 0.7
            
            if fast_ma > slow_ma:
                signal = "buy"  # Golden cross
            elif fast_ma < slow_ma:
                signal = "sell"  # Death cross
            
            return {
                "signal": signal,
                "confidence": confidence,
                "metadata": {
                    "fast_ma": fast_ma,
                    "slow_ma": slow_ma,
                    "ma_diff": fast_ma - slow_ma
                }
            }
            
        except Exception as e:
            logger.error(f"Error in MA crossover signal calculation: {e}")
            return None
    
    def _calculate_rsi_trend_signal(self, results: Dict, params: Dict) -> Optional[Dict[str, Any]]:
        """Calculate RSI Trend strategy signal."""
        try:
            rsi_value = results.get("rsi_14")
            trend_sma = results.get("sma_50")
            
            if rsi_value is None or trend_sma is None:
                return None
            
            rsi_overbought = params.get("rsi_overbought", 70)
            rsi_oversold = params.get("rsi_oversold", 30)
            
            # Generate signals based on RSI levels
            signal = None
            confidence = 0.0
            
            if rsi_value < rsi_oversold:
                signal = "buy"
                confidence = 0.8
            elif rsi_value > rsi_overbought:
                signal = "sell"
                confidence = 0.8
            
            return {
                "signal": signal,
                "confidence": confidence,
                "metadata": {
                    "rsi_value": rsi_value,
                    "trend_sma": trend_sma
                }
            }
            
        except Exception as e:
            logger.error(f"Error in RSI trend signal calculation: {e}")
            return None
    
    def _calculate_bollinger_breakout_signal(self, results: Dict, params: Dict) -> Optional[Dict[str, Any]]:
        """Calculate Bollinger Bands Breakout strategy signal."""
        try:
            bb_data = results.get("bollinger_bands")
            
            if not bb_data:
                return None
            
            # Note: We'd need current price to determine breakout
            # For now, return band squeeze information
            band_width = (bb_data["upper_band"] - bb_data["lower_band"]) / bb_data["middle_band"]
            squeeze_threshold = 0.1
            
            signal = None
            confidence = 0.0
            
            if band_width < squeeze_threshold:
                signal = "prepare"  # Prepare for breakout
                confidence = 0.6
            
            return {
                "signal": signal,
                "confidence": confidence,
                "metadata": {
                    "band_width": band_width,
                    "squeeze": band_width < squeeze_threshold,
                    "bb_data": bb_data
                }
            }
            
        except Exception as e:
            logger.error(f"Error in Bollinger breakout signal calculation: {e}")
            return None
    
    def _calculate_sr_breakout_signal(self, results: Dict, params: Dict) -> Optional[Dict[str, Any]]:
        """Calculate Support/Resistance Breakout strategy signal."""
        try:
            # This would require support/resistance level detection
            # For now, return a placeholder
            return {
                "signal": None,
                "confidence": 0.0,
                "metadata": {
                    "note": "Support/Resistance calculation requires level detection"
                }
            }
            
        except Exception as e:
            logger.error(f"Error in S/R breakout signal calculation: {e}")
            return None
    
    def _store_calculation_results(self, symbol: str, results: Dict):
        """
        Store calculation results in Redis cache.
        
        Args:
            symbol: Trading symbol
            results: Calculation results
        """
        try:
            # Store indicators
            for indicator_name, value in results.items():
                if indicator_name != "signals":
                    cache_key = f"{symbol}_{indicator_name}"
                    cache_manager.set("indicators", cache_key, value, ttl=300)  # 5 minutes
            
            # Store signals separately
            if "signals" in results:
                cache_key = f"{symbol}_signals"
                cache_manager.set("strategies", cache_key, results["signals"], ttl=600)  # 10 minutes
            
            # Store complete results
            cache_key = f"{symbol}_complete"
            cache_manager.set("calculations", cache_key, results, ttl=300)  # 5 minutes
            
        except Exception as e:
            logger.error(f"Error storing calculation results for {symbol}: {e}")
    
    def _update_performance_stats(self, calculation_time_ms: float):
        """Update performance statistics thread-safely."""
        try:
            with self._stats_lock:
                # Update average calculation time (exponential moving average)
                if self._stats["average_calculation_time_ms"] == 0:
                    self._stats["average_calculation_time_ms"] = calculation_time_ms
                else:
                    alpha = 0.1  # Smoothing factor
                    self._stats["average_calculation_time_ms"] = (
                        alpha * calculation_time_ms + 
                        (1 - alpha) * self._stats["average_calculation_time_ms"]
                    )
                
                self._stats["last_calculation_time"] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating performance stats: {e}")
    
    async def _monitor_performance(self):
        """Monitor and log performance statistics."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Report every minute
                
                if self._stats["calculations_performed"] > 0:
                    logger.info(
                        f"Calculation Engine Stats - "
                        f"Calculations: {self._stats['calculations_performed']}, "
                        f"Signals: {self._stats['signals_generated']}, "
                        f"Errors: {self._stats['errors']}, "
                        f"Avg Time: {self._stats['average_calculation_time_ms']:.2f}ms, "
                        f"Symbols: {len(self._stats['symbols_processed'])}"
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
    
    async def _maintain_cache(self):
        """Maintain cache by cleaning up old entries."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # This could be expanded to clean up old cache entries
                # For now, rely on TTL
                logger.debug("Cache maintenance cycle completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache maintenance: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get calculation engine statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        with self._stats_lock:
            # Create a copy to avoid modification during read
            stats_copy = self._stats.copy()
            stats_copy["symbols_processed"] = list(self._stats["symbols_processed"])
            
        return {
            **stats_copy,
            "running": self._running,
            "active_tasks": len(self._tasks),
            "indicator_calculator_state": self.indicator_calculator.get_state_summary()
        }
    
    def get_latest_results(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest calculation results for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Latest calculation results or None
        """
        try:
            cache_key = f"{symbol}_complete"
            return cache_manager.get("calculations", cache_key)
        except Exception as e:
            logger.error(f"Error getting latest results for {symbol}: {e}")
            return None
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if self._redis_client:
                self._redis_client.close()
                logger.info("Redis connection closed")
            
            self._executor.shutdown(wait=True)
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    # Performance optimization helper methods
    
    @lru_cache(maxsize=1)
    def _get_enabled_indicators_cached(self):
        """Get enabled indicators with caching."""
        return self.config_manager.get_enabled_indicators()
    
    def _group_indicators_by_type(self, indicators: List) -> Dict[str, List]:
        """Group indicators by their type for batch processing."""
        grouped = {}
        for indicator in indicators:
            # Extract base type (e.g., 'sma' from 'sma_50')
            base_type = indicator.name.split('_')[0]
            if base_type not in grouped:
                grouped[base_type] = []
            grouped[base_type].append(indicator)
        return grouped
    
    def _calculate_indicator_batch(self, indicator_type: str, configs: List,
                                 symbol: str, price: float, message_data: Dict) -> Dict[str, Any]:
        """Calculate multiple indicators of the same type in batch."""
        results = {}
        
        # Process all indicators of the same type together
        for config in configs:
            try:
                result = self._calculate_single_indicator(config, symbol, price, message_data)
                if result is not None:
                    results[config.name] = result
            except Exception as e:
                logger.error(f"Error in batch calculation for {config.name}: {e}")
        
        return results
    
    async def _process_symbol_batch(self, symbol: str, messages: List[Tuple[str, Dict]]):
        """Process multiple messages for the same symbol."""
        try:
            # Process latest price for the symbol
            if messages:
                # Take the most recent message
                latest_msg_id, latest_data = messages[-1]
                price = float(latest_data.get('price', 0))
                
                if price > 0:
                    # Process in thread pool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        self._executor,
                        self._calculate_indicators_for_symbol,
                        symbol,
                        price,
                        latest_data
                    )
                
                # Acknowledge all messages for this symbol
                for msg_id, _ in messages:
                    self._redis_client.xack("market_data_stream", "calculation_workers", msg_id)
                    
        except Exception as e:
            logger.error(f"Error processing symbol batch for {symbol}: {e}")
            self._increment_error_count()
    
    def _update_indicator_stats(self, indicator_names):
        """Update indicator statistics in batch."""
        with self._stats_lock:
            for name in indicator_names:
                if name not in self._stats["indicators_calculated"]:
                    self._stats["indicators_calculated"][name] = 0
                self._stats["indicators_calculated"][name] += 1
    
    def _increment_error_count(self):
        """Thread-safe error count increment."""
        with self._stats_lock:
            self._stats["errors"] += 1
    
    def _increment_calculation_count(self):
        """Thread-safe calculation count increment."""
        with self._stats_lock:
            self._stats["calculations_performed"] += 1


# Factory function for easy instantiation
def create_calculation_engine(config_manager: ConfigManager = None, 
                            ingestion_service: MarketDataIngestionService = None) -> CalculationEngine:
    """
    Create a calculation engine instance.
    
    Args:
        config_manager: Optional config manager (creates new one if None)
        ingestion_service: Optional ingestion service (creates new one if None)
        
    Returns:
        CalculationEngine instance
    """
    if config_manager is None:
        config_manager = ConfigManager()
    
    if ingestion_service is None:
        from .data_ingestion import create_ingestion_service
        ingestion_service = create_ingestion_service(config_manager)
    
    return CalculationEngine(config_manager, ingestion_service)