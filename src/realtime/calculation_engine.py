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
from src.realtime.config_manager import ConfigManager, IndicatorConfig, StrategyConfig, TradingProfile
from src.realtime.incremental_indicators import IncrementalIndicatorCalculator
from src.realtime.data_ingestion import MarketDataIngestionService
from src.realtime.timeframe_manager import TimeframeManager
from src.realtime.signal_engine import SignalEngine

logger = logging.getLogger(__name__)


class CalculationEngine:
    """
    Advanced real-time calculation engine for multi-strategy multi-timeframe analysis.
    
    This engine processes market data events from Redis Streams and performs
    sophisticated analysis across multiple timeframes simultaneously, generating
    composite signals from multiple strategies with correlation analysis.
    
    Features:
    - Multi-timeframe data aggregation
    - Simultaneous processing of 5 strategies across 3+ timeframes
    - Efficient caching and memory management
    - Signal correlation and composite scoring
    - High-frequency updates with performance optimization
    """
    
    def __init__(self, config_manager: ConfigManager, ingestion_service: MarketDataIngestionService):
        """
        Initialize the advanced calculation engine.
        
        Args:
            config_manager: Configuration manager instance
            ingestion_service: Market data ingestion service
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis package required for streaming. Install with: pip install redis")
        
        self.config_manager = config_manager
        self.ingestion_service = ingestion_service
        
        # Core components for multi-timeframe processing
        self.timeframe_manager = TimeframeManager(config_manager)
        self.signal_engine = SignalEngine(config_manager, self.timeframe_manager)
        
        # Legacy single-timeframe calculator for backward compatibility
        self.indicator_calculator = IncrementalIndicatorCalculator()
        
        self._redis_client = None
        self._running = False
        self._tasks = []
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Enhanced performance tracking for multi-timeframe processing
        self._stats_lock = threading.Lock()
        self._stats = {
            "calculations_performed": 0,
            "signals_generated": 0,
            "composite_signals_generated": 0,
            "timeframe_updates": {},
            "strategy_timeframe_combinations": 0,
            "errors": 0,
            "last_calculation_time": None,
            "average_calculation_time_ms": 0.0,
            "symbols_processed": set(),
            "indicators_calculated": {},
            "strategies_processed": {},
            "timeframes_processed": set(),
            "memory_usage_mb": 0.0,
            "cache_hit_rate": 0.0
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
            from src.core.constants import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD
            
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
            
            # Start cleanup task for timeframe manager and signal engine
            cleanup_task = asyncio.create_task(self._cleanup_old_data())
            self._tasks.append(cleanup_task)
            
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
        Calculate indicators and signals across all timeframes for a symbol.
        
        This method now processes multiple timeframes simultaneously and generates
        composite signals using the advanced signal engine.
        
        Args:
            symbol: Trading symbol
            price: Current price
            message_data: Additional market data
        """
        try:
            # Get symbol configuration
            watchlist = self.config_manager.get_watchlist()
            symbol_config = next((asset for asset in watchlist if asset.symbol == symbol), None)
            
            if not symbol_config or not symbol_config.enabled:
                logger.warning(f"Symbol {symbol} not found or disabled in watchlist")
                return
            
            # Extract timestamp and volume from message data
            timestamp_str = message_data.get('timestamp')
            timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
            volume = float(message_data.get('volume', 0))
            
            # Process multi-timeframe analysis
            timeframe_results = self.timeframe_manager.process_tick(
                symbol=symbol,
                price=price,
                volume=volume,
                timestamp=timestamp,
                timeframes=symbol_config.timeframes,
                trading_profile=symbol_config.trading_profile
            )
            
            # Generate composite signals if we have timeframe results
            composite_signal = None
            if timeframe_results:
                composite_signal = self.signal_engine.generate_signals(
                    symbol=symbol,
                    price=price,
                    timeframe_results=timeframe_results,
                    trading_profile=symbol_config.trading_profile
                )
            
            # Prepare comprehensive calculation results
            calculation_results = {
                "timeframe_results": timeframe_results,
                "composite_signal": composite_signal.get_signal_summary() if composite_signal else None,
                "timestamp": timestamp.isoformat(),
                "symbol_config": {
                    "timeframes": symbol_config.timeframes,
                    "trading_profile": symbol_config.trading_profile.value,
                    "enabled_strategies": symbol_config.enabled_strategies
                }
            }
            
            # Also run legacy single-timeframe calculation for backward compatibility
            legacy_results = self._calculate_legacy_indicators(symbol, price, message_data)
            if legacy_results:
                calculation_results["legacy_indicators"] = legacy_results
                
                # Generate legacy signals
                legacy_signals = self._generate_trading_signals(symbol, legacy_results)
                if legacy_signals:
                    calculation_results["legacy_signals"] = legacy_signals
            
            # Store results in cache with timeframe-specific keys
            self._store_multi_timeframe_results(symbol, calculation_results)
            
            # Update performance statistics
            self._update_multi_timeframe_stats(symbol, timeframe_results, composite_signal)
            
            # Publish calculation event (skip in non-async context)
            try:
                asyncio.create_task(
                    self.ingestion_service.publish_calculation_event(symbol, calculation_results)
                )
            except RuntimeError:
                # No event loop running - skip async publishing in test context
                logger.debug(f"Skipping async event publishing for {symbol} (no event loop)")
            
            self._increment_calculation_count()
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe calculation for {symbol}: {e}")
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
                
                # Clean up old cache entries
                # For now, rely on TTL but could be expanded
                logger.debug("Cache maintenance cycle completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache maintenance: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old data from timeframe manager and signal engine."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Every hour
                
                # Clean up old timeframe data (keep last 24 hours)
                self.timeframe_manager.cleanup_old_data(max_age_hours=24)
                
                # Clean up old signals (keep last 24 hours)
                self.signal_engine.cleanup_old_signals(max_age_hours=24)
                
                # Update memory usage stats
                memory_stats = self.timeframe_manager.get_memory_usage()
                total_memory_mb = sum(
                    data.get('memory_estimate_mb', 0) 
                    for data in memory_stats.get('symbol_breakdown', {}).values()
                )
                
                with self._stats_lock:
                    self._stats['memory_usage_mb'] = total_memory_mb
                
                logger.debug(f"Data cleanup completed. Memory usage: {total_memory_mb:.2f}MB")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in data cleanup: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive calculation engine statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        with self._stats_lock:
            # Create a copy to avoid modification during read
            stats_copy = self._stats.copy()
            stats_copy["symbols_processed"] = list(self._stats["symbols_processed"])
            stats_copy["timeframes_processed"] = list(self._stats["timeframes_processed"])
            
        # Get additional stats from components
        timeframe_stats = self.timeframe_manager.get_stats()
        signal_stats = self.signal_engine.get_stats()
        
        return {
            **stats_copy,
            "running": self._running,
            "active_tasks": len(self._tasks),
            "legacy_indicator_calculator_state": self.indicator_calculator.get_state_summary(),
            "timeframe_manager_stats": timeframe_stats,
            "signal_engine_stats": signal_stats,
            "multi_timeframe_enabled": True
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
    
    def _calculate_legacy_indicators(self, symbol: str, price: float, message_data: Dict) -> Dict[str, Any]:
        """
        Calculate indicators using the legacy single-timeframe approach.
        
        This method provides backward compatibility while the new multi-timeframe
        system is being adopted.
        
        Args:
            symbol: Trading symbol
            price: Current price
            message_data: Additional market data
            
        Returns:
            Dictionary with legacy indicator results
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
            
            return calculation_results
            
        except Exception as e:
            logger.error(f"Error in legacy indicator calculation for {symbol}: {e}")
            return {}
    
    def _store_multi_timeframe_results(self, symbol: str, results: Dict[str, Any]):
        """
        Store multi-timeframe calculation results in cache.
        
        Args:
            symbol: Trading symbol
            results: Comprehensive calculation results
        """
        try:
            # Store complete multi-timeframe results
            cache_key = f"{symbol}_multi_timeframe"
            cache_manager.set("multi_timeframe_calculations", cache_key, results, ttl=300)
            
            # Store timeframe-specific results
            timeframe_results = results.get("timeframe_results", {})
            for timeframe, tf_data in timeframe_results.items():
                tf_cache_key = f"{symbol}_{timeframe}"
                cache_manager.set("timeframe_indicators", tf_cache_key, tf_data, ttl=300)
            
            # Store composite signal separately for quick access
            composite_signal = results.get("composite_signal")
            if composite_signal:
                signal_cache_key = f"{symbol}_composite_signal"
                cache_manager.set("composite_signals", signal_cache_key, composite_signal, ttl=600)
            
            # Maintain backward compatibility - store legacy results
            legacy_indicators = results.get("legacy_indicators", {})
            if legacy_indicators:
                for indicator_name, value in legacy_indicators.items():
                    if indicator_name != "signals":
                        cache_key = f"{symbol}_{indicator_name}"
                        cache_manager.set("indicators", cache_key, value, ttl=300)
                
                # Store legacy signals
                legacy_signals = results.get("legacy_signals")
                if legacy_signals:
                    cache_key = f"{symbol}_signals"
                    cache_manager.set("strategies", cache_key, legacy_signals, ttl=600)
            
        except Exception as e:
            logger.error(f"Error storing multi-timeframe results for {symbol}: {e}")
    
    def _update_multi_timeframe_stats(self, symbol: str, timeframe_results: Dict[str, Any], 
                                    composite_signal: Any):
        """
        Update performance statistics for multi-timeframe processing.
        
        Args:
            symbol: Trading symbol
            timeframe_results: Results from timeframe processing
            composite_signal: Generated composite signal
        """
        try:
            with self._stats_lock:
                # Update timeframe processing stats
                for timeframe in timeframe_results.keys():
                    self._stats["timeframes_processed"].add(timeframe)
                    if timeframe not in self._stats["timeframe_updates"]:
                        self._stats["timeframe_updates"][timeframe] = 0
                    self._stats["timeframe_updates"][timeframe] += 1
                
                # Update signal generation stats
                if composite_signal:
                    self._stats["composite_signals_generated"] += 1
                    
                    # Count strategy-timeframe combinations
                    contributing_signals = getattr(composite_signal, 'contributing_signals', [])
                    self._stats["strategy_timeframe_combinations"] += len(contributing_signals)
                    
                    # Update strategy stats
                    for signal in contributing_signals:
                        strategy_name = getattr(signal, 'strategy', 'unknown')
                        if strategy_name not in self._stats["strategies_processed"]:
                            self._stats["strategies_processed"][strategy_name] = 0
                        self._stats["strategies_processed"][strategy_name] += 1
                
        except Exception as e:
            logger.error(f"Error updating multi-timeframe stats: {e}")
    
    def get_latest_multi_timeframe_results(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest multi-timeframe calculation results for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Latest multi-timeframe results or None
        """
        try:
            cache_key = f"{symbol}_multi_timeframe"
            return cache_manager.get("multi_timeframe_calculations", cache_key)
        except Exception as e:
            logger.error(f"Error getting multi-timeframe results for {symbol}: {e}")
            return None
    
    def get_timeframe_indicators(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """
        Get indicator values for a specific symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string (e.g., '1m', '5m', '1h')
            
        Returns:
            Indicator values for the timeframe or None
        """
        try:
            tf_cache_key = f"{symbol}_{timeframe}"
            return cache_manager.get("timeframe_indicators", tf_cache_key)
        except Exception as e:
            logger.error(f"Error getting timeframe indicators for {symbol}@{timeframe}: {e}")
            return None
    
    def get_composite_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest composite signal for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Latest composite signal or None
        """
        try:
            signal_cache_key = f"{symbol}_composite_signal"
            return cache_manager.get("composite_signals", signal_cache_key)
        except Exception as e:
            logger.error(f"Error getting composite signal for {symbol}: {e}")
            return None
    
    def get_cross_timeframe_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Get cross-timeframe correlation analysis for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Cross-timeframe analysis results
        """
        try:
            # Get symbol configuration
            watchlist = self.config_manager.get_watchlist()
            symbol_config = next((asset for asset in watchlist if asset.symbol == symbol), None)
            
            if not symbol_config:
                return {"error": f"Symbol {symbol} not found in watchlist"}
            
            # Get correlation analysis from timeframe manager
            correlation_data = self.timeframe_manager.get_cross_timeframe_correlation(
                symbol, symbol_config.timeframes
            )
            
            # Get latest composite signal for additional context
            composite_signal = self.get_composite_signal(symbol)
            
            analysis = {
                "symbol": symbol,
                "timeframes_analyzed": symbol_config.timeframes,
                "correlation_data": correlation_data,
                "latest_composite_signal": composite_signal,
                "timestamp": datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting cross-timeframe analysis for {symbol}: {e}")
            return {"error": str(e)}
    
    def reset_symbol_multi_timeframe(self, symbol: str):
        """
        Reset all multi-timeframe data for a symbol.
        
        Args:
            symbol: Trading symbol
        """
        try:
            # Reset timeframe manager data
            self.timeframe_manager.reset_symbol(symbol)
            
            # Reset signal engine data
            self.signal_engine.reset_symbol(symbol)
            
            # Reset legacy indicator calculator
            self.indicator_calculator.reset_symbol(symbol)
            
            logger.info(f"Reset all multi-timeframe data for {symbol}")
            
        except Exception as e:
            logger.error(f"Error resetting multi-timeframe data for {symbol}: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive performance summary of the calculation engine.
        
        Returns:
            Dictionary with performance metrics and recommendations
        """
        try:
            stats = self.get_stats()
            timeframe_stats = stats.get("timeframe_manager_stats", {})
            signal_stats = stats.get("signal_engine_stats", {})
            
            # Calculate performance metrics
            total_calculations = stats.get("calculations_performed", 0)
            total_composite_signals = stats.get("composite_signals_generated", 0)
            avg_processing_time = stats.get("average_calculation_time_ms", 0.0)
            memory_usage = stats.get("memory_usage_mb", 0.0)
            
            # Calculate efficiency metrics
            calculations_per_signal = (
                total_calculations / total_composite_signals 
                if total_composite_signals > 0 else 0
            )
            
            symbols_count = len(stats.get("symbols_processed", []))
            timeframes_count = len(stats.get("timeframes_processed", []))
            
            # Generate performance assessment
            performance_level = "Good"
            recommendations = []
            
            if avg_processing_time > 100:  # > 100ms
                performance_level = "Needs Optimization"
                recommendations.append("Consider increasing batch sizes or reducing update frequency")
            elif avg_processing_time > 50:  # > 50ms
                performance_level = "Fair"
                recommendations.append("Monitor processing times during peak load")
            
            if memory_usage > 500:  # > 500MB
                performance_level = "High Memory Usage"
                recommendations.append("Consider reducing lookback periods or cleaning old data more frequently")
            
            summary = {
                "performance_level": performance_level,
                "recommendations": recommendations,
                "key_metrics": {
                    "total_calculations": total_calculations,
                    "total_composite_signals": total_composite_signals,
                    "calculations_per_signal": calculations_per_signal,
                    "average_processing_time_ms": avg_processing_time,
                    "memory_usage_mb": memory_usage,
                    "symbols_processed": symbols_count,
                    "timeframes_processed": timeframes_count,
                    "strategy_timeframe_combinations": stats.get("strategy_timeframe_combinations", 0)
                },
                "component_stats": {
                    "timeframe_manager": timeframe_stats,
                    "signal_engine": signal_stats
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            return {"error": str(e)}


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