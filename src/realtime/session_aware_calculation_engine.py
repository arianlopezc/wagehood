"""
Session-Aware Calculation Engine - Continuous operation with data integrity

This module provides the main calculation engine with session awareness, gap handling,
and continuous operation across market transitions while maintaining data integrity.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from src.storage.cache import cache_manager
from src.realtime.config_manager import ConfigManager, SignalProfile
from src.realtime.data_ingestion import MarketDataIngestionService
from src.realtime.signal_engine import SignalEngine

from .market_session_manager import (
    MarketSessionManager,
    SessionState,
    SessionTransition,
    MarketType,
)
from .enhanced_timeframe_manager import EnhancedTimeframeManager
from .gap_aware_indicators import GapAwareIndicatorCalculator
from .session_data_synchronizer import SessionDataSynchronizer

logger = logging.getLogger(__name__)


class SessionAwareCalculationEngine:
    """
    Advanced session-aware calculation engine for continuous market data processing.

    This engine provides:
    - Continuous 24/7 operation across all market states
    - Session-aware data processing and candle creation
    - Gap detection and handling
    - Data integrity validation at session boundaries
    - Seamless transitions between market sessions
    - Indicator state management with gap awareness
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        ingestion_service: MarketDataIngestionService,
        market_type: MarketType = MarketType.US_EQUITY,
    ):
        """
        Initialize the session-aware calculation engine.

        Args:
            config_manager: Configuration manager instance
            ingestion_service: Market data ingestion service
            market_type: Type of market for session management
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis package required for streaming. Install with: pip install redis"
            )

        self.config_manager = config_manager
        self.ingestion_service = ingestion_service
        self.market_type = market_type

        # Initialize session-aware components
        self.session_manager = MarketSessionManager(market_type=market_type)
        self.timeframe_manager = EnhancedTimeframeManager(
            config_manager, self.session_manager
        )
        self.signal_engine = SignalEngine(config_manager, self.timeframe_manager)
        self.data_synchronizer = SessionDataSynchronizer(
            self.session_manager, self.timeframe_manager
        )

        # Legacy components for backward compatibility
        self.indicator_calculator = GapAwareIndicatorCalculator()

        # Redis and execution
        self._redis_client = None
        self._running = False
        self._tasks = []

        # Get worker count from environment variable
        import os

        worker_count = int(os.getenv("CALCULATION_WORKERS", "8"))
        logger.info(f"Initializing ThreadPoolExecutor with {worker_count} workers")
        self._executor = ThreadPoolExecutor(max_workers=worker_count)

        # Enhanced performance tracking with session awareness
        self._stats_lock = threading.Lock()
        self._stats = {
            "calculations_performed": 0,
            "signals_generated": 0,
            "composite_signals_generated": 0,
            "timeframe_updates": {},
            "strategy_timeframe_combinations": 0,
            "errors": 0,
            "session_transitions_handled": 0,
            "data_integrity_validations": 0,
            "gap_corrections_applied": 0,
            "continuous_operation_start": datetime.now(),
            "last_calculation_time": None,
            "average_calculation_time_ms": 0.0,
            "session_stats": {},
            "data_quality_score": 100.0,
        }

        logger.info(
            f"SessionAwareCalculationEngine initialized for {market_type.value} market"
        )

    async def initialize(self):
        """Initialize the session-aware calculation engine."""
        try:
            # Initialize Redis connection
            if not self._redis_client:
                self._redis_client = cache_manager.get_redis_client()

            # Start session-aware components
            await self.data_synchronizer.start()

            # Initialize session state
            initial_transition = self.session_manager.update_session_state(
                datetime.now()
            )
            if initial_transition:
                logger.info(
                    f"Initial session state: {initial_transition.new_state.value}"
                )

            logger.info("SessionAwareCalculationEngine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize SessionAwareCalculationEngine: {e}")
            raise

    async def start(self):
        """Start the session-aware calculation engine."""
        if self._running:
            logger.warning("SessionAwareCalculationEngine is already running")
            return

        try:
            await self.initialize()
            self._running = True

            # Start main processing tasks
            self._tasks = [
                asyncio.create_task(self._process_market_data_stream()),
                asyncio.create_task(self._session_monitoring_loop()),
                asyncio.create_task(self._performance_monitoring_loop()),
            ]

            logger.info(
                "SessionAwareCalculationEngine started - continuous operation enabled"
            )

        except Exception as e:
            logger.error(f"Failed to start SessionAwareCalculationEngine: {e}")
            await self.stop()
            raise

    async def stop(self):
        """Stop the session-aware calculation engine."""
        if not self._running:
            return

        logger.info("Stopping SessionAwareCalculationEngine...")
        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # Stop components
        await self.data_synchronizer.stop()

        # Shutdown executor
        self._executor.shutdown(wait=True)

        logger.info("SessionAwareCalculationEngine stopped")

    async def _process_market_data_stream(self):
        """Main market data processing loop with session awareness."""
        logger.info("Started session-aware market data processing")

        while self._running:
            try:
                # Get market data from Redis stream
                stream_data = await self._read_market_data_stream()

                if stream_data:
                    for stream_id, data in stream_data:
                        await self._process_market_tick_with_session_awareness(data)

                # Small delay to prevent overwhelming
                await asyncio.sleep(0.001)  # 1ms delay

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in market data processing: {e}")
                await asyncio.sleep(1)  # Wait on error

        logger.info("Market data processing stopped")

    async def _process_market_tick_with_session_awareness(self, data: Dict[str, Any]):
        """Process a market tick with full session awareness."""
        start_time = time.time()

        try:
            # Extract tick data
            symbol = data.get("symbol")
            price = float(data.get("price", 0))
            volume = float(data.get("volume", 0))
            timestamp = datetime.fromisoformat(
                data.get("timestamp", datetime.now().isoformat())
            )

            if not symbol or price <= 0:
                return

            # Update session state and detect transitions
            session_transition = self.session_manager.update_session_state(timestamp)

            # Get trading profile and timeframes for symbol
            trading_profile = self._get_trading_profile_for_symbol(symbol)
            timeframes = self._get_timeframes_for_symbol(symbol, trading_profile)

            # Process with enhanced timeframe manager (session-aware)
            timeframe_results = self.timeframe_manager.process_tick(
                symbol=symbol,
                price=price,
                volume=volume,
                timestamp=timestamp,
                timeframes=timeframes,
                trading_profile=trading_profile,
            )

            # Handle session transition if occurred
            if session_transition:
                await self._handle_session_transition(
                    symbol, timeframes, session_transition
                )

            # Generate signals with session context
            if timeframe_results:
                await self._generate_signals_with_session_context(
                    symbol, timeframe_results, session_transition
                )

            # Update statistics
            with self._stats_lock:
                self._stats["calculations_performed"] += 1
                self._stats["last_calculation_time"] = datetime.now()

                # Update average calculation time
                calc_time = (time.time() - start_time) * 1000
                if self._stats["average_calculation_time_ms"] == 0:
                    self._stats["average_calculation_time_ms"] = calc_time
                else:
                    alpha = 0.1
                    self._stats["average_calculation_time_ms"] = (
                        alpha * calc_time
                        + (1 - alpha) * self._stats["average_calculation_time_ms"]
                    )

        except Exception as e:
            logger.error(f"Error processing tick with session awareness: {e}")
            with self._stats_lock:
                self._stats["errors"] += 1

    async def _handle_session_transition(
        self, symbol: str, timeframes: List[str], session_transition: SessionTransition
    ):
        """Handle session transition comprehensively."""
        try:
            logger.info(
                f"Handling session transition for {symbol}: "
                f"{session_transition.previous_state.value} → {session_transition.new_state.value}"
            )

            # Update statistics
            with self._stats_lock:
                self._stats["session_transitions_handled"] += 1

            # Handle gap-aware indicators
            self.indicator_calculator.handle_session_transition(
                symbol, session_transition
            )

            # Perform data validation at session boundary
            validation_results = await self.data_synchronizer.handle_session_transition(
                symbol, timeframes, session_transition
            )

            # Update data quality score based on validation results
            self._update_data_quality_score(validation_results)

            # Log significant transitions
            if session_transition.is_significant_gap:
                logger.warning(
                    f"Significant gap detected for {symbol}: "
                    f"{session_transition.gap_duration_hours:.1f} hours"
                )

            if session_transition.is_new_trading_day:
                logger.info(f"New trading day started for {symbol}")

        except Exception as e:
            logger.error(f"Error handling session transition for {symbol}: {e}")

    async def _generate_signals_with_session_context(
        self,
        symbol: str,
        timeframe_results: Dict[str, Any],
        session_transition: Optional[SessionTransition],
    ):
        """Generate trading signals with session context."""
        try:
            # Get session state for signal context
            session_state = self.session_manager.get_current_state()

            # Generate signals using the existing signal engine
            # (The signal engine will use the session-aware timeframe manager)
            signals = await self._run_signal_generation(
                symbol, timeframe_results, session_state
            )

            if signals:
                # Filter signals based on session state
                filtered_signals = self._filter_signals_by_session(
                    signals, session_state
                )

                # Store signals with session metadata
                await self._store_signals_with_session_context(
                    symbol, filtered_signals, session_state, session_transition
                )

                with self._stats_lock:
                    self._stats["signals_generated"] += len(filtered_signals)

        except Exception as e:
            logger.error(
                f"Error generating signals with session context for {symbol}: {e}"
            )

    def _filter_signals_by_session(
        self, signals: List[Dict], session_state: SessionState
    ) -> List[Dict]:
        """Filter signals based on current session state."""
        # For now, allow all signals but add session metadata
        # In production, you might want to filter certain signals during specific sessions

        filtered_signals = []
        for signal in signals:
            # Add session context to signal
            signal["session_state"] = session_state.value
            signal["session_timestamp"] = datetime.now().isoformat()

            # Apply session-specific filtering logic here if needed
            # For example:
            # - Reduce signal strength during pre-market/after-hours
            # - Block certain signal types during closed sessions
            # - Adjust confidence based on session liquidity

            if session_state == SessionState.CLOSED:
                # Reduce signal confidence during closed sessions
                signal["confidence"] = signal.get("confidence", 1.0) * 0.5
                signal["session_adjusted"] = True

            filtered_signals.append(signal)

        return filtered_signals

    async def _store_signals_with_session_context(
        self,
        symbol: str,
        signals: List[Dict],
        session_state: SessionState,
        session_transition: Optional[SessionTransition],
    ):
        """Store signals with session context metadata."""
        try:
            for signal in signals:
                # Add comprehensive session metadata
                signal_with_context = {
                    **signal,
                    "session_metadata": {
                        "session_state": session_state.value,
                        "market_type": self.market_type.value,
                        "session_transition": (
                            {
                                "occurred": session_transition is not None,
                                "gap_duration_hours": (
                                    session_transition.gap_duration_hours
                                    if session_transition
                                    else 0
                                ),
                                "is_new_trading_day": (
                                    session_transition.is_new_trading_day
                                    if session_transition
                                    else False
                                ),
                            }
                            if session_transition
                            else None
                        ),
                        "data_quality_score": self._stats.get(
                            "data_quality_score", 100.0
                        ),
                    },
                }

                # Store in Redis with session-aware key
                signal_key = (
                    f"signals:{symbol}:{session_state.value}:{int(time.time())}"
                )
                await cache_manager.set_async(signal_key, signal_with_context, ttl=3600)

        except Exception as e:
            logger.error(f"Error storing signals with session context: {e}")

    async def _session_monitoring_loop(self):
        """Monitor session state and handle periodic tasks."""
        logger.info("Started session monitoring loop")

        last_session_state = None

        while self._running:
            try:
                current_state = self.session_manager.get_current_state()

                # Log session state changes
                if current_state != last_session_state:
                    logger.info(f"Session state changed to: {current_state.value}")
                    last_session_state = current_state

                # Update session statistics
                with self._stats_lock:
                    self._stats["session_stats"] = self.session_manager.get_stats()

                # Sleep for monitoring interval
                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session monitoring: {e}")
                await asyncio.sleep(60)

        logger.info("Session monitoring stopped")

    async def _performance_monitoring_loop(self):
        """Monitor performance and data quality."""
        logger.info("Started performance monitoring loop")

        while self._running:
            try:
                # Update performance statistics
                sync_stats = self.data_synchronizer.get_sync_stats()
                timeframe_stats = self.timeframe_manager.get_session_stats()

                with self._stats_lock:
                    self._stats.update(
                        {"sync_stats": sync_stats, "timeframe_stats": timeframe_stats}
                    )

                # Calculate overall system health
                self._calculate_system_health()

                # Log performance summary every 10 minutes
                if int(time.time()) % 600 == 0:  # Every 10 minutes
                    self._log_performance_summary()

                await asyncio.sleep(30)  # Update every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)

        logger.info("Performance monitoring stopped")

    def _update_data_quality_score(self, validation_results: List):
        """Update data quality score based on validation results."""
        try:
            if not validation_results:
                return

            total_validations = len(validation_results)
            successful_validations = sum(
                1 for result in validation_results if result.success
            )

            # Calculate quality score (0-100)
            quality_score = (successful_validations / total_validations) * 100

            with self._stats_lock:
                # Use exponential smoothing to update score
                current_score = self._stats.get("data_quality_score", 100.0)
                alpha = 0.1
                self._stats["data_quality_score"] = (
                    alpha * quality_score + (1 - alpha) * current_score
                )
                self._stats["data_integrity_validations"] += total_validations

        except Exception as e:
            logger.error(f"Error updating data quality score: {e}")

    def _calculate_system_health(self):
        """Calculate overall system health score."""
        try:
            with self._stats_lock:
                uptime_hours = (
                    datetime.now() - self._stats["continuous_operation_start"]
                ).total_seconds() / 3600
                error_rate = self._stats["errors"] / max(
                    1, self._stats["calculations_performed"]
                )
                data_quality = self._stats.get("data_quality_score", 100.0)

                # Health score based on multiple factors
                health_score = min(
                    100,
                    (
                        data_quality * 0.4  # 40% weight on data quality
                        + min(100, uptime_hours)
                        * 0.3  # 30% weight on uptime (capped at 100h)
                        + max(0, 100 - error_rate * 1000)
                        * 0.3  # 30% weight on error rate
                    ),
                )

                self._stats["system_health_score"] = health_score

        except Exception as e:
            logger.error(f"Error calculating system health: {e}")

    def _log_performance_summary(self):
        """Log performance summary."""
        try:
            with self._stats_lock:
                uptime = datetime.now() - self._stats["continuous_operation_start"]

                logger.info(
                    f"Performance Summary - "
                    f"Uptime: {uptime.total_seconds()/3600:.1f}h, "
                    f"Calculations: {self._stats['calculations_performed']}, "
                    f"Signals: {self._stats['signals_generated']}, "
                    f"Session Transitions: {self._stats['session_transitions_handled']}, "
                    f"Data Quality: {self._stats.get('data_quality_score', 0):.1f}%, "
                    f"System Health: {self._stats.get('system_health_score', 0):.1f}%"
                )

        except Exception as e:
            logger.error(f"Error logging performance summary: {e}")

    # Helper methods for backward compatibility and configuration

    def _get_trading_profile_for_symbol(self, symbol: str) -> SignalProfile:
        """Get trading profile for symbol."""
        # This could be configurable per symbol
        return SignalProfile.MEDIUM_FREQUENCY

    def _get_timeframes_for_symbol(
        self, symbol: str, trading_profile: SignalProfile
    ) -> List[str]:
        """Get timeframes for symbol based on trading profile."""
        if trading_profile == SignalProfile.HIGH_FREQUENCY:
            return ["1m", "5m", "15m", "1h"]
        elif trading_profile == SignalProfile.MEDIUM_FREQUENCY:
            return ["5m", "15m", "1h", "1d"]
        else:  # LOW_FREQUENCY
            return ["1h", "4h", "1d", "1w"]

    async def _read_market_data_stream(self) -> List[Tuple[str, Dict]]:
        """Read market data from Redis stream."""
        try:
            if not self._redis_client:
                return []

            # Read from market data stream
            streams = self._redis_client.xread(
                {"market_data_stream": "$"}, count=10, block=100
            )

            if streams:
                return streams[0][1]  # Return messages from first stream

            return []

        except Exception as e:
            logger.error(f"Error reading market data stream: {e}")
            return []

    async def _run_signal_generation(
        self,
        symbol: str,
        timeframe_results: Dict[str, Any],
        session_state: SessionState,
    ) -> List[Dict]:
        """Run signal generation in executor."""
        try:
            # This would integrate with the existing signal engine
            # For now, return empty list as placeholder
            return []
        except Exception as e:
            logger.error(f"Error in signal generation: {e}")
            return []

    # Public interface methods

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics including session awareness."""
        with self._stats_lock:
            return {
                **self._stats,
                "market_type": self.market_type.value,
                "current_session_state": self.session_manager.get_current_state().value,
                "is_market_open": self.session_manager.is_trading_hours(),
                "uptime_hours": (
                    datetime.now() - self._stats["continuous_operation_start"]
                ).total_seconds()
                / 3600,
            }

    def force_session_validation(self, symbol: str) -> str:
        """Force session validation for a symbol (useful for testing)."""
        timeframes = self._get_timeframes_for_symbol(
            symbol, SignalProfile.MEDIUM_FREQUENCY
        )
        return self.data_synchronizer.force_validation(symbol, timeframes)

    def get_session_state(self) -> SessionState:
        """Get current session state."""
        return self.session_manager.get_current_state()

    def is_continuous_operation_healthy(self) -> bool:
        """Check if continuous operation is healthy."""
        with self._stats_lock:
            health_score = self._stats.get("system_health_score", 0)
            return health_score > 80  # Consider healthy if > 80%
