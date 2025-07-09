"""
Enhanced TimeframeManager - Session-aware multi-timeframe data aggregation

This module provides session-aware timeframe aggregation that maintains data integrity
across market session boundaries while enabling continuous 24/7 operation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import time
import math

from .config_manager import SignalProfile, TimeframeConfig
from .incremental_indicators import IncrementalIndicatorCalculator, IndicatorState
from .market_session_manager import (
    MarketSessionManager,
    SessionState,
    SessionTransition,
)
from .timeframe_manager import TimeframeData  # Reuse existing TimeframeData

logger = logging.getLogger(__name__)


@dataclass
class SessionAwareTimeframeState:
    """Enhanced timeframe state with session awareness."""

    timeframe: str
    base_interval_seconds: int
    update_interval_seconds: int
    lookback_periods: int
    priority: int

    # Data storage
    candles: deque = field(default_factory=deque)
    current_candle: Optional[TimeframeData] = None
    last_update: Optional[datetime] = None

    # Session tracking
    last_session_state: Optional[SessionState] = None
    last_session_transition: Optional[datetime] = None
    gap_detected: bool = False

    # Indicator calculators (separate for each timeframe)
    indicator_calculator: IncrementalIndicatorCalculator = field(
        default_factory=IncrementalIndicatorCalculator
    )

    # Performance tracking
    updates_count: int = 0
    last_calculation_time: float = 0.0
    session_transitions_handled: int = 0
    gaps_detected: int = 0

    def __post_init__(self):
        """Initialize collections with maxlen for memory efficiency."""
        if not isinstance(self.candles, deque):
            self.candles = deque(maxlen=self.lookback_periods)

    def needs_update(self) -> bool:
        """Check if timeframe needs update based on interval."""
        if self.last_update is None:
            return True

        time_diff = (datetime.now() - self.last_update).total_seconds()
        return time_diff >= self.update_interval_seconds

    def should_create_new_candle(
        self,
        timestamp: datetime,
        session_manager: MarketSessionManager,
        session_transition: Optional[SessionTransition] = None,
    ) -> bool:
        """
        Check if we should create a new candle for this timestamp with session awareness.

        Args:
            timestamp: Current timestamp
            session_manager: Market session manager
            session_transition: Session transition info if any

        Returns:
            True if new candle should be created
        """
        if self.current_candle is None:
            return True

        # Check for session-based candle boundaries
        if session_transition and self._should_create_candle_for_session_transition(
            session_transition
        ):
            logger.debug(
                f"Creating new {self.timeframe} candle due to session transition: "
                f"{session_transition.previous_state.value} → {session_transition.new_state.value}"
            )
            return True

        # Check for significant gaps that warrant new candles
        if self._should_create_candle_for_gap(timestamp, session_manager):
            return True

        # Standard time-based candle logic with session awareness
        return self._should_create_candle_time_based(timestamp, session_manager)

    def _should_create_candle_for_session_transition(
        self, session_transition: SessionTransition
    ) -> bool:
        """Determine if session transition warrants a new candle."""
        if self.timeframe == "1d":
            # Daily candles: New candle on new trading day
            return session_transition.is_new_trading_day
        elif self.timeframe in ["1w"]:
            # Weekly candles: New candle on Monday market open
            return (
                session_transition.is_new_trading_day
                and session_transition.transition_time.weekday() == 0
            )  # Monday
        else:
            # Intraday candles: New candle after significant gaps
            return session_transition.is_significant_gap

    def _should_create_candle_for_gap(
        self, timestamp: datetime, session_manager: MarketSessionManager
    ) -> bool:
        """Check if gap warrants new candle creation."""
        if self.current_candle is None:
            return True

        gap_info = session_manager.get_gap_info(
            self.current_candle.timestamp, timestamp
        )

        # For intraday timeframes, create new candle after significant gaps
        if self.timeframe in ["1m", "5m", "15m", "30m", "1h", "4h"]:
            if gap_info["is_significant"] and gap_info["crosses_session_boundary"]:
                self.gap_detected = True
                self.gaps_detected += 1
                return True

        return False

    def _should_create_candle_time_based(
        self, timestamp: datetime, session_manager: MarketSessionManager
    ) -> bool:
        """Standard time-based candle creation with session alignment."""
        if self.timeframe == "1d":
            # Daily candles align with market sessions, not calendar days
            session_aligned_time = session_manager.get_session_alignment_time(
                self.timeframe, timestamp
            )
            current_aligned_time = session_manager.get_session_alignment_time(
                self.timeframe, self.current_candle.timestamp
            )
            return session_aligned_time.date() != current_aligned_time.date()
        else:
            # Intraday candles use time-based intervals
            seconds_since_epoch = int(timestamp.timestamp())
            interval_start = (
                seconds_since_epoch // self.base_interval_seconds
            ) * self.base_interval_seconds
            current_interval_start = (
                int(self.current_candle.timestamp.timestamp())
                // self.base_interval_seconds
            ) * self.base_interval_seconds
            return interval_start != current_interval_start

    def finalize_current_candle(
        self, session_transition: Optional[SessionTransition] = None
    ):
        """Finalize the current candle and add to history with session context."""
        if self.current_candle is not None:
            # Add session metadata to candle if needed
            if session_transition:
                self.last_session_transition = session_transition.transition_time
                self.session_transitions_handled += 1

            self.candles.append(self.current_candle)
            self.current_candle = None
            self.gap_detected = False

    def handle_session_transition(self, session_transition: SessionTransition):
        """Handle session transition for this timeframe."""
        self.last_session_state = session_transition.new_state
        self.last_session_transition = session_transition.transition_time

        # Log significant transitions
        if session_transition.is_significant_gap:
            logger.info(
                f"Handling significant gap in {self.timeframe}: "
                f"{session_transition.gap_duration_hours:.1f}h gap"
            )


class EnhancedTimeframeManager:
    """
    Enhanced timeframe manager with session awareness and continuous operation.

    This class provides session-aware timeframe aggregation while maintaining
    continuous 24/7 operation across all market states and transitions.

    Key Features:
    - Session-aware candle creation
    - Gap detection and handling
    - Continuous operation during market transitions
    - Data integrity across session boundaries
    - Indicator state management with gap awareness
    """

    def __init__(
        self,
        config_manager,
        market_session_manager: Optional[MarketSessionManager] = None,
    ):
        """
        Initialize the enhanced timeframe manager.

        Args:
            config_manager: Configuration manager instance
            market_session_manager: Market session manager (creates default if None)
        """
        self.config_manager = config_manager
        self.session_manager = market_session_manager or MarketSessionManager()

        self._states: Dict[str, Dict[str, SessionAwareTimeframeState]] = {}
        self._timeframe_configs: Dict[str, int] = {}
        self._lock = threading.RLock()

        # Performance tracking
        self._stats = {
            "total_updates": 0,
            "timeframe_updates": defaultdict(int),
            "symbols_processed": set(),
            "last_update_time": None,
            "average_update_time_ms": 0.0,
            "errors": 0,
            "session_transitions_handled": 0,
            "gaps_detected": 0,
            "continuous_operation_start": datetime.now(),
        }

        # Initialize timeframe configurations
        self._initialize_timeframe_configs()

        logger.info("Enhanced TimeframeManager initialized with session awareness")

    def _initialize_timeframe_configs(self):
        """Initialize timeframe configurations with their interval mappings."""
        self._timeframe_configs = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
            "1w": 604800,
        }

    def _get_timeframe_interval(self, timeframe: str) -> int:
        """Get interval in seconds for a timeframe."""
        return self._timeframe_configs.get(timeframe, 60)

    def _ensure_symbol_initialized(
        self, symbol: str, timeframes: List[str], trading_profile: SignalProfile
    ):
        """Ensure symbol is initialized with required timeframes."""
        with self._lock:
            if symbol not in self._states:
                self._states[symbol] = {}

            # Get timeframe configs for the trading profile
            timeframe_configs = self.config_manager.get_timeframe_configs()
            profile_configs = timeframe_configs.get(trading_profile, [])

            # Initialize each required timeframe
            for timeframe in timeframes:
                if timeframe not in self._states[symbol]:
                    # Find matching config
                    tf_config = next(
                        (c for c in profile_configs if c.timeframe == timeframe), None
                    )
                    if tf_config is None:
                        # Create default config
                        tf_config = TimeframeConfig(
                            timeframe=timeframe,
                            update_interval_seconds=self._get_timeframe_interval(
                                timeframe
                            )
                            // 10,
                            lookback_periods=200,
                            priority=1,
                        )

                    self._states[symbol][timeframe] = SessionAwareTimeframeState(
                        timeframe=timeframe,
                        base_interval_seconds=self._get_timeframe_interval(timeframe),
                        update_interval_seconds=tf_config.update_interval_seconds,
                        lookback_periods=tf_config.lookback_periods,
                        priority=tf_config.priority,
                    )

                    logger.debug(
                        f"Initialized session-aware timeframe {timeframe} for {symbol}"
                    )

    def process_tick(
        self,
        symbol: str,
        price: float,
        volume: float,
        timestamp: datetime,
        timeframes: List[str],
        trading_profile: SignalProfile,
    ) -> Dict[str, Any]:
        """
        Process a new tick for all timeframes with session awareness.

        This method maintains continuous operation while handling session transitions properly.

        Args:
            symbol: Trading symbol
            price: Current price
            volume: Volume (optional)
            timestamp: Tick timestamp
            timeframes: List of timeframes to process
            trading_profile: Trading profile for configuration

        Returns:
            Dictionary with updated indicators for each timeframe
        """
        start_time = time.time()
        results = {}

        try:
            # Update session state (continuous operation)
            session_transition = self.session_manager.update_session_state(timestamp)

            with self._lock:
                # Ensure symbol is initialized
                self._ensure_symbol_initialized(symbol, timeframes, trading_profile)

                # Process each timeframe with session awareness
                for timeframe in timeframes:
                    tf_state = self._states[symbol][timeframe]

                    # Handle session transition if occurred
                    if session_transition:
                        tf_state.handle_session_transition(session_transition)
                        self._stats["session_transitions_handled"] += 1

                    # Check if we need to create a new candle (session-aware)
                    if tf_state.should_create_new_candle(
                        timestamp, self.session_manager, session_transition
                    ):
                        # Finalize current candle
                        tf_state.finalize_current_candle(session_transition)

                        # Create new candle
                        tf_state.current_candle = TimeframeData(
                            timestamp=timestamp,
                            open=price,
                            high=price,
                            low=price,
                            close=price,
                            volume=volume,
                        )

                        logger.debug(
                            f"Created new {timeframe} candle for {symbol} at {timestamp}"
                        )
                    else:
                        # Update current candle
                        if tf_state.current_candle:
                            tf_state.current_candle.update_with_tick(price, volume)

                    # Update indicators if needed (with gap awareness)
                    if tf_state.needs_update():
                        timeframe_results = self._update_timeframe_indicators(
                            symbol, timeframe, price, tf_state, session_transition
                        )
                        results[timeframe] = timeframe_results
                        tf_state.last_update = datetime.now()
                        tf_state.updates_count += 1

                        # Update stats
                        self._stats["timeframe_updates"][timeframe] += 1

                # Update global stats
                self._stats["total_updates"] += 1
                self._stats["symbols_processed"].add(symbol)
                self._stats["last_update_time"] = datetime.now()

                # Update average processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to ms
                if self._stats["average_update_time_ms"] == 0:
                    self._stats["average_update_time_ms"] = processing_time
                else:
                    alpha = 0.1  # Exponential smoothing
                    self._stats["average_update_time_ms"] = (
                        alpha * processing_time
                        + (1 - alpha) * self._stats["average_update_time_ms"]
                    )

            return results

        except Exception as e:
            logger.error(f"Error processing tick for {symbol}: {e}")
            self._stats["errors"] += 1
            return {}

    def _update_timeframe_indicators(
        self,
        symbol: str,
        timeframe: str,
        price: float,
        tf_state: SessionAwareTimeframeState,
        session_transition: Optional[SessionTransition] = None,
    ) -> Dict[str, Any]:
        """
        Update indicators for a specific timeframe with gap awareness.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            price: Current price
            tf_state: Timeframe state
            session_transition: Session transition info if any

        Returns:
            Dictionary with updated indicator values
        """
        try:
            calculation_start = time.time()
            results = {}

            # Determine if gap handling is needed
            gap_detected = tf_state.gap_detected or (
                session_transition and session_transition.is_significant_gap
            )
            gap_duration = (
                session_transition.gap_duration_hours if session_transition else 0.0
            )

            # Get enabled indicators for this timeframe
            enabled_indicators = self.config_manager.get_enabled_indicators()

            # Calculate indicators using the timeframe's calculator with gap awareness
            for indicator_config in enabled_indicators:
                try:
                    indicator_key = f"{symbol}_{timeframe}_{indicator_config.name}"

                    # Apply gap-aware indicator calculations
                    result = self._calculate_indicator_with_gap_awareness(
                        indicator_config,
                        indicator_key,
                        price,
                        tf_state.indicator_calculator,
                        gap_detected,
                        gap_duration,
                    )

                    if result is not None:
                        results[indicator_config.name] = result

                except Exception as e:
                    logger.error(
                        f"Error calculating {indicator_config.name} for {symbol}@{timeframe}: {e}"
                    )

            # Update performance tracking
            tf_state.last_calculation_time = (time.time() - calculation_start) * 1000

            return results

        except Exception as e:
            logger.error(f"Error updating indicators for {symbol}@{timeframe}: {e}")
            return {}

    def _calculate_indicator_with_gap_awareness(
        self,
        indicator_config,
        indicator_key: str,
        price: float,
        calculator: IncrementalIndicatorCalculator,
        gap_detected: bool,
        gap_duration_hours: float,
    ):
        """Calculate indicator with gap awareness."""
        # Handle gap-specific logic for different indicator types
        if gap_detected and gap_duration_hours > 16:  # Weekend gap or longer
            if indicator_config.name.startswith("rsi"):
                # RSI: Reset after significant gaps to avoid false signals
                calculator.reset_indicator_state(indicator_key)
                logger.debug(
                    f"Reset RSI indicator {indicator_key} due to {gap_duration_hours:.1f}h gap"
                )

        # Calculate indicator normally
        if indicator_config.name.startswith("sma"):
            period = indicator_config.parameters.get("period", 50)
            return calculator.calculate_sma_incremental(indicator_key, price, period)

        elif indicator_config.name.startswith("ema"):
            period = indicator_config.parameters.get("period", 50)
            return calculator.calculate_ema_incremental(indicator_key, price, period)

        elif indicator_config.name.startswith("rsi"):
            period = indicator_config.parameters.get("period", 14)
            return calculator.calculate_rsi_incremental(indicator_key, price, period)

        elif indicator_config.name == "macd":
            fast = indicator_config.parameters.get("fast", 12)
            slow = indicator_config.parameters.get("slow", 26)
            signal = indicator_config.parameters.get("signal", 9)
            result = calculator.calculate_macd_incremental(
                indicator_key, price, fast, slow, signal
            )
            if result:
                return {
                    "macd_line": result[0],
                    "signal_line": result[1],
                    "histogram": result[2],
                }

        elif indicator_config.name == "bollinger_bands":
            period = indicator_config.parameters.get("period", 20)
            std_dev = indicator_config.parameters.get("std_dev", 2.0)
            result = calculator.calculate_bollinger_bands_incremental(
                indicator_key, price, period, std_dev
            )
            if result:
                return {
                    "upper_band": result[0],
                    "middle_band": result[1],
                    "lower_band": result[2],
                }

        return None

    def get_timeframe_data(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> List[TimeframeData]:
        """Get historical candle data for a timeframe."""
        try:
            with self._lock:
                if symbol not in self._states or timeframe not in self._states[symbol]:
                    return []

                tf_state = self._states[symbol][timeframe]
                candles = list(tf_state.candles)

                # Include current candle if it exists
                if tf_state.current_candle:
                    candles.append(tf_state.current_candle)

                # Return latest candles up to limit
                return candles[-limit:] if len(candles) > limit else candles

        except Exception as e:
            logger.error(f"Error getting timeframe data for {symbol}@{timeframe}: {e}")
            return []

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session-aware statistics."""
        session_stats = self.session_manager.get_stats()

        with self._lock:
            stats = self._stats.copy()
            stats["symbols_processed"] = list(self._stats["symbols_processed"])
            stats["total_timeframes"] = sum(
                len(tf_states) for tf_states in self._states.values()
            )
            stats["session_manager"] = session_stats

            # Add gap statistics
            total_gaps = sum(
                tf_state.gaps_detected
                for symbol_states in self._states.values()
                for tf_state in symbol_states.values()
            )
            stats["total_gaps_detected"] = total_gaps

            # Calculate uptime
            uptime = datetime.now() - stats["continuous_operation_start"]
            stats["continuous_operation_hours"] = uptime.total_seconds() / 3600.0

            return stats

    def force_session_update(
        self, timestamp: Optional[datetime] = None
    ) -> Optional[SessionTransition]:
        """Force a session state update (useful for testing or manual intervention)."""
        if timestamp is None:
            timestamp = datetime.now()
        return self.session_manager.update_session_state(timestamp)

    # Maintain compatibility with original TimeframeManager interface
    def get_latest_indicators(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get latest indicator values for a symbol and timeframe."""
        try:
            with self._lock:
                if symbol not in self._states or timeframe not in self._states[symbol]:
                    return {}

                tf_state = self._states[symbol][timeframe]
                return tf_state.indicator_calculator.get_all_indicators(
                    f"{symbol}_{timeframe}"
                )

        except Exception as e:
            logger.error(
                f"Error getting latest indicators for {symbol}@{timeframe}: {e}"
            )
            return {}

    def get_stats(self) -> Dict[str, Any]:
        """Get timeframe manager statistics (compatibility method)."""
        return self.get_session_stats()
