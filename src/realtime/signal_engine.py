"""
Signal Engine - Multi-strategy multi-timeframe signal generation

This module provides sophisticated signal generation by combining multiple strategies
across different timeframes, with signal correlation analysis, composite scoring,
and advanced signal quality validation for Discord notifications.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import time
import math

from .config_manager import ConfigManager, StrategyConfig, SignalProfile
from .timeframe_manager import TimeframeManager

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """Signal strength levels."""

    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class SignalDirection(Enum):
    """Signal direction."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    PREPARE = "prepare"


@dataclass
class Signal:
    """Individual strategy signal."""

    strategy: str
    timeframe: str
    direction: SignalDirection
    strength: SignalStrength
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate signal data."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(
                f"Confidence must be between 0 and 1, got {self.confidence}"
            )


@dataclass
class CompositeSignal:
    """Composite signal from multiple strategies and timeframes."""

    symbol: str
    timestamp: datetime
    overall_direction: SignalDirection
    overall_confidence: float
    overall_strength: SignalStrength

    # Individual signals that contributed
    contributing_signals: List[Signal] = field(default_factory=list)

    # Timeframe analysis
    timeframe_consensus: Dict[str, SignalDirection] = field(default_factory=dict)
    timeframe_confidence: Dict[str, float] = field(default_factory=dict)

    # Strategy analysis
    strategy_consensus: Dict[str, SignalDirection] = field(default_factory=dict)
    strategy_confidence: Dict[str, float] = field(default_factory=dict)

    # Correlation metrics
    timeframe_alignment: float = 0.0
    strategy_alignment: float = 0.0
    signal_persistence: float = 0.0

    def get_signal_summary(self) -> Dict[str, Any]:
        """Get a summary of the composite signal."""
        return {
            "symbol": self.symbol,
            "direction": self.overall_direction.value,
            "confidence": self.overall_confidence,
            "strength": self.overall_strength.value,
            "timeframes_analyzed": len(self.timeframe_consensus),
            "strategies_analyzed": len(self.strategy_consensus),
            "contributing_signals": len(self.contributing_signals),
            "timeframe_alignment": self.timeframe_alignment,
            "strategy_alignment": self.strategy_alignment,
            "timestamp": self.timestamp.isoformat(),
        }


class SignalEngine:
    """
    Advanced signal generation engine supporting multiple strategies and timeframes.

    This engine combines signals from different strategies across multiple timeframes
    to generate sophisticated composite signals with confidence scoring, correlation
    analysis, and signal quality validation for Discord notifications.
    """

    def __init__(
        self, config_manager: ConfigManager, timeframe_manager: TimeframeManager
    ):
        """
        Initialize the signal engine.

        Args:
            config_manager: Configuration manager instance
            timeframe_manager: Timeframe manager instance
        """
        self.config_manager = config_manager
        self.timeframe_manager = timeframe_manager

        # Signal history for persistence analysis
        self._signal_history: Dict[str, List[CompositeSignal]] = (
            {}
        )  # {symbol: [signals]}
        self._signal_cache: Dict[str, Dict[str, Dict[str, Signal]]] = (
            {}
        )  # {symbol: {strategy: {timeframe: signal}}}

        # Thread safety
        self._lock = threading.RLock()

        # Performance tracking
        self._stats = {
            "total_signals_generated": 0,
            "composite_signals_generated": 0,
            "high_quality_signals": 0,
            "filtered_signals": 0,
            "discord_notifications_triggered": 0,
            "strategies_processed": set(),
            "timeframes_processed": set(),
            "symbols_processed": set(),
            "last_signal_time": None,
            "average_processing_time_ms": 0.0,
            "errors": 0,
        }

        logger.info("SignalEngine initialized")

    def generate_signals(
        self,
        symbol: str,
        price: float,
        timeframe_results: Dict[str, Dict[str, Any]],
        trading_profile: SignalProfile,
    ) -> Optional[CompositeSignal]:
        """
        Generate composite signals for a symbol across all strategies and timeframes.

        Args:
            symbol: Trading symbol
            price: Current price
            timeframe_results: Results from timeframe processing {timeframe: {indicator: value}}
            trading_profile: Trading profile for the symbol

        Returns:
            CompositeSignal object or None if no signals generated
        """
        start_time = time.time()

        try:
            with self._lock:
                # Initialize signal cache for symbol if needed
                if symbol not in self._signal_cache:
                    self._signal_cache[symbol] = {}

                individual_signals = []

                # Get enabled strategies for this symbol
                enabled_strategies = self.config_manager.get_enabled_strategies()

                # Generate signals for each strategy-timeframe combination
                for strategy_config in enabled_strategies:
                    if symbol not in self._signal_cache:
                        self._signal_cache[symbol] = {}
                    if strategy_config.name not in self._signal_cache[symbol]:
                        self._signal_cache[symbol][strategy_config.name] = {}

                    for timeframe, indicators in timeframe_results.items():
                        # Check if strategy supports this timeframe
                        if not self.config_manager.is_strategy_enabled_for_symbol_timeframe(
                            symbol, strategy_config.name, timeframe
                        ):
                            continue

                        # Generate signal for this strategy-timeframe combination
                        signal = self._generate_strategy_signal(
                            symbol,
                            strategy_config,
                            timeframe,
                            indicators,
                            price,
                            trading_profile,
                        )

                        if signal:
                            individual_signals.append(signal)
                            self._signal_cache[symbol][strategy_config.name][
                                timeframe
                            ] = signal
                            self._stats["strategies_processed"].add(
                                strategy_config.name
                            )
                            self._stats["timeframes_processed"].add(timeframe)

                # Generate composite signal if we have individual signals
                composite_signal = None
                if individual_signals:
                    composite_signal = self._create_composite_signal(
                        symbol, individual_signals
                    )

                    # Validate signal quality
                    signal_quality = self._validate_signal_quality(composite_signal)
                    composite_signal.signal_quality = signal_quality

                    # Store in history
                    if symbol not in self._signal_history:
                        self._signal_history[symbol] = []

                    self._signal_history[symbol].append(composite_signal)

                    # Keep only recent history (last 100 signals)
                    if len(self._signal_history[symbol]) > 100:
                        self._signal_history[symbol] = self._signal_history[symbol][
                            -100:
                        ]

                    self._stats["composite_signals_generated"] += 1

                    # Track signal quality stats
                    if signal_quality.get("is_high_quality", False):
                        self._stats["high_quality_signals"] += 1
                    if signal_quality.get("should_filter", False):
                        self._stats["filtered_signals"] += 1
                    if signal_quality.get("discord_ready", False):
                        self._stats["discord_notifications_triggered"] += 1

                # Update stats
                self._stats["total_signals_generated"] += len(individual_signals)
                self._stats["symbols_processed"].add(symbol)
                self._stats["last_signal_time"] = datetime.now()

                # Update processing time
                processing_time = (time.time() - start_time) * 1000
                if self._stats["average_processing_time_ms"] == 0:
                    self._stats["average_processing_time_ms"] = processing_time
                else:
                    alpha = 0.1
                    self._stats["average_processing_time_ms"] = (
                        alpha * processing_time
                        + (1 - alpha) * self._stats["average_processing_time_ms"]
                    )

                return composite_signal

        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            self._stats["errors"] += 1
            return None

    def _generate_strategy_signal(
        self,
        symbol: str,
        strategy_config: StrategyConfig,
        timeframe: str,
        indicators: Dict[str, Any],
        price: float,
        trading_profile: SignalProfile,
    ) -> Optional[Signal]:
        """
        Generate signal for a specific strategy-timeframe combination.

        Args:
            symbol: Trading symbol
            strategy_config: Strategy configuration
            timeframe: Timeframe string
            indicators: Indicator values for this timeframe
            price: Current price
            trading_profile: Trading profile

        Returns:
            Signal object or None if no signal generated
        """
        try:
            # Get effective parameters for this symbol and timeframe
            params = self.config_manager.get_effective_strategy_parameters(
                symbol, strategy_config.name, timeframe
            )

            # Check if required indicators are available
            for required_indicator in strategy_config.required_indicators:
                if required_indicator not in indicators:
                    return None  # Missing required indicator

            # Generate signal based on strategy type
            signal_data = None

            if strategy_config.name == "macd_rsi_strategy":
                signal_data = self._calculate_macd_rsi_signal(
                    indicators, params, timeframe
                )

            elif strategy_config.name == "ma_crossover_strategy":
                signal_data = self._calculate_ma_crossover_signal(
                    indicators, params, timeframe
                )

            elif strategy_config.name == "rsi_trend_strategy":
                signal_data = self._calculate_rsi_trend_signal(
                    indicators, params, timeframe
                )

            elif strategy_config.name == "bollinger_breakout_strategy":
                signal_data = self._calculate_bollinger_breakout_signal(
                    indicators, params, timeframe, price
                )

            elif strategy_config.name == "sr_breakout_strategy":
                signal_data = self._calculate_sr_breakout_signal(
                    indicators, params, timeframe
                )

            if signal_data:
                direction = (
                    SignalDirection(signal_data["direction"])
                    if signal_data["direction"]
                    else SignalDirection.HOLD
                )
                strength = self._calculate_signal_strength(
                    signal_data["confidence"], timeframe
                )

                return Signal(
                    strategy=strategy_config.name,
                    timeframe=timeframe,
                    direction=direction,
                    strength=strength,
                    confidence=signal_data["confidence"],
                    timestamp=datetime.now(),
                    metadata=signal_data.get("metadata", {}),
                )

            return None

        except Exception as e:
            logger.error(
                f"Error generating {strategy_config.name} signal for {symbol}@{timeframe}: {e}"
            )
            return None

    def _calculate_macd_rsi_signal(
        self, indicators: Dict[str, Any], params: Dict[str, Any], timeframe: str
    ) -> Optional[Dict[str, Any]]:
        """Calculate MACD+RSI strategy signal."""
        try:
            macd_data = indicators.get("macd")
            rsi_value = indicators.get("rsi_14")

            if not macd_data or rsi_value is None:
                return None

            rsi_overbought = params.get("rsi_overbought", 70)
            rsi_oversold = params.get("rsi_oversold", 30)

            # MACD bullish/bearish
            macd_bullish = macd_data["macd_line"] > macd_data["signal_line"]
            macd_bearish = macd_data["macd_line"] < macd_data["signal_line"]

            # Generate signals with timeframe-adjusted confidence
            signal = None
            confidence = 0.0

            # Stronger signals for longer timeframes
            tf_multiplier = self._get_timeframe_confidence_multiplier(timeframe)

            if macd_bullish and rsi_value < rsi_oversold:
                signal = "buy"
                confidence = 0.9 * tf_multiplier
            elif macd_bearish and rsi_value > rsi_overbought:
                signal = "sell"
                confidence = 0.9 * tf_multiplier
            elif macd_bullish and 30 <= rsi_value <= 70:
                signal = "buy"
                confidence = 0.6 * tf_multiplier
            elif macd_bearish and 30 <= rsi_value <= 70:
                signal = "sell"
                confidence = 0.6 * tf_multiplier

            return {
                "direction": signal,
                "confidence": min(confidence, 1.0),
                "metadata": {
                    "macd_bullish": macd_bullish,
                    "rsi_value": rsi_value,
                    "macd_histogram": macd_data["histogram"],
                    "timeframe_multiplier": tf_multiplier,
                },
            }

        except Exception as e:
            logger.error(f"Error in MACD+RSI signal calculation: {e}")
            return None

    def _calculate_ma_crossover_signal(
        self, indicators: Dict[str, Any], params: Dict[str, Any], timeframe: str
    ) -> Optional[Dict[str, Any]]:
        """Calculate Moving Average Crossover strategy signal."""
        try:
            fast_period = params.get("fast_period", 50)
            slow_period = params.get("slow_period", 200)

            fast_ma = indicators.get(f"sma_{fast_period}")
            slow_ma = indicators.get(f"sma_{slow_period}")

            if fast_ma is None or slow_ma is None:
                # Try EMA if SMA not available
                fast_ma = indicators.get(f"ema_{fast_period}")
                slow_ma = indicators.get(f"ema_{slow_period}")

            if fast_ma is None or slow_ma is None:
                return None

            # Calculate signal
            signal = None
            tf_multiplier = self._get_timeframe_confidence_multiplier(timeframe)
            confidence = 0.7 * tf_multiplier

            ma_diff_pct = ((fast_ma - slow_ma) / slow_ma) * 100

            if fast_ma > slow_ma:
                signal = "buy"
                # Stronger signal if significant separation
                if abs(ma_diff_pct) > 2:
                    confidence = 0.8 * tf_multiplier
            elif fast_ma < slow_ma:
                signal = "sell"
                if abs(ma_diff_pct) > 2:
                    confidence = 0.8 * tf_multiplier

            return {
                "direction": signal,
                "confidence": min(confidence, 1.0),
                "metadata": {
                    "fast_ma": fast_ma,
                    "slow_ma": slow_ma,
                    "ma_diff_pct": ma_diff_pct,
                    "timeframe_multiplier": tf_multiplier,
                },
            }

        except Exception as e:
            logger.error(f"Error in MA crossover signal calculation: {e}")
            return None

    def _calculate_rsi_trend_signal(
        self, indicators: Dict[str, Any], params: Dict[str, Any], timeframe: str
    ) -> Optional[Dict[str, Any]]:
        """Calculate RSI Trend strategy signal."""
        try:
            rsi_value = indicators.get("rsi_14")
            trend_sma = indicators.get("sma_50")

            if rsi_value is None:
                return None

            rsi_overbought = params.get("rsi_overbought", 70)
            rsi_oversold = params.get("rsi_oversold", 30)

            signal = None
            tf_multiplier = self._get_timeframe_confidence_multiplier(timeframe)
            confidence = 0.0

            if rsi_value < rsi_oversold:
                signal = "buy"
                # Stronger signal the more oversold
                oversold_intensity = (rsi_oversold - rsi_value) / rsi_oversold
                confidence = (0.7 + 0.2 * oversold_intensity) * tf_multiplier
            elif rsi_value > rsi_overbought:
                signal = "sell"
                # Stronger signal the more overbought
                overbought_intensity = (rsi_value - rsi_overbought) / (
                    100 - rsi_overbought
                )
                confidence = (0.7 + 0.2 * overbought_intensity) * tf_multiplier

            return {
                "direction": signal,
                "confidence": min(confidence, 1.0),
                "metadata": {
                    "rsi_value": rsi_value,
                    "trend_sma": trend_sma,
                    "timeframe_multiplier": tf_multiplier,
                },
            }

        except Exception as e:
            logger.error(f"Error in RSI trend signal calculation: {e}")
            return None

    def _calculate_bollinger_breakout_signal(
        self,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
        timeframe: str,
        price: float,
    ) -> Optional[Dict[str, Any]]:
        """Calculate Bollinger Bands Breakout strategy signal."""
        try:
            bb_data = indicators.get("bollinger_bands")

            if not bb_data or price is None:
                return None

            upper_band = bb_data["upper_band"]
            middle_band = bb_data["middle_band"]
            lower_band = bb_data["lower_band"]

            # Calculate band position and width
            band_width = (upper_band - lower_band) / middle_band
            squeeze_threshold = params.get("squeeze_threshold", 0.02)

            # Calculate price position within bands
            if upper_band != lower_band:
                band_position = (price - lower_band) / (upper_band - lower_band)
            else:
                band_position = 0.5

            signal = None
            tf_multiplier = self._get_timeframe_confidence_multiplier(timeframe)
            confidence = 0.0

            # Breakout signals
            if price > upper_band:
                signal = "buy"
                breakout_strength = (price - upper_band) / upper_band
                confidence = min(0.8 + breakout_strength * 0.2, 1.0) * tf_multiplier
            elif price < lower_band:
                signal = "sell"
                breakout_strength = (lower_band - price) / lower_band
                confidence = min(0.8 + breakout_strength * 0.2, 1.0) * tf_multiplier
            elif band_width < squeeze_threshold:
                signal = "prepare"  # Squeeze - prepare for breakout
                confidence = 0.5 * tf_multiplier

            return {
                "direction": signal,
                "confidence": min(confidence, 1.0),
                "metadata": {
                    "band_width": band_width,
                    "band_position": band_position,
                    "squeeze": band_width < squeeze_threshold,
                    "price": price,
                    "bb_data": bb_data,
                    "timeframe_multiplier": tf_multiplier,
                },
            }

        except Exception as e:
            logger.error(f"Error in Bollinger breakout signal calculation: {e}")
            return None

    def _calculate_sr_breakout_signal(
        self, indicators: Dict[str, Any], params: Dict[str, Any], timeframe: str
    ) -> Optional[Dict[str, Any]]:
        """Calculate Support/Resistance Breakout strategy signal."""
        try:
            # Get required indicators
            sma_20 = indicators.get("sma_20")
            sma_50 = indicators.get("sma_50")
            current_price = indicators.get("current_price")
            volume = indicators.get("volume")
            avg_volume = indicators.get("avg_volume_20")
            
            if not all([sma_20, sma_50, current_price, volume, avg_volume]):
                return None

            tf_multiplier = self._get_timeframe_confidence_multiplier(timeframe)
            
            # Simple S/R breakout logic using SMAs as dynamic levels
            # Upper resistance = higher SMA, Lower support = lower SMA
            upper_level = max(sma_20, sma_50)
            lower_level = min(sma_20, sma_50)
            
            # Calculate breakout strength
            breakout_strength = 0.0
            direction = None
            
            # Resistance breakout (bullish)
            if current_price > upper_level:
                breakout_strength = (current_price - upper_level) / upper_level
                direction = "bullish"
            
            # Support breakout (bearish)  
            elif current_price < lower_level:
                breakout_strength = (lower_level - current_price) / lower_level
                direction = "bearish"
            
            # No breakout
            else:
                return None
            
            # Volume confirmation
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            volume_confirmation = min(volume_ratio / 1.5, 1.0)  # Normalize to 1.0
            
            # Calculate confidence
            base_confidence = min(breakout_strength * 10, 0.8)  # Scale breakout strength
            volume_boost = volume_confirmation * 0.2
            confidence = (base_confidence + volume_boost) * tf_multiplier
            
            # Apply minimum confidence threshold
            min_confidence = params.get("min_confidence", 0.6)
            if confidence < min_confidence:
                return None

            return {
                "direction": direction,
                "confidence": min(confidence, 1.0),  # Cap at 1.0
                "metadata": {
                    "breakout_strength": breakout_strength,
                    "volume_ratio": volume_ratio,
                    "upper_level": upper_level,
                    "lower_level": lower_level,
                    "current_price": current_price,
                    "timeframe_multiplier": tf_multiplier,
                },
            }

        except Exception as e:
            logger.error(f"Error in S/R breakout signal calculation: {e}")
            return None

    def _get_timeframe_confidence_multiplier(self, timeframe: str) -> float:
        """Get confidence multiplier based on timeframe reliability."""
        multipliers = {
            "1m": 0.7,  # Less reliable due to noise
            "5m": 0.8,
            "15m": 0.9,
            "30m": 0.95,
            "1h": 1.0,  # Base reliability
            "4h": 1.1,  # More reliable
            "1d": 1.2,  # Most reliable
            "1w": 1.15,  # Very reliable but less responsive
        }
        return multipliers.get(timeframe, 1.0)

    def _calculate_signal_strength(
        self, confidence: float, timeframe: str
    ) -> SignalStrength:
        """Calculate signal strength based on confidence and timeframe."""
        # Adjust thresholds based on timeframe
        if timeframe in ["1m", "5m"]:
            # Higher thresholds for short timeframes
            if confidence >= 0.85:
                return SignalStrength.VERY_STRONG
            elif confidence >= 0.75:
                return SignalStrength.STRONG
            elif confidence >= 0.65:
                return SignalStrength.MODERATE
            else:
                return SignalStrength.WEAK
        else:
            # Standard thresholds for longer timeframes
            if confidence >= 0.8:
                return SignalStrength.VERY_STRONG
            elif confidence >= 0.7:
                return SignalStrength.STRONG
            elif confidence >= 0.6:
                return SignalStrength.MODERATE
            else:
                return SignalStrength.WEAK

    def _create_composite_signal(
        self, symbol: str, individual_signals: List[Signal]
    ) -> CompositeSignal:
        """
        Create composite signal from individual strategy signals.

        Args:
            symbol: Trading symbol
            individual_signals: List of individual signals

        Returns:
            CompositeSignal object
        """
        try:
            # Analyze timeframe consensus
            timeframe_votes = {}
            timeframe_confidences = {}

            for signal in individual_signals:
                if signal.timeframe not in timeframe_votes:
                    timeframe_votes[signal.timeframe] = []
                    timeframe_confidences[signal.timeframe] = []

                timeframe_votes[signal.timeframe].append(signal.direction)
                timeframe_confidences[signal.timeframe].append(signal.confidence)

            # Calculate timeframe consensus
            timeframe_consensus = {}
            timeframe_confidence = {}

            for tf, votes in timeframe_votes.items():
                # Most common direction
                direction_counts = {}
                for vote in votes:
                    direction_counts[vote] = direction_counts.get(vote, 0) + 1

                consensus_direction = max(direction_counts.items(), key=lambda x: x[1])[
                    0
                ]
                timeframe_consensus[tf] = consensus_direction

                # Average confidence for this timeframe
                timeframe_confidence[tf] = sum(timeframe_confidences[tf]) / len(
                    timeframe_confidences[tf]
                )

            # Analyze strategy consensus
            strategy_votes = {}
            strategy_confidences = {}

            for signal in individual_signals:
                if signal.strategy not in strategy_votes:
                    strategy_votes[signal.strategy] = []
                    strategy_confidences[signal.strategy] = []

                strategy_votes[signal.strategy].append(signal.direction)
                strategy_confidences[signal.strategy].append(signal.confidence)

            strategy_consensus = {}
            strategy_confidence = {}

            for strategy, votes in strategy_votes.items():
                # Most common direction
                direction_counts = {}
                for vote in votes:
                    direction_counts[vote] = direction_counts.get(vote, 0) + 1

                consensus_direction = max(direction_counts.items(), key=lambda x: x[1])[
                    0
                ]
                strategy_consensus[strategy] = consensus_direction

                # Average confidence for this strategy
                strategy_confidence[strategy] = sum(
                    strategy_confidences[strategy]
                ) / len(strategy_confidences[strategy])

            # Calculate overall consensus
            all_directions = [signal.direction for signal in individual_signals]
            all_confidences = [signal.confidence for signal in individual_signals]

            # Weighted voting by confidence
            direction_scores = {}
            for signal in individual_signals:
                if signal.direction not in direction_scores:
                    direction_scores[signal.direction] = 0

                # Weight by confidence and timeframe reliability
                tf_weight = self._get_timeframe_confidence_multiplier(signal.timeframe)
                direction_scores[signal.direction] += signal.confidence * tf_weight

            # Overall direction is highest scoring
            overall_direction = (
                max(direction_scores.items(), key=lambda x: x[1])[0]
                if direction_scores
                else SignalDirection.HOLD
            )

            # Overall confidence is weighted average
            total_weight = sum(
                signal.confidence
                * self._get_timeframe_confidence_multiplier(signal.timeframe)
                for signal in individual_signals
            )
            overall_confidence = (
                total_weight / len(individual_signals) if individual_signals else 0.0
            )

            # Calculate signal strength
            overall_strength = self._calculate_composite_signal_strength(
                individual_signals, overall_confidence
            )

            # Calculate alignment metrics
            timeframe_alignment = self._calculate_timeframe_alignment(
                timeframe_consensus
            )
            strategy_alignment = self._calculate_strategy_alignment(strategy_consensus)
            signal_persistence = self._calculate_signal_persistence(
                symbol, overall_direction
            )

            return CompositeSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                overall_direction=overall_direction,
                overall_confidence=min(overall_confidence, 1.0),
                overall_strength=overall_strength,
                contributing_signals=individual_signals,
                timeframe_consensus=timeframe_consensus,
                timeframe_confidence=timeframe_confidence,
                strategy_consensus=strategy_consensus,
                strategy_confidence=strategy_confidence,
                timeframe_alignment=timeframe_alignment,
                strategy_alignment=strategy_alignment,
                signal_persistence=signal_persistence,
            )

        except Exception as e:
            logger.error(f"Error creating composite signal for {symbol}: {e}")
            # Return default signal
            return CompositeSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                overall_direction=SignalDirection.HOLD,
                overall_confidence=0.0,
                overall_strength=SignalStrength.WEAK,
            )

    def _calculate_composite_signal_strength(
        self, signals: List[Signal], overall_confidence: float
    ) -> SignalStrength:
        """Calculate composite signal strength."""
        try:
            # Consider number of contributing signals and their strength distribution
            strong_signals = sum(
                1
                for s in signals
                if s.strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]
            )
            total_signals = len(signals)

            strong_ratio = strong_signals / total_signals if total_signals > 0 else 0

            # Boost strength if many signals agree
            if overall_confidence >= 0.8 and strong_ratio >= 0.6:
                return SignalStrength.VERY_STRONG
            elif overall_confidence >= 0.7 and strong_ratio >= 0.4:
                return SignalStrength.STRONG
            elif overall_confidence >= 0.6:
                return SignalStrength.MODERATE
            else:
                return SignalStrength.WEAK

        except Exception as e:
            logger.error(f"Error calculating composite signal strength: {e}")
            return SignalStrength.WEAK

    def _calculate_timeframe_alignment(
        self, timeframe_consensus: Dict[str, SignalDirection]
    ) -> float:
        """Calculate how well timeframes are aligned."""
        try:
            if len(timeframe_consensus) < 2:
                return 1.0  # Perfect alignment if only one timeframe

            directions = list(timeframe_consensus.values())
            # Count how many agree with the most common direction
            direction_counts = {}
            for direction in directions:
                direction_counts[direction] = direction_counts.get(direction, 0) + 1

            max_agreement = max(direction_counts.values())
            return max_agreement / len(directions)

        except Exception as e:
            logger.error(f"Error calculating timeframe alignment: {e}")
            return 0.0

    def _calculate_strategy_alignment(
        self, strategy_consensus: Dict[str, SignalDirection]
    ) -> float:
        """Calculate how well strategies are aligned."""
        try:
            if len(strategy_consensus) < 2:
                return 1.0  # Perfect alignment if only one strategy

            directions = list(strategy_consensus.values())
            # Count how many agree with the most common direction
            direction_counts = {}
            for direction in directions:
                direction_counts[direction] = direction_counts.get(direction, 0) + 1

            max_agreement = max(direction_counts.values())
            return max_agreement / len(directions)

        except Exception as e:
            logger.error(f"Error calculating strategy alignment: {e}")
            return 0.0

    def _calculate_signal_persistence(
        self, symbol: str, current_direction: SignalDirection
    ) -> float:
        """Calculate how persistent the signal has been."""
        try:
            if symbol not in self._signal_history or not self._signal_history[symbol]:
                return 0.0

            # Look at last 10 signals
            recent_signals = self._signal_history[symbol][-10:]

            # Count how many had the same direction
            same_direction_count = sum(
                1
                for signal in recent_signals
                if signal.overall_direction == current_direction
            )

            return same_direction_count / len(recent_signals)

        except Exception as e:
            logger.error(f"Error calculating signal persistence for {symbol}: {e}")
            return 0.0

    def get_latest_signal(self, symbol: str) -> Optional[CompositeSignal]:
        """Get the latest composite signal for a symbol."""
        try:
            with self._lock:
                if symbol in self._signal_history and self._signal_history[symbol]:
                    return self._signal_history[symbol][-1]
                return None

        except Exception as e:
            logger.error(f"Error getting latest signal for {symbol}: {e}")
            return None

    def get_signal_history(self, symbol: str, limit: int = 50) -> List[CompositeSignal]:
        """Get signal history for a symbol."""
        try:
            with self._lock:
                if symbol in self._signal_history:
                    history = self._signal_history[symbol]
                    return history[-limit:] if len(history) > limit else history
                return []

        except Exception as e:
            logger.error(f"Error getting signal history for {symbol}: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get signal engine statistics."""
        with self._lock:
            stats = self._stats.copy()
            stats["strategies_processed"] = list(self._stats["strategies_processed"])
            stats["timeframes_processed"] = list(self._stats["timeframes_processed"])
            stats["symbols_processed"] = list(self._stats["symbols_processed"])
            stats["symbols_with_history"] = len(self._signal_history)
            stats["total_historical_signals"] = sum(
                len(history) for history in self._signal_history.values()
            )

            return stats

    def cleanup_old_signals(self, max_age_hours: int = 24):
        """Clean up old signals to manage memory."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            cleaned_count = 0

            with self._lock:
                for symbol in self._signal_history:
                    original_count = len(self._signal_history[symbol])
                    self._signal_history[symbol] = [
                        signal
                        for signal in self._signal_history[symbol]
                        if signal.timestamp > cutoff_time
                    ]
                    cleaned_count += original_count - len(self._signal_history[symbol])

            if cleaned_count > 0:
                logger.info(f"Cleaned {cleaned_count} old signals from memory")

        except Exception as e:
            logger.error(f"Error cleaning old signals: {e}")

    def reset_symbol(self, symbol: str):
        """Reset all signals for a symbol."""
        try:
            with self._lock:
                if symbol in self._signal_history:
                    del self._signal_history[symbol]
                if symbol in self._signal_cache:
                    del self._signal_cache[symbol]
                logger.info(f"Reset signals for {symbol}")

        except Exception as e:
            logger.error(f"Error resetting signals for {symbol}: {e}")

    def _validate_signal_quality(
        self, composite_signal: CompositeSignal
    ) -> Dict[str, Any]:
        """
        Validate signal quality for noise filtering and Discord notifications.

        Args:
            composite_signal: The composite signal to validate

        Returns:
            Dictionary with signal quality metrics
        """
        try:
            if not composite_signal:
                return {
                    "is_high_quality": False,
                    "should_filter": True,
                    "discord_ready": False,
                    "quality_score": 0.0,
                    "noise_level": "high",
                    "validation_reasons": ["no_signal"],
                }

            validation_reasons = []
            quality_score = 0.0

            # Check confidence threshold
            confidence_weight = 0.3
            if composite_signal.overall_confidence >= 0.7:
                quality_score += confidence_weight * 1.0
            elif composite_signal.overall_confidence >= 0.5:
                quality_score += confidence_weight * 0.7
            else:
                quality_score += confidence_weight * 0.3
                validation_reasons.append("low_confidence")

            # Check timeframe alignment
            alignment_weight = 0.25
            if composite_signal.timeframe_alignment >= 0.8:
                quality_score += alignment_weight * 1.0
            elif composite_signal.timeframe_alignment >= 0.6:
                quality_score += alignment_weight * 0.7
            else:
                quality_score += alignment_weight * 0.3
                validation_reasons.append("poor_timeframe_alignment")

            # Check strategy alignment
            strategy_weight = 0.25
            if composite_signal.strategy_alignment >= 0.8:
                quality_score += strategy_weight * 1.0
            elif composite_signal.strategy_alignment >= 0.6:
                quality_score += strategy_weight * 0.7
            else:
                quality_score += strategy_weight * 0.3
                validation_reasons.append("poor_strategy_alignment")

            # Check contributing signals count
            signals_weight = 0.2
            contributing_count = len(composite_signal.contributing_signals)
            if contributing_count >= 4:
                quality_score += signals_weight * 1.0
            elif contributing_count >= 2:
                quality_score += signals_weight * 0.7
            else:
                quality_score += signals_weight * 0.3
                validation_reasons.append("few_contributing_signals")

            # Check signal persistence
            persistence_bonus = 0.1
            if composite_signal.signal_persistence >= 0.6:
                quality_score += persistence_bonus
            elif composite_signal.signal_persistence >= 0.4:
                quality_score += persistence_bonus * 0.5

            # Normalize quality score
            quality_score = min(quality_score, 1.0)

            # Determine quality levels
            is_high_quality = quality_score >= 0.7
            should_filter = quality_score < 0.4

            # Discord notification readiness
            discord_ready = (
                is_high_quality
                and composite_signal.overall_direction
                not in [SignalDirection.HOLD, SignalDirection.PREPARE]
                and composite_signal.overall_confidence >= 0.6
                and composite_signal.timeframe_alignment >= 0.5
                and len(composite_signal.contributing_signals) >= 2
            )

            # Determine noise level
            if quality_score >= 0.7:
                noise_level = "low"
            elif quality_score >= 0.5:
                noise_level = "medium"
            else:
                noise_level = "high"

            return {
                "is_high_quality": is_high_quality,
                "should_filter": should_filter,
                "discord_ready": discord_ready,
                "quality_score": quality_score,
                "noise_level": noise_level,
                "validation_reasons": validation_reasons,
                "confidence_score": composite_signal.overall_confidence,
                "timeframe_alignment_score": composite_signal.timeframe_alignment,
                "strategy_alignment_score": composite_signal.strategy_alignment,
                "contributing_signals_count": contributing_count,
                "signal_persistence_score": composite_signal.signal_persistence,
            }

        except Exception as e:
            logger.error(f"Error validating signal quality: {e}")
            return {
                "is_high_quality": False,
                "should_filter": True,
                "discord_ready": False,
                "quality_score": 0.0,
                "noise_level": "high",
                "validation_reasons": ["validation_error"],
            }

    def get_signal_quality_stats(self) -> Dict[str, Any]:
        """
        Get signal quality statistics.

        Returns:
            Dictionary with signal quality metrics
        """
        try:
            with self._lock:
                total_composite = self._stats["composite_signals_generated"]
                if total_composite == 0:
                    return {
                        "total_composite_signals": 0,
                        "high_quality_rate": 0.0,
                        "filter_rate": 0.0,
                        "discord_ready_rate": 0.0,
                        "quality_distribution": {"high": 0, "medium": 0, "low": 0},
                    }

                high_quality_rate = (
                    self._stats["high_quality_signals"] / total_composite
                )
                filter_rate = self._stats["filtered_signals"] / total_composite
                discord_ready_rate = (
                    self._stats["discord_notifications_triggered"] / total_composite
                )

                return {
                    "total_composite_signals": total_composite,
                    "high_quality_signals": self._stats["high_quality_signals"],
                    "filtered_signals": self._stats["filtered_signals"],
                    "discord_notifications_triggered": self._stats[
                        "discord_notifications_triggered"
                    ],
                    "high_quality_rate": high_quality_rate,
                    "filter_rate": filter_rate,
                    "discord_ready_rate": discord_ready_rate,
                    "quality_distribution": {
                        "high": self._stats["high_quality_signals"],
                        "medium": total_composite
                        - self._stats["high_quality_signals"]
                        - self._stats["filtered_signals"],
                        "low": self._stats["filtered_signals"],
                    },
                }

        except Exception as e:
            logger.error(f"Error getting signal quality stats: {e}")
            return {}
