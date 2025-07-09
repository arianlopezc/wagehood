"""
RSI Trend Following Strategy

Uses RSI for trend confirmation and pullback identification within established trends.
This strategy focuses on trend following rather than traditional RSI reversal signals.
"""

from typing import List, Dict, Any, Union
import numpy as np
import logging
from datetime import datetime

from .base import TradingStrategy
from ..core.models import Signal, SignalType, MarketData
from ..indicators.momentum import calculate_rsi

logger = logging.getLogger(__name__)


class RSITrendFollowing(TradingStrategy):
    """
    RSI Trend Following Strategy

    Uses RSI for trend confirmation and pullback timing:
    - Uptrend: RSI generally between 50-80, buy on pullbacks to 40-50
    - Downtrend: RSI generally between 20-50, sell on rallies to 50-60
    - Includes divergence detection for potential reversal signals
    """

    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the RSI Trend Following strategy

        Args:
            parameters: Strategy parameters including:
                - rsi_period: RSI period (default: 14)
                - rsi_main_period: Main RSI period for trend (default: 21)
                - uptrend_threshold: RSI threshold for uptrend (default: 50)
                - downtrend_threshold: RSI threshold for downtrend (default: 50)
                - uptrend_pullback_low: Lower bound for uptrend pullback (default: 40)
                - uptrend_pullback_high: Upper bound for uptrend pullback (default: 50)
                - downtrend_pullback_low: Lower bound for downtrend pullback (default: 50)
                - downtrend_pullback_high: Upper bound for downtrend pullback (default: 60)
                - oversold_level: RSI oversold level (default: 30)
                - overbought_level: RSI overbought level (default: 70)
                - min_confidence: Minimum confidence threshold (default: 0.6)
                - divergence_detection: Enable divergence detection (default: True)
                - trend_confirmation_periods: Periods for trend confirmation (default: 10)
        """
        default_params = {
            "rsi_period": 14,
            "rsi_main_period": 21,
            "uptrend_threshold": 50,
            "downtrend_threshold": 50,
            "uptrend_pullback_low": 40,
            "uptrend_pullback_high": 50,
            "downtrend_pullback_low": 50,
            "downtrend_pullback_high": 60,
            "oversold_level": 30,
            "overbought_level": 70,
            "min_confidence": 0.6,
            "divergence_detection": True,
            "trend_confirmation_periods": 10,
        }

        if parameters:
            default_params.update(parameters)

        super().__init__("RSITrendFollowing", default_params)

    def generate_signals(
        self, data: MarketData, indicators: Dict[str, Any]
    ) -> List[Signal]:
        """
        Generate RSI trend following signals

        Args:
            data: Market data
            indicators: Pre-calculated indicators

        Returns:
            List of trading signals
        """
        signals = []

        try:
            # Get RSI data from indicators
            rsi_data = indicators.get("rsi", {})
            rsi_14 = rsi_data.get(f'rsi_{self.parameters["rsi_period"]}')
            rsi_21 = rsi_data.get(f'rsi_{self.parameters["rsi_main_period"]}')

            if rsi_14 is None:
                logger.warning(f"Missing RSI data for {self.name}")
                return signals

            # Use RSI-21 for trend confirmation if available, otherwise use RSI-14
            trend_rsi = rsi_21 if rsi_21 is not None else rsi_14

            rsi_values = np.array(rsi_14)
            trend_rsi_values = np.array(trend_rsi)

            # Need sufficient data for trend analysis
            if len(rsi_values) < self.parameters["trend_confirmation_periods"]:
                return signals

            # Analyze each period for signals
            for i in range(
                self.parameters["trend_confirmation_periods"], len(rsi_values)
            ):
                # Determine current trend
                current_trend = self._determine_trend(trend_rsi_values, i)

                # Check for trend following signals
                if current_trend == "uptrend":
                    signal = self._check_uptrend_signal(data, rsi_values, i)
                    if signal:
                        signals.append(signal)

                elif current_trend == "downtrend":
                    signal = self._check_downtrend_signal(data, rsi_values, i)
                    if signal:
                        signals.append(signal)

                # Check for reversal signals if divergence detection is enabled
                if self.parameters["divergence_detection"]:
                    divergence_signal = self._check_divergence_signal(
                        data, rsi_values, i
                    )
                    if divergence_signal:
                        signals.append(divergence_signal)

            return self.validate_signals(signals)

        except Exception as e:
            logger.error(f"Error generating signals for {self.name}: {e}")
            return []

    def _determine_trend(self, rsi_values: np.ndarray, index: int) -> str:
        """Determine current trend based on RSI behavior"""

        # Look at recent RSI values
        lookback = self.parameters["trend_confirmation_periods"]
        start_idx = max(0, index - lookback)
        recent_rsi = rsi_values[start_idx : index + 1]

        # Calculate trend characteristics
        above_50_count = np.sum(recent_rsi > self.parameters["uptrend_threshold"])
        below_50_count = np.sum(recent_rsi < self.parameters["downtrend_threshold"])

        # Determine trend based on RSI position
        if above_50_count >= lookback * 0.7:  # 70% of time above 50
            return "uptrend"
        elif below_50_count >= lookback * 0.7:  # 70% of time below 50
            return "downtrend"
        else:
            return "sideways"

    def _check_uptrend_signal(
        self, data: MarketData, rsi_values: np.ndarray, index: int
    ) -> Signal:
        """Check for uptrend pullback signal"""

        current_rsi = rsi_values[index]
        previous_rsi = rsi_values[index - 1] if index > 0 else current_rsi

        # Check for pullback entry conditions
        pullback_entry = (
            # RSI pullback to support zone
            (
                self.parameters["uptrend_pullback_low"]
                <= current_rsi
                <= self.parameters["uptrend_pullback_high"]
            )
            and
            # RSI starting to turn up
            (
                current_rsi > previous_rsi
                or current_rsi > self.parameters["uptrend_pullback_low"]
            )
        )

        if not pullback_entry:
            return None

        # Calculate confidence
        confidence_factors = self._calculate_uptrend_confidence(data, rsi_values, index)
        confidence = self.calculate_confidence(confidence_factors)

        if confidence < self.parameters["min_confidence"]:
            return None

        # Get current price
        arrays = data.to_arrays()
        current_price = arrays["close"][index]

        # Create signal
        signal = Signal(
            timestamp=arrays["timestamp"][index],
            symbol=data.symbol,
            signal_type=SignalType.BUY,
            price=current_price,
            confidence=confidence,
            strategy_name=self.name,
            metadata=self.get_signal_metadata(
                signal_name="RSI Uptrend Pullback",
                rsi_value=current_rsi,
                trend="uptrend",
                entry_type="pullback",
                **confidence_factors,
            ),
        )

        return signal

    def _check_downtrend_signal(
        self, data: MarketData, rsi_values: np.ndarray, index: int
    ) -> Signal:
        """Check for downtrend rally signal"""

        current_rsi = rsi_values[index]
        previous_rsi = rsi_values[index - 1] if index > 0 else current_rsi

        # Check for rally entry conditions
        rally_entry = (
            # RSI rally to resistance zone
            (
                self.parameters["downtrend_pullback_low"]
                <= current_rsi
                <= self.parameters["downtrend_pullback_high"]
            )
            and
            # RSI starting to turn down
            (
                current_rsi < previous_rsi
                or current_rsi < self.parameters["downtrend_pullback_high"]
            )
        )

        if not rally_entry:
            return None

        # Calculate confidence
        confidence_factors = self._calculate_downtrend_confidence(
            data, rsi_values, index
        )
        confidence = self.calculate_confidence(confidence_factors)

        if confidence < self.parameters["min_confidence"]:
            return None

        # Get current price
        arrays = data.to_arrays()
        current_price = arrays["close"][index]

        # Create signal
        signal = Signal(
            timestamp=arrays["timestamp"][index],
            symbol=data.symbol,
            signal_type=SignalType.SELL,
            price=current_price,
            confidence=confidence,
            strategy_name=self.name,
            metadata=self.get_signal_metadata(
                signal_name="RSI Downtrend Rally",
                rsi_value=current_rsi,
                trend="downtrend",
                entry_type="rally",
                **confidence_factors,
            ),
        )

        return signal

    def _calculate_uptrend_confidence(
        self, data: MarketData, rsi_values: np.ndarray, index: int
    ) -> Dict[str, float]:
        """Calculate confidence factors for uptrend signals"""

        confidence_factors = {}
        current_rsi = rsi_values[index]

        # RSI position factor (closer to pullback low = higher confidence)
        rsi_range = (
            self.parameters["uptrend_pullback_high"]
            - self.parameters["uptrend_pullback_low"]
        )
        rsi_position = 1.0 - (
            (current_rsi - self.parameters["uptrend_pullback_low"]) / rsi_range
        )
        confidence_factors["rsi_position"] = max(0.0, min(1.0, rsi_position))

        # Trend strength factor
        trend_strength = self._calculate_trend_strength(rsi_values, index, "uptrend")
        confidence_factors["trend_strength"] = trend_strength

        # Price momentum factor
        price_momentum = self._calculate_price_momentum(data, index, "uptrend")
        confidence_factors["price_momentum"] = price_momentum

        # RSI momentum factor (RSI starting to turn up)
        rsi_momentum = self._calculate_rsi_momentum(rsi_values, index, "uptrend")
        confidence_factors["rsi_momentum"] = rsi_momentum

        return confidence_factors

    def _calculate_downtrend_confidence(
        self, data: MarketData, rsi_values: np.ndarray, index: int
    ) -> Dict[str, float]:
        """Calculate confidence factors for downtrend signals"""

        confidence_factors = {}
        current_rsi = rsi_values[index]

        # RSI position factor (closer to rally high = higher confidence)
        rsi_range = (
            self.parameters["downtrend_pullback_high"]
            - self.parameters["downtrend_pullback_low"]
        )
        rsi_position = (
            current_rsi - self.parameters["downtrend_pullback_low"]
        ) / rsi_range
        confidence_factors["rsi_position"] = max(0.0, min(1.0, rsi_position))

        # Trend strength factor
        trend_strength = self._calculate_trend_strength(rsi_values, index, "downtrend")
        confidence_factors["trend_strength"] = trend_strength

        # Price momentum factor
        price_momentum = self._calculate_price_momentum(data, index, "downtrend")
        confidence_factors["price_momentum"] = price_momentum

        # RSI momentum factor (RSI starting to turn down)
        rsi_momentum = self._calculate_rsi_momentum(rsi_values, index, "downtrend")
        confidence_factors["rsi_momentum"] = rsi_momentum

        return confidence_factors

    def _calculate_trend_strength(
        self, rsi_values: np.ndarray, index: int, trend_type: str
    ) -> float:
        """Calculate trend strength based on RSI consistency"""

        lookback = min(self.parameters["trend_confirmation_periods"], index)
        if lookback < 3:
            return 0.5

        start_idx = index - lookback
        recent_rsi = rsi_values[start_idx : index + 1]

        if trend_type == "uptrend":
            # Count periods where RSI was above 50
            above_threshold = np.sum(recent_rsi > self.parameters["uptrend_threshold"])
            strength = above_threshold / len(recent_rsi)
        else:  # downtrend
            # Count periods where RSI was below 50
            below_threshold = np.sum(
                recent_rsi < self.parameters["downtrend_threshold"]
            )
            strength = below_threshold / len(recent_rsi)

        return strength

    def _calculate_price_momentum(
        self, data: MarketData, index: int, trend_type: str
    ) -> float:
        """Calculate price momentum supporting the trend"""

        arrays = data.to_arrays()
        close_prices = arrays["close"]

        lookback = min(5, index)
        if lookback < 2:
            return 0.5

        start_idx = index - lookback
        recent_closes = close_prices[start_idx : index + 1]

        # Calculate momentum
        price_change = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]

        if trend_type == "uptrend":
            # Positive momentum supports uptrend
            momentum = max(0.0, min(1.0, price_change * 20 + 0.5))
        else:  # downtrend
            # Negative momentum supports downtrend
            momentum = max(0.0, min(1.0, -price_change * 20 + 0.5))

        return momentum

    def _calculate_rsi_momentum(
        self, rsi_values: np.ndarray, index: int, trend_type: str
    ) -> float:
        """Calculate RSI momentum"""

        if index < 2:
            return 0.5

        current_rsi = rsi_values[index]
        previous_rsi = rsi_values[index - 1]

        rsi_change = current_rsi - previous_rsi

        if trend_type == "uptrend":
            # Positive RSI momentum supports uptrend entry
            momentum = max(0.0, min(1.0, rsi_change / 10 + 0.5))
        else:  # downtrend
            # Negative RSI momentum supports downtrend entry
            momentum = max(0.0, min(1.0, -rsi_change / 10 + 0.5))

        return momentum

    def _check_divergence_signal(
        self, data: MarketData, rsi_values: np.ndarray, index: int
    ) -> Signal:
        """Check for RSI divergence signals"""

        # Need at least 20 periods for divergence detection
        if index < 20:
            return None

        arrays = data.to_arrays()
        close_prices = arrays["close"]

        # Look for divergences in recent period
        lookback = min(10, index)
        start_idx = index - lookback

        price_segment = close_prices[start_idx : index + 1]
        rsi_segment = rsi_values[start_idx : index + 1]

        # Check for bullish divergence
        if self._detect_bullish_divergence(price_segment, rsi_segment):
            return self._create_divergence_signal(data, index, "bullish")

        # Check for bearish divergence
        if self._detect_bearish_divergence(price_segment, rsi_segment):
            return self._create_divergence_signal(data, index, "bearish")

        return None

    def _detect_bullish_divergence(
        self, prices: np.ndarray, rsi_values: np.ndarray
    ) -> bool:
        """Detect bullish divergence (price lower low, RSI higher low)"""

        # Find recent lows
        price_lows = []
        rsi_lows = []

        for i in range(1, len(prices) - 1):
            if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
                price_lows.append((i, prices[i]))
                rsi_lows.append((i, rsi_values[i]))

        if len(price_lows) < 2:
            return False

        # Compare the two most recent lows
        price_low1, price_val1 = price_lows[-2]
        price_low2, price_val2 = price_lows[-1]

        rsi_low1, rsi_val1 = rsi_lows[-2]
        rsi_low2, rsi_val2 = rsi_lows[-1]

        # Bullish divergence: price makes lower low, RSI makes higher low
        return price_val2 < price_val1 and rsi_val2 > rsi_val1

    def _detect_bearish_divergence(
        self, prices: np.ndarray, rsi_values: np.ndarray
    ) -> bool:
        """Detect bearish divergence (price higher high, RSI lower high)"""

        # Find recent highs
        price_highs = []
        rsi_highs = []

        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
                price_highs.append((i, prices[i]))
                rsi_highs.append((i, rsi_values[i]))

        if len(price_highs) < 2:
            return False

        # Compare the two most recent highs
        price_high1, price_val1 = price_highs[-2]
        price_high2, price_val2 = price_highs[-1]

        rsi_high1, rsi_val1 = rsi_highs[-2]
        rsi_high2, rsi_val2 = rsi_highs[-1]

        # Bearish divergence: price makes higher high, RSI makes lower high
        return price_val2 > price_val1 and rsi_val2 < rsi_val1

    def _create_divergence_signal(
        self, data: MarketData, index: int, direction: str
    ) -> Signal:
        """Create divergence signal"""

        arrays = data.to_arrays()
        current_price = arrays["close"][index]

        # Divergence signals have moderate confidence
        confidence = 0.7

        if confidence < self.parameters["min_confidence"]:
            return None

        signal_type = SignalType.BUY if direction == "bullish" else SignalType.SELL

        # Create signal
        signal = Signal(
            timestamp=arrays["timestamp"][index],
            symbol=data.symbol,
            signal_type=signal_type,
            price=current_price,
            confidence=confidence,
            strategy_name=self.name,
            metadata=self.get_signal_metadata(
                signal_name=f"RSI {direction.title()} Divergence",
                divergence_type=direction,
                entry_type="divergence",
            ),
        )

        return signal

    def get_required_indicators(self) -> List[str]:
        """Get list of required indicators"""
        return ["rsi"]

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters with descriptions"""
        return {
            "rsi_period": {
                "value": self.parameters["rsi_period"],
                "description": "Primary RSI period",
                "type": "int",
                "min": 9,
                "max": 25,
                "default": 14,
            },
            "rsi_main_period": {
                "value": self.parameters["rsi_main_period"],
                "description": "Main RSI period for trend confirmation",
                "type": "int",
                "min": 14,
                "max": 30,
                "default": 21,
            },
            "uptrend_threshold": {
                "value": self.parameters["uptrend_threshold"],
                "description": "RSI threshold for uptrend identification",
                "type": "float",
                "min": 45,
                "max": 55,
                "default": 50,
            },
            "downtrend_threshold": {
                "value": self.parameters["downtrend_threshold"],
                "description": "RSI threshold for downtrend identification",
                "type": "float",
                "min": 45,
                "max": 55,
                "default": 50,
            },
            "uptrend_pullback_low": {
                "value": self.parameters["uptrend_pullback_low"],
                "description": "Lower bound for uptrend pullback entry",
                "type": "float",
                "min": 30,
                "max": 45,
                "default": 40,
            },
            "uptrend_pullback_high": {
                "value": self.parameters["uptrend_pullback_high"],
                "description": "Upper bound for uptrend pullback entry",
                "type": "float",
                "min": 45,
                "max": 55,
                "default": 50,
            },
            "downtrend_pullback_low": {
                "value": self.parameters["downtrend_pullback_low"],
                "description": "Lower bound for downtrend rally entry",
                "type": "float",
                "min": 45,
                "max": 55,
                "default": 50,
            },
            "downtrend_pullback_high": {
                "value": self.parameters["downtrend_pullback_high"],
                "description": "Upper bound for downtrend rally entry",
                "type": "float",
                "min": 55,
                "max": 70,
                "default": 60,
            },
            "min_confidence": {
                "value": self.parameters["min_confidence"],
                "description": "Minimum confidence threshold",
                "type": "float",
                "min": 0.1,
                "max": 1.0,
                "default": 0.6,
            },
            "divergence_detection": {
                "value": self.parameters["divergence_detection"],
                "description": "Enable divergence detection",
                "type": "bool",
                "default": True,
            },
            "trend_confirmation_periods": {
                "value": self.parameters["trend_confirmation_periods"],
                "description": "Periods for trend confirmation",
                "type": "int",
                "min": 5,
                "max": 20,
                "default": 10,
            },
        }

    def _get_parameter_ranges(self) -> Dict[str, List]:
        """Get parameter ranges for optimization"""
        return {
            "rsi_period": [9, 14, 18, 21],
            "rsi_main_period": [14, 21, 28],
            "uptrend_pullback_low": [35, 40, 45],
            "uptrend_pullback_high": [45, 50, 55],
            "downtrend_pullback_low": [45, 50, 55],
            "downtrend_pullback_high": [55, 60, 65],
            "min_confidence": [0.5, 0.6, 0.7, 0.8],
            "trend_confirmation_periods": [5, 10, 15],
        }

    def _validate_signal_strategy(self, signal: Signal) -> bool:
        """Strategy-specific signal validation"""

        # Check if signal type is appropriate
        if signal.signal_type not in [SignalType.BUY, SignalType.SELL]:
            return False

        # Check confidence threshold
        if signal.confidence < self.parameters["min_confidence"]:
            return False

        # Check metadata for required fields
        metadata = signal.metadata
        if "signal_name" not in metadata or "entry_type" not in metadata:
            return False

        # Validate signal name
        valid_signal_names = [
            "RSI Uptrend Pullback",
            "RSI Downtrend Rally",
            "RSI Bullish Divergence",
            "RSI Bearish Divergence",
        ]

        if metadata["signal_name"] not in valid_signal_names:
            return False

        return True
