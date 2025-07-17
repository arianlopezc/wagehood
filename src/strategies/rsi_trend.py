"""
RSI Trend Following Strategy - Optimized Version

A cleaned, simplified, and performance-optimized implementation of the RSI Trend strategy.
Maintains all confidence calculation logic while improving readability and performance.
"""

from typing import List, Dict, Any
import numpy as np
import logging
from datetime import datetime

from .base import TradingStrategy
from ..indicators.talib_wrapper import calculate_rsi

logger = logging.getLogger(__name__)


class RSITrendFollowing(TradingStrategy):
    """
    RSI Trend Following Strategy - Optimized Implementation

    Uses RSI for trend confirmation and pullback timing:
    - Uptrend: RSI 50-80, buy on pullbacks to 40-50
    - Downtrend: RSI 20-50, sell on rallies to 50-60
    - Includes divergence detection for reversal signals

    Confidence Factors:
    1. RSI Position: Proximity to optimal entry level
    2. Trend Strength: Consistency of trend direction
    3. Price Momentum: Recent price movement alignment
    4. RSI Momentum: RSI directional change
    """

    # Default parameters - centralized for easy modification
    DEFAULT_PARAMS = {
        "rsi_period": 14,
        "rsi_main_period": 21,
        "uptrend_threshold": 50,
        "downtrend_threshold": 50,
        "uptrend_pullback_low": 30,  # Widened from 40
        "uptrend_pullback_high": 55,  # Widened from 50
        "downtrend_pullback_low": 45,  # Adjusted from 50
        "downtrend_pullback_high": 70,  # Widened from 60
        "oversold_level": 30,
        "overbought_level": 70,
        "min_confidence": 0.45,  # Lowered from 0.6
        "divergence_detection": True,
        "trend_confirmation_periods": 10,
    }

    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize RSI Trend Following strategy with optimized parameters."""
        params = self.DEFAULT_PARAMS.copy()
        if parameters:
            params.update(parameters)

        # Validate parameters
        self._validate_parameters(params)

        super().__init__("RSITrendFollowing", params)

        # Cache frequently used parameters
        self._rsi_period = params["rsi_period"]
        self._rsi_main_period = params["rsi_main_period"]
        self._min_confidence = params["min_confidence"]
        self._trend_periods = params["trend_confirmation_periods"]

        # Pre-calculate ranges for performance (with zero-division protection)
        self._uptrend_range = max(
            params["uptrend_pullback_high"] - params["uptrend_pullback_low"], 0.01
        )
        self._downtrend_range = max(
            params["downtrend_pullback_high"] - params["downtrend_pullback_low"], 0.01
        )

    def _validate_parameters(self, params: Dict[str, Any]) -> None:
        """Validate strategy parameters to prevent edge cases."""
        # Validate RSI periods
        if not isinstance(params["rsi_period"], int) or params["rsi_period"] < 2:
            raise ValueError("rsi_period must be an integer >= 2")
        if (
            not isinstance(params["rsi_main_period"], int)
            or params["rsi_main_period"] < 2
        ):
            raise ValueError("rsi_main_period must be an integer >= 2")

        # Validate confidence thresholds
        if not 0 <= params["min_confidence"] <= 1:
            raise ValueError("min_confidence must be between 0 and 1")

        # Validate trend confirmation periods
        if (
            not isinstance(params["trend_confirmation_periods"], int)
            or params["trend_confirmation_periods"] < 1
        ):
            raise ValueError("trend_confirmation_periods must be an integer >= 1")

        # Validate pullback ranges
        if params["uptrend_pullback_low"] >= params["uptrend_pullback_high"]:
            raise ValueError(
                "uptrend_pullback_low must be less than uptrend_pullback_high"
            )
        if params["downtrend_pullback_low"] >= params["downtrend_pullback_high"]:
            raise ValueError(
                "downtrend_pullback_low must be less than downtrend_pullback_high"
            )

        # Validate RSI thresholds
        if not 0 <= params["uptrend_threshold"] <= 100:
            raise ValueError("uptrend_threshold must be between 0 and 100")
        if not 0 <= params["downtrend_threshold"] <= 100:
            raise ValueError("downtrend_threshold must be between 0 and 100")

    def generate_signals(
        self, data: Dict[str, Any], indicators: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate RSI trend following signals with optimized processing.

        Args:
            data: Market data
            indicators: Pre-calculated indicators

        Returns:
            List of validated trading signals
        """
        try:
            # Extract RSI data efficiently
            rsi_data = indicators.get("rsi", {})
            rsi_14 = rsi_data.get(f"rsi_{self._rsi_period}")
            rsi_21 = rsi_data.get(f"rsi_{self._rsi_main_period}")

            if rsi_14 is None:
                logger.warning(f"Missing RSI data for {self.name}")
                return []

            # Use main RSI for trend, fallback to primary RSI
            trend_rsi = rsi_21 if rsi_21 is not None else rsi_14

            # Convert to numpy arrays for vectorized operations
            rsi_values = np.asarray(rsi_14)
            trend_rsi_values = np.asarray(trend_rsi)

            # Check data sufficiency with proper validation
            if len(rsi_values) < max(self._trend_periods, self._rsi_period) + 5:
                logger.warning(
                    f"Insufficient data for {self.name}: need at least {max(self._trend_periods, self._rsi_period) + 5} points"
                )
                return []

            # Check for too many NaN values
            valid_rsi_count = np.sum(~np.isnan(rsi_values))
            if valid_rsi_count < self._trend_periods:
                logger.warning(f"Too many NaN values in RSI data for {self.name}")
                return []

            # Process signals efficiently
            signals = self._process_signals_vectorized(
                data, rsi_values, trend_rsi_values
            )

            # Sort signals by timestamp in descending order (newest first)
            signals.sort(key=lambda s: s["timestamp"], reverse=True)

            return self.validate_signals(signals)

        except Exception as e:
            logger.error(f"Error generating signals for {self.name}: {e}")
            return []

    def _process_signals_vectorized(
        self, data: Dict[str, Any], rsi_values: np.ndarray, trend_rsi_values: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Process signals using vectorized operations for better performance."""
        signals = []

        # Pre-calculate trend for all periods (vectorized)
        trends = self._calculate_trends_vectorized(trend_rsi_values)

        # Process each valid period
        for i in range(self._trend_periods, len(rsi_values)):
            current_trend = trends[i]

            # Check for trend-following signals
            if current_trend == "uptrend":
                signal = self._check_uptrend_signal_optimized(data, rsi_values, i)
            elif current_trend == "downtrend":
                signal = self._check_downtrend_signal_optimized(data, rsi_values, i)
            else:
                signal = None

            if signal:
                signals.append(signal)

            # Check divergence if enabled
            if self.parameters["divergence_detection"]:
                div_signal = self._check_divergence_signal_optimized(
                    data, rsi_values, i
                )
                if div_signal:
                    signals.append(div_signal)

        return signals

    def _calculate_trends_vectorized(self, rsi_values: np.ndarray) -> List[str]:
        """Calculate trends for all periods using vectorized operations."""
        trends = ["sideways"] * len(rsi_values)

        for i in range(self._trend_periods, len(rsi_values)):
            start_idx = i - self._trend_periods
            recent_rsi = rsi_values[start_idx : i + 1]

            above_threshold = np.sum(recent_rsi > self.parameters["uptrend_threshold"])
            below_threshold = np.sum(
                recent_rsi < self.parameters["downtrend_threshold"]
            )

            threshold_count = self._trend_periods * 0.7

            if above_threshold >= threshold_count:
                trends[i] = "uptrend"
            elif below_threshold >= threshold_count:
                trends[i] = "downtrend"

        return trends

    def _check_uptrend_signal_optimized(
        self, data: Dict[str, Any], rsi_values: np.ndarray, index: int
    ) -> Dict[str, Any]:
        """Optimized uptrend signal checking with confidence calculation."""
        current_rsi = rsi_values[index]
        previous_rsi = rsi_values[index - 1] if index > 0 else current_rsi

        # Quick exit if not in pullback range
        if not (
            self.parameters["uptrend_pullback_low"]
            <= current_rsi
            <= self.parameters["uptrend_pullback_high"]
        ):
            return None

        # Check momentum condition
        if not (
            current_rsi > previous_rsi
            or current_rsi > self.parameters["uptrend_pullback_low"]
        ):
            return None

        # Calculate confidence efficiently
        confidence = self._calculate_uptrend_confidence_optimized(
            data, rsi_values, index
        )

        if confidence < self._min_confidence:
            return None

        # Create signal
        arrays = data.to_arrays()
        return {
            "timestamp": arrays["timestamp"][index],
            "symbol": data.get("symbol", "UNKNOWN"),
            "signal_type": "BUY",
            "price": arrays["close"][index],
            "confidence": confidence,
            "strategy_name": self.name,
            "metadata": {
                "signal_name": "RSI Uptrend Pullback",
                "rsi_value": current_rsi,
                "trend": "uptrend",
                "entry_type": "pullback",
            },
        }

    def _check_downtrend_signal_optimized(
        self, data: Dict[str, Any], rsi_values: np.ndarray, index: int
    ) -> Dict[str, Any]:
        """Optimized downtrend signal checking with confidence calculation."""
        current_rsi = rsi_values[index]
        previous_rsi = rsi_values[index - 1] if index > 0 else current_rsi

        # Quick exit if not in rally range
        if not (
            self.parameters["downtrend_pullback_low"]
            <= current_rsi
            <= self.parameters["downtrend_pullback_high"]
        ):
            return None

        # Check momentum condition
        if not (
            current_rsi < previous_rsi
            or current_rsi < self.parameters["downtrend_pullback_high"]
        ):
            return None

        # Calculate confidence efficiently
        confidence = self._calculate_downtrend_confidence_optimized(
            data, rsi_values, index
        )

        if confidence < self._min_confidence:
            return None

        # Create signal
        arrays = data.to_arrays()
        return {
            "timestamp": arrays["timestamp"][index],
            "symbol": data.get("symbol", "UNKNOWN"),
            "signal_type": "SELL",
            "price": arrays["close"][index],
            "confidence": confidence,
            "strategy_name": self.name,
            "metadata": {
                "signal_name": "RSI Downtrend Rally",
                "rsi_value": current_rsi,
                "trend": "downtrend",
                "entry_type": "rally",
            },
        }

    def _calculate_uptrend_confidence_optimized(
        self, data: Dict[str, Any], rsi_values: np.ndarray, index: int
    ) -> float:
        """Optimized uptrend confidence calculation."""
        current_rsi = rsi_values[index]

        # 1. RSI position factor (vectorized calculation)
        rsi_position = 1.0 - (
            (current_rsi - self.parameters["uptrend_pullback_low"])
            / self._uptrend_range
        )
        rsi_position = np.clip(rsi_position, 0.0, 1.0)

        # 2. Trend strength factor (optimized lookback)
        trend_strength = self._calculate_trend_strength_optimized(
            rsi_values, index, "uptrend"
        )

        # 3. Price momentum factor (cached arrays)
        price_momentum = self._calculate_price_momentum_optimized(
            data, index, "uptrend"
        )

        # 4. RSI momentum factor (simple calculation)
        rsi_momentum = self._calculate_rsi_momentum_optimized(
            rsi_values, index, "uptrend"
        )

        # Calculate weighted confidence
        confidence_factors = {
            "rsi_position": rsi_position,
            "trend_strength": trend_strength,
            "price_momentum": price_momentum,
            "rsi_momentum": rsi_momentum,
        }

        return self.calculate_confidence(confidence_factors)

    def _calculate_uptrend_confidence_factors(
        self, data: Dict[str, Any], rsi_values: np.ndarray, index: int
    ) -> Dict[str, float]:
        """Calculate uptrend confidence factors (for testing)."""
        current_rsi = rsi_values[index]

        # 1. RSI position factor
        rsi_position = 1.0 - (
            (current_rsi - self.parameters["uptrend_pullback_low"])
            / self._uptrend_range
        )
        rsi_position = np.clip(rsi_position, 0.0, 1.0)

        # 2. Trend strength factor
        trend_strength = self._calculate_trend_strength_optimized(
            rsi_values, index, "uptrend"
        )

        # 3. Price momentum factor
        price_momentum = self._calculate_price_momentum_optimized(
            data, index, "uptrend"
        )

        # 4. RSI momentum factor
        rsi_momentum = self._calculate_rsi_momentum_optimized(
            rsi_values, index, "uptrend"
        )

        return {
            "rsi_position": rsi_position,
            "trend_strength": trend_strength,
            "price_momentum": price_momentum,
            "rsi_momentum": rsi_momentum,
        }

    def _calculate_downtrend_confidence_optimized(
        self, data: Dict[str, Any], rsi_values: np.ndarray, index: int
    ) -> float:
        """Optimized downtrend confidence calculation."""
        current_rsi = rsi_values[index]

        # 1. RSI position factor (vectorized calculation)
        rsi_position = (
            current_rsi - self.parameters["downtrend_pullback_low"]
        ) / self._downtrend_range
        rsi_position = np.clip(rsi_position, 0.0, 1.0)

        # 2. Trend strength factor (optimized lookback)
        trend_strength = self._calculate_trend_strength_optimized(
            rsi_values, index, "downtrend"
        )

        # 3. Price momentum factor (cached arrays)
        price_momentum = self._calculate_price_momentum_optimized(
            data, index, "downtrend"
        )

        # 4. RSI momentum factor (simple calculation)
        rsi_momentum = self._calculate_rsi_momentum_optimized(
            rsi_values, index, "downtrend"
        )

        # Calculate weighted confidence
        confidence_factors = {
            "rsi_position": rsi_position,
            "trend_strength": trend_strength,
            "price_momentum": price_momentum,
            "rsi_momentum": rsi_momentum,
        }

        return self.calculate_confidence(confidence_factors)

    def _calculate_downtrend_confidence_factors(
        self, data: Dict[str, Any], rsi_values: np.ndarray, index: int
    ) -> Dict[str, float]:
        """Calculate downtrend confidence factors (for testing)."""
        current_rsi = rsi_values[index]

        # 1. RSI position factor
        rsi_position = (
            current_rsi - self.parameters["downtrend_pullback_low"]
        ) / self._downtrend_range
        rsi_position = np.clip(rsi_position, 0.0, 1.0)

        # 2. Trend strength factor
        trend_strength = self._calculate_trend_strength_optimized(
            rsi_values, index, "downtrend"
        )

        # 3. Price momentum factor
        price_momentum = self._calculate_price_momentum_optimized(
            data, index, "downtrend"
        )

        # 4. RSI momentum factor
        rsi_momentum = self._calculate_rsi_momentum_optimized(
            rsi_values, index, "downtrend"
        )

        return {
            "rsi_position": rsi_position,
            "trend_strength": trend_strength,
            "price_momentum": price_momentum,
            "rsi_momentum": rsi_momentum,
        }

    def _calculate_trend_strength_optimized(
        self, rsi_values: np.ndarray, index: int, trend_type: str
    ) -> float:
        """Optimized trend strength calculation using vectorized operations."""
        lookback = min(self._trend_periods, index)
        if lookback < 3:
            return 0.5

        start_idx = index - lookback
        recent_rsi = rsi_values[start_idx : index + 1]

        if trend_type == "uptrend":
            above_count = np.sum(recent_rsi > self.parameters["uptrend_threshold"])
            return above_count / len(recent_rsi)
        else:  # downtrend
            below_count = np.sum(recent_rsi < self.parameters["downtrend_threshold"])
            return below_count / len(recent_rsi)

    def _calculate_price_momentum_optimized(
        self, data: Dict[str, Any], index: int, trend_type: str
    ) -> float:
        """Calculate price momentum."""
        # Get arrays directly
        arrays = data.to_arrays()
        close_prices = arrays["close"]
        lookback = min(5, index)

        if lookback < 2 or index >= len(close_prices):
            return 0.5

        start_idx = max(0, index - lookback)  # Ensure positive index
        if start_idx >= len(close_prices) or close_prices[start_idx] == 0:
            return 0.5

        price_change = (close_prices[index] - close_prices[start_idx]) / close_prices[
            start_idx
        ]

        if trend_type == "uptrend":
            return np.clip(price_change * 20 + 0.5, 0.0, 1.0)
        else:  # downtrend
            return np.clip(-price_change * 20 + 0.5, 0.0, 1.0)

    def _calculate_rsi_momentum_optimized(
        self, rsi_values: np.ndarray, index: int, trend_type: str
    ) -> float:
        """Optimized RSI momentum calculation."""
        if index < 2:
            return 0.5

        rsi_change = rsi_values[index] - rsi_values[index - 1]

        if trend_type == "uptrend":
            return np.clip(rsi_change / 10 + 0.5, 0.0, 1.0)
        else:  # downtrend
            return np.clip(-rsi_change / 10 + 0.5, 0.0, 1.0)

    def _check_divergence_signal_optimized(
        self, data: Dict[str, Any], rsi_values: np.ndarray, index: int
    ) -> Dict[str, Any]:
        """Optimized divergence detection with early exit conditions."""
        # Early exit for insufficient data
        if index < 20:
            return None

        # Get arrays directly
        arrays = data.to_arrays()
        close_prices = arrays["close"]
        lookback = min(10, index)
        start_idx = index - lookback

        price_segment = close_prices[start_idx : index + 1]
        rsi_segment = rsi_values[start_idx : index + 1]

        # Check for divergences
        if self._detect_bullish_divergence_optimized(price_segment, rsi_segment):
            return self._create_divergence_signal_optimized(data, index, "bullish")
        elif self._detect_bearish_divergence_optimized(price_segment, rsi_segment):
            return self._create_divergence_signal_optimized(data, index, "bearish")

        return None

    def _detect_bullish_divergence_optimized(
        self, prices: np.ndarray, rsi_values: np.ndarray
    ) -> bool:
        """Optimized bullish divergence detection."""
        # Find lows efficiently
        lows = []
        for i in range(1, len(prices) - 1):
            if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
                lows.append((i, prices[i], rsi_values[i]))

        if len(lows) < 2:
            return False

        # Check most recent two lows
        _, price1, rsi1 = lows[-2]
        _, price2, rsi2 = lows[-1]

        return price2 < price1 and rsi2 > rsi1

    def _detect_bearish_divergence_optimized(
        self, prices: np.ndarray, rsi_values: np.ndarray
    ) -> bool:
        """Optimized bearish divergence detection."""
        # Find highs efficiently
        highs = []
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
                highs.append((i, prices[i], rsi_values[i]))

        if len(highs) < 2:
            return False

        # Check most recent two highs
        _, price1, rsi1 = highs[-2]
        _, price2, rsi2 = highs[-1]

        return price2 > price1 and rsi2 < rsi1

    def _create_divergence_signal_optimized(
        self, data: Dict[str, Any], index: int, direction: str
    ) -> Dict[str, Any]:
        """Optimized divergence signal creation."""
        confidence = 0.7  # Fixed confidence for divergence signals

        if confidence < self._min_confidence:
            return None

        # Get arrays directly
        arrays = data.to_arrays()

        return {
            "timestamp": arrays["timestamp"][index],
            "symbol": data.get("symbol", "UNKNOWN"),
            "signal_type": "BUY" if direction == "bullish" else "SELL",
            "price": arrays["close"][index],
            "confidence": confidence,
            "strategy_name": self.name,
            "metadata": {
                "signal_name": f"RSI {direction.title()} Divergence",
                "divergence_type": direction,
                "entry_type": "divergence",
            },
        }

    def get_required_indicators(self) -> List[str]:
        """Get list of required indicators."""
        return ["rsi"]

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters with descriptions."""
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
            "min_confidence": {
                "value": self.parameters["min_confidence"],
                "description": "Minimum confidence threshold",
                "type": "float",
                "min": 0.1,
                "max": 1.0,
                "default": 0.6,
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

    def _validate_signal_strategy(self, signal: Dict[str, Any]) -> bool:
        """Strategy-specific signal validation ensuring confidence thresholds."""
        # Validate signal type
        if signal.get("signal_type") not in ["BUY", "SELL"]:
            return False

        # Critical: Validate confidence threshold
        if signal.get("confidence", 0) < self._min_confidence:
            return False

        # Validate metadata
        metadata = signal.get("metadata", {})
        if "signal_name" not in metadata or "entry_type" not in metadata:
            return False

        # Validate signal names
        valid_names = [
            "RSI Uptrend Pullback",
            "RSI Downtrend Rally",
            "RSI Bullish Divergence",
            "RSI Bearish Divergence",
        ]

        return metadata["signal_name"] in valid_names

    def _determine_trend(self, rsi_values: np.ndarray, index: int) -> str:
        """Determine trend for a specific index (for backward compatibility)."""
        if index < self._trend_periods:
            return "sideways"

        start_idx = index - self._trend_periods
        recent_rsi = rsi_values[start_idx : index + 1]

        above_threshold = np.sum(recent_rsi > self.parameters["uptrend_threshold"])
        below_threshold = np.sum(recent_rsi < self.parameters["downtrend_threshold"])

        threshold_count = self._trend_periods * 0.7

        if above_threshold >= threshold_count:
            return "uptrend"
        elif below_threshold >= threshold_count:
            return "downtrend"
        else:
            return "sideways"
