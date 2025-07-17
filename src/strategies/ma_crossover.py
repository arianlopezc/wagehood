"""
Moving Average Crossover Strategy (Golden Cross/Death Cross)

Implements the dual moving average crossover strategy using 50-day and 200-day EMAs.
This is the most popular and proven trend following strategy for retail investors.
"""

from typing import List, Dict, Any, Union
import numpy as np
import logging
from datetime import datetime

from .base import TradingStrategy

# Note: Signal types are simplified to basic Python types since core.models doesn't exist
from ..indicators.moving_averages import calculate_ema

logger = logging.getLogger(__name__)


class MovingAverageCrossover(TradingStrategy):
    """
    Moving Average Crossover Strategy

    Uses two EMAs (default 50 and 200 periods) to generate Golden Cross and Death Cross signals.
    Buy signals occur when the short EMA crosses above the long EMA.
    Sell signals occur when the short EMA crosses below the long EMA.
    """

    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the Moving Average Crossover strategy

        Args:
            parameters: Strategy parameters including:
                - short_period: Short EMA period (default: 50)
                - long_period: Long EMA period (default: 200)
                - min_confidence: Minimum confidence threshold (default: 0.6)
                - volume_confirmation: Require volume confirmation (default: True)
                - volume_threshold: Volume threshold multiplier (default: 1.2)
        """
        default_params = {
            "short_period": 50,
            "long_period": 200,
            "min_confidence": 0.6,
            "volume_confirmation": True,
            "volume_threshold": 1.2,  # 20% above average volume
        }

        if parameters:
            default_params.update(parameters)

        super().__init__("MovingAverageCrossover", default_params)

    def generate_signals(
        self, data: Dict[str, Any], indicators: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate Golden Cross and Death Cross signals

        Args:
            data: Market data
            indicators: Pre-calculated indicators

        Returns:
            List of trading signals
        """
        signals = []

        try:
            # Get EMAs from indicators
            short_ema = indicators.get("ema", {}).get(
                f'ema_{self.parameters["short_period"]}'
            )
            long_ema = indicators.get("ema", {}).get(
                f'ema_{self.parameters["long_period"]}'
            )

            if short_ema is None or long_ema is None:
                logger.warning(f"Missing EMA data for {self.name}")
                return signals

            # Convert to numpy arrays for easier calculation
            short_ema = np.array(short_ema)
            long_ema = np.array(long_ema)

            # Need at least 2 periods to detect crossover
            if len(short_ema) < 2 or len(long_ema) < 2:
                return signals

            # Get volume data if volume confirmation is enabled
            volume_data = None
            avg_volume = None
            if self.parameters["volume_confirmation"]:
                arrays = data.to_arrays()
                volume_data = np.array(arrays["volume"])
                if len(volume_data) >= 20:
                    avg_volume = np.mean(volume_data[-20:])  # 20-day average volume

            # Check for crossovers
            for i in range(1, len(short_ema)):
                # Skip if we don't have enough data
                if np.isnan(short_ema[i]) or np.isnan(long_ema[i]):
                    continue
                if np.isnan(short_ema[i - 1]) or np.isnan(long_ema[i - 1]):
                    continue

                # Current and previous values
                short_curr = short_ema[i]
                short_prev = short_ema[i - 1]
                long_curr = long_ema[i]
                long_prev = long_ema[i - 1]

                # Check for Golden Cross (bullish crossover)
                if short_prev <= long_prev and short_curr > long_curr:
                    signal = self._create_golden_cross_signal(
                        data, i, short_curr, long_curr, volume_data, avg_volume
                    )
                    if signal:
                        signals.append(signal)

                # Check for Death Cross (bearish crossover)
                elif short_prev >= long_prev and short_curr < long_curr:
                    signal = self._create_death_cross_signal(
                        data, i, short_curr, long_curr, volume_data, avg_volume
                    )
                    if signal:
                        signals.append(signal)

            # Sort signals by timestamp in descending order (newest first)
            signals.sort(key=lambda s: s["timestamp"], reverse=True)

            return self.validate_signals(signals)

        except Exception as e:
            logger.error(f"Error generating signals for {self.name}: {e}")
            return []

    def _create_golden_cross_signal(
        self,
        data: Dict[str, Any],
        index: int,
        short_ema: float,
        long_ema: float,
        volume_data: np.ndarray = None,
        avg_volume: float = None,
    ) -> Dict[str, Any]:
        """Create Golden Cross (bullish) signal"""

        # Calculate confidence based on multiple factors
        confidence_factors = {}

        # EMA separation factor (wider separation = higher confidence)
        ema_separation = abs(short_ema - long_ema) / long_ema
        confidence_factors["ema_separation"] = min(
            1.0, ema_separation * 10
        )  # Scale to 0-1

        # Volume confirmation
        volume_confidence = 1.0
        if (
            self.parameters["volume_confirmation"]
            and volume_data is not None
            and avg_volume is not None
        ):
            if index < len(volume_data):
                current_volume = volume_data[index]
                volume_ratio = current_volume / avg_volume
                volume_confidence = min(
                    1.0, volume_ratio / self.parameters["volume_threshold"]
                )

        confidence_factors["volume"] = volume_confidence

        # Trend strength (how consistently the short EMA has been rising)
        trend_strength = self._calculate_trend_strength(data, index, "bullish")
        confidence_factors["trend_strength"] = trend_strength

        # Calculate overall confidence
        weights = {"ema_separation": 0.4, "volume": 0.3, "trend_strength": 0.3}

        confidence = self.calculate_confidence(confidence_factors, weights)

        # Only generate signal if confidence meets threshold
        if confidence < self.parameters["min_confidence"]:
            return None

        # Get current price
        arrays = data.to_arrays()
        current_price = arrays["close"][index]

        # Create signal
        signal = {
            "timestamp": arrays["timestamp"][index],
            "symbol": data.get("symbol", "UNKNOWN"),
            "signal_type": "BUY",
            "price": current_price,
            "confidence": confidence,
            "strategy_name": self.name,
            "metadata": self.get_signal_metadata(
                signal_name="Golden Cross",
                short_ema=short_ema,
                long_ema=long_ema,
                ema_separation=ema_separation,
                volume_confirmation=volume_confidence,
                trend_strength=trend_strength,
            ),
        }

        return signal

    def _create_death_cross_signal(
        self,
        data: Dict[str, Any],
        index: int,
        short_ema: float,
        long_ema: float,
        volume_data: np.ndarray = None,
        avg_volume: float = None,
    ) -> Dict[str, Any]:
        """Create Death Cross (bearish) signal"""

        # Calculate confidence based on multiple factors
        confidence_factors = {}

        # EMA separation factor
        ema_separation = abs(short_ema - long_ema) / long_ema
        confidence_factors["ema_separation"] = min(1.0, ema_separation * 10)

        # Volume confirmation
        volume_confidence = 1.0
        if (
            self.parameters["volume_confirmation"]
            and volume_data is not None
            and avg_volume is not None
        ):
            if index < len(volume_data):
                current_volume = volume_data[index]
                volume_ratio = current_volume / avg_volume
                volume_confidence = min(
                    1.0, volume_ratio / self.parameters["volume_threshold"]
                )

        confidence_factors["volume"] = volume_confidence

        # Trend strength (how consistently the short EMA has been falling)
        trend_strength = self._calculate_trend_strength(data, index, "bearish")
        confidence_factors["trend_strength"] = trend_strength

        # Calculate overall confidence
        weights = {"ema_separation": 0.4, "volume": 0.3, "trend_strength": 0.3}

        confidence = self.calculate_confidence(confidence_factors, weights)

        # Only generate signal if confidence meets threshold
        if confidence < self.parameters["min_confidence"]:
            return None

        # Get current price
        arrays = data.to_arrays()
        current_price = arrays["close"][index]

        # Create signal
        signal = {
            "timestamp": arrays["timestamp"][index],
            "symbol": data.get("symbol", "UNKNOWN"),
            "signal_type": "SELL",
            "price": current_price,
            "confidence": confidence,
            "strategy_name": self.name,
            "metadata": self.get_signal_metadata(
                signal_name="Death Cross",
                short_ema=short_ema,
                long_ema=long_ema,
                ema_separation=ema_separation,
                volume_confirmation=volume_confidence,
                trend_strength=trend_strength,
            ),
        }

        return signal

    def _calculate_trend_strength(
        self, data: Dict[str, Any], index: int, direction: str
    ) -> float:
        """Calculate trend strength based on recent price action"""

        arrays = data.to_arrays()
        close_prices = arrays["close"]

        # Look back 10 periods to assess trend strength
        lookback = min(10, index)
        if lookback < 3:
            return 0.5  # Neutral confidence

        start_idx = index - lookback
        recent_closes = close_prices[start_idx : index + 1]

        # Calculate trend direction consistency
        if direction == "bullish":
            # Count periods where price moved up
            up_periods = sum(
                1
                for i in range(1, len(recent_closes))
                if recent_closes[i] > recent_closes[i - 1]
            )
        else:  # bearish
            # Count periods where price moved down
            up_periods = sum(
                1
                for i in range(1, len(recent_closes))
                if recent_closes[i] < recent_closes[i - 1]
            )

        # Calculate strength as percentage of consistent moves
        strength = up_periods / (len(recent_closes) - 1)

        return strength

    def get_required_indicators(self) -> List[str]:
        """Get list of required indicators"""
        return ["ema"]

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters with descriptions"""
        return {
            "short_period": {
                "value": self.parameters["short_period"],
                "description": "Short EMA period",
                "type": "int",
                "min": 10,
                "max": 100,
                "default": 50,
            },
            "long_period": {
                "value": self.parameters["long_period"],
                "description": "Long EMA period",
                "type": "int",
                "min": 100,
                "max": 300,
                "default": 200,
            },
            "min_confidence": {
                "value": self.parameters["min_confidence"],
                "description": "Minimum confidence threshold",
                "type": "float",
                "min": 0.1,
                "max": 1.0,
                "default": 0.6,
            },
            "volume_confirmation": {
                "value": self.parameters["volume_confirmation"],
                "description": "Require volume confirmation",
                "type": "bool",
                "default": True,
            },
            "volume_threshold": {
                "value": self.parameters["volume_threshold"],
                "description": "Volume threshold multiplier",
                "type": "float",
                "min": 1.0,
                "max": 3.0,
                "default": 1.2,
            },
        }

    def _get_parameter_ranges(self) -> Dict[str, List]:
        """Get parameter ranges for optimization"""
        return {
            "short_period": [20, 30, 50, 100],
            "long_period": [100, 150, 200, 250],
            "min_confidence": [0.5, 0.6, 0.7, 0.8],
            "volume_threshold": [1.1, 1.2, 1.5, 2.0],
        }

    def _validate_signal_strategy(self, signal: Dict[str, Any]) -> bool:
        """Strategy-specific signal validation"""

        # Check if signal type is appropriate
        if signal.get("signal_type") not in ["BUY", "SELL"]:
            return False

        # Check confidence threshold
        if signal.get("confidence", 0) < self.parameters["min_confidence"]:
            return False

        # Check metadata for required fields
        metadata = signal.get("metadata", {})
        if not metadata.get("short_ema") or not metadata.get("long_ema"):
            return False

        # Validate EMA separation (should be meaningful)
        ema_separation = metadata.get("ema_separation", 0)
        if ema_separation < 0.001:  # Less than 0.1% separation
            return False

        return True
