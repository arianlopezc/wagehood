"""
MACD + RSI Combined Signal Detection Strategy

Implements high-quality signal detection using MACD trend analysis and RSI momentum.
Combines MACD trend detection with RSI momentum analysis for optimal signal timing.
Focuses on signal quality and confidence rather than trading performance.
"""

from typing import List, Dict, Any, Union
import numpy as np
import logging
from datetime import datetime

from .base import TradingStrategy

# Note: Signal types are simplified to basic Python types since core.models doesn't exist
from ..indicators.talib_wrapper import calculate_macd, calculate_rsi

logger = logging.getLogger(__name__)


class MACDRSIStrategy(TradingStrategy):
    """
    MACD + RSI Combined Signal Detection Strategy

    Uses MACD for trend direction and RSI for momentum timing.
    Focuses on generating high-quality signals with strong confidence indicators.

    Signal conditions:
    - Bullish: MACD line crosses above signal line + RSI momentum confirmation
    - Bearish: MACD line crosses below signal line + RSI momentum confirmation
    - Enhanced with volume confirmation and trend validation
    """

    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the MACD+RSI signal detection strategy

        Args:
            parameters: Strategy parameters including:
                - macd_fast: MACD fast EMA period (default: 12)
                - macd_slow: MACD slow EMA period (default: 26)
                - macd_signal: MACD signal line period (default: 9)
                - rsi_period: RSI period (default: 14)
                - rsi_oversold: RSI oversold level (default: 30)
                - rsi_overbought: RSI overbought level (default: 70)
                - min_confidence: Minimum confidence threshold (default: 0.5)
                - volume_confirmation: Require volume confirmation (default: True)
                - volume_threshold: Volume threshold multiplier (default: 1.1)
                - divergence_detection: Enable divergence detection (default: True)
                - trend_confirmation: Enable trend confirmation (default: True)
        """
        default_params = {
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "min_confidence": 0.5,  # Lowered for more signals
            "volume_confirmation": True,
            "volume_threshold": 1.1,  # Relaxed volume requirement
            "divergence_detection": True,
            "trend_confirmation": True,
        }

        if parameters:
            default_params.update(parameters)

        super().__init__("MACDRSIStrategy", default_params)

    def generate_signals(
        self, data: Dict[str, Any], indicators: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate high-quality MACD+RSI combination signals

        Args:
            data: Market data
            indicators: Pre-calculated indicators

        Returns:
            List of validated market signals with enhanced quality metrics
        """
        signals = []

        # Debug logging
        logger.info(
            f"[MACD+RSI] generate_signals called for {data.get('symbol', 'UNKNOWN')}"
        )
        logger.info(
            f"[MACD+RSI] Data length: {len(data) if hasattr(data, '__len__') else 'N/A'}"
        )
        logger.info(f"[MACD+RSI] Indicators keys: {list(indicators.keys())}")

        try:
            # Get MACD and RSI data from indicators
            macd_data = indicators.get("macd", {})
            rsi_data = indicators.get("rsi", {}).get(
                f'rsi_{self.parameters["rsi_period"]}'
            )

            # Debug logging
            logger.debug(
                f"MACD data keys: {list(macd_data.keys()) if macd_data else 'None'}"
            )
            logger.debug(
                f"RSI data keys: {list(indicators.get('rsi', {}).keys()) if 'rsi' in indicators else 'None'}"
            )
            logger.debug(f"Looking for RSI key: rsi_{self.parameters['rsi_period']}")

            if not macd_data or rsi_data is None:
                logger.error(f"Missing MACD or RSI data for {self.name}")
                logger.error(f"MACD data: {macd_data}")
                logger.error(f"RSI data: {rsi_data}")
                logger.error(f"All indicators: {indicators}")
                return signals

            # Extract MACD components - handle nested structure from base class
            if "macd" in macd_data and isinstance(macd_data["macd"], dict):
                # Base class returns nested structure: {'macd': {'macd': ..., 'signal': ..., 'histogram': ...}}
                macd_components = macd_data["macd"]
                macd_line = np.array(macd_components["macd"])
                signal_line = np.array(macd_components["signal"])
                histogram = np.array(macd_components["histogram"])
            else:
                # Direct structure: {'macd': ..., 'signal': ..., 'histogram': ...}
                macd_line = np.array(macd_data["macd"])
                signal_line = np.array(macd_data["signal"])
                histogram = np.array(macd_data["histogram"])

            rsi_values = np.array(rsi_data)

            # Need at least 2 periods to detect crossovers
            if len(macd_line) < 2 or len(rsi_values) < 2:
                return signals

            # Get volume data if volume confirmation is enabled
            volume_data = None
            avg_volume = None
            if self.parameters["volume_confirmation"]:
                arrays = data.to_arrays()
                volume_data = np.array(arrays["volume"])
                if len(volume_data) >= 20:
                    avg_volume = np.mean(volume_data[-20:])

            # Check for signal combinations
            logger.info(f"[MACD+RSI] Checking {len(macd_line)-1} periods for signals")
            for i in range(1, len(macd_line)):
                # Skip if we don't have enough data
                if (
                    np.isnan(macd_line[i])
                    or np.isnan(signal_line[i])
                    or np.isnan(rsi_values[i])
                    or i >= len(rsi_values)
                ):
                    continue

                # Current and previous values
                macd_curr = macd_line[i]
                macd_prev = macd_line[i - 1]
                signal_curr = signal_line[i]
                signal_prev = signal_line[i - 1]
                rsi_curr = rsi_values[i]
                rsi_prev = rsi_values[i - 1] if i > 0 else rsi_curr

                # Check for bullish signal
                bullish_signal = self._check_bullish_signal(
                    macd_curr,
                    macd_prev,
                    signal_curr,
                    signal_prev,
                    rsi_curr,
                    rsi_prev,
                    histogram[i],
                )

                # Check for bearish signal
                bearish_signal = self._check_bearish_signal(
                    macd_curr,
                    macd_prev,
                    signal_curr,
                    signal_prev,
                    rsi_curr,
                    rsi_prev,
                    histogram[i],
                )

                # IMPORTANT: Only generate one signal per timestamp to avoid conflicts
                # If both conditions are met, choose based on signal strength
                if bullish_signal and bearish_signal:
                    # This should not happen with proper logic, but if it does,
                    # choose the signal with stronger MACD divergence
                    macd_divergence = abs(macd_curr - signal_curr)
                    if histogram[i] > 0:
                        # Positive histogram favors bullish
                        bearish_signal = False
                    else:
                        # Negative histogram favors bearish
                        bullish_signal = False
                    logger.warning(f"[MACD+RSI] Both BUY and SELL conditions met at index {i}, choosing based on histogram: {'BUY' if bullish_signal else 'SELL'}")

                if bullish_signal:
                    logger.info(f"[MACD+RSI] Bullish signal detected at index {i}")
                    signal = self._create_bullish_signal(
                        data,
                        i,
                        macd_curr,
                        signal_curr,
                        rsi_curr,
                        histogram[i],
                        volume_data,
                        avg_volume,
                    )
                    if signal:
                        logger.info(
                            f"[MACD+RSI] Bullish signal created with confidence {signal['confidence']}"
                        )
                        signals.append(signal)
                    else:
                        logger.info(
                            f"[MACD+RSI] Bullish signal filtered out (low confidence)"
                        )

                elif bearish_signal:
                    signal = self._create_bearish_signal(
                        data,
                        i,
                        macd_curr,
                        signal_curr,
                        rsi_curr,
                        histogram[i],
                        volume_data,
                        avg_volume,
                    )
                    if signal:
                        signals.append(signal)

            # Add divergence signals if enabled
            if self.parameters["divergence_detection"]:
                divergence_signals = self._detect_divergence_signals(
                    data, macd_line, rsi_values, volume_data, avg_volume
                )
                signals.extend(divergence_signals)

            # Sort signals by timestamp in descending order (newest first)
            signals.sort(key=lambda s: s["timestamp"], reverse=True)

            return self.validate_signals(signals)

        except Exception as e:
            logger.error(f"Error generating signals for {self.name}: {e}")
            return []

    def _check_bullish_signal(
        self,
        macd_curr: float,
        macd_prev: float,
        signal_curr: float,
        signal_prev: float,
        rsi_curr: float,
        rsi_prev: float,
        histogram: float,
    ) -> bool:
        """Check for bullish signal conditions"""

        # MACD bullish crossover
        macd_crossover = macd_prev <= signal_prev and macd_curr > signal_curr

        # RSI moving above oversold level
        rsi_oversold_exit = (
            rsi_prev <= self.parameters["rsi_oversold"]
            and rsi_curr > self.parameters["rsi_oversold"]
        )

        # Alternative RSI condition: RSI in uptrend zone (above 50)
        rsi_uptrend = rsi_curr > 50

        # MACD histogram trending positive (momentum confirmation)
        histogram_positive = histogram > 0

        # Primary signal: MACD crossover + RSI exit from oversold
        primary_signal = macd_crossover and rsi_oversold_exit

        # Secondary signal: MACD crossover + RSI uptrend + positive histogram
        secondary_signal = macd_crossover and rsi_uptrend and histogram_positive

        # Tertiary signal: Strong MACD crossover with directional confirmation
        macd_strength = (
            abs(macd_curr - signal_curr) / abs(signal_curr) if signal_curr != 0 else 0
        )
        # For bullish, MACD must be above signal line
        strong_macd_signal = macd_crossover and macd_strength > 0.02 and macd_curr > signal_curr

        return primary_signal or secondary_signal or strong_macd_signal

    def _check_bearish_signal(
        self,
        macd_curr: float,
        macd_prev: float,
        signal_curr: float,
        signal_prev: float,
        rsi_curr: float,
        rsi_prev: float,
        histogram: float,
    ) -> bool:
        """Check for bearish signal conditions"""

        # MACD bearish crossover
        macd_crossover = macd_prev >= signal_prev and macd_curr < signal_curr

        # RSI moving below overbought level
        rsi_overbought_exit = (
            rsi_prev >= self.parameters["rsi_overbought"]
            and rsi_curr < self.parameters["rsi_overbought"]
        )

        # Alternative RSI condition: RSI in downtrend zone (below 50)
        rsi_downtrend = rsi_curr < 50

        # MACD histogram trending negative (momentum confirmation)
        histogram_negative = histogram < 0

        # Primary signal: MACD crossover + RSI exit from overbought
        primary_signal = macd_crossover and rsi_overbought_exit

        # Secondary signal: MACD crossover + RSI downtrend + negative histogram
        secondary_signal = macd_crossover and rsi_downtrend and histogram_negative

        # Tertiary signal: Strong MACD crossover with directional confirmation
        macd_strength = (
            abs(macd_curr - signal_curr) / abs(signal_curr) if signal_curr != 0 else 0
        )
        # For bearish, MACD must be below signal line
        strong_macd_signal = macd_crossover and macd_strength > 0.02 and macd_curr < signal_curr

        return primary_signal or secondary_signal or strong_macd_signal

    def _create_bullish_signal(
        self,
        data: Dict[str, Any],
        index: int,
        macd_value: float,
        signal_value: float,
        rsi_value: float,
        histogram: float,
        volume_data: np.ndarray = None,
        avg_volume: float = None,
    ) -> Dict[str, Any]:
        """Create bullish signal"""

        # Calculate enhanced confidence factors for signal quality
        confidence_factors = {}

        # MACD strength (distance between MACD and signal line)
        macd_strength = (
            abs(macd_value - signal_value) / abs(signal_value)
            if signal_value != 0
            else 0
        )
        confidence_factors["macd_strength"] = min(1.0, macd_strength * 100)

        # RSI position quality (closer to oversold = higher confidence for bullish)
        rsi_factor = 1.0 - (
            (rsi_value - self.parameters["rsi_oversold"])
            / (self.parameters["rsi_overbought"] - self.parameters["rsi_oversold"])
        )
        confidence_factors["rsi_position"] = max(0.0, min(1.0, rsi_factor))

        # Histogram momentum confirmation
        histogram_factor = min(1.0, max(0.0, histogram * 10)) if histogram > 0 else 0.0
        confidence_factors["histogram"] = histogram_factor

        # Volume confirmation with enhanced validation
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

        # Market momentum with trend validation
        momentum_factor = self._calculate_momentum_factor(data, index, "bullish")
        confidence_factors["momentum"] = momentum_factor

        # Trend confirmation factor
        if self.parameters["trend_confirmation"]:
            trend_factor = self._calculate_trend_confirmation(data, index, "bullish")
            confidence_factors["trend_confirmation"] = trend_factor

        # Signal timing quality
        timing_factor = self._calculate_signal_timing_quality(data, index)
        confidence_factors["timing_quality"] = timing_factor

        # Calculate overall confidence with enhanced weighting
        weights = {
            "macd_strength": 0.2,
            "rsi_position": 0.2,
            "histogram": 0.15,
            "volume": 0.15,
            "momentum": 0.15,
            "trend_confirmation": 0.1 if self.parameters["trend_confirmation"] else 0.0,
            "timing_quality": 0.05,
        }

        confidence = self.calculate_confidence(confidence_factors, weights)

        # Only generate signal if confidence meets enhanced threshold
        if confidence < self.parameters["min_confidence"]:
            return None

        # Get current price
        arrays = data.to_arrays()
        current_price = arrays["close"][index]

        # Create enhanced signal with quality indicators
        signal = {
            "timestamp": arrays["timestamp"][index],
            "symbol": data.get("symbol", "UNKNOWN"),
            "signal_type": "BUY",
            "price": current_price,
            "confidence": confidence,
            "strategy_name": self.name,
            "metadata": self.get_signal_metadata(
                signal_name="MACD+RSI Bullish",
                macd_value=macd_value,
                signal_value=signal_value,
                rsi_value=rsi_value,
                histogram=histogram,
                macd_strength=macd_strength,
                volume_confirmation=volume_confidence,
                momentum_factor=momentum_factor,
                trend_confirmation=confidence_factors.get("trend_confirmation", 0.0),
                timing_quality=timing_factor,
                signal_quality_score=confidence,
            ),
        }

        return signal

    def _create_bearish_signal(
        self,
        data: Dict[str, Any],
        index: int,
        macd_value: float,
        signal_value: float,
        rsi_value: float,
        histogram: float,
        volume_data: np.ndarray = None,
        avg_volume: float = None,
    ) -> Dict[str, Any]:
        """Create enhanced bearish signal with quality validation"""

        # Calculate enhanced confidence factors for signal quality
        confidence_factors = {}

        # MACD strength
        macd_strength = (
            abs(macd_value - signal_value) / abs(signal_value)
            if signal_value != 0
            else 0
        )
        confidence_factors["macd_strength"] = min(1.0, macd_strength * 100)

        # RSI position quality (closer to overbought = higher confidence for bearish)
        rsi_factor = (rsi_value - self.parameters["rsi_oversold"]) / (
            self.parameters["rsi_overbought"] - self.parameters["rsi_oversold"]
        )
        confidence_factors["rsi_position"] = max(0.0, min(1.0, rsi_factor))

        # Histogram momentum confirmation
        histogram_factor = min(1.0, max(0.0, -histogram * 10)) if histogram < 0 else 0.0
        confidence_factors["histogram"] = histogram_factor

        # Volume confirmation with enhanced validation
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

        # Market momentum with trend validation
        momentum_factor = self._calculate_momentum_factor(data, index, "bearish")
        confidence_factors["momentum"] = momentum_factor

        # Trend confirmation factor
        if self.parameters["trend_confirmation"]:
            trend_factor = self._calculate_trend_confirmation(data, index, "bearish")
            confidence_factors["trend_confirmation"] = trend_factor

        # Signal timing quality
        timing_factor = self._calculate_signal_timing_quality(data, index)
        confidence_factors["timing_quality"] = timing_factor

        # Calculate overall confidence with enhanced weighting
        weights = {
            "macd_strength": 0.2,
            "rsi_position": 0.2,
            "histogram": 0.15,
            "volume": 0.15,
            "momentum": 0.15,
            "trend_confirmation": 0.1 if self.parameters["trend_confirmation"] else 0.0,
            "timing_quality": 0.05,
        }

        confidence = self.calculate_confidence(confidence_factors, weights)

        # Only generate signal if confidence meets enhanced threshold
        if confidence < self.parameters["min_confidence"]:
            return None

        # Get current price
        arrays = data.to_arrays()
        current_price = arrays["close"][index]

        # Create enhanced signal with quality indicators
        signal = {
            "timestamp": arrays["timestamp"][index],
            "symbol": data.get("symbol", "UNKNOWN"),
            "signal_type": "SELL",
            "price": current_price,
            "confidence": confidence,
            "strategy_name": self.name,
            "metadata": self.get_signal_metadata(
                signal_name="MACD+RSI Bearish",
                macd_value=macd_value,
                signal_value=signal_value,
                rsi_value=rsi_value,
                histogram=histogram,
                macd_strength=macd_strength,
                volume_confirmation=volume_confidence,
                momentum_factor=momentum_factor,
                trend_confirmation=confidence_factors.get("trend_confirmation", 0.0),
                timing_quality=timing_factor,
                signal_quality_score=confidence,
            ),
        }

        return signal

    def _calculate_momentum_factor(
        self, data: Dict[str, Any], index: int, direction: str
    ) -> float:
        """Calculate momentum factor based on recent price action"""

        arrays = data.to_arrays()
        close_prices = arrays["close"]

        # Look back 5 periods for momentum
        lookback = min(5, index)
        if lookback < 2:
            return 0.5

        start_idx = index - lookback
        recent_closes = close_prices[start_idx : index + 1]

        # Calculate momentum based on price change
        price_change = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]

        if direction == "bullish":
            # Positive momentum supports bullish signal
            momentum = max(0.0, min(1.0, price_change * 10 + 0.5))
        else:  # bearish
            # Negative momentum supports bearish signal
            momentum = max(0.0, min(1.0, -price_change * 10 + 0.5))

        return momentum

    def _calculate_trend_confirmation(
        self, data: Dict[str, Any], index: int, direction: str
    ) -> float:
        """Calculate trend confirmation factor based on multiple timeframe analysis"""

        arrays = data.to_arrays()
        close_prices = arrays["close"]

        # Look at longer-term trend
        lookback = min(20, index)
        if lookback < 5:
            return 0.5

        start_idx = index - lookback
        long_term_trend = close_prices[start_idx : index + 1]

        # Calculate trend strength
        if len(long_term_trend) < 3:
            return 0.5

        # Linear regression slope approximation
        if direction == "bullish":
            # Count upward moves
            up_moves = sum(
                1
                for i in range(1, len(long_term_trend))
                if long_term_trend[i] > long_term_trend[i - 1]
            )
            trend_strength = up_moves / (len(long_term_trend) - 1)
        else:
            # Count downward moves
            down_moves = sum(
                1
                for i in range(1, len(long_term_trend))
                if long_term_trend[i] < long_term_trend[i - 1]
            )
            trend_strength = down_moves / (len(long_term_trend) - 1)

        return trend_strength

    def _calculate_signal_timing_quality(
        self, data: Dict[str, Any], index: int
    ) -> float:
        """Calculate signal timing quality based on market conditions"""

        arrays = data.to_arrays()
        close_prices = arrays["close"]

        # Check for clean price action (less noise)
        lookback = min(5, index)
        if lookback < 3:
            return 0.5

        start_idx = index - lookback
        recent_prices = close_prices[start_idx : index + 1]

        # Calculate price volatility (lower is better for timing)
        if len(recent_prices) < 2:
            return 0.5

        price_changes = [
            abs(recent_prices[i] - recent_prices[i - 1]) / recent_prices[i - 1]
            for i in range(1, len(recent_prices))
        ]

        avg_volatility = sum(price_changes) / len(price_changes)

        # Lower volatility = better timing quality
        timing_quality = max(0.0, 1.0 - avg_volatility * 50)

        return min(1.0, timing_quality)

    def _detect_divergence_signals(
        self,
        data: Dict[str, Any],
        macd_line: np.ndarray,
        rsi_values: np.ndarray,
        volume_data: np.ndarray = None,
        avg_volume: float = None,
    ) -> List[Dict[str, Any]]:
        """Detect price-indicator divergence signals"""

        signals = []
        arrays = data.to_arrays()
        close_prices = arrays["close"]

        # Need at least 20 periods for divergence detection
        if len(close_prices) < 20:
            return signals

        # Look for divergences in the last 20 periods
        for i in range(20, len(close_prices)):
            # Find recent highs and lows
            recent_period = 10
            start_idx = max(0, i - recent_period)

            price_segment = close_prices[start_idx : i + 1]
            macd_segment = macd_line[start_idx : i + 1]
            rsi_segment = rsi_values[start_idx : i + 1]

            # Check for bullish divergence (price makes lower low, indicators make higher low)
            bullish_divergence = self._check_bullish_divergence(
                price_segment, macd_segment, rsi_segment
            )

            if bullish_divergence:
                signal = self._create_divergence_signal(
                    data, i, "bullish", volume_data, avg_volume
                )
                if signal:
                    signals.append(signal)

            # Check for bearish divergence (price makes higher high, indicators make lower high)
            bearish_divergence = self._check_bearish_divergence(
                price_segment, macd_segment, rsi_segment
            )

            if bearish_divergence:
                signal = self._create_divergence_signal(
                    data, i, "bearish", volume_data, avg_volume
                )
                if signal:
                    signals.append(signal)

        return signals

    def _check_bullish_divergence(
        self, prices: np.ndarray, macd: np.ndarray, rsi: np.ndarray
    ) -> bool:
        """Check for bullish divergence pattern"""

        # Find the two most recent lows
        price_lows = []
        for i in range(1, len(prices) - 1):
            if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
                price_lows.append((i, prices[i]))

        if len(price_lows) < 2:
            return False

        # Get the two most recent lows
        low1_idx, low1_price = price_lows[-2]
        low2_idx, low2_price = price_lows[-1]

        # Check if price made lower low
        if low2_price >= low1_price:
            return False

        # Check if MACD made higher low
        macd_low1 = macd[low1_idx]
        macd_low2 = macd[low2_idx]

        # Check if RSI made higher low
        rsi_low1 = rsi[low1_idx]
        rsi_low2 = rsi[low2_idx]

        # Bullish divergence: price lower low, but MACD and RSI higher lows
        return macd_low2 > macd_low1 and rsi_low2 > rsi_low1

    def _check_bearish_divergence(
        self, prices: np.ndarray, macd: np.ndarray, rsi: np.ndarray
    ) -> bool:
        """Check for bearish divergence pattern"""

        # Find the two most recent highs
        price_highs = []
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
                price_highs.append((i, prices[i]))

        if len(price_highs) < 2:
            return False

        # Get the two most recent highs
        high1_idx, high1_price = price_highs[-2]
        high2_idx, high2_price = price_highs[-1]

        # Check if price made higher high
        if high2_price <= high1_price:
            return False

        # Check if MACD made lower high
        macd_high1 = macd[high1_idx]
        macd_high2 = macd[high2_idx]

        # Check if RSI made lower high
        rsi_high1 = rsi[high1_idx]
        rsi_high2 = rsi[high2_idx]

        # Bearish divergence: price higher high, but MACD and RSI lower highs
        return macd_high2 < macd_high1 and rsi_high2 < rsi_high1

    def _create_divergence_signal(
        self,
        data: Dict[str, Any],
        index: int,
        direction: str,
        volume_data: np.ndarray = None,
        avg_volume: float = None,
    ) -> Dict[str, Any]:
        """Create divergence signal"""

        # Divergence signals have moderate confidence
        base_confidence = 0.7

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

        confidence = base_confidence * volume_confidence

        # Only generate signal if confidence meets threshold
        if confidence < self.parameters["min_confidence"]:
            return None

        # Get current price
        arrays = data.to_arrays()
        current_price = arrays["close"][index]

        signal_type = "BUY" if direction == "bullish" else "SELL"

        # Create signal
        signal = {
            "timestamp": arrays["timestamp"][index],
            "symbol": data.get("symbol", "UNKNOWN"),
            "signal_type": signal_type,
            "price": current_price,
            "confidence": confidence,
            "strategy_name": self.name,
            "metadata": self.get_signal_metadata(
                signal_name=f"MACD+RSI {direction.title()} Divergence",
                divergence_type=direction,
                volume_confirmation=volume_confidence,
            ),
        }

        return signal

    def get_required_indicators(self) -> List[str]:
        """Get list of required indicators"""
        return ["macd", "rsi"]

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters with descriptions"""
        return {
            "macd_fast": {
                "value": self.parameters["macd_fast"],
                "description": "MACD fast EMA period",
                "type": "int",
                "min": 8,
                "max": 20,
                "default": 12,
            },
            "macd_slow": {
                "value": self.parameters["macd_slow"],
                "description": "MACD slow EMA period",
                "type": "int",
                "min": 20,
                "max": 35,
                "default": 26,
            },
            "macd_signal": {
                "value": self.parameters["macd_signal"],
                "description": "MACD signal line period",
                "type": "int",
                "min": 7,
                "max": 15,
                "default": 9,
            },
            "rsi_period": {
                "value": self.parameters["rsi_period"],
                "description": "RSI period",
                "type": "int",
                "min": 10,
                "max": 25,
                "default": 14,
            },
            "rsi_oversold": {
                "value": self.parameters["rsi_oversold"],
                "description": "RSI oversold level",
                "type": "float",
                "min": 20,
                "max": 40,
                "default": 30,
            },
            "rsi_overbought": {
                "value": self.parameters["rsi_overbought"],
                "description": "RSI overbought level",
                "type": "float",
                "min": 60,
                "max": 80,
                "default": 70,
            },
            "min_confidence": {
                "value": self.parameters["min_confidence"],
                "description": "Minimum confidence threshold for signal quality",
                "type": "float",
                "min": 0.3,
                "max": 1.0,
                "default": 0.5,
            },
            "volume_confirmation": {
                "value": self.parameters["volume_confirmation"],
                "description": "Require volume confirmation",
                "type": "bool",
                "default": True,
            },
            "volume_threshold": {
                "value": self.parameters["volume_threshold"],
                "description": "Volume threshold multiplier for signal validation",
                "type": "float",
                "min": 1.0,
                "max": 3.0,
                "default": 1.1,
            },
            "divergence_detection": {
                "value": self.parameters["divergence_detection"],
                "description": "Enable divergence detection",
                "type": "bool",
                "default": True,
            },
            "trend_confirmation": {
                "value": self.parameters["trend_confirmation"],
                "description": "Enable trend confirmation analysis",
                "type": "bool",
                "default": True,
            },
        }

    def _get_parameter_ranges(self) -> Dict[str, List]:
        """Get parameter ranges for signal quality optimization"""
        return {
            "macd_fast": [8, 10, 12, 15],
            "macd_slow": [20, 24, 26, 30],
            "macd_signal": [7, 9, 11, 14],
            "rsi_period": [10, 14, 18, 21],
            "rsi_oversold": [25, 30, 35],
            "rsi_overbought": [65, 70, 75],
            "min_confidence": [0.6, 0.7, 0.8, 0.9],  # Higher range for quality
            "volume_threshold": [1.2, 1.3, 1.5, 2.0],  # Stricter volume requirements
        }

    def _validate_signal_strategy(self, signal: Dict[str, Any]) -> bool:
        """Enhanced strategy-specific signal validation for quality assurance"""

        # Check if signal type is appropriate
        if signal.get("signal_type") not in ["BUY", "SELL"]:
            return False

        # Check enhanced confidence threshold
        if signal.get("confidence", 0) < self.parameters["min_confidence"]:
            return False

        # Check metadata for required fields
        metadata = signal.get("metadata", {})
        if "signal_name" not in metadata:
            return False

        # Validate signal name
        valid_signal_names = [
            "MACD+RSI Bullish",
            "MACD+RSI Bearish",
            "MACD+RSI Bullish Divergence",
            "MACD+RSI Bearish Divergence",
        ]

        if metadata["signal_name"] not in valid_signal_names:
            return False

        # Additional quality checks
        if "signal_quality_score" in metadata:
            if metadata["signal_quality_score"] < 0.5:
                return False

        # Check for required technical indicators
        required_indicators = ["macd_value", "rsi_value", "histogram"]
        if not all(indicator in metadata for indicator in required_indicators):
            return False

        # Validate indicator values are reasonable
        if "rsi_value" in metadata:
            rsi_val = metadata["rsi_value"]
            if not (0 <= rsi_val <= 100):
                return False

        # Check for trend confirmation if enabled
        if (
            self.parameters["trend_confirmation"]
            and "trend_confirmation" not in metadata
        ):
            return False

        return True
