"""
Bollinger Band Breakout Strategy

Uses Bollinger Bands to identify volatility expansion and breakout opportunities.
Focuses on price movements beyond normal trading ranges with volume confirmation.
"""

from typing import List, Dict, Any, Union, Optional
import numpy as np
import logging
from datetime import datetime
from ..utils.timezone_utils import utc_now

from .base import TradingStrategy

# Note: Signal types are simplified to basic Python types since core.models doesn't exist
from ..indicators.talib_wrapper import calculate_bb as calculate_bollinger_bands

logger = logging.getLogger(__name__)


class BollingerBandBreakout(TradingStrategy):
    """
    Bollinger Band Breakout Strategy

    Identifies breakout opportunities based on:
    - Price breaking above/below Bollinger Bands
    - Volume confirmation
    - Band width (volatility) conditions
    - Consolidation patterns before breakouts
    """

    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the Bollinger Band Breakout strategy

        Args:
            parameters: Strategy parameters including:
                - bb_period: Bollinger Band period (default: 20)
                - bb_std: Standard deviation multiplier (default: 2.0)
                - consolidation_periods: Minimum consolidation periods (default: 5)
                - volume_confirmation: Require volume confirmation (default: True)
                - volume_threshold: Volume threshold multiplier (default: 1.1)
                - min_confidence: Minimum confidence threshold (default: 0.5)
                - squeeze_threshold: Band width threshold for squeeze (default: 0.1)
                - breakout_strength: Minimum breakout strength (default: 0.5)
                - walking_bands: Enable walking the bands detection (default: True)
        """
        default_params = {
            "bb_period": 20,
            "bb_std": 2.0,
            "consolidation_periods": 5,  # Reduced from 10
            "volume_confirmation": True,
            "volume_threshold": 1.1,  # Relaxed from 1.5
            "min_confidence": 0.5,  # Lowered from 0.6
            "squeeze_threshold": 0.1,
            "breakout_strength": 0.5,
            "walking_bands": True,
        }

        if parameters:
            default_params.update(parameters)

        super().__init__("BollingerBandBreakout", default_params)

    def generate_signals(
        self, data: Dict[str, Any], indicators: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate Bollinger Band breakout signals

        Args:
            data: Market data
            indicators: Pre-calculated indicators

        Returns:
            List of trading signals
        """
        signals = []

        try:
            # Get Bollinger Bands data from indicators
            bb_data = indicators.get("bollinger", {}).get(
                f'bollinger_{self.parameters["bb_period"]}'
            )

            if bb_data is None:
                logger.warning(f"Missing Bollinger Bands data for {self.name}")
                return signals

            # Extract Bollinger Bands components
            upper_band = np.array(bb_data["upper"])
            middle_band = np.array(bb_data["middle"])
            lower_band = np.array(bb_data["lower"])

            # Get price and volume data
            arrays = data.to_arrays()
            close_prices = np.array(arrays["close"])
            high_prices = np.array(arrays["high"])
            low_prices = np.array(arrays["low"])
            volume_data = np.array(arrays["volume"])

            # Need sufficient data for analysis
            if len(close_prices) < self.parameters["consolidation_periods"]:
                return signals

            # Calculate average volume for confirmation
            avg_volume = None
            if self.parameters["volume_confirmation"] and len(volume_data) >= 20:
                avg_volume = np.mean(volume_data[-20:])

            # Analyze each period for breakout signals
            for i in range(self.parameters["consolidation_periods"], len(close_prices)):
                # Skip if we don't have valid data
                if (
                    np.isnan(upper_band[i])
                    or np.isnan(lower_band[i])
                    or np.isnan(middle_band[i])
                    or np.isnan(close_prices[i])
                ):
                    continue

                # Check for bullish breakout
                bullish_signal = self._check_bullish_breakout(
                    close_prices,
                    high_prices,
                    upper_band,
                    middle_band,
                    lower_band,
                    volume_data,
                    avg_volume,
                    i,
                )

                if bullish_signal:
                    signal = self._create_bullish_signal(
                        market_data=data,
                        index=i,
                        current_price=close_prices[i],
                        upper_band=upper_band[i],
                        middle_band=middle_band[i],
                        lower_band=lower_band[i],
                        volume_data=volume_data,
                        avg_volume=avg_volume,
                    )
                    if signal:
                        signals.append(signal)

                # Check for bearish breakout
                bearish_signal = self._check_bearish_breakout(
                    close_prices,
                    low_prices,
                    upper_band,
                    middle_band,
                    lower_band,
                    volume_data,
                    avg_volume,
                    i,
                )

                if bearish_signal:
                    signal = self._create_bearish_signal(
                        market_data=data,
                        index=i,
                        current_price=close_prices[i],
                        upper_band=upper_band[i],
                        middle_band=middle_band[i],
                        lower_band=lower_band[i],
                        volume_data=volume_data,
                        avg_volume=avg_volume,
                    )
                    if signal:
                        signals.append(signal)

                # Check for squeeze breakout if enabled
                squeeze_signal = self._check_squeeze_breakout(
                    data,
                    close_prices,
                    upper_band,
                    middle_band,
                    lower_band,
                    volume_data,
                    avg_volume,
                    i,
                )

                if squeeze_signal:
                    signals.append(squeeze_signal)

            # Sort signals by timestamp in descending order (newest first)
            signals.sort(key=lambda s: s["timestamp"], reverse=True)

            return self.validate_signals(signals)

        except Exception as e:
            logger.error(f"Error generating signals for {self.name}: {e}")
            return []

    def _check_bullish_breakout(
        self,
        close_prices: np.ndarray,
        high_prices: np.ndarray,
        upper_band: np.ndarray,
        middle_band: np.ndarray,
        lower_band: np.ndarray,
        volume_data: np.ndarray,
        avg_volume: float,
        index: int,
    ) -> bool:
        """Check for bullish breakout conditions"""

        # Current values
        current_close = close_prices[index]
        current_high = high_prices[index]
        current_upper = upper_band[index]
        current_middle = middle_band[index]

        # Previous values
        prev_close = close_prices[index - 1]
        prev_upper = upper_band[index - 1]

        # Primary breakout condition: close above upper band
        breakout_condition = current_close > current_upper

        # Alternative condition: high touches upper band with strong close
        touch_condition = (
            current_high >= current_upper
            and current_close > current_middle
            and current_close > prev_close
        )

        if not (breakout_condition or touch_condition):
            return False

        # Check for consolidation before breakout
        consolidation = self._check_consolidation(
            close_prices, upper_band, lower_band, index
        )

        if not consolidation:
            return False

        # Volume confirmation
        if self._requires_volume_confirmation(volume_data, avg_volume, index):
            return False

        return True

    def _check_bearish_breakout(
        self,
        close_prices: np.ndarray,
        low_prices: np.ndarray,
        upper_band: np.ndarray,
        middle_band: np.ndarray,
        lower_band: np.ndarray,
        volume_data: np.ndarray,
        avg_volume: float,
        index: int,
    ) -> bool:
        """Check for bearish breakout conditions"""

        # Current values
        current_close = close_prices[index]
        current_low = low_prices[index]
        current_lower = lower_band[index]
        current_middle = middle_band[index]

        # Previous values
        prev_close = close_prices[index - 1]
        prev_lower = lower_band[index - 1]

        # Primary breakout condition: close below lower band
        breakout_condition = current_close < current_lower

        # Alternative condition: low touches lower band with weak close
        touch_condition = (
            current_low <= current_lower
            and current_close < current_middle
            and current_close < prev_close
        )

        if not (breakout_condition or touch_condition):
            return False

        # Check for consolidation before breakout
        consolidation = self._check_consolidation(
            close_prices, upper_band, lower_band, index
        )

        if not consolidation:
            return False

        # Volume confirmation
        if self._requires_volume_confirmation(volume_data, avg_volume, index):
            return False

        return True

    def _check_consolidation(
        self,
        close_prices: np.ndarray,
        upper_band: np.ndarray,
        lower_band: np.ndarray,
        index: int,
    ) -> bool:
        """Check for consolidation period before breakout"""

        lookback = self.parameters["consolidation_periods"]
        start_idx = max(0, index - lookback)

        # Check if price has been consolidating within bands
        consolidation_prices = close_prices[start_idx:index]
        consolidation_upper = upper_band[start_idx:index]
        consolidation_lower = lower_band[start_idx:index]

        # Count periods where price stayed within bands
        within_bands = 0
        for i in range(len(consolidation_prices)):
            if i < len(consolidation_lower) and i < len(consolidation_upper):
                if (
                    consolidation_lower[i]
                    <= consolidation_prices[i]
                    <= consolidation_upper[i]
                ):
                    within_bands += 1

        # Require at least 50% of periods within bands (relaxed from 70%)
        if len(consolidation_prices) == 0:
            return False
        consolidation_ratio = within_bands / len(consolidation_prices)

        return consolidation_ratio >= 0.5

    def _requires_volume_confirmation(
        self, volume_data: np.ndarray, avg_volume: float, index: int
    ) -> bool:
        """Check if volume confirmation is required but not met"""

        if not self.parameters["volume_confirmation"] or avg_volume is None:
            return False

        if index >= len(volume_data):
            return True  # No volume data available

        current_volume = volume_data[index]
        volume_ratio = current_volume / avg_volume

        # Return True if volume confirmation is required but not met
        return volume_ratio < self.parameters["volume_threshold"]

    def _check_squeeze_breakout(
        self,
        data: Dict[str, Any],
        close_prices: np.ndarray,
        upper_band: np.ndarray,
        middle_band: np.ndarray,
        lower_band: np.ndarray,
        volume_data: np.ndarray,
        avg_volume: float,
        index: int,
    ) -> Optional[Dict[str, Any]]:
        """Check for Bollinger Band squeeze breakout"""

        # Calculate band width (volatility)
        if middle_band[index] == 0:
            return None
        band_width = (upper_band[index] - lower_band[index]) / middle_band[index]

        # Check if we're in a squeeze (low volatility)
        if band_width > self.parameters["squeeze_threshold"]:
            return None

        # Look for expansion after squeeze
        if index < 5:
            return None

        # Check recent band widths
        recent_widths = []
        for i in range(index - 4, index + 1):
            if (
                i >= 0
                and i < len(upper_band)
                and i < len(lower_band)
                and i < len(middle_band)
            ):
                if middle_band[i] != 0:  # Avoid division by zero
                    width = (upper_band[i] - lower_band[i]) / middle_band[i]
                    recent_widths.append(width)

        # Check if band width is expanding
        if len(recent_widths) < 5:
            return None

        # Band width should be increasing
        if not (
            recent_widths[-1] > recent_widths[-2]
            and recent_widths[-2] > recent_widths[-3]
        ):
            return None

        # Determine breakout direction
        current_close = close_prices[index]
        current_middle = middle_band[index]

        if current_close > current_middle:
            # Bullish squeeze breakout
            return self._create_squeeze_signal(
                market_data=data,
                index=index,
                direction="bullish",
                upper_band=upper_band[index],
                middle_band=middle_band[index],
                lower_band=lower_band[index],
                band_width=band_width,
            )
        else:
            # Bearish squeeze breakout
            return self._create_squeeze_signal(
                market_data=data,
                index=index,
                direction="bearish",
                upper_band=upper_band[index],
                middle_band=middle_band[index],
                lower_band=lower_band[index],
                band_width=band_width,
            )

    def _create_bullish_signal(
        self,
        market_data: Dict[str, Any],
        index: int,
        current_price: float,
        upper_band: float,
        middle_band: float,
        lower_band: float,
        volume_data: np.ndarray,
        avg_volume: float,
    ) -> Optional[Dict[str, Any]]:
        """Create bullish breakout signal"""

        # Calculate confidence factors
        confidence_factors = {}

        # Breakout strength (how far above upper band)
        breakout_strength = (current_price - upper_band) / upper_band
        confidence_factors["breakout_strength"] = min(1.0, breakout_strength * 100)

        # Band width (narrower bands = higher confidence)
        band_width = (upper_band - lower_band) / middle_band
        band_width_factor = max(0.0, 1.0 - band_width * 5)  # Inverse relationship
        confidence_factors["band_width"] = band_width_factor

        # Volume confirmation
        volume_factor = 1.0
        if self.parameters["volume_confirmation"] and avg_volume is not None:
            if index < len(volume_data):
                current_volume = volume_data[index]
                volume_ratio = current_volume / avg_volume
                volume_factor = min(
                    1.0, volume_ratio / self.parameters["volume_threshold"]
                )

        confidence_factors["volume"] = volume_factor

        # Price momentum
        momentum_factor = self._calculate_momentum_factor(market_data, index, "bullish")
        confidence_factors["momentum"] = momentum_factor

        # Calculate overall confidence
        weights = {
            "breakout_strength": 0.3,
            "band_width": 0.2,
            "volume": 0.3,
            "momentum": 0.2,
        }

        confidence = self.calculate_confidence(confidence_factors, weights)

        if confidence < self.parameters["min_confidence"]:
            return None

        # Get timestamp
        arrays = market_data.to_arrays()

        # Create signal
        signal = {
            "timestamp": arrays["timestamp"][index],
            "symbol": (
                market_data.symbol
                if hasattr(market_data, "symbol")
                else market_data.get("symbol", "UNKNOWN")
            ),
            "signal_type": "BUY",
            "price": current_price,
            "confidence": confidence,
            "strategy_name": self.name,
            "metadata": self.get_signal_metadata(
                signal_name="Bollinger Band Bullish Breakout",
                upper_band=upper_band,
                middle_band=middle_band,
                lower_band=lower_band,
                band_width=band_width,
                breakout_strength=breakout_strength,
                volume_factor=volume_factor,
                momentum_factor=momentum_factor,
            ),
        }

        return signal

    def _create_bearish_signal(
        self,
        market_data: Dict[str, Any],
        index: int,
        current_price: float,
        upper_band: float,
        middle_band: float,
        lower_band: float,
        volume_data: np.ndarray,
        avg_volume: float,
    ) -> Optional[Dict[str, Any]]:
        """Create bearish breakout signal"""

        # Calculate confidence factors
        confidence_factors = {}

        # Breakout strength (how far below lower band)
        breakout_strength = (lower_band - current_price) / lower_band
        confidence_factors["breakout_strength"] = min(1.0, breakout_strength * 100)

        # Band width (narrower bands = higher confidence)
        band_width = (upper_band - lower_band) / middle_band
        band_width_factor = max(0.0, 1.0 - band_width * 5)
        confidence_factors["band_width"] = band_width_factor

        # Volume confirmation
        volume_factor = 1.0
        if self.parameters["volume_confirmation"] and avg_volume is not None:
            if index < len(volume_data):
                current_volume = volume_data[index]
                volume_ratio = current_volume / avg_volume
                volume_factor = min(
                    1.0, volume_ratio / self.parameters["volume_threshold"]
                )

        confidence_factors["volume"] = volume_factor

        # Price momentum
        momentum_factor = self._calculate_momentum_factor(market_data, index, "bearish")
        confidence_factors["momentum"] = momentum_factor

        # Calculate overall confidence
        weights = {
            "breakout_strength": 0.3,
            "band_width": 0.2,
            "volume": 0.3,
            "momentum": 0.2,
        }

        confidence = self.calculate_confidence(confidence_factors, weights)

        if confidence < self.parameters["min_confidence"]:
            return None

        # Get timestamp
        arrays = market_data.to_arrays()

        # Create signal
        signal = {
            "timestamp": arrays["timestamp"][index],
            "symbol": (
                market_data.symbol
                if hasattr(market_data, "symbol")
                else market_data.get("symbol", "UNKNOWN")
            ),
            "signal_type": "SELL",
            "price": current_price,
            "confidence": confidence,
            "strategy_name": self.name,
            "metadata": self.get_signal_metadata(
                signal_name="Bollinger Band Bearish Breakout",
                upper_band=upper_band,
                middle_band=middle_band,
                lower_band=lower_band,
                band_width=band_width,
                breakout_strength=breakout_strength,
                volume_factor=volume_factor,
                momentum_factor=momentum_factor,
            ),
        }

        return signal

    def _create_squeeze_signal(
        self,
        market_data: Dict[str, Any],
        index: int,
        direction: str,
        upper_band: float,
        middle_band: float,
        lower_band: float,
        band_width: float,
    ) -> Optional[Dict[str, Any]]:
        """Create squeeze breakout signal"""

        # Squeeze signals have moderate confidence
        confidence = 0.7

        if confidence < self.parameters["min_confidence"]:
            return None

        # Get timestamp and price from data
        arrays = market_data.to_arrays()
        current_price = arrays["close"][index]
        signal_type = "BUY" if direction == "bullish" else "SELL"

        # Create signal
        signal = {
            "timestamp": arrays["timestamp"][index],
            "symbol": (
                market_data.symbol
                if hasattr(market_data, "symbol")
                else market_data.get("symbol", "UNKNOWN")
            ),
            "signal_type": signal_type,
            "price": current_price,
            "confidence": confidence,
            "strategy_name": self.name,
            "metadata": self.get_signal_metadata(
                signal_name=f"Bollinger Band {direction.title()} Squeeze",
                upper_band=upper_band,
                middle_band=middle_band,
                lower_band=lower_band,
                band_width=band_width,
                squeeze_type=direction,
            ),
        }

        return signal

    def _calculate_momentum_factor(
        self, market_data: Dict[str, Any], index: int, direction: str
    ) -> float:
        """Calculate momentum factor based on recent price action"""

        arrays = market_data.to_arrays()
        close_prices = arrays["close"]

        # Look back 3 periods for momentum
        lookback = min(3, index)
        if lookback < 2:
            return 0.5

        start_idx = index - lookback
        if start_idx < 0 or index + 1 > len(close_prices):
            return 0.5
        recent_closes = close_prices[start_idx : index + 1]

        # Calculate momentum
        if recent_closes[0] == 0:
            return 0.5
        price_change = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]

        if direction == "bullish":
            # Positive momentum supports bullish breakout
            momentum = max(0.0, min(1.0, price_change * 20 + 0.5))
        else:  # bearish
            # Negative momentum supports bearish breakout
            momentum = max(0.0, min(1.0, -price_change * 20 + 0.5))

        return momentum

    def get_required_indicators(self) -> List[str]:
        """Get list of required indicators"""
        return ["bollinger"]

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters with descriptions"""
        return {
            "bb_period": {
                "value": self.parameters["bb_period"],
                "description": "Bollinger Band period",
                "type": "int",
                "min": 10,
                "max": 50,
                "default": 20,
            },
            "bb_std": {
                "value": self.parameters["bb_std"],
                "description": "Standard deviation multiplier",
                "type": "float",
                "min": 1.5,
                "max": 3.0,
                "default": 2.0,
            },
            "consolidation_periods": {
                "value": self.parameters["consolidation_periods"],
                "description": "Minimum consolidation periods",
                "type": "int",
                "min": 5,
                "max": 20,
                "default": 10,
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
                "default": 1.5,
            },
            "min_confidence": {
                "value": self.parameters["min_confidence"],
                "description": "Minimum confidence threshold",
                "type": "float",
                "min": 0.1,
                "max": 1.0,
                "default": 0.6,
            },
            "squeeze_threshold": {
                "value": self.parameters["squeeze_threshold"],
                "description": "Band width threshold for squeeze",
                "type": "float",
                "min": 0.05,
                "max": 0.2,
                "default": 0.1,
            },
            "walking_bands": {
                "value": self.parameters["walking_bands"],
                "description": "Enable walking the bands detection",
                "type": "bool",
                "default": True,
            },
        }

    def _get_parameter_ranges(self) -> Dict[str, List]:
        """Get parameter ranges for optimization"""
        return {
            "bb_period": [10, 15, 20, 25, 30],
            "bb_std": [1.5, 1.8, 2.0, 2.2, 2.5],
            "consolidation_periods": [5, 10, 15, 20],
            "volume_threshold": [1.2, 1.5, 2.0, 2.5],
            "min_confidence": [0.5, 0.6, 0.7, 0.8],
            "squeeze_threshold": [0.05, 0.1, 0.15, 0.2],
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
        if "signal_name" not in metadata:
            return False

        # Validate signal name
        valid_signal_names = [
            "Bollinger Band Bullish Breakout",
            "Bollinger Band Bearish Breakout",
            "Bollinger Band Bullish Squeeze",
            "Bollinger Band Bearish Squeeze",
        ]

        if metadata["signal_name"] not in valid_signal_names:
            return False

        # Check for required Bollinger Band data
        required_fields = ["upper_band", "middle_band", "lower_band", "band_width"]
        if not all(field in metadata for field in required_fields):
            return False

        return True
