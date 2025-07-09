"""
Support and Resistance Breakout Strategy

Identifies key support and resistance levels and trades breakouts from these levels
with volume confirmation. This is the most advanced strategy requiring experience.
"""

from typing import List, Dict, Any, Union, Tuple
import numpy as np
import logging
from datetime import datetime

from .base import TradingStrategy
from ..core.models import Signal, SignalType, MarketData
from ..indicators.levels import calculate_support_resistance

logger = logging.getLogger(__name__)


class SupportResistanceBreakout(TradingStrategy):
    """
    Support and Resistance Breakout Strategy

    Identifies key S/R levels and trades breakouts:
    - Resistance breakout: Price closes above resistance with volume
    - Support breakout: Price closes below support with volume
    - Multiple touch confirmation for stronger levels
    - Measured move targets based on consolidation height
    """

    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the Support/Resistance Breakout strategy

        Args:
            parameters: Strategy parameters including:
                - lookback_periods: Periods for S/R level identification (default: 50)
                - min_touches: Minimum touches for valid S/R level (default: 2)
                - touch_tolerance: Price tolerance for level touches (default: 0.02)
                - volume_confirmation: Require volume confirmation (default: True)
                - volume_threshold: Volume threshold multiplier (default: 1.5)
                - min_confidence: Minimum confidence threshold (default: 0.7)
                - consolidation_periods: Minimum consolidation periods (default: 10)
                - level_strength_weight: Weight for level strength in confidence (default: 0.3)
                - breakout_strength_weight: Weight for breakout strength (default: 0.4)
                - false_breakout_filter: Enable false breakout filtering (default: True)
        """
        default_params = {
            "lookback_periods": 50,
            "min_touches": 2,
            "touch_tolerance": 0.02,  # 2% tolerance
            "volume_confirmation": True,
            "volume_threshold": 1.5,
            "min_confidence": 0.7,
            "consolidation_periods": 10,
            "level_strength_weight": 0.3,
            "breakout_strength_weight": 0.4,
            "false_breakout_filter": True,
        }

        if parameters:
            default_params.update(parameters)

        super().__init__("SupportResistanceBreakout", default_params)

    def generate_signals(
        self, data: MarketData, indicators: Dict[str, Any]
    ) -> List[Signal]:
        """
        Generate support/resistance breakout signals

        Args:
            data: Market data
            indicators: Pre-calculated indicators

        Returns:
            List of trading signals
        """
        signals = []

        try:
            # Get support/resistance levels from indicators
            sr_data = indicators.get("support_resistance")

            if sr_data is None:
                logger.warning(f"Missing support/resistance data for {self.name}")
                return signals

            # Get price and volume data
            arrays = data.to_arrays()
            close_prices = np.array(arrays["close"])
            high_prices = np.array(arrays["high"])
            low_prices = np.array(arrays["low"])
            volume_data = np.array(arrays["volume"])

            # Need sufficient data for analysis
            if len(close_prices) < self.parameters["lookback_periods"]:
                return signals

            # Calculate average volume for confirmation
            avg_volume = None
            if self.parameters["volume_confirmation"] and len(volume_data) >= 20:
                avg_volume = np.mean(volume_data[-20:])

            # Identify support and resistance levels
            support_levels = self._identify_support_levels(
                close_prices, low_prices, high_prices
            )
            resistance_levels = self._identify_resistance_levels(
                close_prices, low_prices, high_prices
            )

            # Check for breakout signals
            for i in range(self.parameters["consolidation_periods"], len(close_prices)):
                # Check resistance breakouts
                for level_info in resistance_levels:
                    signal = self._check_resistance_breakout(
                        data,
                        i,
                        level_info,
                        close_prices,
                        high_prices,
                        volume_data,
                        avg_volume,
                    )
                    if signal:
                        signals.append(signal)

                # Check support breakouts
                for level_info in support_levels:
                    signal = self._check_support_breakout(
                        data,
                        i,
                        level_info,
                        close_prices,
                        low_prices,
                        volume_data,
                        avg_volume,
                    )
                    if signal:
                        signals.append(signal)

            return self.validate_signals(signals)

        except Exception as e:
            logger.error(f"Error generating signals for {self.name}: {e}")
            return []

    def _identify_support_levels(
        self, close_prices: np.ndarray, low_prices: np.ndarray, high_prices: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Identify significant support levels"""

        support_levels = []

        # Find local lows
        local_lows = self._find_local_lows(low_prices)

        # Group similar levels together
        level_groups = self._group_similar_levels(local_lows, close_prices)

        # Analyze each level group
        for group in level_groups:
            if len(group) >= self.parameters["min_touches"]:
                level_info = self._analyze_level_strength(
                    group, close_prices, low_prices, high_prices, "support"
                )
                if level_info:
                    support_levels.append(level_info)

        return support_levels

    def _identify_resistance_levels(
        self, close_prices: np.ndarray, low_prices: np.ndarray, high_prices: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Identify significant resistance levels"""

        resistance_levels = []

        # Find local highs
        local_highs = self._find_local_highs(high_prices)

        # Group similar levels together
        level_groups = self._group_similar_levels(local_highs, close_prices)

        # Analyze each level group
        for group in level_groups:
            if len(group) >= self.parameters["min_touches"]:
                level_info = self._analyze_level_strength(
                    group, close_prices, low_prices, high_prices, "resistance"
                )
                if level_info:
                    resistance_levels.append(level_info)

        return resistance_levels

    def _find_local_lows(self, prices: np.ndarray) -> List[Tuple[int, float]]:
        """Find local low points in price data"""

        local_lows = []

        # Use a simple approach to find local lows
        for i in range(2, len(prices) - 2):
            if (
                prices[i] < prices[i - 1]
                and prices[i] < prices[i + 1]
                and prices[i] < prices[i - 2]
                and prices[i] < prices[i + 2]
            ):
                local_lows.append((i, prices[i]))

        return local_lows

    def _find_local_highs(self, prices: np.ndarray) -> List[Tuple[int, float]]:
        """Find local high points in price data"""

        local_highs = []

        # Use a simple approach to find local highs
        for i in range(2, len(prices) - 2):
            if (
                prices[i] > prices[i - 1]
                and prices[i] > prices[i + 1]
                and prices[i] > prices[i - 2]
                and prices[i] > prices[i + 2]
            ):
                local_highs.append((i, prices[i]))

        return local_highs

    def _group_similar_levels(
        self, levels: List[Tuple[int, float]], close_prices: np.ndarray
    ) -> List[List[Tuple[int, float]]]:
        """Group similar price levels together"""

        if not levels:
            return []

        # Sort levels by price
        sorted_levels = sorted(levels, key=lambda x: x[1])

        groups = []
        current_group = [sorted_levels[0]]

        for i in range(1, len(sorted_levels)):
            prev_price = sorted_levels[i - 1][1]
            curr_price = sorted_levels[i][1]

            # Calculate tolerance based on current price level
            tolerance = curr_price * self.parameters["touch_tolerance"]

            if abs(curr_price - prev_price) <= tolerance:
                current_group.append(sorted_levels[i])
            else:
                groups.append(current_group)
                current_group = [sorted_levels[i]]

        groups.append(current_group)

        return groups

    def _analyze_level_strength(
        self,
        level_group: List[Tuple[int, float]],
        close_prices: np.ndarray,
        low_prices: np.ndarray,
        high_prices: np.ndarray,
        level_type: str,
    ) -> Dict[str, Any]:
        """Analyze the strength of a support/resistance level"""

        if not level_group:
            return None

        # Calculate average price for the level
        avg_price = np.mean([price for _, price in level_group])

        # Calculate level strength factors
        touch_count = len(level_group)
        time_span = max(idx for idx, _ in level_group) - min(
            idx for idx, _ in level_group
        )

        # Calculate how often the level held
        holds = 0
        tests = 0

        for idx, price in level_group:
            tests += 1
            if level_type == "support":
                # Check if price bounced from support
                if (
                    idx + 3 < len(close_prices)
                    and close_prices[idx + 3] > close_prices[idx]
                ):
                    holds += 1
            else:  # resistance
                # Check if price fell from resistance
                if (
                    idx + 3 < len(close_prices)
                    and close_prices[idx + 3] < close_prices[idx]
                ):
                    holds += 1

        hold_ratio = holds / tests if tests > 0 else 0

        # Calculate level strength score
        strength_score = (
            (touch_count / 10) * 0.4  # Touch count factor
            + (min(time_span / 50, 1.0)) * 0.3  # Time span factor
            + hold_ratio * 0.3  # Hold ratio factor
        )

        return {
            "price": avg_price,
            "type": level_type,
            "touch_count": touch_count,
            "time_span": time_span,
            "hold_ratio": hold_ratio,
            "strength_score": strength_score,
            "indices": [idx for idx, _ in level_group],
        }

    def _check_resistance_breakout(
        self,
        data: MarketData,
        index: int,
        level_info: Dict[str, Any],
        close_prices: np.ndarray,
        high_prices: np.ndarray,
        volume_data: np.ndarray,
        avg_volume: float,
    ) -> Signal:
        """Check for resistance breakout signal"""

        resistance_level = level_info["price"]
        current_close = close_prices[index]
        current_high = high_prices[index]

        # Check breakout conditions
        breakout_condition = current_close > resistance_level

        # Additional confirmation: high should also break resistance
        high_breakout = current_high > resistance_level

        if not (breakout_condition and high_breakout):
            return None

        # Check for consolidation before breakout
        consolidation = self._check_consolidation_near_level(
            close_prices, resistance_level, index, "resistance"
        )

        if not consolidation:
            return None

        # Volume confirmation
        if self._requires_volume_confirmation(volume_data, avg_volume, index):
            return None

        # Calculate confidence
        confidence = self._calculate_breakout_confidence(
            level_info,
            current_close,
            resistance_level,
            "resistance",
            volume_data,
            avg_volume,
            index,
        )

        if confidence < self.parameters["min_confidence"]:
            return None

        # Get timestamp
        arrays = data.to_arrays()

        # Create signal
        signal = Signal(
            timestamp=arrays["timestamp"][index],
            symbol=data.symbol,
            signal_type=SignalType.BUY,
            price=current_close,
            confidence=confidence,
            strategy_name=self.name,
            metadata=self.get_signal_metadata(
                signal_name="Resistance Breakout",
                level_price=resistance_level,
                level_type="resistance",
                touch_count=level_info["touch_count"],
                strength_score=level_info["strength_score"],
                breakout_strength=(current_close - resistance_level) / resistance_level,
                consolidation=consolidation,
            ),
        )

        return signal

    def _check_support_breakout(
        self,
        data: MarketData,
        index: int,
        level_info: Dict[str, Any],
        close_prices: np.ndarray,
        low_prices: np.ndarray,
        volume_data: np.ndarray,
        avg_volume: float,
    ) -> Signal:
        """Check for support breakout signal"""

        support_level = level_info["price"]
        current_close = close_prices[index]
        current_low = low_prices[index]

        # Check breakout conditions
        breakout_condition = current_close < support_level

        # Additional confirmation: low should also break support
        low_breakout = current_low < support_level

        if not (breakout_condition and low_breakout):
            return None

        # Check for consolidation before breakout
        consolidation = self._check_consolidation_near_level(
            close_prices, support_level, index, "support"
        )

        if not consolidation:
            return None

        # Volume confirmation
        if self._requires_volume_confirmation(volume_data, avg_volume, index):
            return None

        # Calculate confidence
        confidence = self._calculate_breakout_confidence(
            level_info,
            current_close,
            support_level,
            "support",
            volume_data,
            avg_volume,
            index,
        )

        if confidence < self.parameters["min_confidence"]:
            return None

        # Get timestamp
        arrays = data.to_arrays()

        # Create signal
        signal = Signal(
            timestamp=arrays["timestamp"][index],
            symbol=data.symbol,
            signal_type=SignalType.SELL,
            price=current_close,
            confidence=confidence,
            strategy_name=self.name,
            metadata=self.get_signal_metadata(
                signal_name="Support Breakout",
                level_price=support_level,
                level_type="support",
                touch_count=level_info["touch_count"],
                strength_score=level_info["strength_score"],
                breakout_strength=(support_level - current_close) / support_level,
                consolidation=consolidation,
            ),
        )

        return signal

    def _check_consolidation_near_level(
        self, close_prices: np.ndarray, level_price: float, index: int, level_type: str
    ) -> bool:
        """Check for consolidation near the S/R level"""

        lookback = self.parameters["consolidation_periods"]
        start_idx = max(0, index - lookback)

        recent_prices = close_prices[start_idx:index]

        if len(recent_prices) < lookback // 2:
            return False

        # Calculate how many periods were near the level
        tolerance = level_price * self.parameters["touch_tolerance"]
        near_level_count = 0

        for price in recent_prices:
            if abs(price - level_price) <= tolerance:
                near_level_count += 1

        # Require at least 30% of periods to be near the level
        consolidation_ratio = near_level_count / len(recent_prices)

        return consolidation_ratio >= 0.3

    def _requires_volume_confirmation(
        self, volume_data: np.ndarray, avg_volume: float, index: int
    ) -> bool:
        """Check if volume confirmation is required but not met"""

        if not self.parameters["volume_confirmation"] or avg_volume is None:
            return False

        if index >= len(volume_data):
            return True

        current_volume = volume_data[index]
        volume_ratio = current_volume / avg_volume

        return volume_ratio < self.parameters["volume_threshold"]

    def _calculate_breakout_confidence(
        self,
        level_info: Dict[str, Any],
        current_price: float,
        level_price: float,
        level_type: str,
        volume_data: np.ndarray,
        avg_volume: float,
        index: int,
    ) -> float:
        """Calculate confidence for breakout signal"""

        confidence_factors = {}

        # Level strength factor
        strength_factor = min(1.0, level_info["strength_score"])
        confidence_factors["level_strength"] = strength_factor

        # Breakout strength factor
        if level_type == "resistance":
            breakout_strength = (current_price - level_price) / level_price
        else:  # support
            breakout_strength = (level_price - current_price) / level_price

        breakout_factor = min(1.0, breakout_strength * 50)  # Scale to 0-1
        confidence_factors["breakout_strength"] = breakout_factor

        # Volume confirmation factor
        volume_factor = 1.0
        if self.parameters["volume_confirmation"] and avg_volume is not None:
            if index < len(volume_data):
                current_volume = volume_data[index]
                volume_ratio = current_volume / avg_volume
                volume_factor = min(
                    1.0, volume_ratio / self.parameters["volume_threshold"]
                )

        confidence_factors["volume"] = volume_factor

        # Touch count factor (more touches = stronger level)
        touch_factor = min(1.0, level_info["touch_count"] / 5)
        confidence_factors["touch_count"] = touch_factor

        # Calculate overall confidence
        weights = {
            "level_strength": self.parameters["level_strength_weight"],
            "breakout_strength": self.parameters["breakout_strength_weight"],
            "volume": 0.2,
            "touch_count": 0.1,
        }

        return self.calculate_confidence(confidence_factors, weights)

    def get_required_indicators(self) -> List[str]:
        """Get list of required indicators"""
        return ["support_resistance"]

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters with descriptions"""
        return {
            "lookback_periods": {
                "value": self.parameters["lookback_periods"],
                "description": "Periods for S/R level identification",
                "type": "int",
                "min": 20,
                "max": 100,
                "default": 50,
            },
            "min_touches": {
                "value": self.parameters["min_touches"],
                "description": "Minimum touches for valid S/R level",
                "type": "int",
                "min": 2,
                "max": 5,
                "default": 2,
            },
            "touch_tolerance": {
                "value": self.parameters["touch_tolerance"],
                "description": "Price tolerance for level touches (as percentage)",
                "type": "float",
                "min": 0.01,
                "max": 0.05,
                "default": 0.02,
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
                "default": 0.7,
            },
            "consolidation_periods": {
                "value": self.parameters["consolidation_periods"],
                "description": "Minimum consolidation periods",
                "type": "int",
                "min": 5,
                "max": 20,
                "default": 10,
            },
            "level_strength_weight": {
                "value": self.parameters["level_strength_weight"],
                "description": "Weight for level strength in confidence calculation",
                "type": "float",
                "min": 0.1,
                "max": 0.5,
                "default": 0.3,
            },
            "breakout_strength_weight": {
                "value": self.parameters["breakout_strength_weight"],
                "description": "Weight for breakout strength in confidence calculation",
                "type": "float",
                "min": 0.1,
                "max": 0.5,
                "default": 0.4,
            },
            "false_breakout_filter": {
                "value": self.parameters["false_breakout_filter"],
                "description": "Enable false breakout filtering",
                "type": "bool",
                "default": True,
            },
        }

    def _get_parameter_ranges(self) -> Dict[str, List]:
        """Get parameter ranges for optimization"""
        return {
            "lookback_periods": [30, 40, 50, 60, 80],
            "min_touches": [2, 3, 4],
            "touch_tolerance": [0.01, 0.015, 0.02, 0.025, 0.03],
            "volume_threshold": [1.2, 1.5, 2.0, 2.5],
            "min_confidence": [0.6, 0.7, 0.8],
            "consolidation_periods": [5, 10, 15, 20],
            "level_strength_weight": [0.2, 0.3, 0.4],
            "breakout_strength_weight": [0.3, 0.4, 0.5],
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
        required_fields = ["signal_name", "level_price", "level_type", "touch_count"]

        if not all(field in metadata for field in required_fields):
            return False

        # Validate signal name
        valid_signal_names = ["Resistance Breakout", "Support Breakout"]

        if metadata["signal_name"] not in valid_signal_names:
            return False

        # Validate level type matches signal type
        if (
            metadata["level_type"] == "resistance"
            and signal.signal_type != SignalType.BUY
        ):
            return False

        if (
            metadata["level_type"] == "support"
            and signal.signal_type != SignalType.SELL
        ):
            return False

        # Validate touch count
        if metadata["touch_count"] < self.parameters["min_touches"]:
            return False

        return True
