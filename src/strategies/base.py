"""
Base strategy class for signal detection strategies
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple, TYPE_CHECKING
import logging
from datetime import datetime
from ..utils.timezone_utils import utc_now
import itertools

if TYPE_CHECKING:
    import numpy as np

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Note: These types are simplified to basic Python types since core.models doesn't exist
from ..indicators import IndicatorCalculator

logger = logging.getLogger(__name__)


class TradingStrategy(ABC):
    """Abstract base class for all signal detection strategies"""

    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        """
        Initialize the signal detection strategy

        Args:
            name: Strategy name
            parameters: Strategy parameters dictionary
        """
        self.name = name
        self.parameters = parameters or {}
        self.indicator_calculator = IndicatorCalculator()
        self._last_signal_time = None

    @abstractmethod
    def generate_signals(
        self, data: Dict[str, Any], indicators: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate market signals based on market data and indicators

        Args:
            data: Market data dictionary including OHLCV
            indicators: Pre-calculated indicators

        Returns:
            List of high-quality market signal dictionaries
        """
        pass

    @abstractmethod
    def get_required_indicators(self) -> List[str]:
        """
        Get list of required indicators for this strategy

        Returns:
            List of indicator names
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get strategy parameters with descriptions and defaults

        Returns:
            Dictionary of parameters with metadata
        """
        pass

    def validate_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and filter signals based on quality criteria

        Args:
            signals: List of raw signal dictionaries

        Returns:
            List of validated, high-quality signal dictionaries
        """
        # Use list comprehension for better performance
        validated_signals = [
            signal
            for signal in signals
            if self._validate_signal_basic(signal)
            and self._validate_signal_strategy(signal)
        ]

        # Additional quality validation
        return self._enhance_signal_quality(validated_signals)

    def _validate_signal_basic(self, signal: Dict[str, Any]) -> bool:
        """Basic signal validation with enhanced quality checks"""
        # Enhanced validation for signal quality
        return (
            signal.get("confidence", 0) >= 0.3  # Higher minimum confidence
            and signal.get("price", 0) > 0
            and signal.get("timestamp") is not None
            and signal.get("signal_type") in ["BUY", "SELL"]
            and signal.get("metadata") is not None
        )

    def _validate_signal_strategy(self, signal: Dict[str, Any]) -> bool:
        """Strategy-specific signal validation - override in subclasses"""
        return True

    def optimize_parameters(
        self, data: Dict[str, Any], metric: str = "signal_quality"
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters for signal quality

        Args:
            data: Historical market data dictionary
            metric: Optimization metric ('signal_quality', 'signal_count', 'confidence_avg')

        Returns:
            Optimized parameters dictionary
        """
        logger.info(f"Optimizing parameters for {self.name} using {metric} metric")

        # Get parameter ranges for optimization
        param_ranges = self._get_parameter_ranges()

        if not param_ranges:
            logger.warning(f"No parameter ranges defined for {self.name}")
            return self.parameters

        # Simple grid search optimization
        best_params = self.parameters.copy()
        best_score = float("-inf")

        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(param_ranges)

        for params in param_combinations:
            try:
                # Test strategy with these parameters
                old_params = self.parameters.copy()
                self.parameters.update(params)

                # Calculate signal quality score
                score = self._calculate_signal_quality_score(data, metric)

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

                # Restore original parameters
                self.parameters = old_params

            except Exception as e:
                logger.error(f"Error testing parameters {params}: {e}")
                continue

        logger.info(
            f"Best parameters for {self.name}: {best_params} (score: {best_score:.4f})"
        )
        return best_params

    def _get_parameter_ranges(self) -> Dict[str, List]:
        """Get parameter ranges for optimization - override in subclasses"""
        return {}

    def _generate_parameter_combinations(
        self, param_ranges: Dict[str, List]
    ) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for optimization"""
        if not param_ranges:
            return [{}]

        keys = list(param_ranges.keys())
        values = list(param_ranges.values())

        # Use list comprehension for better performance
        return [
            dict(zip(keys, combination)) for combination in itertools.product(*values)
        ]

    def _calculate_signal_quality_score(
        self, data: Dict[str, Any], metric: str
    ) -> float:
        """Calculate signal quality score for optimization"""
        try:
            # Generate signals for the data
            indicators = self._calculate_indicators(data)
            signals = self.generate_signals(data, indicators)

            if not signals:
                return float("-inf")

            # Calculate signal quality metrics
            if metric == "signal_quality":
                return self._calculate_overall_signal_quality(signals)
            elif metric == "signal_count":
                return len(signals)
            elif metric == "confidence_avg":
                return sum(s.get("confidence", 0) for s in signals) / len(signals)
            else:
                return float("-inf")

        except Exception as e:
            logger.error(f"Error calculating signal quality score: {e}")
            return float("-inf")

    def _calculate_indicators(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate all required indicators for the strategy"""
        indicators = {}
        required_indicators = self.get_required_indicators()

        # Convert data to arrays once
        arrays = data.get("data", [])
        close_prices = [item.get("close", 0) for item in arrays] if arrays else []

        # Define indicator mapping for cleaner code
        indicator_calculators = {
            "sma": lambda: self._calculate_sma(close_prices),
            "ema": lambda: self._calculate_ema(close_prices),
            "rsi": lambda: self._calculate_rsi(close_prices),
            "macd": lambda: self._calculate_macd(close_prices),
            "bollinger": lambda: self._calculate_bollinger_bands(close_prices),
            "support_resistance": lambda: self._calculate_support_resistance(arrays),
        }

        for indicator in required_indicators:
            if indicator in indicator_calculators:
                try:
                    indicators[indicator] = indicator_calculators[indicator]()
                except Exception as e:
                    logger.error(f"Error calculating {indicator}: {e}")
                    continue

        return indicators

    def _get_common_periods(self, indicator_type: str) -> Tuple[int, ...]:
        """Get common periods for indicators"""
        if indicator_type == "sma":
            return (20, 50, 100, 200)
        elif indicator_type == "ema":
            return (12, 26, 50, 200)
        elif indicator_type == "rsi":
            return (14, 21)
        elif indicator_type == "bollinger":
            return (20, 50)
        return ()

    def _calculate_sma(
        self, close_prices: Union[List[float], "np.ndarray"]
    ) -> Dict[str, Any]:
        """Calculate Simple Moving Average"""
        from ..indicators.moving_averages import calculate_sma

        results = {}
        data_len = len(close_prices)

        # Only calculate for periods that have enough data
        for period in self._get_common_periods("sma"):
            if period <= data_len:
                results[f"sma_{period}"] = calculate_sma(close_prices, period)

        return results

    def _calculate_ema(
        self, close_prices: Union[List[float], "np.ndarray"]
    ) -> Dict[str, Any]:
        """Calculate Exponential Moving Average"""
        from ..indicators.moving_averages import calculate_ema

        results = {}
        data_len = len(close_prices)

        for period in self._get_common_periods("ema"):
            if period <= data_len:
                results[f"ema_{period}"] = calculate_ema(close_prices, period)

        return results

    def _calculate_rsi(
        self, close_prices: Union[List[float], "np.ndarray"]
    ) -> Dict[str, Any]:
        """Calculate RSI"""
        from ..indicators.talib_wrapper import calculate_rsi

        results = {}
        data_len = len(close_prices)

        # Calculate RSI for common periods
        for period in self._get_common_periods("rsi"):
            if period <= data_len:
                results[f"rsi_{period}"] = calculate_rsi(close_prices, period)

        # Calculate RSI for strategy-specific periods if they exist
        if hasattr(self, "parameters") and self.parameters:
            # Check for strategy-specific RSI period
            if "rsi_period" in self.parameters:
                rsi_period = self.parameters["rsi_period"]
                if rsi_period <= data_len and f"rsi_{rsi_period}" not in results:
                    results[f"rsi_{rsi_period}"] = calculate_rsi(
                        close_prices, rsi_period
                    )

        return results

    def _calculate_macd(
        self, close_prices: Union[List[float], np.ndarray]
    ) -> Dict[str, Any]:
        """Calculate MACD"""
        from ..indicators.talib_wrapper import calculate_macd

        # Use strategy-specific parameters if available, otherwise use defaults
        fast_period = 12
        slow_period = 26
        signal_period = 9

        if hasattr(self, "parameters") and self.parameters:
            fast_period = self.parameters.get("macd_fast", 12)
            slow_period = self.parameters.get("macd_slow", 26)
            signal_period = self.parameters.get("macd_signal", 9)

        if len(close_prices) >= slow_period:
            macd_line, signal_line, histogram = calculate_macd(
                close_prices, fast_period, slow_period, signal_period
            )
            # Structure the data as expected by strategies
            return {
                "macd": {
                    "macd": macd_line,
                    "signal": signal_line,
                    "histogram": histogram,
                }
            }

        return {}

    def _calculate_bollinger_bands(
        self, close_prices: Union[List[float], "np.ndarray"]
    ) -> Dict[str, Any]:
        """Calculate Bollinger Bands"""
        from ..indicators.talib_wrapper import calculate_bb as calculate_bollinger_bands

        results = {}
        data_len = len(close_prices)

        for period in self._get_common_periods("bollinger"):
            if period <= data_len:
                upper_band, middle_band, lower_band = calculate_bollinger_bands(
                    close_prices, period, 2.0
                )
                # Structure the data as expected by strategies
                results[f"bollinger_{period}"] = {
                    "upper": upper_band,
                    "middle": middle_band,
                    "lower": lower_band,
                }

        return results

    def _calculate_support_resistance(self, arrays: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Support and Resistance levels"""
        from ..indicators.levels import calculate_support_resistance

        if len(arrays["close"]) >= 20:
            # Use close prices for support/resistance calculation
            return calculate_support_resistance(
                arrays["close"], lookback=20, min_touches=3
            )

        return {}

    def _calculate_overall_signal_quality(self, signals: List[Dict[str, Any]]) -> float:
        """Calculate overall signal quality score"""
        if not signals:
            return 0.0

        quality_factors = []

        for signal in signals:
            # Calculate individual signal quality
            confidence_score = signal.get("confidence", 0)

            # Check for signal timing quality
            timing_score = self._assess_signal_timing(signal)

            # Check for signal context quality
            context_score = self._assess_signal_context(signal)

            # Check for signal validation quality
            validation_score = self._assess_signal_validation(signal)

            # Combined quality score
            signal_quality = (
                confidence_score * 0.4
                + timing_score * 0.25
                + context_score * 0.2
                + validation_score * 0.15
            )

            quality_factors.append(signal_quality)

        return sum(quality_factors) / len(quality_factors)

    def calculate_confidence(
        self, conditions: Dict[str, float], weights: Dict[str, float] = None
    ) -> float:
        """
        Calculate signal confidence based on multiple conditions with enhanced quality assessment

        Args:
            conditions: Dictionary of condition names and their strength (0.0 to 1.0)
            weights: Optional weights for each condition

        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not conditions:
            return 0.0

        # Use default weights if not provided
        if weights is None:
            total_weight = len(conditions)
            weighted_sum = sum(conditions.values())
        else:
            # Calculate weighted sum and total weight in single pass
            weighted_sum = 0.0
            total_weight = 0.0
            for key, value in conditions.items():
                weight = weights.get(key, 1.0)
                weighted_sum += value * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        # Calculate base confidence
        base_confidence = weighted_sum / total_weight

        # Apply quality enhancement
        enhanced_confidence = self._enhance_confidence_quality(
            base_confidence, conditions
        )

        # Clamp between 0 and 1
        return max(0.0, min(1.0, enhanced_confidence))

    def get_signal_metadata(self, **kwargs) -> Dict[str, Any]:
        """
        Generate enhanced metadata for signals with quality indicators

        Args:
            **kwargs: Additional metadata fields

        Returns:
            Enhanced metadata dictionary
        """
        metadata = {
            "strategy": self.name,
            "parameters": self.parameters.copy(),
            "timestamp": utc_now().isoformat(),
            "signal_quality": self._calculate_signal_quality_metadata(kwargs),
            "market_context": self._get_market_context_metadata(),
        }
        metadata.update(kwargs)
        return metadata

    def _calculate_signal_quality_metadata(
        self, signal_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate signal quality metadata"""
        quality_indicators = {
            "indicator_count": len(
                [k for k in signal_data.keys() if "indicator" in k.lower()]
            ),
            "confirmation_count": len(
                [k for k in signal_data.keys() if "confirmation" in k.lower()]
            ),
            "has_volume_confirmation": "volume_confirmation" in signal_data,
            "has_trend_confirmation": any(
                "trend" in k.lower() for k in signal_data.keys()
            ),
        }

        # Calculate overall quality score
        quality_score = (
            quality_indicators["indicator_count"] * 0.3
            + quality_indicators["confirmation_count"] * 0.3
            + (1.0 if quality_indicators["has_volume_confirmation"] else 0.0) * 0.2
            + (1.0 if quality_indicators["has_trend_confirmation"] else 0.0) * 0.2
        )

        quality_indicators["quality_score"] = min(1.0, quality_score)
        return quality_indicators

    def _get_market_context_metadata(self) -> Dict[str, Any]:
        """Get market context metadata"""
        return {
            "analysis_time": utc_now().isoformat(),
            "strategy_type": "signal_detection",
            "optimization_target": "signal_quality",
        }

    def _enhance_signal_quality(
        self, signals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enhance signal quality through additional filtering and validation"""
        if not signals:
            return signals

        # Sort by timestamp (most recent first)
        sorted_signals = sorted(
            signals, key=lambda s: s.get("timestamp", datetime.min), reverse=True
        )

        # Apply quality enhancements without filtering
        enhanced_signals = []
        for signal in sorted_signals:
            # Add signal strength indicators
            if "metadata" in signal:
                signal["metadata"]["signal_strength"] = self._calculate_signal_strength(
                    signal
                )
                # Add market timing assessment
                signal["metadata"]["timing_quality"] = self._assess_signal_timing(
                    signal
                )

            enhanced_signals.append(signal)

        return enhanced_signals

    def _calculate_signal_strength(self, signal: Dict[str, Any]) -> str:
        """Calculate signal strength category"""
        if signal.get("confidence", 0) >= 0.8:
            return "strong"
        elif signal.get("confidence", 0) >= 0.6:
            return "moderate"
        else:
            return "weak"

    def _assess_signal_timing(self, signal: Dict[str, Any]) -> float:
        """Assess signal timing quality"""
        # Basic timing assessment - can be overridden by subclasses
        if signal.get("timestamp") is None:
            return 0.0

        # Signal is recent and well-timed
        return 0.8

    def _assess_signal_context(self, signal: Dict[str, Any]) -> float:
        """Assess signal context quality"""
        # Check metadata richness and context
        if not signal.get("metadata"):
            return 0.0

        # Count meaningful metadata fields
        meaningful_fields = ["signal_name", "trend", "momentum", "volume_confirmation"]
        present_fields = sum(
            1 for field in meaningful_fields if field in signal.get("metadata", {})
        )

        return present_fields / len(meaningful_fields)

    def _assess_signal_validation(self, signal: Dict[str, Any]) -> float:
        """Assess signal validation quality"""
        # Check if signal meets validation criteria
        if signal.get("confidence", 0) < 0.5:
            return 0.0

        # Higher confidence signals get better validation scores
        return min(1.0, signal.get("confidence", 0) * 1.2)

    def _enhance_confidence_quality(
        self, base_confidence: float, conditions: Dict[str, float]
    ) -> float:
        """Enhance confidence based on signal quality factors"""
        # Quality multiplier based on conditions diversity
        condition_count = len(conditions)
        diversity_bonus = min(0.1, condition_count * 0.02)

        # Consistency bonus if all conditions are strong
        min_condition = min(conditions.values())
        consistency_bonus = min(0.1, min_condition * 0.2)

        # Apply enhancements
        enhanced = base_confidence + diversity_bonus + consistency_bonus

        # Penalty for very low base confidence
        if base_confidence < 0.3:
            enhanced *= 0.8

        return enhanced

    def __str__(self) -> str:
        return f"{self.name}({self.parameters})"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"
