"""
Signal analysis engine for strategy evaluation
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from dataclasses import dataclass
from collections import defaultdict

from ..core.models import Signal, SignalType, SignalAnalysisResult, MarketData
from ..strategies.base import TradingStrategy

logger = logging.getLogger(__name__)


@dataclass
class SignalAnalysisConfig:
    """Configuration for signal analysis"""

    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_confidence: float = 0.0
    max_signals_per_day: Optional[int] = None
    timeframe_filter: Optional[str] = None


class SignalAnalysisEngine:
    """Main signal analysis engine for signal-detection-only service"""

    def __init__(self, config: SignalAnalysisConfig = None):
        """
        Initialize signal analysis engine

        Args:
            config: Signal analysis configuration
        """
        self.config = config or SignalAnalysisConfig()
        self.results_cache: Dict[str, SignalAnalysisResult] = {}

    def run_signal_analysis(
        self, strategy: TradingStrategy, data: MarketData, min_confidence: float = None
    ) -> SignalAnalysisResult:
        """
        Run signal analysis for a strategy

        Args:
            strategy: Trading strategy to analyze
            data: Market data
            min_confidence: Minimum confidence threshold (overrides config)

        Returns:
            SignalAnalysisResult with signal analysis results
        """
        logger.info(f"Starting signal analysis for {strategy.name}")

        # Use provided parameters or defaults from config
        confidence_threshold = min_confidence or self.config.min_confidence

        # Generate signals
        signals = self._generate_signals(strategy, data)
        logger.info(f"Generated {len(signals)} signals")

        # Filter signals by confidence if specified
        if confidence_threshold > 0:
            filtered_signals = [
                s for s in signals if s.confidence >= confidence_threshold
            ]
            logger.info(
                f"Filtered to {len(filtered_signals)} signals with confidence >= {confidence_threshold}"
            )
            signals = filtered_signals

        # Filter by date range if specified
        if self.config.start_date or self.config.end_date:
            signals = self._filter_signals_by_date(signals)
            logger.info(f"Date filtered to {len(signals)} signals")

        # Limit signals per day if specified
        if self.config.max_signals_per_day:
            signals = self._limit_signals_per_day(signals)
            logger.info(f"Daily limit filtered to {len(signals)} signals")

        # Analyze signals
        analysis_result = self._analyze_signals(strategy, data, signals)

        # Cache result
        cache_key = f"{strategy.name}_{data.symbol}_{confidence_threshold}"
        self.results_cache[cache_key] = analysis_result

        logger.info(f"Signal analysis completed for {strategy.name}")
        return analysis_result

    def _generate_signals(
        self, strategy: TradingStrategy, data: MarketData
    ) -> List[Signal]:
        """Generate trading signals from strategy"""
        signals = []

        # Calculate required indicators
        indicators = strategy._calculate_indicators(data)

        # Generate signals
        try:
            raw_signals = strategy.generate_signals(data, indicators)
            validated_signals = strategy.validate_signals(raw_signals)
            signals.extend(validated_signals)
        except Exception as e:
            logger.error(f"Error generating signals: {e}")

        return signals

    def _filter_signals_by_date(self, signals: List[Signal]) -> List[Signal]:
        """Filter signals by date range"""
        filtered_signals = []

        for signal in signals:
            if self.config.start_date and signal.timestamp < self.config.start_date:
                continue
            if self.config.end_date and signal.timestamp > self.config.end_date:
                continue
            filtered_signals.append(signal)

        return filtered_signals

    def _limit_signals_per_day(self, signals: List[Signal]) -> List[Signal]:
        """Limit number of signals per day"""
        if not self.config.max_signals_per_day:
            return signals

        # Group signals by date
        daily_signals = defaultdict(list)
        for signal in signals:
            date = signal.timestamp.date()
            daily_signals[date].append(signal)

        # Limit each day and sort by confidence
        limited_signals = []
        for date, day_signals in daily_signals.items():
            # Sort by confidence (highest first) then by timestamp
            day_signals.sort(key=lambda s: (-s.confidence, s.timestamp))
            limited_signals.extend(day_signals[: self.config.max_signals_per_day])

        # Sort final result by timestamp
        limited_signals.sort(key=lambda s: s.timestamp)
        return limited_signals

    def _analyze_signals(
        self, strategy: TradingStrategy, data: MarketData, signals: List[Signal]
    ) -> SignalAnalysisResult:
        """Analyze signals and create result"""
        if not signals:
            return SignalAnalysisResult(
                strategy_name=strategy.name,
                symbol=data.symbol,
                start_date=data.data[0].timestamp if data.data else datetime.now(),
                end_date=data.data[-1].timestamp if data.data else datetime.now(),
                signals=[],
                signal_timestamps=[],
                total_signals=0,
                buy_signals=0,
                sell_signals=0,
                hold_signals=0,
                avg_confidence=0.0,
                signal_frequency=0.0,
                metadata={},
            )

        # Analyze signal statistics
        signal_timestamps = [s.timestamp for s in signals]
        buy_signals = len([s for s in signals if s.signal_type == SignalType.BUY])
        sell_signals = len([s for s in signals if s.signal_type == SignalType.SELL])
        hold_signals = len([s for s in signals if s.signal_type == SignalType.HOLD])

        # Calculate average confidence
        avg_confidence = sum(s.confidence for s in signals) / len(signals)

        # Calculate signal frequency (signals per day)
        if len(signal_timestamps) > 1:
            time_span = (signal_timestamps[-1] - signal_timestamps[0]).days
            signal_frequency = len(signals) / max(time_span, 1)
        else:
            signal_frequency = 0.0

        # Create metadata with additional analysis
        metadata = {
            "confidence_distribution": self._calculate_confidence_distribution(signals),
            "signal_type_distribution": {
                "buy": buy_signals,
                "sell": sell_signals,
                "hold": hold_signals,
            },
            "hourly_distribution": self._calculate_hourly_distribution(signals),
            "strategy_parameters": strategy.parameters.copy(),
        }

        return SignalAnalysisResult(
            strategy_name=strategy.name,
            symbol=data.symbol,
            start_date=data.data[0].timestamp if data.data else datetime.now(),
            end_date=data.data[-1].timestamp if data.data else datetime.now(),
            signals=signals,
            signal_timestamps=signal_timestamps,
            total_signals=len(signals),
            buy_signals=buy_signals,
            sell_signals=sell_signals,
            hold_signals=hold_signals,
            avg_confidence=avg_confidence,
            signal_frequency=signal_frequency,
            metadata=metadata,
        )

    def _calculate_confidence_distribution(
        self, signals: List[Signal]
    ) -> Dict[str, float]:
        """Calculate confidence distribution statistics"""
        if not signals:
            return {}

        confidences = [s.confidence for s in signals]

        return {
            "min": min(confidences),
            "max": max(confidences),
            "median": sorted(confidences)[len(confidences) // 2],
            "std": self._calculate_std(confidences),
            "high_confidence_count": len([c for c in confidences if c >= 0.8]),
            "medium_confidence_count": len([c for c in confidences if 0.5 <= c < 0.8]),
            "low_confidence_count": len([c for c in confidences if c < 0.5]),
        }

    def _calculate_hourly_distribution(self, signals: List[Signal]) -> Dict[int, int]:
        """Calculate distribution of signals by hour of day"""
        hourly_counts = defaultdict(int)

        for signal in signals:
            hour = signal.timestamp.hour
            hourly_counts[hour] += 1

        return dict(hourly_counts)

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance**0.5

    def generate_signal_log(self, signals: List[Signal]) -> List[Dict[str, Any]]:
        """Generate detailed signal log"""
        signal_log = []

        for signal in signals:
            log_entry = {
                "timestamp": signal.timestamp.isoformat(),
                "symbol": signal.symbol,
                "signal_type": signal.signal_type.value,
                "price": signal.price,
                "confidence": signal.confidence,
                "strategy_name": signal.strategy_name,
                "metadata": signal.metadata,
            }
            signal_log.append(log_entry)

        return signal_log

    def run_parameter_optimization(
        self,
        strategy: TradingStrategy,
        data: MarketData,
        parameter_ranges: Dict[str, List[Any]],
        optimization_metric: str = "avg_confidence",
    ) -> Dict[str, Any]:
        """
        Run parameter optimization for a strategy based on signal quality

        Args:
            strategy: Strategy to optimize
            data: Market data
            parameter_ranges: Dictionary of parameter names and their ranges
            optimization_metric: Metric to optimize ('avg_confidence', 'total_signals', 'signal_frequency')

        Returns:
            Dictionary with best parameters and results
        """
        logger.info(f"Starting parameter optimization for {strategy.name}")

        best_params = None
        best_score = float("-inf")
        best_result = None

        # Generate parameter combinations
        import itertools

        keys = list(parameter_ranges.keys())
        values = list(parameter_ranges.values())

        for combination in itertools.product(*values):
            params = dict(zip(keys, combination))

            # Update strategy parameters
            original_params = strategy.parameters.copy()
            strategy.parameters.update(params)

            try:
                # Run signal analysis
                result = self.run_signal_analysis(strategy, data)

                # Extract optimization metric
                if optimization_metric == "avg_confidence":
                    score = result.avg_confidence
                elif optimization_metric == "total_signals":
                    score = result.total_signals
                elif optimization_metric == "signal_frequency":
                    score = result.signal_frequency
                elif optimization_metric == "buy_signals":
                    score = result.buy_signals
                else:
                    score = result.avg_confidence

                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_result = result

            except Exception as e:
                logger.error(f"Error in optimization iteration {params}: {e}")
                continue
            finally:
                # Restore original parameters
                strategy.parameters = original_params

        logger.info(f"Optimization completed. Best parameters: {best_params}")

        return {
            "best_parameters": best_params,
            "best_score": best_score,
            "best_result": best_result,
            "optimization_metric": optimization_metric,
        }

    # Alias for backward compatibility
    def run_backtest(
        self, strategy: TradingStrategy, data: MarketData, **kwargs
    ) -> SignalAnalysisResult:
        """Backward compatibility alias for run_signal_analysis"""
        return self.run_signal_analysis(strategy, data, **kwargs)


# Alias for backward compatibility
BacktestEngine = SignalAnalysisEngine
BacktestConfig = SignalAnalysisConfig
