"""
Sample OHLCV data and market scenarios for testing.

This module provides realistic sample data for testing trading strategies,
backtesting engines, and analysis components.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from src.core.models import OHLCV, Signal, Trade, SignalType, TimeFrame
from src.data.mock_generator import MockDataGenerator


@dataclass
class SampleScenario:
    """Container for sample market scenario data."""

    name: str
    description: str
    data: List[OHLCV]
    expected_signals: List[Signal]
    expected_performance: Dict[str, float]


class SampleDataProvider:
    """Provider for sample test data and scenarios."""

    def __init__(self, seed: int = 42):
        """Initialize with fixed seed for reproducible data."""
        self.generator = MockDataGenerator(seed=seed)
        self.base_date = datetime(2023, 1, 1)

    def get_trending_bull_market(self, periods: int = 252) -> List[OHLCV]:
        """
        Get sample data for a trending bull market.

        Strong upward trend with moderate volatility, typical of a bull market.
        """
        return self.generator.generate_trending_data(
            periods=periods,
            trend_strength=0.08 / 252,  # 8% annual return
            volatility=0.15,
            start_price=100.0,
            start_date=self.base_date,
        )

    def get_trending_bear_market(self, periods: int = 252) -> List[OHLCV]:
        """
        Get sample data for a trending bear market.

        Strong downward trend with high volatility, typical of a bear market.
        """
        return self.generator.generate_trending_data(
            periods=periods,
            trend_strength=-0.20 / 252,  # -20% annual return
            volatility=0.25,
            start_price=100.0,
            start_date=self.base_date,
        )

    def get_sideways_market(self, periods: int = 252) -> List[OHLCV]:
        """
        Get sample data for a sideways/ranging market.

        No clear trend, price oscillates within a range.
        """
        return self.generator.generate_ranging_data(
            periods=periods,
            range_width=0.20,  # 20% range
            volatility=0.12,
            center_price=100.0,
            start_date=self.base_date,
        )

    def get_high_volatility_market(self, periods: int = 100) -> List[OHLCV]:
        """
        Get sample data for high volatility market conditions.

        Extreme price swings, typical during market stress.
        """
        return self.generator.generate_trending_data(
            periods=periods,
            trend_strength=0.02 / 252,
            volatility=0.40,  # Very high volatility
            start_price=100.0,
            start_date=self.base_date,
        )

    def get_low_volatility_market(self, periods: int = 100) -> List[OHLCV]:
        """
        Get sample data for low volatility market conditions.

        Stable price movement, typical during calm periods.
        """
        return self.generator.generate_trending_data(
            periods=periods,
            trend_strength=0.04 / 252,
            volatility=0.05,  # Very low volatility
            start_price=100.0,
            start_date=self.base_date,
        )

    def get_flash_crash_scenario(self, periods: int = 50) -> List[OHLCV]:
        """
        Get sample data for a flash crash scenario.

        Rapid market decline followed by partial recovery.
        """
        # Generate normal market first
        normal_data = self.generator.generate_trending_data(
            periods=20,
            trend_strength=0.01 / 252,
            volatility=0.12,
            start_price=100.0,
            start_date=self.base_date,
        )

        # Add crash period
        crash_data = self.generator.generate_trending_data(
            periods=10,
            trend_strength=-0.30 / 10,  # 30% decline over 10 days
            volatility=0.50,
            start_price=normal_data[-1].close,
            start_date=normal_data[-1].timestamp + timedelta(days=1),
        )

        # Add recovery period
        recovery_data = self.generator.generate_trending_data(
            periods=20,
            trend_strength=0.15 / 20,  # Partial recovery
            volatility=0.25,
            start_price=crash_data[-1].close,
            start_date=crash_data[-1].timestamp + timedelta(days=1),
        )

        return normal_data + crash_data + recovery_data

    def get_golden_cross_scenario(self, periods: int = 100) -> SampleScenario:
        """
        Get scenario specifically designed to trigger MA golden cross signals.

        Creates conditions favorable for moving average crossover strategy.
        """
        # Generate data with strong uptrend after initial consolidation
        consolidation_data = self.generator.generate_ranging_data(
            periods=50,
            range_width=0.10,
            volatility=0.08,
            center_price=100.0,
            start_date=self.base_date,
        )

        trend_data = self.generator.generate_trending_data(
            periods=50,
            trend_strength=0.15 / 50,  # Strong trend
            volatility=0.12,
            start_price=consolidation_data[-1].close,
            start_date=consolidation_data[-1].timestamp + timedelta(days=1),
        )

        data = consolidation_data + trend_data

        # Expected signals (approximate)
        expected_signals = [
            Signal(
                timestamp=data[60].timestamp,  # After trend starts
                symbol="AAPL",
                signal_type=SignalType.BUY,
                price=data[60].close,
                confidence=0.8,
                strategy_name="MovingAverageCrossover",
                metadata={"signal_type": "golden_cross"},
            )
        ]

        expected_performance = {
            "total_return_pct": 12.0,
            "win_rate": 0.75,
            "sharpe_ratio": 1.2,
            "max_drawdown_pct": 5.0,
        }

        return SampleScenario(
            name="Golden Cross",
            description="Market scenario designed to trigger MA golden cross signals",
            data=data,
            expected_signals=expected_signals,
            expected_performance=expected_performance,
        )

    def get_rsi_oversold_scenario(self, periods: int = 100) -> SampleScenario:
        """
        Get scenario designed to trigger RSI oversold signals.

        Creates conditions favorable for RSI-based strategies.
        """
        # Generate data with sharp decline followed by recovery
        decline_data = self.generator.generate_trending_data(
            periods=20,
            trend_strength=-0.15 / 20,  # Sharp decline
            volatility=0.20,
            start_price=100.0,
            start_date=self.base_date,
        )

        recovery_data = self.generator.generate_trending_data(
            periods=30,
            trend_strength=0.20 / 30,  # Strong recovery
            volatility=0.15,
            start_price=decline_data[-1].close,
            start_date=decline_data[-1].timestamp + timedelta(days=1),
        )

        consolidation_data = self.generator.generate_ranging_data(
            periods=50,
            range_width=0.15,
            volatility=0.10,
            center_price=recovery_data[-1].close,
            start_date=recovery_data[-1].timestamp + timedelta(days=1),
        )

        data = decline_data + recovery_data + consolidation_data

        # Expected RSI oversold signal
        expected_signals = [
            Signal(
                timestamp=data[20].timestamp,  # End of decline
                symbol="AAPL",
                signal_type=SignalType.BUY,
                price=data[20].close,
                confidence=0.85,
                strategy_name="RSITrend",
                metadata={"rsi_value": 25, "signal_type": "oversold_recovery"},
            )
        ]

        expected_performance = {
            "total_return_pct": 18.0,
            "win_rate": 0.80,
            "sharpe_ratio": 1.5,
            "max_drawdown_pct": 8.0,
        }

        return SampleScenario(
            name="RSI Oversold Recovery",
            description="Market scenario with oversold conditions followed by recovery",
            data=data,
            expected_signals=expected_signals,
            expected_performance=expected_performance,
        )

    def get_bollinger_breakout_scenario(self, periods: int = 100) -> SampleScenario:
        """
        Get scenario designed to trigger Bollinger Band breakout signals.

        Creates conditions with low volatility squeeze followed by breakout.
        """
        # Low volatility consolidation (squeeze)
        squeeze_data = self.generator.generate_ranging_data(
            periods=60,
            range_width=0.05,  # Very tight range
            volatility=0.04,  # Very low volatility
            center_price=100.0,
            start_date=self.base_date,
        )

        # Breakout phase
        breakout_data = self.generator.generate_trending_data(
            periods=40,
            trend_strength=0.12 / 40,  # Strong breakout
            volatility=0.18,
            start_price=squeeze_data[-1].close,
            start_date=squeeze_data[-1].timestamp + timedelta(days=1),
        )

        data = squeeze_data + breakout_data

        # Expected breakout signal
        expected_signals = [
            Signal(
                timestamp=data[65].timestamp,  # Early in breakout
                symbol="AAPL",
                signal_type=SignalType.BUY,
                price=data[65].close,
                confidence=0.75,
                strategy_name="BollingerBreakout",
                metadata={"breakout_type": "upper_band", "squeeze_duration": 60},
            )
        ]

        expected_performance = {
            "total_return_pct": 10.0,
            "win_rate": 0.70,
            "sharpe_ratio": 1.1,
            "max_drawdown_pct": 4.0,
        }

        return SampleScenario(
            name="Bollinger Breakout",
            description="Low volatility squeeze followed by strong breakout",
            data=data,
            expected_signals=expected_signals,
            expected_performance=expected_performance,
        )

    def get_support_resistance_scenario(self, periods: int = 120) -> SampleScenario:
        """
        Get scenario with clear support and resistance levels.

        Creates conditions favorable for support/resistance breakout strategies.
        """
        # Generate data with multiple tests of support/resistance
        price_levels = [95, 100, 105, 100, 95, 100, 105, 110]  # Key levels
        data = []

        current_date = self.base_date
        current_price = 100.0

        for i, target_price in enumerate(price_levels):
            if i == 0:
                # Initial move to first level
                segment_data = self.generator.generate_trending_data(
                    periods=15,
                    trend_strength=(target_price - current_price) / current_price / 15,
                    volatility=0.08,
                    start_price=current_price,
                    start_date=current_date,
                )
            else:
                # Move between levels
                prev_price = data[-1].close
                segment_data = self.generator.generate_trending_data(
                    periods=15,
                    trend_strength=(target_price - prev_price) / prev_price / 15,
                    volatility=0.10,
                    start_price=prev_price,
                    start_date=current_date,
                )

            data.extend(segment_data)
            current_date = segment_data[-1].timestamp + timedelta(days=1)
            current_price = target_price

        # Expected resistance breakout signal
        expected_signals = [
            Signal(
                timestamp=data[-10].timestamp,  # During final breakout
                symbol="AAPL",
                signal_type=SignalType.BUY,
                price=data[-10].close,
                confidence=0.82,
                strategy_name="SRBreakout",
                metadata={"resistance_level": 105, "breakout_strength": "strong"},
            )
        ]

        expected_performance = {
            "total_return_pct": 8.0,
            "win_rate": 0.65,
            "sharpe_ratio": 0.9,
            "max_drawdown_pct": 6.0,
        }

        return SampleScenario(
            name="Support Resistance Breakout",
            description="Clear support/resistance levels with eventual breakout",
            data=data,
            expected_signals=expected_signals,
            expected_performance=expected_performance,
        )

    def get_multi_timeframe_data(self) -> Dict[TimeFrame, List[OHLCV]]:
        """
        Get sample data for multiple timeframes.

        Useful for testing multi-timeframe analysis.
        """
        # Generate base daily data
        daily_data = self.get_trending_bull_market(periods=100)

        # Create hourly data (simplified - normally would be more detailed)
        hourly_data = []
        for daily_bar in daily_data[:10]:  # Just first 10 days
            for hour in range(24):
                if 9 <= hour <= 16:  # Market hours only
                    timestamp = daily_bar.timestamp.replace(
                        hour=hour, minute=0, second=0
                    )

                    # Simulate intraday movement
                    price_variation = np.random.normal(0, 0.005)  # 0.5% std
                    close_price = daily_bar.close * (1 + price_variation)
                    open_price = close_price * (1 + np.random.normal(0, 0.002))
                    high_price = max(open_price, close_price) * (
                        1 + np.random.uniform(0, 0.01)
                    )
                    low_price = min(open_price, close_price) * (
                        1 - np.random.uniform(0, 0.01)
                    )
                    volume = daily_bar.volume / 8  # Distribute daily volume

                    hourly_data.append(
                        OHLCV(
                            timestamp=timestamp,
                            open=open_price,
                            high=high_price,
                            low=low_price,
                            close=close_price,
                            volume=volume,
                        )
                    )

        # Create weekly data (aggregate from daily)
        weekly_data = []
        for i in range(0, len(daily_data), 5):  # 5 days per week
            week_data = daily_data[i : i + 5]
            if len(week_data) >= 5:
                weekly_bar = OHLCV(
                    timestamp=week_data[0].timestamp,
                    open=week_data[0].open,
                    high=max(bar.high for bar in week_data),
                    low=min(bar.low for bar in week_data),
                    close=week_data[-1].close,
                    volume=sum(bar.volume for bar in week_data),
                )
                weekly_data.append(weekly_bar)

        return {
            TimeFrame.DAILY: daily_data,
            TimeFrame.HOUR_1: hourly_data,
            TimeFrame.WEEKLY: weekly_data,
        }

    def get_stress_test_scenarios(self) -> Dict[str, List[OHLCV]]:
        """
        Get various stress test scenarios.

        Includes extreme market conditions for robustness testing.
        """
        scenarios = {}

        # Scenario 1: Market crash
        scenarios["crash"] = self.get_flash_crash_scenario(50)

        # Scenario 2: Extreme volatility
        scenarios["high_volatility"] = self.get_high_volatility_market(60)

        # Scenario 3: Prolonged bear market
        scenarios["bear_market"] = self.get_trending_bear_market(200)

        # Scenario 4: Whipsaw market (frequent reversals)
        whipsaw_data = []
        current_price = 100.0
        current_date = self.base_date

        for i in range(20):  # 20 reversals
            trend_direction = 1 if i % 2 == 0 else -1
            segment_data = self.generator.generate_trending_data(
                periods=5,
                trend_strength=trend_direction * 0.05 / 5,  # 5% moves
                volatility=0.15,
                start_price=current_price,
                start_date=current_date,
            )
            whipsaw_data.extend(segment_data)
            current_price = segment_data[-1].close
            current_date = segment_data[-1].timestamp + timedelta(days=1)

        scenarios["whipsaw"] = whipsaw_data

        # Scenario 5: Gap-heavy market
        gap_data = []
        current_price = 100.0
        current_date = self.base_date

        for i in range(50):
            # Random gap
            gap_size = np.random.uniform(-0.05, 0.05)  # ±5% gaps
            gap_price = current_price * (1 + gap_size)

            daily_bar = OHLCV(
                timestamp=current_date,
                open=gap_price,
                high=gap_price * (1 + np.random.uniform(0, 0.02)),
                low=gap_price * (1 - np.random.uniform(0, 0.02)),
                close=gap_price * (1 + np.random.uniform(-0.01, 0.01)),
                volume=np.random.uniform(500000, 2000000),
            )
            gap_data.append(daily_bar)

            current_price = daily_bar.close
            current_date += timedelta(days=1)

        scenarios["gaps"] = gap_data

        return scenarios

    def get_known_indicator_test_data(self) -> Dict[str, Any]:
        """
        Get test data with known indicator values for validation.

        Useful for testing indicator calculations against expected results.
        """
        # Simple trending data for predictable indicator values
        prices = [
            100,
            102,
            104,
            103,
            105,
            107,
            106,
            108,
            110,
            109,
            111,
            113,
            112,
            114,
            116,
        ]

        data = []
        current_date = self.base_date

        for price in prices:
            bar = OHLCV(
                timestamp=current_date,
                open=price * 0.995,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=1000000,
            )
            data.append(bar)
            current_date += timedelta(days=1)

        # Known expected values (approximate)
        expected_indicators = {
            "sma_5": {
                "values": [
                    None,
                    None,
                    None,
                    None,
                    102.8,
                    104.2,
                    105.0,
                    105.8,
                    107.2,
                    108.0,
                    108.8,
                    110.2,
                    111.0,
                    112.0,
                    113.2,
                ],
                "description": "5-period Simple Moving Average",
            },
            "ema_5": {
                "values": [
                    None,
                    None,
                    None,
                    None,
                    102.8,
                    104.37,
                    105.25,
                    106.5,
                    108.0,
                    108.33,
                    109.55,
                    111.03,
                    111.69,
                    112.79,
                    114.19,
                ],
                "description": "5-period Exponential Moving Average",
            },
            "rsi_5": {
                "values": [None] * 5
                + [75.0, 60.0, 70.0, 80.0, 65.0, 75.0, 85.0, 70.0, 80.0, 90.0],
                "description": "5-period RSI (approximate values)",
            },
        }

        return {
            "data": data,
            "expected_indicators": expected_indicators,
            "description": "Simple trending data with known indicator values",
        }

    def get_performance_test_data(self, size: str = "medium") -> List[OHLCV]:
        """
        Get data for performance testing.

        Args:
            size: 'small' (100 points), 'medium' (1000 points), 'large' (10000 points)
        """
        sizes = {"small": 100, "medium": 1000, "large": 10000}

        periods = sizes.get(size, 1000)

        return self.generator.generate_realistic_data(
            symbol="PERF_TEST", periods=periods, start_date=self.base_date
        )

    def get_edge_case_data(self) -> Dict[str, List[OHLCV]]:
        """
        Get edge case data for testing robustness.

        Includes various edge cases that might break strategies or indicators.
        """
        edge_cases = {}

        # Case 1: Constant prices (no movement)
        constant_data = []
        for i in range(50):
            bar = OHLCV(
                timestamp=self.base_date + timedelta(days=i),
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=1000000,
            )
            constant_data.append(bar)
        edge_cases["constant_prices"] = constant_data

        # Case 2: Zero volume
        zero_volume_data = []
        for i in range(50):
            bar = OHLCV(
                timestamp=self.base_date + timedelta(days=i),
                open=100.0 + i * 0.1,
                high=100.5 + i * 0.1,
                low=99.5 + i * 0.1,
                close=100.0 + i * 0.1,
                volume=0.0,  # Zero volume
            )
            zero_volume_data.append(bar)
        edge_cases["zero_volume"] = zero_volume_data

        # Case 3: Extreme price jumps
        extreme_jumps_data = []
        prices = [100, 200, 50, 300, 25, 400, 10]  # Extreme movements
        for i, price in enumerate(prices):
            bar = OHLCV(
                timestamp=self.base_date + timedelta(days=i),
                open=price * 0.95,
                high=price * 1.05,
                low=price * 0.90,
                close=price,
                volume=1000000,
            )
            extreme_jumps_data.append(bar)
        edge_cases["extreme_jumps"] = extreme_jumps_data

        # Case 4: Very small numbers
        micro_prices_data = []
        for i in range(50):
            price = 0.0001 + i * 0.00001  # Very small prices
            bar = OHLCV(
                timestamp=self.base_date + timedelta(days=i),
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=1000000,
            )
            micro_prices_data.append(bar)
        edge_cases["micro_prices"] = micro_prices_data

        # Case 5: Very large numbers
        macro_prices_data = []
        for i in range(50):
            price = 1000000 + i * 10000  # Very large prices
            bar = OHLCV(
                timestamp=self.base_date + timedelta(days=i),
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=1000000,
            )
            macro_prices_data.append(bar)
        edge_cases["macro_prices"] = macro_prices_data

        return edge_cases


# Global instance for easy access
sample_data = SampleDataProvider(seed=42)


def get_sample_bull_market(periods: int = 252) -> List[OHLCV]:
    """Quick access to bull market data."""
    return sample_data.get_trending_bull_market(periods)


def get_sample_bear_market(periods: int = 252) -> List[OHLCV]:
    """Quick access to bear market data."""
    return sample_data.get_trending_bear_market(periods)


def get_sample_sideways_market(periods: int = 252) -> List[OHLCV]:
    """Quick access to sideways market data."""
    return sample_data.get_sideways_market(periods)


def get_all_sample_scenarios() -> Dict[str, SampleScenario]:
    """Get all predefined sample scenarios."""
    return {
        "golden_cross": sample_data.get_golden_cross_scenario(),
        "rsi_oversold": sample_data.get_rsi_oversold_scenario(),
        "bollinger_breakout": sample_data.get_bollinger_breakout_scenario(),
        "support_resistance": sample_data.get_support_resistance_scenario(),
    }


def get_comprehensive_test_suite() -> Dict[str, Any]:
    """Get comprehensive test data suite for all testing needs."""
    return {
        "market_types": {
            "bull": get_sample_bull_market(100),
            "bear": get_sample_bear_market(100),
            "sideways": get_sample_sideways_market(100),
            "high_vol": sample_data.get_high_volatility_market(100),
            "low_vol": sample_data.get_low_volatility_market(100),
        },
        "scenarios": get_all_sample_scenarios(),
        "stress_tests": sample_data.get_stress_test_scenarios(),
        "edge_cases": sample_data.get_edge_case_data(),
        "multi_timeframe": sample_data.get_multi_timeframe_data(),
        "known_indicators": sample_data.get_known_indicator_test_data(),
        "performance_data": {
            "small": sample_data.get_performance_test_data("small"),
            "medium": sample_data.get_performance_test_data("medium"),
            "large": sample_data.get_performance_test_data("large"),
        },
    }
