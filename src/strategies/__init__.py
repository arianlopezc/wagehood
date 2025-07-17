"""
Trading Strategies Module

This module provides a comprehensive set of proven trading strategies for retail investors,
implementing the most popular and effective trend following methods based on extensive research.

Strategy Priority (based on effectiveness and simplicity):
1. MovingAverageCrossover (Golden Cross/Death Cross) - Highest Priority
2. MACDRSIStrategy (73% win rate documented) - High Priority
3. RSITrendFollowing - Medium Priority
4. BollingerBandBreakout - Medium Priority
5. SupportResistanceBreakout - Lower Priority (Advanced)

All strategies inherit from the TradingStrategy base class and provide:
- Signal generation with confidence scoring
- Parameter optimization capabilities
- Signal validation
- Comprehensive metadata and performance tracking
"""

from .base import TradingStrategy
from .ma_crossover import MovingAverageCrossover
from .macd_rsi import MACDRSIStrategy
from .rsi_trend import RSITrendFollowing
from .bollinger_breakout import BollingerBandBreakout
from .sr_breakout import SupportResistanceBreakout

# Strategy registry for easy access
STRATEGY_REGISTRY = {
    "ma_crossover": MovingAverageCrossover,
    "moving_average_crossover": MovingAverageCrossover,
    "golden_cross": MovingAverageCrossover,
    "macd_rsi": MACDRSIStrategy,
    "macd_rsi_strategy": MACDRSIStrategy,
    "rsi_trend": RSITrendFollowing,
    "rsi_trend_following": RSITrendFollowing,
    "bollinger_breakout": BollingerBandBreakout,
    "bollinger_band_breakout": BollingerBandBreakout,
    "sr_breakout": SupportResistanceBreakout,
    "support_resistance_breakout": SupportResistanceBreakout,
}

# Default parameters for each strategy (proven configurations from research)
DEFAULT_STRATEGY_PARAMS = {
    "ma_crossover": {
        "short_period": 50,
        "long_period": 200,
        "min_confidence": 0.6,
        "volume_confirmation": True,
        "volume_threshold": 1.2,
    },
    "macd_rsi": {
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "min_confidence": 0.7,  # Higher threshold for quality signals
        "volume_confirmation": True,
        "volume_threshold": 1.2,
        "divergence_detection": True,
    },
    "rsi_trend": {
        "rsi_period": 14,
        "rsi_main_period": 21,
        "uptrend_threshold": 50,
        "downtrend_threshold": 50,
        "uptrend_pullback_low": 40,
        "uptrend_pullback_high": 50,
        "downtrend_pullback_low": 50,
        "downtrend_pullback_high": 60,
        "min_confidence": 0.6,
        "divergence_detection": True,
    },
    "bollinger_breakout": {
        "bb_period": 20,
        "bb_std": 2.0,
        "consolidation_periods": 10,
        "volume_confirmation": True,
        "volume_threshold": 1.5,
        "min_confidence": 0.6,
        "squeeze_threshold": 0.1,
    },
    "sr_breakout": {
        "lookback_periods": 50,
        "min_touches": 2,
        "touch_tolerance": 0.02,
        "volume_confirmation": True,
        "volume_threshold": 1.5,
        "min_confidence": 0.7,
        "consolidation_periods": 10,
    },
}

# Strategy metadata for UI and documentation
STRATEGY_METADATA = {
    "ma_crossover": {
        "name": "Moving Average Crossover",
        "description": "Golden Cross and Death Cross strategy using 50/200 EMA",
        "difficulty": "Beginner",
        "priority": 1,
        "documented_performance": {
            "signals_per_year": 0.5,  # Very low frequency
            "avg_trade_duration": 350,  # days
            "market_exposure": 0.67,  # 2/3 of time invested
            "risk_reduction": "Significant drawdown reduction vs buy-and-hold",
        },
        "best_conditions": ["Strong trending markets", "Bull/bear markets"],
        "poor_conditions": ["Range-bound markets", "Choppy sideways markets"],
    },
    "macd_rsi": {
        "name": "MACD + RSI Combined",
        "description": "High-performance dual indicator strategy",
        "difficulty": "Intermediate",
        "priority": 2,
        "documented_performance": {
            "win_rate": 0.73,  # 73% win rate
            "avg_gain_per_trade": 0.0088,  # 0.88% per trade
            "total_trades": 235,
            "market_conditions": "Effective in trending and volatile environments",
        },
        "best_conditions": [
            "Trending markets",
            "Volatile markets with clear direction",
        ],
        "poor_conditions": ["Extremely volatile conditions", "Conflicting signals"],
    },
    "rsi_trend": {
        "name": "RSI Trend Following",
        "description": "RSI for trend confirmation and pullback timing",
        "difficulty": "Intermediate",
        "priority": 3,
        "documented_performance": {
            "reliability": "High",
            "consistency": "High performance across testing periods",
            "market_adaptation": "Effective in range-bound and trending markets",
        },
        "best_conditions": ["Established trends", "Pullback opportunities"],
        "poor_conditions": ["Volatile whipsaw markets", "Trend reversals"],
    },
    "bollinger_breakout": {
        "name": "Bollinger Band Breakout",
        "description": "Volatility-based breakout strategy",
        "difficulty": "Intermediate",
        "priority": 4,
        "documented_performance": {
            "reliability": "High",
            "volatility_adaptation": "Excellent",
            "breakout_accuracy": "High with volume confirmation",
        },
        "best_conditions": ["Volatility expansion", "Clear breakout patterns"],
        "poor_conditions": ["False breakouts", "Choppy markets"],
    },
    "sr_breakout": {
        "name": "Support/Resistance Breakout",
        "description": "Advanced level-based breakout strategy",
        "difficulty": "Advanced",
        "priority": 5,
        "documented_performance": {
            "breakout_success": "Higher with longer consolidation",
            "volume_importance": "Critical for true vs false breakouts",
            "trend_initiation": "Excellent for catching new trends",
        },
        "best_conditions": ["Clear S/R levels", "Volume confirmation"],
        "poor_conditions": ["Subjective level identification", "False breakouts"],
    },
}


def get_strategy_class(strategy_name: str) -> type:
    """
    Get strategy class by name

    Args:
        strategy_name: Name of the strategy

    Returns:
        Strategy class

    Raises:
        KeyError: If strategy not found
    """
    strategy_name = strategy_name.lower()
    if strategy_name not in STRATEGY_REGISTRY:
        raise KeyError(
            f"Strategy '{strategy_name}' not found. Available strategies: {list(STRATEGY_REGISTRY.keys())}"
        )

    return STRATEGY_REGISTRY[strategy_name]


def create_strategy(strategy_name: str, parameters: dict = None) -> TradingStrategy:
    """
    Create strategy instance with optional parameters

    Args:
        strategy_name: Name of the strategy
        parameters: Optional parameters dict

    Returns:
        Strategy instance
    """
    strategy_class = get_strategy_class(strategy_name)

    # Merge with default parameters if provided
    if parameters is None:
        parameters = DEFAULT_STRATEGY_PARAMS.get(strategy_name.lower(), {})
    else:
        default_params = DEFAULT_STRATEGY_PARAMS.get(strategy_name.lower(), {})
        merged_params = default_params.copy()
        merged_params.update(parameters)
        parameters = merged_params

    return strategy_class(parameters)


def get_all_strategies() -> dict:
    """
    Get all available strategies with their metadata

    Returns:
        Dictionary of strategy names and their metadata
    """
    return STRATEGY_METADATA.copy()


def get_strategy_by_priority() -> list:
    """
    Get strategies ordered by priority (effectiveness and simplicity)

    Returns:
        List of strategy names ordered by priority
    """
    strategies = list(STRATEGY_METADATA.items())
    strategies.sort(key=lambda x: x[1]["priority"])
    return [name for name, _ in strategies]


def get_beginner_strategies() -> list:
    """
    Get strategies suitable for beginners

    Returns:
        List of beginner-friendly strategy names
    """
    return [
        name
        for name, metadata in STRATEGY_METADATA.items()
        if metadata["difficulty"] == "Beginner"
    ]


def get_intermediate_strategies() -> list:
    """
    Get strategies suitable for intermediate traders

    Returns:
        List of intermediate strategy names
    """
    return [
        name
        for name, metadata in STRATEGY_METADATA.items()
        if metadata["difficulty"] == "Intermediate"
    ]


def get_advanced_strategies() -> list:
    """
    Get strategies suitable for advanced traders

    Returns:
        List of advanced strategy names
    """
    return [
        name
        for name, metadata in STRATEGY_METADATA.items()
        if metadata["difficulty"] == "Advanced"
    ]


__all__ = [
    # Base class
    "TradingStrategy",
    # Strategy classes
    "MovingAverageCrossover",
    "MACDRSIStrategy",
    "RSITrendFollowing",
    "BollingerBandBreakout",
    "SupportResistanceBreakout",
    # Registry and metadata
    "STRATEGY_REGISTRY",
    "DEFAULT_STRATEGY_PARAMS",
    "STRATEGY_METADATA",
    # Utility functions
    "get_strategy_class",
    "create_strategy",
    "get_all_strategies",
    "get_strategy_by_priority",
    "get_beginner_strategies",
    "get_intermediate_strategies",
    "get_advanced_strategies",
]
