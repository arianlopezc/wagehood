"""
Constants for Discord notifications system.

Centralized constants to reduce code duplication and improve maintainability.
"""

from typing import Dict, Final
from enum import Enum


class StrategyType(Enum):
    """Enum for supported trading strategy types."""
    MACD_RSI = "macd_rsi"
    RSI_TREND = "rsi_trend"
    BOLLINGER_BREAKOUT = "bollinger_breakout"
    SR_BREAKOUT = "sr_breakout"


# Company names mapping - centralized to avoid duplication
COMPANY_NAMES: Final[Dict[str, str]] = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corp.',
    'GOOGL': 'Alphabet Inc.',
    'TSLA': 'Tesla Inc.',
    'SPY': 'SPDR S&P 500 ETF',
    'QQQ': 'Invesco QQQ ETF',
    'IWM': 'iShares Russell 2000 ETF'
}

# Strategy configurations - centralized for consistency
STRATEGY_CONFIGS: Final[Dict[str, Dict[str, any]]] = {
    StrategyType.MACD_RSI.value: {
        'name': 'MACD+RSI Combined',
        'color': 3447003,  # Blue
        'emoji': '📊',
        'default_rate_limit': 8
    },
    StrategyType.RSI_TREND.value: {
        'name': 'RSI Trend Following', 
        'color': 15105570,  # Orange
        'emoji': '📈',
        'default_rate_limit': 6
    },
    StrategyType.BOLLINGER_BREAKOUT.value: {
        'name': 'Bollinger Band Breakout',
        'color': 15158332,  # Red
        'emoji': '💥', 
        'default_rate_limit': 10
    },
    StrategyType.SR_BREAKOUT.value: {
        'name': 'Support/Resistance Breakout',
        'color': 3066993,  # Green
        'emoji': '🏗️',
        'default_rate_limit': 3
    }
}

# Strategy name normalization mappings
STRATEGY_NAME_MAPPINGS: Final[Dict[str, str]] = {
    # MACD+RSI variations
    'macd+rsi': StrategyType.MACD_RSI.value,
    'macd_rsi': StrategyType.MACD_RSI.value,
    'macd+rsi combined': StrategyType.MACD_RSI.value,
    'macdrsi': StrategyType.MACD_RSI.value,
    'macdrsistrategy': StrategyType.MACD_RSI.value,
    
    # RSI Trend variations
    'rsi trend': StrategyType.RSI_TREND.value,
    'rsi_trend': StrategyType.RSI_TREND.value,
    'rsi trend following': StrategyType.RSI_TREND.value,
    'rsitrendstrategy': StrategyType.RSI_TREND.value,
    'rsitrend': StrategyType.RSI_TREND.value,
    
    # Bollinger Breakout variations
    'bollinger breakout': StrategyType.BOLLINGER_BREAKOUT.value,
    'bollinger_breakout': StrategyType.BOLLINGER_BREAKOUT.value,
    'bollinger band breakout': StrategyType.BOLLINGER_BREAKOUT.value,
    'bollinger bands breakout': StrategyType.BOLLINGER_BREAKOUT.value,
    'bollingerbreakout': StrategyType.BOLLINGER_BREAKOUT.value,
    'bollingerbreakoutstrategy': StrategyType.BOLLINGER_BREAKOUT.value,
    
    # Support/Resistance variations
    'support resistance breakout': StrategyType.SR_BREAKOUT.value,
    'support/resistance breakout': StrategyType.SR_BREAKOUT.value,
    'sr breakout': StrategyType.SR_BREAKOUT.value,
    'sr_breakout': StrategyType.SR_BREAKOUT.value,
    'support resistance': StrategyType.SR_BREAKOUT.value,
    'supportresistancebreakout': StrategyType.SR_BREAKOUT.value,
    's/r breakout': StrategyType.SR_BREAKOUT.value,
    'srbreakout': StrategyType.SR_BREAKOUT.value
}

# Discord embed colors for signal types (compatible with existing tests)
SIGNAL_COLORS: Final[Dict[str, int]] = {
    'BUY': 65280,      # Green (same as original tests)
    'SELL': 16711680,  # Red (same as original tests)
    'HOLD': 3447003,   # Blue
    'UNKNOWN': 3447003 # Blue
}

# Discord embed limits (Discord API constraints)
DISCORD_LIMITS: Final[Dict[str, int]] = {
    'title_max_length': 256,
    'description_max_length': 2048,
    'field_name_max_length': 256,
    'field_value_max_length': 1024,
    'footer_max_length': 2048,
    'total_embed_max_length': 6000,
    'max_fields': 25
}