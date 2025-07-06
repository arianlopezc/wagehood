"""
Strategy Explanations Database

This module contains detailed explanations for all trading strategies
including signal generation logic, parameters, and usage guidelines.
"""

from typing import Dict, Any, List

STRATEGY_EXPLANATIONS = {
    "macd_rsi": {
        "name": "MACD + RSI Combined Strategy",
        "description": "High-performance momentum strategy combining MACD trend detection with RSI timing. Documented 73% win rate.",
        
        "buy_signals": {
            "primary": {
                "description": "MACD Bullish Crossover + RSI Exit from Oversold",
                "conditions": [
                    "MACD line crosses ABOVE signal line",
                    "RSI moves ABOVE 30 (from below 30)"
                ]
            },
            "secondary": {
                "description": "MACD Bullish Crossover + RSI Uptrend + Positive Momentum",
                "conditions": [
                    "MACD line crosses ABOVE signal line",
                    "RSI is ABOVE 50 (uptrend zone)",
                    "MACD histogram is POSITIVE"
                ]
            },
            "divergence": {
                "description": "Bullish Divergence Pattern",
                "conditions": [
                    "Price makes LOWER LOW",
                    "MACD AND RSI both make HIGHER LOWS",
                    "Suggests weakening downtrend → reversal up"
                ]
            }
        },
        
        "sell_signals": {
            "primary": {
                "description": "MACD Bearish Crossover + RSI Exit from Overbought",
                "conditions": [
                    "MACD line crosses BELOW signal line",
                    "RSI moves BELOW 70 (from above 70)"
                ]
            },
            "secondary": {
                "description": "MACD Bearish Crossover + RSI Downtrend + Negative Momentum",
                "conditions": [
                    "MACD line crosses BELOW signal line",
                    "RSI is BELOW 50 (downtrend zone)",
                    "MACD histogram is NEGATIVE"
                ]
            },
            "divergence": {
                "description": "Bearish Divergence Pattern",
                "conditions": [
                    "Price makes HIGHER HIGH",
                    "MACD AND RSI both make LOWER HIGHS",
                    "Suggests weakening uptrend → reversal down"
                ]
            }
        },
        
        "parameters": {
            "macd_fast": {"default": 12, "description": "MACD fast EMA period"},
            "macd_slow": {"default": 26, "description": "MACD slow EMA period"},
            "macd_signal": {"default": 9, "description": "MACD signal line period"},
            "rsi_period": {"default": 14, "description": "RSI calculation period"},
            "rsi_oversold": {"default": 30, "description": "RSI oversold threshold"},
            "rsi_overbought": {"default": 70, "description": "RSI overbought threshold"},
            "min_confidence": {"default": 0.6, "description": "Minimum signal confidence (60%)"},
            "volume_confirmation": {"default": True, "description": "Require volume confirmation"},
            "volume_threshold": {"default": 1.2, "description": "Volume must be 1.2x average"}
        },
        
        "confidence_factors": {
            "macd_strength": {"weight": "25%", "description": "Distance between MACD and signal lines"},
            "rsi_position": {"weight": "25%", "description": "How close RSI is to oversold/overbought"},
            "histogram": {"weight": "20%", "description": "MACD histogram supporting direction"},
            "volume": {"weight": "15%", "description": "Current volume vs 20-day average"},
            "momentum": {"weight": "15%", "description": "Recent 5-day price momentum"}
        },
        
        "special_features": [
            "Only executes signals with ≥60% confidence",
            "Divergence detection for reversal opportunities",
            "Volume confirmation with 20-day average",
            "Multiple signal types for different market conditions"
        ],
        
        "best_for": ["Momentum trading", "Trend following", "Short to medium-term trades"],
        "difficulty": "Intermediate",
        "frequency": "Medium (selective entries)"
    },
    
    "ma_crossover": {
        "name": "Moving Average Crossover Strategy",
        "description": "Classic trend-following strategy using 50-day and 200-day EMAs. Known as Golden Cross (buy) and Death Cross (sell).",
        
        "buy_signals": {
            "golden_cross": {
                "description": "Golden Cross Formation",
                "conditions": [
                    "Short EMA (50) crosses ABOVE Long EMA (200)",
                    "Volume exceeds 1.2x average (if enabled)",
                    "Signal confidence ≥ 60%"
                ]
            }
        },
        
        "sell_signals": {
            "death_cross": {
                "description": "Death Cross Formation",
                "conditions": [
                    "Short EMA (50) crosses BELOW Long EMA (200)",
                    "Volume exceeds 1.2x average (if enabled)",
                    "Signal confidence ≥ 60%"
                ]
            }
        },
        
        "parameters": {
            "short_period": {"default": 50, "description": "Short EMA period"},
            "long_period": {"default": 200, "description": "Long EMA period"},
            "min_confidence": {"default": 0.6, "description": "Minimum signal confidence"},
            "volume_confirmation": {"default": True, "description": "Require volume confirmation"},
            "volume_threshold": {"default": 1.2, "description": "Volume threshold multiplier"}
        },
        
        "confidence_factors": {
            "ema_separation": {"weight": "40%", "description": "Distance between short and long EMAs"},
            "volume": {"weight": "30%", "description": "Current volume vs average"},
            "trend_strength": {"weight": "30%", "description": "10-period trend consistency"}
        },
        
        "special_features": [
            "Most popular retail trading strategy",
            "Clear trend identification",
            "Low false signal rate in trending markets",
            "Works best in strong trending conditions"
        ],
        
        "best_for": ["Long-term trend following", "Position trading", "Low-frequency trading"],
        "difficulty": "Beginner",
        "frequency": "Low (few signals per year)"
    },
    
    "rsi_trend": {
        "name": "RSI Trend Following Strategy",
        "description": "Uses RSI for trend confirmation and pullback identification. Focuses on trend following rather than traditional RSI reversal signals.",
        
        "buy_signals": {
            "uptrend_pullback": {
                "description": "Uptrend Pullback Entry",
                "conditions": [
                    "RSI consistently above 50 (70% of recent periods)",
                    "RSI pulls back to 40-50 zone",
                    "RSI starts turning up (RSI > previous RSI)"
                ]
            },
            "bullish_divergence": {
                "description": "Bullish Divergence",
                "conditions": [
                    "Price makes LOWER LOW",
                    "RSI makes HIGHER LOW",
                    "Suggests trend reversal"
                ]
            }
        },
        
        "sell_signals": {
            "downtrend_rally": {
                "description": "Downtrend Rally Entry",
                "conditions": [
                    "RSI consistently below 50 (70% of recent periods)",
                    "RSI rallies to 50-60 zone",
                    "RSI starts turning down (RSI < previous RSI)"
                ]
            },
            "bearish_divergence": {
                "description": "Bearish Divergence",
                "conditions": [
                    "Price makes HIGHER HIGH",
                    "RSI makes LOWER HIGH",
                    "Suggests trend reversal"
                ]
            }
        },
        
        "parameters": {
            "rsi_period": {"default": 14, "description": "Primary RSI period"},
            "rsi_main_period": {"default": 21, "description": "Trend confirmation RSI period"},
            "uptrend_threshold": {"default": 50, "description": "RSI level defining uptrend"},
            "downtrend_threshold": {"default": 50, "description": "RSI level defining downtrend"},
            "uptrend_pullback_low": {"default": 40, "description": "Uptrend pullback entry zone low"},
            "uptrend_pullback_high": {"default": 50, "description": "Uptrend pullback entry zone high"},
            "downtrend_pullback_low": {"default": 50, "description": "Downtrend pullback entry zone low"},
            "downtrend_pullback_high": {"default": 60, "description": "Downtrend pullback entry zone high"},
            "min_confidence": {"default": 0.6, "description": "Minimum signal confidence"}
        },
        
        "confidence_factors": {
            "rsi_position": {"weight": "25%", "description": "How close RSI is to optimal entry zone"},
            "trend_strength": {"weight": "25%", "description": "Percentage of periods RSI stayed in trend zone"},
            "price_momentum": {"weight": "25%", "description": "Recent price movement supporting trend"},
            "rsi_momentum": {"weight": "25%", "description": "RSI direction change supporting entry"}
        },
        
        "special_features": [
            "Dual RSI periods for better trend identification",
            "Divergence detection for reversal signals",
            "Trend determination requires 70% consistency",
            "Separate logic for pullback vs divergence entries"
        ],
        
        "best_for": ["Trend following", "Pullback trading", "Medium-term positions"],
        "difficulty": "Intermediate",
        "frequency": "Medium (trend-dependent)"
    },
    
    "bollinger_breakout": {
        "name": "Bollinger Band Breakout Strategy",
        "description": "Uses Bollinger Bands to identify volatility expansion and breakout opportunities beyond normal trading ranges.",
        
        "buy_signals": {
            "upper_band_breakout": {
                "description": "Bullish Breakout Above Upper Band",
                "conditions": [
                    "Close price breaks ABOVE upper Bollinger Band",
                    "Volume > 1.5x average volume",
                    "70% of recent periods were within bands (consolidation)"
                ]
            },
            "upper_band_touch": {
                "description": "Alternative Upper Band Touch",
                "conditions": [
                    "High touches upper band",
                    "Close > middle band (20-period SMA)",
                    "Close > previous close (strength confirmation)"
                ]
            },
            "squeeze_breakout": {
                "description": "Bollinger Band Squeeze Breakout",
                "conditions": [
                    "Band width < 10% of normal (low volatility)",
                    "Price breaks above upper band",
                    "Volume confirmation"
                ]
            }
        },
        
        "sell_signals": {
            "lower_band_breakout": {
                "description": "Bearish Breakout Below Lower Band",
                "conditions": [
                    "Close price breaks BELOW lower Bollinger Band",
                    "Volume > 1.5x average volume",
                    "70% of recent periods were within bands"
                ]
            },
            "lower_band_touch": {
                "description": "Alternative Lower Band Touch",
                "conditions": [
                    "Low touches lower band",
                    "Close < middle band",
                    "Close < previous close (weakness confirmation)"
                ]
            },
            "squeeze_breakdown": {
                "description": "Bollinger Band Squeeze Breakdown",
                "conditions": [
                    "Band width < 10% of normal",
                    "Price breaks below lower band",
                    "Volume confirmation"
                ]
            }
        },
        
        "parameters": {
            "bb_period": {"default": 20, "description": "Bollinger Band calculation period"},
            "bb_std": {"default": 2.0, "description": "Standard deviation multiplier"},
            "consolidation_periods": {"default": 10, "description": "Periods to check for consolidation"},
            "volume_threshold": {"default": 1.5, "description": "Volume threshold for breakouts"},
            "min_confidence": {"default": 0.6, "description": "Minimum signal confidence"},
            "squeeze_threshold": {"default": 0.1, "description": "Band width threshold for squeeze"},
            "breakout_strength": {"default": 0.5, "description": "Minimum breakout strength"}
        },
        
        "confidence_factors": {
            "breakout_strength": {"weight": "30%", "description": "(price - band) / band * 100"},
            "band_width": {"weight": "20%", "description": "Narrower bands = higher confidence"},
            "volume": {"weight": "30%", "description": "Volume ratio to threshold"},
            "momentum": {"weight": "20%", "description": "Recent price momentum in breakout direction"}
        },
        
        "special_features": [
            "Bollinger Band squeeze detection",
            "Consolidation pattern recognition",
            "Alternative entry conditions for band touches",
            "Higher volume requirement (1.5x vs 1.2x)"
        ],
        
        "best_for": ["Volatility breakouts", "Range trading", "Short-term momentum"],
        "difficulty": "Intermediate",
        "frequency": "Medium (volatility-dependent)"
    },
    
    "sr_breakout": {
        "name": "Support/Resistance Breakout Strategy",
        "description": "Identifies key support and resistance levels and trades breakouts with volume confirmation. Most advanced strategy.",
        
        "buy_signals": {
            "resistance_breakout": {
                "description": "Resistance Level Breakout",
                "conditions": [
                    "Close price breaks ABOVE resistance level",
                    "High also breaks above resistance",
                    "Resistance level tested minimum 2 times",
                    "Volume > 1.5x average",
                    "30% of recent periods near the level (consolidation)"
                ]
            }
        },
        
        "sell_signals": {
            "support_breakout": {
                "description": "Support Level Breakdown",
                "conditions": [
                    "Close price breaks BELOW support level",
                    "Low also breaks below support",
                    "Support level tested minimum 2 times",
                    "Volume > 1.5x average",
                    "30% of recent periods near the level"
                ]
            }
        },
        
        "parameters": {
            "lookback_periods": {"default": 50, "description": "Periods to look back for S/R levels"},
            "min_touches": {"default": 2, "description": "Minimum times level must be tested"},
            "touch_tolerance": {"default": 0.02, "description": "2% tolerance for level touches"},
            "volume_threshold": {"default": 1.5, "description": "Volume threshold for breakouts"},
            "min_confidence": {"default": 0.7, "description": "Highest confidence requirement (70%)"},
            "consolidation_periods": {"default": 10, "description": "Periods for consolidation check"},
            "level_strength_weight": {"default": 0.3, "description": "Weight for level strength in confidence"},
            "breakout_strength_weight": {"default": 0.4, "description": "Weight for breakout strength"}
        },
        
        "confidence_factors": {
            "level_strength": {"weight": "30%", "description": "Based on touches, time span, hold ratio"},
            "breakout_strength": {"weight": "40%", "description": "(price - level) / level * 50"},
            "volume": {"weight": "20%", "description": "Volume ratio to threshold"},
            "touch_count": {"weight": "10%", "description": "Number of times level was tested"}
        },
        
        "special_features": [
            "Automatic S/R level identification",
            "Level grouping with tolerance-based clustering",
            "Level strength scoring based on multiple factors",
            "Highest default confidence requirement (70%)",
            "Complex validation including level type matching"
        ],
        
        "best_for": ["Breakout trading", "Key level trading", "Experienced traders"],
        "difficulty": "Advanced",
        "frequency": "Low (selective key levels only)"
    }
}

def get_strategy_explanation(strategy_name: str) -> Dict[str, Any]:
    """Get detailed explanation for a specific strategy"""
    strategy_key = strategy_name.lower().replace(" ", "_").replace("+", "_").replace("-", "_")
    
    # Handle alternative names
    name_mappings = {
        "macdrsi": "macd_rsi",
        "macd_rsi_strategy": "macd_rsi",
        "movingaveragecrossover": "ma_crossover",
        "ma_crossover_strategy": "ma_crossover",
        "rsitrendfollow": "rsi_trend", 
        "rsi_trend_following": "rsi_trend",
        "bollingerbreakout": "bollinger_breakout",
        "bollinger_band_breakout": "bollinger_breakout",
        "supportresistancebreakout": "sr_breakout",
        "sr_breakout_strategy": "sr_breakout"
    }
    
    strategy_key = name_mappings.get(strategy_key, strategy_key)
    
    return STRATEGY_EXPLANATIONS.get(strategy_key)

def list_available_strategies() -> List[str]:
    """Get list of all available strategy names"""
    return list(STRATEGY_EXPLANATIONS.keys())

def get_strategy_summary() -> Dict[str, Dict[str, str]]:
    """Get summary information for all strategies"""
    summaries = {}
    for key, strategy in STRATEGY_EXPLANATIONS.items():
        summaries[key] = {
            "name": strategy["name"],
            "description": strategy["description"],
            "difficulty": strategy["difficulty"],
            "frequency": strategy["frequency"]
        }
    return summaries