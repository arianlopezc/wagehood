"""
Incremental Technical Indicators

This module provides incremental calculation algorithms for technical indicators
that can be updated with new data points without recalculating the entire dataset.
This is essential for real-time market data processing where efficiency is critical.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import math

from ..core.constants import (
    RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD_DEV,
    MA_FAST, MA_SLOW
)

logger = logging.getLogger(__name__)


@dataclass
class IndicatorState:
    """Base class for indicator state storage."""
    last_updated: datetime
    period: int
    value: float


@dataclass
class SMAState(IndicatorState):
    """Simple Moving Average state."""
    sum: float
    count: int
    values: List[float]  # Rolling window


@dataclass
class EMAState(IndicatorState):
    """Exponential Moving Average state."""
    alpha: float  # Smoothing factor
    

@dataclass
class RSIState(IndicatorState):
    """RSI state using Wilder's smoothing."""
    avg_gain: float
    avg_loss: float
    last_price: float


@dataclass
class MACDState(IndicatorState):
    """MACD state."""
    fast_ema: float
    slow_ema: float
    signal_ema: float
    fast_alpha: float
    slow_alpha: float
    signal_alpha: float
    macd_line: float
    signal_line: float
    histogram: float


@dataclass
class BollingerBandsState(IndicatorState):
    """Bollinger Bands state."""
    sma_state: SMAState
    sum_squares: float  # For variance calculation
    upper_band: float
    middle_band: float
    lower_band: float
    std_dev_multiplier: float


class IncrementalIndicatorCalculator:
    """
    High-performance incremental technical indicator calculator.
    
    This class maintains state for various technical indicators and updates
    them incrementally as new market data arrives, avoiding full recalculation.
    """
    
    def __init__(self):
        """Initialize the incremental calculator."""
        self._states: Dict[str, Dict[str, IndicatorState]] = {}  # {symbol: {indicator: state}}
        self._initialized: Dict[str, set] = {}  # Track which indicators are initialized per symbol
    
    def _get_state_key(self, symbol: str, indicator: str, **params) -> str:
        """Generate unique key for indicator state."""
        param_str = "_".join(f"{k}_{v}" for k, v in sorted(params.items()))
        return f"{indicator}_{param_str}" if param_str else indicator
    
    def _ensure_symbol_initialized(self, symbol: str):
        """Ensure symbol is initialized in state tracking."""
        if symbol not in self._states:
            self._states[symbol] = {}
            self._initialized[symbol] = set()
    
    def calculate_sma_incremental(self, symbol: str, price: float, period: int = MA_FAST) -> Optional[float]:
        """
        Calculate Simple Moving Average incrementally.
        
        Args:
            symbol: Trading symbol
            price: New price value
            period: SMA period
            
        Returns:
            Updated SMA value or None if not enough data
        """
        try:
            self._ensure_symbol_initialized(symbol)
            state_key = self._get_state_key(symbol, "sma", period=period)
            
            if state_key not in self._states[symbol]:
                # Initialize new SMA state
                self._states[symbol][state_key] = SMAState(
                    last_updated=datetime.now(),
                    period=period,
                    value=price,
                    sum=price,
                    count=1,
                    values=[price]
                )
                return None  # Not enough data yet
            
            state = self._states[symbol][state_key]
            
            # Add new value
            state.values.append(price)
            state.sum += price
            state.count += 1
            
            # Remove old values if window is full
            if len(state.values) > period:
                old_value = state.values.pop(0)
                state.sum -= old_value
                state.count -= 1
            
            # Calculate SMA
            if len(state.values) >= period:
                state.value = state.sum / period
                state.last_updated = datetime.now()
                return state.value
            
            return None  # Not enough data yet
            
        except Exception as e:
            logger.error(f"Error calculating SMA for {symbol}: {e}")
            return None
    
    def calculate_ema_incremental(self, symbol: str, price: float, period: int = MA_FAST) -> Optional[float]:
        """
        Calculate Exponential Moving Average incrementally.
        
        Args:
            symbol: Trading symbol
            price: New price value
            period: EMA period
            
        Returns:
            Updated EMA value or None if not initialized
        """
        try:
            self._ensure_symbol_initialized(symbol)
            state_key = self._get_state_key(symbol, "ema", period=period)
            
            alpha = 2.0 / (period + 1)
            
            if state_key not in self._states[symbol]:
                # Initialize with first price
                self._states[symbol][state_key] = EMAState(
                    last_updated=datetime.now(),
                    period=period,
                    value=price,
                    alpha=alpha
                )
                return price
            
            state = self._states[symbol][state_key]
            
            # EMA formula: EMA_today = (Price_today * alpha) + (EMA_yesterday * (1 - alpha))
            state.value = (price * alpha) + (state.value * (1 - alpha))
            state.last_updated = datetime.now()
            
            return state.value
            
        except Exception as e:
            logger.error(f"Error calculating EMA for {symbol}: {e}")
            return None
    
    def calculate_rsi_incremental(self, symbol: str, price: float, period: int = RSI_PERIOD) -> Optional[float]:
        """
        Calculate RSI incrementally using Wilder's smoothing method.
        
        Args:
            symbol: Trading symbol
            price: New price value
            period: RSI period
            
        Returns:
            Updated RSI value or None if not enough data
        """
        try:
            self._ensure_symbol_initialized(symbol)
            state_key = self._get_state_key(symbol, "rsi", period=period)
            
            if state_key not in self._states[symbol]:
                # Initialize new RSI state
                self._states[symbol][state_key] = RSIState(
                    last_updated=datetime.now(),
                    period=period,
                    value=50.0,  # Neutral RSI
                    avg_gain=0.0,
                    avg_loss=0.0,
                    last_price=price
                )
                return None  # Need at least one price change
            
            state = self._states[symbol][state_key]
            
            # Calculate price change
            price_change = price - state.last_price
            state.last_price = price
            
            # Separate gains and losses
            gain = max(price_change, 0)
            loss = max(-price_change, 0)
            
            # Wilder's smoothing (similar to EMA with alpha = 1/period)
            alpha = 1.0 / period
            state.avg_gain = alpha * gain + (1 - alpha) * state.avg_gain
            state.avg_loss = alpha * loss + (1 - alpha) * state.avg_loss
            
            # Calculate RSI
            if state.avg_loss == 0:
                state.value = 100.0
            else:
                rs = state.avg_gain / state.avg_loss
                state.value = 100.0 - (100.0 / (1.0 + rs))
            
            state.last_updated = datetime.now()
            return state.value
            
        except Exception as e:
            logger.error(f"Error calculating RSI for {symbol}: {e}")
            return None
    
    def calculate_macd_incremental(self, symbol: str, price: float, 
                                 fast: int = MACD_FAST, slow: int = MACD_SLOW, 
                                 signal: int = MACD_SIGNAL) -> Optional[Tuple[float, float, float]]:
        """
        Calculate MACD incrementally.
        
        Args:
            symbol: Trading symbol
            price: New price value
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal EMA period
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram) or None if not enough data
        """
        try:
            self._ensure_symbol_initialized(symbol)
            state_key = self._get_state_key(symbol, "macd", fast=fast, slow=slow, signal=signal)
            
            fast_alpha = 2.0 / (fast + 1)
            slow_alpha = 2.0 / (slow + 1)
            signal_alpha = 2.0 / (signal + 1)
            
            if state_key not in self._states[symbol]:
                # Initialize new MACD state
                self._states[symbol][state_key] = MACDState(
                    last_updated=datetime.now(),
                    period=max(fast, slow, signal),
                    value=0.0,
                    fast_ema=price,
                    slow_ema=price,
                    signal_ema=0.0,
                    fast_alpha=fast_alpha,
                    slow_alpha=slow_alpha,
                    signal_alpha=signal_alpha,
                    macd_line=0.0,
                    signal_line=0.0,
                    histogram=0.0
                )
                return None  # Need more data
            
            state = self._states[symbol][state_key]
            
            # Update fast and slow EMAs
            state.fast_ema = (price * fast_alpha) + (state.fast_ema * (1 - fast_alpha))
            state.slow_ema = (price * slow_alpha) + (state.slow_ema * (1 - slow_alpha))
            
            # Calculate MACD line
            state.macd_line = state.fast_ema - state.slow_ema
            
            # Update signal line EMA
            state.signal_line = (state.macd_line * signal_alpha) + (state.signal_line * (1 - signal_alpha))
            
            # Calculate histogram
            state.histogram = state.macd_line - state.signal_line
            
            state.last_updated = datetime.now()
            return (state.macd_line, state.signal_line, state.histogram)
            
        except Exception as e:
            logger.error(f"Error calculating MACD for {symbol}: {e}")
            return None
    
    def calculate_bollinger_bands_incremental(self, symbol: str, price: float, 
                                            period: int = BB_PERIOD, 
                                            std_dev: float = BB_STD_DEV) -> Optional[Tuple[float, float, float]]:
        """
        Calculate Bollinger Bands incrementally.
        
        Args:
            symbol: Trading symbol
            price: New price value
            period: BB period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (Upper band, Middle band, Lower band) or None if not enough data
        """
        try:
            self._ensure_symbol_initialized(symbol)
            state_key = self._get_state_key(symbol, "bb", period=period, std_dev=std_dev)
            
            if state_key not in self._states[symbol]:
                # Initialize new Bollinger Bands state
                sma_state = SMAState(
                    last_updated=datetime.now(),
                    period=period,
                    value=price,
                    sum=price,
                    count=1,
                    values=[price]
                )
                
                self._states[symbol][state_key] = BollingerBandsState(
                    last_updated=datetime.now(),
                    period=period,
                    value=price,
                    sma_state=sma_state,
                    sum_squares=price * price,
                    upper_band=price,
                    middle_band=price,
                    lower_band=price,
                    std_dev_multiplier=std_dev
                )
                return None  # Not enough data yet
            
            state = self._states[symbol][state_key]
            
            # Update SMA state
            state.sma_state.values.append(price)
            state.sma_state.sum += price
            state.sma_state.count += 1
            state.sum_squares += price * price
            
            # Remove old values if window is full
            if len(state.sma_state.values) > period:
                old_value = state.sma_state.values.pop(0)
                state.sma_state.sum -= old_value
                state.sma_state.count -= 1
                state.sum_squares -= old_value * old_value
            
            # Calculate if we have enough data
            if len(state.sma_state.values) >= period:
                # Calculate SMA (middle band)
                state.middle_band = state.sma_state.sum / period
                
                # Calculate standard deviation
                mean_squares = state.sum_squares / period
                variance = mean_squares - (state.middle_band * state.middle_band)
                std_deviation = math.sqrt(max(variance, 0))  # Avoid negative variance due to floating point errors
                
                # Calculate bands
                band_width = std_deviation * std_dev
                state.upper_band = state.middle_band + band_width
                state.lower_band = state.middle_band - band_width
                
                state.last_updated = datetime.now()
                return (state.upper_band, state.middle_band, state.lower_band)
            
            return None  # Not enough data yet
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands for {symbol}: {e}")
            return None
    
    def get_indicator_value(self, symbol: str, indicator: str, **params) -> Optional[float]:
        """
        Get current value of an indicator.
        
        Args:
            symbol: Trading symbol
            indicator: Indicator name
            **params: Indicator parameters
            
        Returns:
            Current indicator value or None if not available
        """
        try:
            if symbol not in self._states:
                return None
            
            state_key = self._get_state_key(symbol, indicator, **params)
            if state_key not in self._states[symbol]:
                return None
            
            return self._states[symbol][state_key].value
            
        except Exception as e:
            logger.error(f"Error getting indicator value for {symbol}: {e}")
            return None
    
    def get_all_indicators(self, symbol: str) -> Dict[str, Any]:
        """
        Get all current indicator values for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with all indicator values
        """
        try:
            if symbol not in self._states:
                return {}
            
            results = {}
            for state_key, state in self._states[symbol].items():
                if isinstance(state, SMAState):
                    results[state_key] = state.value
                elif isinstance(state, EMAState):
                    results[state_key] = state.value
                elif isinstance(state, RSIState):
                    results[state_key] = state.value
                elif isinstance(state, MACDState):
                    results[state_key] = {
                        "macd_line": state.macd_line,
                        "signal_line": state.signal_line,
                        "histogram": state.histogram
                    }
                elif isinstance(state, BollingerBandsState):
                    results[state_key] = {
                        "upper_band": state.upper_band,
                        "middle_band": state.middle_band,
                        "lower_band": state.lower_band
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting all indicators for {symbol}: {e}")
            return {}
    
    def reset_indicator(self, symbol: str, indicator: str, **params):
        """
        Reset a specific indicator for a symbol.
        
        Args:
            symbol: Trading symbol
            indicator: Indicator name
            **params: Indicator parameters
        """
        try:
            if symbol in self._states:
                state_key = self._get_state_key(symbol, indicator, **params)
                if state_key in self._states[symbol]:
                    del self._states[symbol][state_key]
                    logger.info(f"Reset indicator {state_key} for {symbol}")
                    
        except Exception as e:
            logger.error(f"Error resetting indicator for {symbol}: {e}")
    
    def reset_symbol(self, symbol: str):
        """
        Reset all indicators for a symbol.
        
        Args:
            symbol: Trading symbol
        """
        try:
            if symbol in self._states:
                del self._states[symbol]
                del self._initialized[symbol]
                logger.info(f"Reset all indicators for {symbol}")
                
        except Exception as e:
            logger.error(f"Error resetting symbol {symbol}: {e}")
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get summary of current state.
        
        Returns:
            Dictionary with state summary
        """
        try:
            summary = {
                "total_symbols": len(self._states),
                "symbols": list(self._states.keys()),
                "total_indicators": sum(len(indicators) for indicators in self._states.values()),
                "indicators_by_symbol": {
                    symbol: list(indicators.keys()) 
                    for symbol, indicators in self._states.items()
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting state summary: {e}")
            return {"error": str(e)}
    
    def generate_trading_signals(self, symbol: str) -> Dict[str, Any]:
        """
        Generate trading signals based on current indicator values.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with trading signals
        """
        try:
            signals = {}
            
            if symbol not in self._states:
                return signals
            
            # RSI signals
            rsi_state = self._get_indicator_state(symbol, "rsi")
            if rsi_state:
                signals["rsi_overbought"] = rsi_state.value > RSI_OVERBOUGHT
                signals["rsi_oversold"] = rsi_state.value < RSI_OVERSOLD
                signals["rsi_value"] = rsi_state.value
            
            # MACD signals
            macd_state = self._get_indicator_state(symbol, "macd")
            if macd_state:
                signals["macd_bullish"] = macd_state.macd_line > macd_state.signal_line
                signals["macd_bearish"] = macd_state.macd_line < macd_state.signal_line
                signals["macd_histogram_positive"] = macd_state.histogram > 0
                signals["macd_values"] = {
                    "macd_line": macd_state.macd_line,
                    "signal_line": macd_state.signal_line,
                    "histogram": macd_state.histogram
                }
            
            # Moving Average signals
            sma_50_state = self._get_indicator_state(symbol, "sma", period=50)
            sma_200_state = self._get_indicator_state(symbol, "sma", period=200)
            
            if sma_50_state and sma_200_state:
                signals["golden_cross"] = sma_50_state.value > sma_200_state.value
                signals["death_cross"] = sma_50_state.value < sma_200_state.value
            
            # Bollinger Bands signals
            bb_state = self._get_indicator_state(symbol, "bb")
            if bb_state:
                # We'd need current price to determine if it's above/below bands
                # This would be passed in from the calculation engine
                signals["bb_squeeze"] = (bb_state.upper_band - bb_state.lower_band) / bb_state.middle_band < 0.1
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals for {symbol}: {e}")
            return {}
    
    def _get_indicator_state(self, symbol: str, indicator: str, **params) -> Optional[IndicatorState]:
        """Get indicator state by name and parameters."""
        try:
            if symbol not in self._states:
                return None
            
            state_key = self._get_state_key(symbol, indicator, **params)
            return self._states[symbol].get(state_key)
            
        except Exception as e:
            logger.error(f"Error getting indicator state: {e}")
            return None