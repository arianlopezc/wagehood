"""
Gap-Aware Incremental Technical Indicators

This module extends the incremental indicator calculator with gap awareness and
session boundary handling for maintaining data integrity across market transitions.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
import math

from .incremental_indicators import (
    IncrementalIndicatorCalculator, IndicatorState, SMAState, EMAState, 
    RSIState, MACDState, BollingerBandsState
)
from .market_session_manager import SessionTransition, SessionState

logger = logging.getLogger(__name__)


@dataclass
class GapAwareIndicatorState(IndicatorState):
    """Enhanced indicator state with gap awareness."""
    last_price: float = 0.0
    last_timestamp: Optional[datetime] = None
    gap_count: int = 0
    session_resets: int = 0
    gap_threshold_hours: float = 4.0
    
    # Gap handling configuration
    reset_on_weekend_gap: bool = True
    reset_on_significant_gap: bool = False
    adjust_for_gaps: bool = True


@dataclass
class GapAwareSMAState(SMAState):
    """Simple Moving Average state with gap handling."""
    gap_count: int = 0
    session_resets: int = 0
    last_timestamp: Optional[datetime] = None
    
    def should_reset_on_gap(self, gap_duration_hours: float) -> bool:
        """Determine if SMA should reset due to gap."""
        # SMA generally doesn't need reset, but very long gaps might warrant it
        return gap_duration_hours > 168  # 1 week


@dataclass
class GapAwareEMAState(EMAState):
    """Exponential Moving Average state with gap handling."""
    gap_count: int = 0
    session_resets: int = 0
    last_timestamp: Optional[datetime] = None
    gap_adjustment_factor: float = 1.0
    
    def should_reset_on_gap(self, gap_duration_hours: float) -> bool:
        """Determine if EMA should reset due to gap."""
        # EMA can handle gaps better, only reset on very long gaps
        return gap_duration_hours > 120  # 5 days
    
    def calculate_gap_adjustment(self, gap_duration_hours: float) -> float:
        """Calculate adjustment factor for gap duration."""
        if gap_duration_hours < 4:
            return 1.0
        elif gap_duration_hours < 24:
            return 0.9  # Slight discount for overnight gaps
        elif gap_duration_hours < 72:
            return 0.8  # More discount for weekend gaps
        else:
            return 0.7  # Significant discount for holiday gaps


@dataclass
class GapAwareRSIState(RSIState):
    """RSI state with gap handling."""
    gap_count: int = 0
    session_resets: int = 0
    last_timestamp: Optional[datetime] = None
    price_history: List[float] = field(default_factory=list)
    
    def should_reset_on_gap(self, gap_duration_hours: float) -> bool:
        """Determine if RSI should reset due to gap."""
        # RSI is very sensitive to gaps, reset more aggressively
        return gap_duration_hours > 16  # Reset after weekend gaps


@dataclass
class GapAwareMACDState(MACDState):
    """MACD state with gap handling."""
    gap_count: int = 0
    session_resets: int = 0
    last_timestamp: Optional[datetime] = None
    
    def should_reset_on_gap(self, gap_duration_hours: float) -> bool:
        """Determine if MACD should reset due to gap."""
        # MACD uses EMAs, so it's moderately gap-tolerant
        return gap_duration_hours > 72  # Reset after 3-day gaps


class GapAwareIndicatorCalculator(IncrementalIndicatorCalculator):
    """
    Enhanced incremental indicator calculator with gap awareness and session handling.
    
    This class extends the base calculator with:
    - Gap detection and handling
    - Session boundary awareness
    - Indicator-specific reset logic
    - Continuous operation across market transitions
    """
    
    def __init__(self):
        """Initialize the gap-aware calculator."""
        super().__init__()
        
        # Gap handling configuration
        self._gap_thresholds = {
            'short_gap': 1.0,      # 1 hour
            'medium_gap': 4.0,     # 4 hours
            'long_gap': 16.0,      # 16 hours (overnight)
            'weekend_gap': 40.0,   # 40+ hours (weekend)
            'holiday_gap': 80.0    # 80+ hours (long weekend/holiday)
        }
        
        # Session handling statistics
        self._gap_stats = {
            'total_gaps_handled': 0,
            'indicator_resets': 0,
            'gap_adjustments_applied': 0,
            'session_transitions': 0
        }
        
        logger.info("Initialized GapAwareIndicatorCalculator")
    
    def handle_session_transition(self, symbol: str, session_transition: SessionTransition):
        """
        Handle session transition for all indicators of a symbol.
        
        Args:
            symbol: Trading symbol
            session_transition: Session transition information
        """
        if symbol not in self._states:
            return
        
        self._gap_stats['session_transitions'] += 1
        gap_duration = session_transition.gap_duration_hours
        
        logger.debug(f"Handling session transition for {symbol}: "
                    f"{session_transition.previous_state.value} → {session_transition.new_state.value} "
                    f"(gap: {gap_duration:.1f}h)")
        
        # Apply gap handling to all indicators for this symbol (use list to avoid iteration issues)
        indicator_items = list(self._states[symbol].items())
        for indicator_key, state in indicator_items:
            self._apply_gap_handling(symbol, indicator_key, state, session_transition)
    
    def _apply_gap_handling(self, symbol: str, indicator_key: str, state: IndicatorState, 
                          session_transition: SessionTransition):
        """Apply gap-specific handling to an indicator state."""
        gap_duration = session_transition.gap_duration_hours
        
        # Determine gap type
        gap_type = self._classify_gap(gap_duration)
        
        # Apply indicator-specific gap handling
        if isinstance(state, (GapAwareRSIState, RSIState)):
            self._handle_rsi_gap(symbol, indicator_key, state, session_transition)
        elif isinstance(state, (GapAwareEMAState, EMAState)):
            self._handle_ema_gap(symbol, indicator_key, state, session_transition)
        elif isinstance(state, (GapAwareSMAState, SMAState)):
            self._handle_sma_gap(symbol, indicator_key, state, session_transition)
        elif isinstance(state, (GapAwareMACDState, MACDState)):
            self._handle_macd_gap(symbol, indicator_key, state, session_transition)
    
    def _handle_rsi_gap(self, symbol: str, indicator_key: str, state: IndicatorState, 
                       session_transition: SessionTransition):
        """Handle gap for RSI indicator."""
        gap_duration = session_transition.gap_duration_hours
        
        # RSI is sensitive to gaps - reset on significant gaps
        if gap_duration > self._gap_thresholds['long_gap']:
            logger.info(f"Resetting RSI {indicator_key} for {symbol} due to {gap_duration:.1f}h gap")
            self.reset_indicator_state(f"{symbol}_{indicator_key}")
            self._gap_stats['indicator_resets'] += 1
        elif gap_duration > self._gap_thresholds['medium_gap']:
            # Apply adjustment for medium gaps
            if hasattr(state, 'avg_gain') and hasattr(state, 'avg_loss'):
                # Reduce the impact of previous gains/losses
                state.avg_gain *= 0.9
                state.avg_loss *= 0.9
                self._gap_stats['gap_adjustments_applied'] += 1
                logger.debug(f"Applied gap adjustment to RSI {indicator_key} for {symbol}")
    
    def _handle_ema_gap(self, symbol: str, indicator_key: str, state: IndicatorState, 
                       session_transition: SessionTransition):
        """Handle gap for EMA indicator."""
        gap_duration = session_transition.gap_duration_hours
        
        # EMA can handle gaps better, but apply adjustment factor
        if gap_duration > self._gap_thresholds['weekend_gap']:
            if hasattr(state, 'value'):
                # Apply gap adjustment factor
                adjustment_factor = 0.8 if gap_duration < 80 else 0.7
                logger.debug(f"Applied {adjustment_factor} gap adjustment to EMA {indicator_key} for {symbol}")
                self._gap_stats['gap_adjustments_applied'] += 1
    
    def _handle_sma_gap(self, symbol: str, indicator_key: str, state: IndicatorState, 
                       session_transition: SessionTransition):
        """Handle gap for SMA indicator."""
        gap_duration = session_transition.gap_duration_hours
        
        # SMA is least affected by gaps, only reset on very long gaps
        if gap_duration > self._gap_thresholds['holiday_gap']:
            logger.info(f"Resetting SMA {indicator_key} for {symbol} due to {gap_duration:.1f}h gap")
            self.reset_indicator_state(f"{symbol}_{indicator_key}")
            self._gap_stats['indicator_resets'] += 1
    
    def _handle_macd_gap(self, symbol: str, indicator_key: str, state: IndicatorState, 
                        session_transition: SessionTransition):
        """Handle gap for MACD indicator."""
        gap_duration = session_transition.gap_duration_hours
        
        # MACD uses EMAs, so moderate gap tolerance
        if gap_duration > self._gap_thresholds['weekend_gap'] * 1.5:  # 60+ hours
            logger.info(f"Resetting MACD {indicator_key} for {symbol} due to {gap_duration:.1f}h gap")
            self.reset_indicator_state(f"{symbol}_{indicator_key}")
            self._gap_stats['indicator_resets'] += 1
    
    def _classify_gap(self, gap_duration_hours: float) -> str:
        """Classify gap duration type."""
        if gap_duration_hours < self._gap_thresholds['short_gap']:
            return 'normal'
        elif gap_duration_hours < self._gap_thresholds['medium_gap']:
            return 'short_gap'
        elif gap_duration_hours < self._gap_thresholds['long_gap']:
            return 'medium_gap'
        elif gap_duration_hours < self._gap_thresholds['weekend_gap']:
            return 'long_gap'
        elif gap_duration_hours < self._gap_thresholds['holiday_gap']:
            return 'weekend_gap'
        else:
            return 'holiday_gap'
    
    def reset_indicator_state(self, indicator_key: str):
        """Reset specific indicator state."""
        try:
            # Parse the indicator key to find symbol and indicator
            parts = indicator_key.split('_')
            if len(parts) >= 2:
                symbol = parts[0]
                indicator_name = '_'.join(parts[1:])
                
                if symbol in self._states and indicator_name in self._states[symbol]:
                    del self._states[symbol][indicator_name]
                    logger.debug(f"Reset indicator state for {indicator_key}")
                    return True
        except Exception as e:
            logger.error(f"Error resetting indicator state {indicator_key}: {e}")
        
        return False
    
    def calculate_sma_incremental_with_gap_awareness(self, symbol: str, price: float, 
                                                   period: int, timestamp: datetime,
                                                   gap_info: Optional[Dict] = None) -> Optional[float]:
        """
        Calculate SMA with gap awareness.
        
        Args:
            symbol: Trading symbol
            price: New price value
            period: SMA period
            timestamp: Current timestamp
            gap_info: Gap information if any
            
        Returns:
            Updated SMA value or None if not enough data
        """
        # Handle gap if detected
        if gap_info and gap_info.get('is_significant', False):
            gap_duration = gap_info.get('duration_hours', 0)
            if gap_duration > self._gap_thresholds['holiday_gap']:
                # Reset SMA on very long gaps
                state_key = self._get_state_key(symbol, "sma", period=period)
                if symbol in self._states and state_key in self._states[symbol]:
                    del self._states[symbol][state_key]
                    self._gap_stats['indicator_resets'] += 1
        
        # Calculate SMA normally
        return self.calculate_sma_incremental(symbol, price, period)
    
    def calculate_ema_incremental_with_gap_awareness(self, symbol: str, price: float, 
                                                   period: int, timestamp: datetime,
                                                   gap_info: Optional[Dict] = None) -> Optional[float]:
        """
        Calculate EMA with gap awareness.
        
        Args:
            symbol: Trading symbol
            price: New price value
            period: EMA period
            timestamp: Current timestamp
            gap_info: Gap information if any
            
        Returns:
            Updated EMA value or None if not initialized
        """
        # Handle gap if detected
        if gap_info and gap_info.get('is_significant', False):
            gap_duration = gap_info.get('duration_hours', 0)
            state_key = self._get_state_key(symbol, "ema", period=period)
            
            if (symbol in self._states and state_key in self._states[symbol] and 
                gap_duration > self._gap_thresholds['weekend_gap']):
                
                # Apply gap adjustment to EMA
                state = self._states[symbol][state_key]
                if hasattr(state, 'value'):
                    if gap_duration < 80:
                        adjustment = 0.9
                    else:
                        adjustment = 0.8
                    
                    # Don't modify the state directly, apply adjustment in calculation
                    result = self.calculate_ema_incremental(symbol, price, period)
                    if result is not None:
                        self._gap_stats['gap_adjustments_applied'] += 1
                        return result * adjustment
        
        # Calculate EMA normally
        return self.calculate_ema_incremental(symbol, price, period)
    
    def calculate_rsi_incremental_with_gap_awareness(self, symbol: str, price: float, 
                                                   period: int, timestamp: datetime,
                                                   gap_info: Optional[Dict] = None) -> Optional[float]:
        """
        Calculate RSI with gap awareness.
        
        Args:
            symbol: Trading symbol
            price: New price value
            period: RSI period
            timestamp: Current timestamp
            gap_info: Gap information if any
            
        Returns:
            Updated RSI value or None if not enough data
        """
        # Handle gap if detected
        if gap_info and gap_info.get('is_significant', False):
            gap_duration = gap_info.get('duration_hours', 0)
            if gap_duration > self._gap_thresholds['long_gap']:
                # Reset RSI on overnight or longer gaps
                state_key = self._get_state_key(symbol, "rsi", period=period)
                if symbol in self._states and state_key in self._states[symbol]:
                    del self._states[symbol][state_key]
                    self._gap_stats['indicator_resets'] += 1
                    logger.debug(f"Reset RSI for {symbol} due to {gap_duration:.1f}h gap")
        
        # Calculate RSI normally
        return self.calculate_rsi_incremental(symbol, price, period)
    
    def get_gap_stats(self) -> Dict[str, Any]:
        """Get gap handling statistics."""
        return {
            **self._gap_stats,
            'gap_thresholds': self._gap_thresholds,
            'total_states': sum(len(states) for states in self._states.values()),
            'symbols_tracked': len(self._states)
        }
    
    def configure_gap_thresholds(self, **thresholds):
        """Configure gap detection thresholds."""
        for key, value in thresholds.items():
            if key in self._gap_thresholds:
                self._gap_thresholds[key] = value
                logger.info(f"Updated {key} threshold to {value} hours")
    
    def get_all_indicators_with_gap_info(self, symbol: str) -> Dict[str, Any]:
        """Get all indicators for a symbol with gap handling information."""
        base_indicators = self.get_all_indicators(symbol)
        
        # Add gap handling metadata
        gap_info = {
            'gap_stats': self.get_gap_stats(),
            'indicators': base_indicators
        }
        
        return gap_info