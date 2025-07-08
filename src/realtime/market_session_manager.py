"""
Market Session Manager - Session-aware data processing

This module provides market session tracking and gap detection for maintaining
data integrity across market open/close cycles while enabling continuous 24/7
real-time operation.
"""

import logging
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import pytz
from threading import Lock

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Market session states."""
    PRE_MARKET = "pre_market"
    REGULAR_HOURS = "regular_hours"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"  # Weekends, holidays, overnight


class MarketType(Enum):
    """Supported market types."""
    US_EQUITY = "us_equity"
    CRYPTO = "crypto"


@dataclass
class SessionTransition:
    """Information about a session transition."""
    previous_state: SessionState
    new_state: SessionState
    transition_time: datetime
    gap_duration_hours: float
    is_new_trading_day: bool
    is_weekend_gap: bool
    is_significant_gap: bool  # > 4 hours


@dataclass
class TradingSession:
    """Represents a trading session with open/close times."""
    date: datetime
    pre_market_open: time
    regular_open: time
    regular_close: time
    after_hours_close: time
    is_trading_day: bool


class MarketSessionManager:
    """
    Manages market session states and transitions for continuous operation.
    
    This class provides session-aware processing while maintaining 24/7 operation:
    - Tracks current market session state
    - Detects session transitions and gaps
    - Provides gap-aware data processing logic
    - Maintains continuous operation across all market states
    """
    
    def __init__(self, market_type: MarketType = MarketType.US_EQUITY, timezone: str = "America/New_York"):
        """
        Initialize market session manager.
        
        Args:
            market_type: Type of market to track
            timezone: Market timezone
        """
        self.market_type = market_type
        self.timezone = pytz.timezone(timezone)
        self._lock = Lock()
        
        # Current session state
        self._current_state = SessionState.CLOSED
        self._last_state_change = None
        self._last_tick_time = None
        
        # Session tracking
        self._current_session = None
        self._session_cache = {}  # Cache for session data
        
        # Gap detection thresholds
        self._significant_gap_hours = 4.0
        self._weekend_gap_hours = 40.0  # Friday close to Monday open
        
        # US Equity market schedule (can be extended for other markets)
        if market_type == MarketType.US_EQUITY:
            self._schedule = {
                'pre_market_open': time(4, 0),      # 4:00 AM ET
                'regular_open': time(9, 30),        # 9:30 AM ET
                'regular_close': time(16, 0),       # 4:00 PM ET
                'after_hours_close': time(20, 0),   # 8:00 PM ET
            }
        elif market_type == MarketType.CRYPTO:
            # Crypto markets are 24/7, but we can still track traditional sessions
            self._schedule = {
                'pre_market_open': time(0, 0),
                'regular_open': time(9, 30),
                'regular_close': time(16, 0),
                'after_hours_close': time(23, 59),
            }
        
        logger.info(f"Initialized MarketSessionManager for {market_type.value} in {timezone}")
    
    def update_session_state(self, timestamp: datetime) -> Optional[SessionTransition]:
        """
        Update session state based on current timestamp.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            SessionTransition if state changed, None otherwise
        """
        with self._lock:
            # Convert to market timezone
            market_time = self._to_market_time(timestamp)
            new_state = self._determine_session_state(market_time)
            
            transition = None
            if new_state != self._current_state or self._last_tick_time is None:
                # Calculate gap information
                gap_duration = 0.0
                if self._last_tick_time:
                    gap_duration = (timestamp - self._last_tick_time).total_seconds() / 3600.0
                
                transition = SessionTransition(
                    previous_state=self._current_state,
                    new_state=new_state,
                    transition_time=timestamp,
                    gap_duration_hours=gap_duration,
                    is_new_trading_day=self._is_new_trading_day(market_time),
                    is_weekend_gap=self._is_weekend_gap(gap_duration),
                    is_significant_gap=gap_duration > self._significant_gap_hours
                )
                
                logger.info(f"Session transition: {self._current_state.value} → {new_state.value} "
                           f"(gap: {gap_duration:.1f}h)")
                
                self._current_state = new_state
                self._last_state_change = timestamp
            
            self._last_tick_time = timestamp
            return transition
    
    def get_current_state(self) -> SessionState:
        """Get current session state."""
        return self._current_state
    
    def is_trading_hours(self, timestamp: Optional[datetime] = None) -> bool:
        """Check if timestamp is during trading hours (pre-market through after-hours)."""
        if timestamp is None:
            timestamp = datetime.now()
        
        market_time = self._to_market_time(timestamp)
        state = self._determine_session_state(market_time)
        return state != SessionState.CLOSED
    
    def is_regular_hours(self, timestamp: Optional[datetime] = None) -> bool:
        """Check if timestamp is during regular trading hours."""
        if timestamp is None:
            timestamp = datetime.now()
        
        market_time = self._to_market_time(timestamp)
        state = self._determine_session_state(market_time)
        return state == SessionState.REGULAR_HOURS
    
    def should_handle_gap(self, last_timestamp: datetime, current_timestamp: datetime) -> bool:
        """
        Determine if there's a significant gap that needs special handling.
        
        Args:
            last_timestamp: Previous data timestamp
            current_timestamp: Current data timestamp
            
        Returns:
            True if gap should be handled specially
        """
        gap_hours = (current_timestamp - last_timestamp).total_seconds() / 3600.0
        return gap_hours > self._significant_gap_hours
    
    def get_gap_info(self, last_timestamp: datetime, current_timestamp: datetime) -> Dict:
        """
        Get detailed information about a gap between timestamps.
        
        Args:
            last_timestamp: Previous data timestamp
            current_timestamp: Current data timestamp
            
        Returns:
            Dictionary with gap analysis
        """
        gap_duration = (current_timestamp - last_timestamp).total_seconds() / 3600.0
        
        return {
            'duration_hours': gap_duration,
            'is_significant': gap_duration > self._significant_gap_hours,
            'is_weekend_gap': self._is_weekend_gap(gap_duration),
            'crosses_session_boundary': self._crosses_session_boundary(last_timestamp, current_timestamp),
            'gap_type': self._classify_gap(gap_duration),
            'last_session_state': self._determine_session_state(self._to_market_time(last_timestamp)),
            'current_session_state': self._determine_session_state(self._to_market_time(current_timestamp))
        }
    
    def get_session_alignment_time(self, timeframe: str, timestamp: datetime) -> datetime:
        """
        Get the session-aligned timestamp for a given timeframe.
        
        For daily timeframes, aligns to market open.
        For intraday timeframes, uses regular time alignment but respects session boundaries.
        
        Args:
            timeframe: Timeframe string (1m, 5m, 1h, 1d, etc.)
            timestamp: Current timestamp
            
        Returns:
            Session-aligned timestamp
        """
        market_time = self._to_market_time(timestamp)
        
        if timeframe == "1d":
            # Daily candles align with market open
            session_date = market_time.date()
            market_open = datetime.combine(session_date, self._schedule['regular_open'])
            return self.timezone.localize(market_open).astimezone(pytz.UTC)
        else:
            # Intraday timeframes use regular time alignment
            return timestamp
    
    def _to_market_time(self, timestamp: datetime) -> datetime:
        """Convert timestamp to market timezone."""
        if timestamp.tzinfo is None:
            # Assume UTC if no timezone
            timestamp = pytz.UTC.localize(timestamp)
        return timestamp.astimezone(self.timezone)
    
    def _determine_session_state(self, market_time: datetime) -> SessionState:
        """Determine session state for market time."""
        if self.market_type == MarketType.CRYPTO:
            # Crypto is always "open" but we can track traditional sessions
            current_time = market_time.time()
            if self._schedule['regular_open'] <= current_time < self._schedule['regular_close']:
                return SessionState.REGULAR_HOURS
            else:
                return SessionState.AFTER_HOURS
        
        # US Equity logic
        weekday = market_time.weekday()
        if weekday >= 5:  # Weekend (Saturday=5, Sunday=6)
            return SessionState.CLOSED
        
        current_time = market_time.time()
        
        if current_time < self._schedule['pre_market_open']:
            return SessionState.CLOSED
        elif current_time < self._schedule['regular_open']:
            return SessionState.PRE_MARKET
        elif current_time < self._schedule['regular_close']:
            return SessionState.REGULAR_HOURS
        elif current_time < self._schedule['after_hours_close']:
            return SessionState.AFTER_HOURS
        else:
            return SessionState.CLOSED
    
    def _is_new_trading_day(self, market_time: datetime) -> bool:
        """Check if this represents a new trading day."""
        if not self._current_session:
            return True
        
        current_date = market_time.date()
        last_session_date = getattr(self._current_session, 'date', None)
        
        return current_date != last_session_date and market_time.weekday() < 5
    
    def _is_weekend_gap(self, gap_duration_hours: float) -> bool:
        """Check if gap duration suggests a weekend gap."""
        return gap_duration_hours > self._weekend_gap_hours
    
    def _crosses_session_boundary(self, start_time: datetime, end_time: datetime) -> bool:
        """Check if time range crosses a session boundary."""
        start_state = self._determine_session_state(self._to_market_time(start_time))
        end_state = self._determine_session_state(self._to_market_time(end_time))
        return start_state != end_state
    
    def _classify_gap(self, gap_duration_hours: float) -> str:
        """Classify gap type based on duration."""
        if gap_duration_hours < 0.1:
            return "normal"
        elif gap_duration_hours < 1.0:
            return "short_gap"
        elif gap_duration_hours < 4.0:
            return "medium_gap"
        elif gap_duration_hours < 20.0:
            return "long_gap"
        elif gap_duration_hours < 50.0:
            return "weekend_gap"
        else:
            return "extended_gap"
    
    def get_stats(self) -> Dict:
        """Get session manager statistics."""
        with self._lock:
            return {
                'current_state': self._current_state.value,
                'market_type': self.market_type.value,
                'timezone': str(self.timezone),
                'last_state_change': self._last_state_change.isoformat() if self._last_state_change else None,
                'last_tick_time': self._last_tick_time.isoformat() if self._last_tick_time else None,
                'significant_gap_threshold_hours': self._significant_gap_hours,
                'weekend_gap_threshold_hours': self._weekend_gap_hours
            }