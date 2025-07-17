"""
Extended Hours Market Calendar

Handles market session detection supporting pre-market (4 AM) through after-hours (8 PM EST).
"""

from datetime import datetime, time, timedelta
import pandas_market_calendars as mcal
from typing import Optional
import pytz
import logging

logger = logging.getLogger(__name__)


class ExtendedHoursCalendar:
    """
    Extended hours market calendar supporting pre-market and after-hours.
    Provides mathematical accuracy for all trading sessions.
    """
    
    def __init__(self):
        self.nyse_calendar = mcal.get_calendar('NYSE')
        self.est = pytz.timezone('US/Eastern')
        
        # Extended hours definitions
        self.premarket_start = time(4, 0)      # 4:00 AM EST
        self.market_open = time(9, 30)         # 9:30 AM EST  
        self.market_close = time(16, 0)        # 4:00 PM EST
        self.afterhours_end = time(20, 0)      # 8:00 PM EST
        
        # Cache for trading day lookups (improves performance)
        self._trading_day_cache = {}
        
        logger.info("Extended hours calendar initialized (4 AM - 8 PM EST)")
        
    def is_extended_trading_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is within extended trading hours."""
        # Convert to EST if needed
        est_time = self._to_est(timestamp)
        time_only = est_time.time()
        
        # Check if it's a trading day
        if not self._is_trading_day(est_time.date()):
            return False
            
        # Check if within extended hours (4 AM - 8 PM EST)
        return self.premarket_start <= time_only <= self.afterhours_end
        
    def get_trading_session(self, timestamp: datetime) -> Optional[str]:
        """Determine which trading session the timestamp belongs to."""
        if not self.is_extended_trading_hours(timestamp):
            return None
            
        est_time = self._to_est(timestamp)
        time_only = est_time.time()
        
        if self.premarket_start <= time_only < self.market_open:
            return 'premarket'
        elif self.market_open <= time_only < self.market_close:
            return 'regular'
        elif self.market_close <= time_only <= self.afterhours_end:
            return 'afterhours'
            
        return None
        
    def is_new_trading_period(self, last_timestamp: datetime, 
                            current_timestamp: datetime, 
                            timeframe: str) -> bool:
        """Detect new trading periods with extended hours awareness."""
        if timeframe == '1h':
            return self._is_new_hour_boundary(last_timestamp, current_timestamp)
        elif timeframe == '1d':
            return self._is_new_extended_trading_day(last_timestamp, current_timestamp)
            
        return False
        
    def _is_new_hour_boundary(self, last_ts: datetime, current_ts: datetime) -> bool:
        """Check for new hour boundary within extended hours."""
        last_est = self._to_est(last_ts)
        current_est = self._to_est(current_ts)
        
        # Must be within extended hours
        if not (self.is_extended_trading_hours(last_est) and 
                self.is_extended_trading_hours(current_est)):
            return False
            
        return last_est.hour != current_est.hour
        
    def _is_new_extended_trading_day(self, last_ts: datetime, 
                                   current_ts: datetime) -> bool:
        """Check for new extended trading day (4 AM boundary)."""
        last_est = self._to_est(last_ts)
        current_est = self._to_est(current_ts)
        
        # Extended trading day starts at 4 AM
        last_trading_day = self._get_extended_trading_date(last_est)
        current_trading_day = self._get_extended_trading_date(current_est)
        
        return last_trading_day != current_trading_day
        
    def _get_extended_trading_date(self, timestamp: datetime) -> datetime.date:
        """Get the extended trading date (considers 4 AM start)."""
        est_time = self._to_est(timestamp)
        
        # If before 4 AM, it belongs to previous trading day
        if est_time.time() < self.premarket_start:
            return (est_time - timedelta(days=1)).date()
        else:
            return est_time.date()
            
    def _to_est(self, timestamp: datetime) -> datetime:
        """Convert timestamp to EST timezone."""
        if timestamp.tzinfo is None:
            # For naive datetimes, assume local system timezone
            # This is safer than assuming UTC for production deployments
            from ..utils.timezone_utils import get_local_timezone
            local_tz = get_local_timezone()
            try:
                timestamp = local_tz.localize(timestamp)
            except:
                # Fallback to UTC if local timezone detection fails
                timestamp = pytz.utc.localize(timestamp)
                logger.warning("Failed to detect local timezone, assuming UTC")
        return timestamp.astimezone(self.est)
        
    def _is_trading_day(self, date) -> bool:
        """Check if date is a trading day with caching for performance."""
        # Convert to string for cache key
        date_key = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
        
        # Check cache first
        if date_key in self._trading_day_cache:
            return self._trading_day_cache[date_key]
        
        try:
            schedule = self.nyse_calendar.schedule(
                start_date=date,
                end_date=date
            )
            is_trading = not schedule.empty
            
            # Cache the result
            self._trading_day_cache[date_key] = is_trading
            return is_trading
            
        except Exception as e:
            logger.error(f"Error checking trading day: {e}")
            # Default to True to avoid missing signals, but don't cache errors
            return True