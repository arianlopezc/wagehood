"""
Session Volume Logger - Minimal & Non-Invasive
Provides optional session-aware volume logging without impacting existing functionality.
"""

import logging
from datetime import datetime
from typing import Optional
from src.utils.market_calendar import ExtendedHoursCalendar

class SessionVolumeLogger:
    """Non-invasive session volume logger that logs volume information by trading session"""
    
    def __init__(self, enabled: bool = False):
        """
        Initialize session volume logger
        
        Args:
            enabled: Whether logging is enabled (default: False for safety)
        """
        self.enabled = enabled
        self.market_calendar = ExtendedHoursCalendar()  # Use existing implementation
        
    def log_session_volume(self, symbol: str, volume: float, timestamp: Optional[datetime] = None):
        """
        Log volume information by session - non-invasive
        
        Args:
            symbol: Trading symbol
            volume: Volume to log
            timestamp: Timestamp (default: current time)
        """
        if not self.enabled:
            return
            
        try:
            if timestamp is None:
                timestamp = datetime.now()
                
            session_type = self.market_calendar.get_trading_session(timestamp)
            
            if session_type is None:
                session_type = 'closed'
                
            logging.info(f"Session volume: {symbol} | {session_type} | volume: {volume}")
            
        except Exception as e:
            logging.debug(f"Session volume logging error: {e}")
            # Silently continue - this is just logging
            
    def set_enabled(self, enabled: bool):
        """Enable or disable session volume logging"""
        self.enabled = enabled