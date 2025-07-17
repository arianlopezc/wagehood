"""
Timezone utilities for consistent datetime handling across the application.

This module provides timezone-aware datetime functions to ensure proper
market hour detection regardless of the server's local timezone.
"""

from datetime import datetime
import pytz
from typing import Optional

# Define timezone constants
UTC = pytz.UTC
EST = pytz.timezone('US/Eastern')


def utc_now() -> datetime:
    """
    Get current UTC time as timezone-aware datetime.
    
    Returns:
        Current UTC time with timezone info
    """
    return datetime.now(UTC)


def est_now() -> datetime:
    """
    Get current Eastern time as timezone-aware datetime.
    
    Returns:
        Current Eastern time with timezone info
    """
    return datetime.now(EST)


def to_utc(dt: datetime) -> datetime:
    """
    Convert datetime to UTC timezone.
    
    Args:
        dt: Datetime to convert (can be naive or timezone-aware)
        
    Returns:
        UTC datetime with timezone info
        
    Note:
        If input is naive, it's assumed to be in the system's local timezone
    """
    if dt.tzinfo is None:
        # For naive datetimes, we need to localize to the system timezone first
        # This is safer than assuming UTC
        local_tz = get_local_timezone()
        
        # Handle both pytz and standard library timezone objects
        if hasattr(local_tz, 'localize'):
            # pytz timezone
            dt = local_tz.localize(dt)
        else:
            # Standard library timezone
            dt = dt.replace(tzinfo=local_tz)
    
    return dt.astimezone(UTC)


def to_est(dt: datetime) -> datetime:
    """
    Convert datetime to Eastern timezone.
    
    Args:
        dt: Datetime to convert (can be naive or timezone-aware)
        
    Returns:
        Eastern datetime with timezone info
        
    Note:
        If input is naive, it's assumed to be in the system's local timezone
    """
    if dt.tzinfo is None:
        # For naive datetimes, we need to localize to the system timezone first
        local_tz = get_local_timezone()
        
        # Handle both pytz and standard library timezone objects
        if hasattr(local_tz, 'localize'):
            # pytz timezone
            dt = local_tz.localize(dt)
        else:
            # Standard library timezone
            dt = dt.replace(tzinfo=local_tz)
    
    return dt.astimezone(EST)


def get_local_timezone():
    """
    Get the system's local timezone.
    
    Returns:
        Local timezone object
    """
    # Try to get the system's timezone using multiple methods
    import time
    
    # Method 1: Try using datetime's astimezone() which is most reliable
    try:
        local_dt = datetime.now().astimezone()
        return local_dt.tzinfo
    except:
        pass
    
    # Method 2: Try using time.tzname with known mappings
    try:
        import time
        tz_name = time.tzname[1] if time.daylight else time.tzname[0]
        
        # Common timezone mappings for abbreviations
        tz_mappings = {
            'EDT': 'US/Eastern',
            'EST': 'US/Eastern', 
            'CDT': 'US/Central',
            'CST': 'US/Central',
            'MDT': 'US/Mountain',
            'MST': 'US/Mountain',
            'PDT': 'US/Pacific',
            'PST': 'US/Pacific',
        }
        
        if tz_name in tz_mappings:
            return pytz.timezone(tz_mappings[tz_name])
        else:
            # Try direct pytz lookup
            return pytz.timezone(tz_name)
    except:
        pass
    
    # Method 3: Fallback to UTC
    import logging
    logging.warning("Could not detect local timezone, falling back to UTC")
    return UTC


def ensure_timezone_aware(dt: datetime, default_tz = None) -> datetime:
    """
    Ensure datetime is timezone-aware.
    
    Args:
        dt: Datetime to check
        default_tz: Timezone to use if datetime is naive (defaults to UTC)
        
    Returns:
        Timezone-aware datetime
    """
    if dt.tzinfo is None:
        if default_tz is None:
            default_tz = UTC
            
        # Handle both pytz and standard library timezone objects
        if hasattr(default_tz, 'localize'):
            # pytz timezone
            return default_tz.localize(dt)
        else:
            # Standard library timezone
            return dt.replace(tzinfo=default_tz)
    
    return dt


def trading_time_now() -> datetime:
    """
    Get current time in trading timezone (Eastern).
    
    This is the preferred function for market-related operations.
    
    Returns:
        Current time in Eastern timezone
    """
    return est_now()