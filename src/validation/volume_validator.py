"""
Volume Validator - Minimal & Non-Invasive
Provides optional volume validation without impacting existing signal generation.
"""

import logging
from typing import Tuple, Optional

class VolumeValidator:
    """Non-invasive volume validator that logs warnings but doesn't block signals"""
    
    def __init__(self, enabled: bool = False):
        """
        Initialize volume validator
        
        Args:
            enabled: Whether validation is enabled (default: False for safety)
        """
        self.enabled = enabled
        self.min_volume_threshold = 1000
        
    def validate_volume(self, symbol: str, volume: Optional[float]) -> Tuple[bool, str]:
        """
        Validate volume - non-invasive, log-only
        
        Args:
            symbol: Trading symbol
            volume: Volume to validate
            
        Returns:
            Tuple of (is_valid, message) - always returns True to not block signals
        """
        if not self.enabled:
            return True, "Volume validation disabled"
            
        if volume is None:
            logging.warning(f"Volume validation: No volume data for {symbol}")
            return True, "No volume data"  # Don't block signals
            
        if volume < self.min_volume_threshold:
            logging.warning(f"Volume validation: {symbol} volume {volume} below threshold {self.min_volume_threshold}")
            return True, f"Low volume: {volume}"  # Log but don't block
            
        return True, "Volume OK"
        
    def set_threshold(self, threshold: float):
        """Update volume threshold"""
        self.min_volume_threshold = threshold
        
    def set_enabled(self, enabled: bool):
        """Enable or disable volume validation"""
        self.enabled = enabled