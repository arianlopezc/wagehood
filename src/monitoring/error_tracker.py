"""
Error Tracker - Minimal & Non-Invasive
Provides optional error tracking without impacting existing error handling.
"""

import logging
import time
from collections import defaultdict, deque
from typing import Optional, Dict, Any, List

class ErrorTracker:
    """Non-invasive error tracker for debugging and monitoring"""
    
    def __init__(self, enabled: bool = False, max_history: int = 100):
        """
        Initialize error tracker
        
        Args:
            enabled: Whether tracking is enabled (default: False for safety)
            max_history: Maximum number of error records to keep
        """
        self.enabled = enabled
        self.error_history = deque(maxlen=max_history)
        self.error_counts = defaultdict(int)
        
    def track_error(self, component: str, symbol: str, error_message: str, error_type: str = "general"):
        """
        Track error information - non-invasive
        
        Args:
            component: Name of the component where error occurred
            symbol: Trading symbol (if applicable)
            error_message: Error message
            error_type: Type of error (default: "general")
        """
        if not self.enabled:
            return
            
        try:
            error_info = {
                'component': component,
                'symbol': symbol,
                'error_message': error_message,
                'error_type': error_type,
                'timestamp': time.time()
            }
            
            self.error_history.append(error_info)
            key = f"{component}_{error_type}"
            self.error_counts[key] += 1
            
            logging.info(f"Error tracked: {component} | {symbol} | {error_type} | {error_message}")
            
        except Exception as e:
            logging.debug(f"Error tracking error: {e}")
            # Silently continue - this is just tracking
            
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get error summary - for debugging
        
        Returns:
            Dictionary with error statistics
        """
        if not self.enabled:
            return {"status": "disabled"}
            
        recent_errors = list(self.error_history)[-5:]  # Last 5 errors
        
        return {
            "enabled": self.enabled,
            "total_errors": len(self.error_history),
            "error_counts": dict(self.error_counts),
            "recent_errors": recent_errors
        }
        
    def get_errors_by_component(self, component: str) -> List[Dict[str, Any]]:
        """
        Get errors for a specific component
        
        Args:
            component: Component name
            
        Returns:
            List of error records for the component
        """
        if not self.enabled:
            return []
            
        return [error for error in self.error_history if error['component'] == component]
        
    def clear_history(self):
        """Clear error history"""
        self.error_history.clear()
        self.error_counts.clear()
        
    def set_enabled(self, enabled: bool):
        """Enable or disable error tracking"""
        self.enabled = enabled