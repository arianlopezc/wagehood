"""
Timing Collector - Minimal & Non-Invasive
Provides optional timing metrics collection without impacting existing functionality.
"""

import time
import logging
from collections import deque
from typing import Optional, Dict, Any

class TimingCollector:
    """Non-invasive timing collector for performance monitoring"""
    
    def __init__(self, enabled: bool = False, max_history: int = 100):
        """
        Initialize timing collector
        
        Args:
            enabled: Whether collection is enabled (default: False for safety)
            max_history: Maximum number of timing records to keep
        """
        self.enabled = enabled
        self.timings = deque(maxlen=max_history)
        
    def record_timing(self, operation: str, duration: float, metadata: Optional[Dict[str, Any]] = None):
        """
        Record timing information - non-invasive
        
        Args:
            operation: Name of the operation being timed
            duration: Duration in seconds
            metadata: Optional metadata about the operation
        """
        if not self.enabled:
            return
            
        try:
            timing_info = {
                'operation': operation,
                'duration': duration,
                'timestamp': time.time(),
                'metadata': metadata or {}
            }
            self.timings.append(timing_info)
            
        except Exception as e:
            logging.debug(f"Timing collection error: {e}")
            # Silently continue - this is just monitoring
            
    def get_summary(self) -> Dict[str, Any]:
        """
        Get timing summary - for debugging
        
        Returns:
            Dictionary with timing statistics
        """
        if not self.enabled or not self.timings:
            return {"status": "disabled or no data"}
            
        durations = [t['duration'] for t in self.timings]
        operations = [t['operation'] for t in self.timings]
        
        # Calculate statistics
        total_count = len(durations)
        avg_duration = sum(durations) / total_count
        max_duration = max(durations)
        min_duration = min(durations)
        
        # Count by operation type
        operation_counts = {}
        for op in operations:
            operation_counts[op] = operation_counts.get(op, 0) + 1
        
        return {
            "enabled": self.enabled,
            "total_operations": total_count,
            "avg_duration": avg_duration,
            "max_duration": max_duration,
            "min_duration": min_duration,
            "operation_counts": operation_counts
        }
        
    def clear_history(self):
        """Clear timing history"""
        self.timings.clear()
        
    def set_enabled(self, enabled: bool):
        """Enable or disable timing collection"""
        self.enabled = enabled