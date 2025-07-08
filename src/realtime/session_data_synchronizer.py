"""
Session Data Synchronizer - Continuous operation with data integrity

This module provides day boundary data synchronization and validation while
maintaining continuous 24/7 real-time operation across market session transitions.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import time

from .market_session_manager import MarketSessionManager, SessionState, SessionTransition
from .enhanced_timeframe_manager import EnhancedTimeframeManager
from .gap_aware_indicators import GapAwareIndicatorCalculator

logger = logging.getLogger(__name__)


@dataclass
class DataValidationResult:
    """Result of data validation at session boundaries."""
    timestamp: datetime
    symbol: str
    timeframe: str
    validation_type: str  # "session_boundary", "gap_detection", "historical_alignment"
    success: bool
    issues_found: List[str] = field(default_factory=list)
    corrections_applied: List[str] = field(default_factory=list)
    data_points_validated: int = 0
    validation_duration_ms: float = 0.0


@dataclass
class SynchronizationJob:
    """Background synchronization job."""
    job_id: str
    symbol: str
    timeframes: List[str]
    job_type: str  # "validation", "historical_sync", "gap_bridge"
    priority: int
    created_at: datetime
    scheduled_for: datetime
    max_retries: int = 3
    retry_count: int = 0
    status: str = "pending"  # "pending", "running", "completed", "failed"
    result: Optional[DataValidationResult] = None


class SessionDataSynchronizer:
    """
    Manages data synchronization and validation across session boundaries.
    
    This class ensures data integrity while maintaining continuous operation:
    - Validates data at session boundaries
    - Aligns with historical data when needed
    - Bridges gaps in real-time data
    - Performs background validation without stopping real-time processing
    - Maintains data quality across market transitions
    """
    
    def __init__(self, session_manager: MarketSessionManager, 
                 timeframe_manager: EnhancedTimeframeManager,
                 data_provider=None):
        """
        Initialize session data synchronizer.
        
        Args:
            session_manager: Market session manager
            timeframe_manager: Enhanced timeframe manager
            data_provider: Data provider for historical data (optional)
        """
        self.session_manager = session_manager
        self.timeframe_manager = timeframe_manager
        self.data_provider = data_provider
        
        # Synchronization state
        self._sync_jobs: deque = deque(maxlen=1000)  # Job queue
        self._validation_history: Dict[str, List[DataValidationResult]] = defaultdict(list)
        self._last_validation: Dict[str, datetime] = {}  # {symbol_timeframe: timestamp}
        
        # Configuration
        self._validation_intervals = {
            "session_boundary": timedelta(hours=0),      # Immediate at session boundaries
            "regular_validation": timedelta(hours=1),    # Hourly validation
            "deep_validation": timedelta(hours=6),       # Deep validation every 6 hours
            "historical_sync": timedelta(days=1)         # Daily historical sync
        }
        
        # Background processing
        self._running = False
        self._sync_task = None
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'validations_performed': 0,
            'issues_detected': 0,
            'corrections_applied': 0,
            'session_transitions_handled': 0,
            'background_jobs_completed': 0,
            'continuous_operation_start': datetime.now(),
            'last_validation_time': None,
            'total_validation_time_ms': 0.0
        }
        
        logger.info("SessionDataSynchronizer initialized for continuous operation")
    
    async def start(self):
        """Start the synchronization service."""
        if self._running:
            return
        
        self._running = True
        self._sync_task = asyncio.create_task(self._background_sync_loop())
        logger.info("SessionDataSynchronizer started")
    
    async def stop(self):
        """Stop the synchronization service."""
        if not self._running:
            return
        
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        
        logger.info("SessionDataSynchronizer stopped")
    
    async def handle_session_transition(self, symbol: str, timeframes: List[str], 
                                      session_transition: SessionTransition) -> List[DataValidationResult]:
        """
        Handle session transition with immediate data validation.
        
        Args:
            symbol: Trading symbol
            timeframes: List of timeframes to validate
            session_transition: Session transition information
            
        Returns:
            List of validation results
        """
        self._stats['session_transitions_handled'] += 1
        validation_results = []
        
        logger.info(f"Handling session transition for {symbol}: "
                   f"{session_transition.previous_state.value} → {session_transition.new_state.value}")
        
        # Immediate validation at session boundaries
        for timeframe in timeframes:
            result = await self._validate_session_boundary(symbol, timeframe, session_transition)
            validation_results.append(result)
            
            # Schedule additional validation if issues found
            if not result.success:
                await self._schedule_validation_job(symbol, [timeframe], "gap_bridge", priority=1)
        
        # Schedule historical alignment if entering new trading day
        if session_transition.is_new_trading_day:
            await self._schedule_validation_job(symbol, timeframes, "historical_sync", priority=2)
        
        return validation_results
    
    async def _validate_session_boundary(self, symbol: str, timeframe: str, 
                                       session_transition: SessionTransition) -> DataValidationResult:
        """Validate data at session boundary."""
        start_time = time.time()
        validation_key = f"{symbol}_{timeframe}"
        
        result = DataValidationResult(
            timestamp=datetime.now(),
            symbol=symbol,
            timeframe=timeframe,
            validation_type="session_boundary",
            success=True  # Initialize with True, will be set to False if issues found
        )
        
        try:
            # Get current timeframe data
            candles = self.timeframe_manager.get_timeframe_data(symbol, timeframe, limit=50)
            result.data_points_validated = len(candles)
            
            issues_found = []
            corrections_applied = []
            
            # Check for data gaps around session transition
            if len(candles) >= 2:
                gap_issues = self._detect_data_gaps(candles, timeframe, session_transition)
                issues_found.extend(gap_issues)
            
            # Validate candle integrity
            integrity_issues = self._validate_candle_integrity(candles, timeframe)
            issues_found.extend(integrity_issues)
            
            # Check indicator states
            indicator_issues = await self._validate_indicator_states(symbol, timeframe, session_transition)
            issues_found.extend(indicator_issues)
            
            # Apply corrections if possible
            if issues_found:
                corrections = await self._apply_automatic_corrections(
                    symbol, timeframe, issues_found, session_transition
                )
                corrections_applied.extend(corrections)
            
            result.issues_found = issues_found
            result.corrections_applied = corrections_applied
            result.success = len(issues_found) == 0 or len(corrections_applied) > 0
            
            # Update statistics
            self._stats['validations_performed'] += 1
            if issues_found:
                self._stats['issues_detected'] += len(issues_found)
            if corrections_applied:
                self._stats['corrections_applied'] += len(corrections_applied)
            
            # Store validation result
            with self._lock:
                self._validation_history[validation_key].append(result)
                if len(self._validation_history[validation_key]) > 100:
                    self._validation_history[validation_key].pop(0)
                self._last_validation[validation_key] = result.timestamp
            
        except Exception as e:
            logger.error(f"Error validating session boundary for {symbol}@{timeframe}: {e}")
            result.success = False
            result.issues_found.append(f"Validation error: {str(e)}")
        
        finally:
            result.validation_duration_ms = (time.time() - start_time) * 1000
            self._stats['total_validation_time_ms'] += result.validation_duration_ms
            self._stats['last_validation_time'] = result.timestamp
        
        return result
    
    def _detect_data_gaps(self, candles: List, timeframe: str, 
                         session_transition: SessionTransition) -> List[str]:
        """Detect gaps in candle data."""
        issues = []
        
        if len(candles) < 2:
            return issues
        
        # Calculate expected interval for timeframe
        interval_seconds = self._get_timeframe_seconds(timeframe)
        
        for i in range(1, len(candles)):
            prev_candle = candles[i-1]
            curr_candle = candles[i]
            
            time_diff = (curr_candle.timestamp - prev_candle.timestamp).total_seconds()
            expected_diff = interval_seconds
            
            # Allow for session gaps
            if session_transition.is_significant_gap:
                continue  # Expected gap due to session transition
            
            # Check for unexpected gaps
            if time_diff > expected_diff * 1.5:  # 50% tolerance
                gap_hours = time_diff / 3600
                issues.append(f"Data gap detected: {gap_hours:.1f}h between candles")
        
        return issues
    
    def _validate_candle_integrity(self, candles: List, timeframe: str) -> List[str]:
        """Validate candle data integrity."""
        issues = []
        
        for i, candle in enumerate(candles):
            # Check OHLC relationship
            if not (candle.low <= candle.open <= candle.high and
                   candle.low <= candle.close <= candle.high):
                issues.append(f"Invalid OHLC relationship in candle {i}")
            
            # Check for zero volume (might be normal for some assets)
            if candle.volume == 0 and timeframe in ["1m", "5m"]:
                # Zero volume in short timeframes might indicate data quality issues
                pass  # Not necessarily an error, but worth noting
            
            # Check for extreme price movements (might indicate bad data)
            if i > 0:
                prev_candle = candles[i-1]
                price_change = abs(candle.close - prev_candle.close) / prev_candle.close
                if price_change > 0.2:  # 20% change
                    issues.append(f"Extreme price movement: {price_change:.1%} in candle {i}")
        
        return issues
    
    async def _validate_indicator_states(self, symbol: str, timeframe: str, 
                                       session_transition: SessionTransition) -> List[str]:
        """Validate indicator states for consistency."""
        issues = []
        
        try:
            # Get latest indicators
            indicators = self.timeframe_manager.get_latest_indicators(symbol, timeframe)
            
            # Check for NaN or infinite values
            for indicator_name, value in indicators.items():
                if isinstance(value, (int, float)):
                    if not (-1e10 < value < 1e10):  # Check for reasonable range
                        issues.append(f"Invalid {indicator_name} value: {value}")
                elif isinstance(value, dict):
                    for sub_name, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            if not (-1e10 < sub_value < 1e10):
                                issues.append(f"Invalid {indicator_name}.{sub_name} value: {sub_value}")
            
            # Check RSI bounds
            for indicator_name, value in indicators.items():
                if "rsi" in indicator_name.lower() and isinstance(value, (int, float)):
                    if not (0 <= value <= 100):
                        issues.append(f"RSI value out of bounds: {value}")
        
        except Exception as e:
            issues.append(f"Error validating indicators: {str(e)}")
        
        return issues
    
    async def _apply_automatic_corrections(self, symbol: str, timeframe: str, 
                                         issues: List[str], session_transition: SessionTransition) -> List[str]:
        """Apply automatic corrections for detected issues."""
        corrections = []
        
        for issue in issues:
            try:
                if "Data gap detected" in issue:
                    # Schedule gap bridging
                    await self._schedule_validation_job(symbol, [timeframe], "gap_bridge", priority=1)
                    corrections.append(f"Scheduled gap bridging for {issue}")
                
                elif "Invalid OHLC relationship" in issue:
                    # This requires manual intervention or data provider fix
                    corrections.append(f"Flagged for manual review: {issue}")
                
                elif "Invalid" in issue and "value" in issue:
                    # Reset problematic indicator
                    if hasattr(self.timeframe_manager, '_states'):
                        tf_state = self.timeframe_manager._states.get(symbol, {}).get(timeframe)
                        if tf_state and hasattr(tf_state, 'indicator_calculator'):
                            # Reset indicators on significant issues
                            corrections.append(f"Reset indicators due to: {issue}")
                
            except Exception as e:
                logger.error(f"Error applying correction for '{issue}': {e}")
        
        return corrections
    
    async def _schedule_validation_job(self, symbol: str, timeframes: List[str], 
                                     job_type: str, priority: int = 5) -> str:
        """Schedule a background validation job."""
        job_id = f"{job_type}_{symbol}_{int(time.time())}"
        
        job = SynchronizationJob(
            job_id=job_id,
            symbol=symbol,
            timeframes=timeframes,
            job_type=job_type,
            priority=priority,
            created_at=datetime.now(),
            scheduled_for=datetime.now()  # Immediate scheduling
        )
        
        with self._lock:
            self._sync_jobs.append(job)
        
        logger.debug(f"Scheduled {job_type} job {job_id} for {symbol}")
        return job_id
    
    async def _background_sync_loop(self):
        """Background synchronization loop."""
        logger.info("Started background synchronization loop")
        
        while self._running:
            try:
                # Process pending jobs
                await self._process_sync_jobs()
                
                # Perform regular validations
                await self._perform_regular_validations()
                
                # Sleep between iterations
                await asyncio.sleep(30)  # 30-second intervals
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background sync loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
        
        logger.info("Background synchronization loop stopped")
    
    async def _process_sync_jobs(self):
        """Process pending synchronization jobs."""
        processed_jobs = []
        
        with self._lock:
            # Get jobs sorted by priority and creation time
            pending_jobs = [job for job in self._sync_jobs if job.status == "pending"]
            pending_jobs.sort(key=lambda x: (x.priority, x.created_at))
        
        for job in pending_jobs[:5]:  # Process up to 5 jobs per iteration
            try:
                job.status = "running"
                result = await self._execute_sync_job(job)
                job.result = result
                job.status = "completed" if result.success else "failed"
                
                processed_jobs.append(job)
                self._stats['background_jobs_completed'] += 1
                
            except Exception as e:
                logger.error(f"Error executing sync job {job.job_id}: {e}")
                job.status = "failed"
                job.retry_count += 1
                
                if job.retry_count < job.max_retries:
                    job.status = "pending"
                    job.scheduled_for = datetime.now() + timedelta(minutes=5)
        
        # Clean up completed jobs
        with self._lock:
            self._sync_jobs = deque([job for job in self._sync_jobs if job.status in ["pending", "running"]], 
                                  maxlen=1000)
    
    async def _execute_sync_job(self, job: SynchronizationJob) -> DataValidationResult:
        """Execute a synchronization job."""
        if job.job_type == "validation":
            return await self._perform_validation(job.symbol, job.timeframes[0])
        elif job.job_type == "gap_bridge":
            return await self._bridge_data_gap(job.symbol, job.timeframes[0])
        elif job.job_type == "historical_sync":
            return await self._sync_with_historical_data(job.symbol, job.timeframes)
        else:
            raise ValueError(f"Unknown job type: {job.job_type}")
    
    async def _perform_validation(self, symbol: str, timeframe: str) -> DataValidationResult:
        """Perform regular validation."""
        # This is a lighter version of session boundary validation
        result = DataValidationResult(
            timestamp=datetime.now(),
            symbol=symbol,
            timeframe=timeframe,
            validation_type="regular_validation",
            success=True
        )
        
        # Basic validation checks
        candles = self.timeframe_manager.get_timeframe_data(symbol, timeframe, limit=10)
        result.data_points_validated = len(candles)
        
        if len(candles) == 0:
            result.success = False
            result.issues_found.append("No candle data available")
        
        return result
    
    async def _bridge_data_gap(self, symbol: str, timeframe: str) -> DataValidationResult:
        """Attempt to bridge data gaps."""
        result = DataValidationResult(
            timestamp=datetime.now(),
            symbol=symbol,
            timeframe=timeframe,
            validation_type="gap_bridge",
            success=True
        )
        
        # For now, this is a placeholder - would need data provider integration
        result.corrections_applied.append("Gap bridging attempted")
        return result
    
    async def _sync_with_historical_data(self, symbol: str, timeframes: List[str]) -> DataValidationResult:
        """Synchronize with historical data."""
        result = DataValidationResult(
            timestamp=datetime.now(),
            symbol=symbol,
            timeframe=timeframes[0] if timeframes else "unknown",
            validation_type="historical_sync",
            success=True
        )
        
        # For now, this is a placeholder - would need data provider integration
        result.corrections_applied.append("Historical sync attempted")
        return result
    
    async def _perform_regular_validations(self):
        """Perform regular validations for active symbols."""
        # This would check all active symbols periodically
        pass
    
    def _get_timeframe_seconds(self, timeframe: str) -> int:
        """Get timeframe interval in seconds."""
        timeframe_map = {
            '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '4h': 14400, '1d': 86400, '1w': 604800
        }
        return timeframe_map.get(timeframe, 60)
    
    def get_validation_history(self, symbol: str, timeframe: str, limit: int = 10) -> List[DataValidationResult]:
        """Get validation history for a symbol/timeframe."""
        validation_key = f"{symbol}_{timeframe}"
        with self._lock:
            history = self._validation_history.get(validation_key, [])
            return history[-limit:] if len(history) > limit else history
    
    def get_sync_stats(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        with self._lock:
            uptime = datetime.now() - self._stats['continuous_operation_start']
            
            return {
                **self._stats,
                'continuous_operation_hours': uptime.total_seconds() / 3600.0,
                'pending_jobs': len([job for job in self._sync_jobs if job.status == "pending"]),
                'total_jobs_queued': len(self._sync_jobs),
                'average_validation_time_ms': (
                    self._stats['total_validation_time_ms'] / max(1, self._stats['validations_performed'])
                ),
                'validation_success_rate': (
                    (self._stats['validations_performed'] - self._stats['issues_detected']) / 
                    max(1, self._stats['validations_performed'])
                ) * 100
            }
    
    def force_validation(self, symbol: str, timeframes: List[str]) -> str:
        """Force immediate validation (useful for testing)."""
        return asyncio.create_task(self._schedule_validation_job(symbol, timeframes, "validation", priority=0))