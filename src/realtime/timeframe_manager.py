"""
TimeframeManager - Multi-timeframe data aggregation and indicator management

This module provides efficient timeframe aggregation and manages separate indicator
states for each timeframe. It handles data aggregation from base timeframes to
higher timeframes and maintains sliding windows for memory efficiency.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import time
import math

from .config_manager import TradingProfile, TimeframeConfig
from .incremental_indicators import IncrementalIndicatorCalculator, IndicatorState

logger = logging.getLogger(__name__)


@dataclass
class TimeframeData:
    """Data structure for a specific timeframe candle."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    
    def update_with_tick(self, price: float, volume: float = 0.0):
        """Update candle with new tick data."""
        if self.high is None or price > self.high:
            self.high = price
        if self.low is None or price < self.low:
            self.low = price
        self.close = price
        self.volume += volume


@dataclass
class TimeframeState:
    """State for a specific timeframe."""
    timeframe: str
    base_interval_seconds: int
    update_interval_seconds: int
    lookback_periods: int
    priority: int
    
    # Data storage
    candles: deque = field(default_factory=deque)
    current_candle: Optional[TimeframeData] = None
    last_update: Optional[datetime] = None
    
    # Indicator calculators (separate for each timeframe)
    indicator_calculator: IncrementalIndicatorCalculator = field(default_factory=IncrementalIndicatorCalculator)
    
    # Performance tracking
    updates_count: int = 0
    last_calculation_time: float = 0.0
    
    def __post_init__(self):
        """Initialize collections with maxlen for memory efficiency."""
        if not isinstance(self.candles, deque):
            self.candles = deque(maxlen=self.lookback_periods)
    
    def needs_update(self) -> bool:
        """Check if timeframe needs update based on interval."""
        if self.last_update is None:
            return True
        
        time_diff = (datetime.now() - self.last_update).total_seconds()
        return time_diff >= self.update_interval_seconds
    
    def should_create_new_candle(self, timestamp: datetime) -> bool:
        """Check if we should create a new candle for this timestamp."""
        if self.current_candle is None:
            return True
        
        # Calculate the start of the current interval
        seconds_since_epoch = int(timestamp.timestamp())
        interval_start = (seconds_since_epoch // self.base_interval_seconds) * self.base_interval_seconds
        current_interval_start = (int(self.current_candle.timestamp.timestamp()) // self.base_interval_seconds) * self.base_interval_seconds
        
        return interval_start != current_interval_start
    
    def finalize_current_candle(self):
        """Finalize the current candle and add to history."""
        if self.current_candle is not None:
            self.candles.append(self.current_candle)
            self.current_candle = None


class TimeframeManager:
    """
    Manages multiple timeframes for symbols and handles data aggregation.
    
    This class efficiently aggregates tick data into different timeframes,
    maintains separate indicator states for each timeframe, and provides
    coordinated signal generation across timeframes.
    """
    
    def __init__(self, config_manager):
        """
        Initialize the TimeframeManager.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self._states: Dict[str, Dict[str, TimeframeState]] = {}  # {symbol: {timeframe: state}}
        self._timeframe_configs: Dict[str, int] = {}  # {timeframe: interval_seconds}
        self._lock = threading.RLock()
        
        # Performance tracking
        self._stats = {
            'total_updates': 0,
            'timeframe_updates': defaultdict(int),
            'symbols_processed': set(),
            'last_update_time': None,
            'average_update_time_ms': 0.0,
            'errors': 0
        }
        
        # Initialize timeframe configurations
        self._initialize_timeframe_configs()
        
        logger.info("TimeframeManager initialized")
    
    def _initialize_timeframe_configs(self):
        """Initialize timeframe configurations with their interval mappings."""
        self._timeframe_configs = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400,
            '1w': 604800
        }
    
    def _get_timeframe_interval(self, timeframe: str) -> int:
        """Get interval in seconds for a timeframe."""
        return self._timeframe_configs.get(timeframe, 60)
    
    def _ensure_symbol_initialized(self, symbol: str, timeframes: List[str], trading_profile: TradingProfile):
        """Ensure symbol is initialized with required timeframes."""
        with self._lock:
            if symbol not in self._states:
                self._states[symbol] = {}
            
            # Get timeframe configs for the trading profile
            timeframe_configs = self.config_manager.get_timeframe_configs()
            profile_configs = timeframe_configs.get(trading_profile, [])
            
            # Initialize each required timeframe
            for timeframe in timeframes:
                if timeframe not in self._states[symbol]:
                    # Find matching config
                    tf_config = next((c for c in profile_configs if c.timeframe == timeframe), None)
                    if tf_config is None:
                        # Create default config
                        tf_config = TimeframeConfig(
                            timeframe=timeframe,
                            update_interval_seconds=self._get_timeframe_interval(timeframe) // 10,
                            lookback_periods=200,
                            priority=1
                        )
                    
                    self._states[symbol][timeframe] = TimeframeState(
                        timeframe=timeframe,
                        base_interval_seconds=self._get_timeframe_interval(timeframe),
                        update_interval_seconds=tf_config.update_interval_seconds,
                        lookback_periods=tf_config.lookback_periods,
                        priority=tf_config.priority
                    )
                    
                    logger.debug(f"Initialized timeframe {timeframe} for {symbol}")
    
    def process_tick(self, symbol: str, price: float, volume: float, timestamp: datetime,
                    timeframes: List[str], trading_profile: TradingProfile) -> Dict[str, Any]:
        """
        Process a new tick for all timeframes of a symbol.
        
        Args:
            symbol: Trading symbol
            price: Current price
            volume: Volume (optional)
            timestamp: Tick timestamp
            timeframes: List of timeframes to process
            trading_profile: Trading profile for configuration
            
        Returns:
            Dictionary with updated indicators for each timeframe
        """
        start_time = time.time()
        results = {}
        
        try:
            with self._lock:
                # Ensure symbol is initialized
                self._ensure_symbol_initialized(symbol, timeframes, trading_profile)
                
                # Process each timeframe
                for timeframe in timeframes:
                    tf_state = self._states[symbol][timeframe]
                    
                    # Check if we need to create a new candle
                    if tf_state.should_create_new_candle(timestamp):
                        # Finalize current candle
                        tf_state.finalize_current_candle()
                        
                        # Create new candle
                        tf_state.current_candle = TimeframeData(
                            timestamp=timestamp,
                            open=price,
                            high=price,
                            low=price,
                            close=price,
                            volume=volume
                        )
                    else:
                        # Update current candle
                        if tf_state.current_candle:
                            tf_state.current_candle.update_with_tick(price, volume)
                    
                    # Update indicators if needed
                    if tf_state.needs_update():
                        timeframe_results = self._update_timeframe_indicators(
                            symbol, timeframe, price, tf_state
                        )
                        results[timeframe] = timeframe_results
                        tf_state.last_update = datetime.now()
                        tf_state.updates_count += 1
                        
                        # Update stats
                        self._stats['timeframe_updates'][timeframe] += 1
                
                # Update global stats
                self._stats['total_updates'] += 1
                self._stats['symbols_processed'].add(symbol)
                self._stats['last_update_time'] = datetime.now()
                
                # Update average processing time
                processing_time = (time.time() - start_time) * 1000  # Convert to ms
                if self._stats['average_update_time_ms'] == 0:
                    self._stats['average_update_time_ms'] = processing_time
                else:
                    alpha = 0.1  # Exponential smoothing
                    self._stats['average_update_time_ms'] = (
                        alpha * processing_time + 
                        (1 - alpha) * self._stats['average_update_time_ms']
                    )
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing tick for {symbol}: {e}")
            self._stats['errors'] += 1
            return {}
    
    def _update_timeframe_indicators(self, symbol: str, timeframe: str, price: float, 
                                   tf_state: TimeframeState) -> Dict[str, Any]:
        """
        Update indicators for a specific timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            price: Current price
            tf_state: Timeframe state
            
        Returns:
            Dictionary with updated indicator values
        """
        try:
            calculation_start = time.time()
            results = {}
            
            # Get enabled indicators for this timeframe
            enabled_indicators = self.config_manager.get_enabled_indicators()
            
            # Calculate indicators using the timeframe's calculator
            for indicator_config in enabled_indicators:
                try:
                    indicator_key = f"{symbol}_{timeframe}_{indicator_config.name}"
                    
                    if indicator_config.name.startswith("sma"):
                        period = indicator_config.parameters.get("period", 50)
                        result = tf_state.indicator_calculator.calculate_sma_incremental(
                            indicator_key, price, period
                        )
                        if result is not None:
                            results[indicator_config.name] = result
                    
                    elif indicator_config.name.startswith("ema"):
                        period = indicator_config.parameters.get("period", 50)
                        result = tf_state.indicator_calculator.calculate_ema_incremental(
                            indicator_key, price, period
                        )
                        if result is not None:
                            results[indicator_config.name] = result
                    
                    elif indicator_config.name.startswith("rsi"):
                        period = indicator_config.parameters.get("period", 14)
                        result = tf_state.indicator_calculator.calculate_rsi_incremental(
                            indicator_key, price, period
                        )
                        if result is not None:
                            results[indicator_config.name] = result
                    
                    elif indicator_config.name == "macd":
                        fast = indicator_config.parameters.get("fast", 12)
                        slow = indicator_config.parameters.get("slow", 26)
                        signal = indicator_config.parameters.get("signal", 9)
                        result = tf_state.indicator_calculator.calculate_macd_incremental(
                            indicator_key, price, fast, slow, signal
                        )
                        if result:
                            results[indicator_config.name] = {
                                "macd_line": result[0],
                                "signal_line": result[1],
                                "histogram": result[2]
                            }
                    
                    elif indicator_config.name == "bollinger_bands":
                        period = indicator_config.parameters.get("period", 20)
                        std_dev = indicator_config.parameters.get("std_dev", 2.0)
                        result = tf_state.indicator_calculator.calculate_bollinger_bands_incremental(
                            indicator_key, price, period, std_dev
                        )
                        if result:
                            results[indicator_config.name] = {
                                "upper_band": result[0],
                                "middle_band": result[1],
                                "lower_band": result[2]
                            }
                
                except Exception as e:
                    logger.error(f"Error calculating {indicator_config.name} for {symbol}@{timeframe}: {e}")
            
            # Update performance tracking
            tf_state.last_calculation_time = (time.time() - calculation_start) * 1000
            
            return results
            
        except Exception as e:
            logger.error(f"Error updating indicators for {symbol}@{timeframe}: {e}")
            return {}
    
    def get_timeframe_data(self, symbol: str, timeframe: str, limit: int = 100) -> List[TimeframeData]:
        """
        Get historical candle data for a timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            limit: Maximum number of candles to return
            
        Returns:
            List of TimeframeData objects
        """
        try:
            with self._lock:
                if symbol not in self._states or timeframe not in self._states[symbol]:
                    return []
                
                tf_state = self._states[symbol][timeframe]
                candles = list(tf_state.candles)
                
                # Include current candle if it exists
                if tf_state.current_candle:
                    candles.append(tf_state.current_candle)
                
                # Return latest candles up to limit
                return candles[-limit:] if len(candles) > limit else candles
                
        except Exception as e:
            logger.error(f"Error getting timeframe data for {symbol}@{timeframe}: {e}")
            return []
    
    def get_latest_indicators(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Get latest indicator values for a symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            
        Returns:
            Dictionary with latest indicator values
        """
        try:
            with self._lock:
                if symbol not in self._states or timeframe not in self._states[symbol]:
                    return {}
                
                tf_state = self._states[symbol][timeframe]
                return tf_state.indicator_calculator.get_all_indicators(f"{symbol}_{timeframe}")
                
        except Exception as e:
            logger.error(f"Error getting latest indicators for {symbol}@{timeframe}: {e}")
            return {}
    
    def get_cross_timeframe_correlation(self, symbol: str, timeframes: List[str]) -> Dict[str, Any]:
        """
        Get correlation analysis across multiple timeframes.
        
        Args:
            symbol: Trading symbol
            timeframes: List of timeframes to analyze
            
        Returns:
            Dictionary with correlation analysis
        """
        try:
            correlation_data = {}
            
            # Get latest indicators for each timeframe
            for timeframe in timeframes:
                indicators = self.get_latest_indicators(symbol, timeframe)
                if indicators:
                    correlation_data[timeframe] = indicators
            
            # Basic correlation analysis
            analysis = {
                'timeframes_analyzed': len(correlation_data),
                'available_indicators': [],
                'trend_alignment': {},
                'signal_strength': {}
            }
            
            # Find common indicators
            if correlation_data:
                first_tf = next(iter(correlation_data.keys()))
                common_indicators = set(correlation_data[first_tf].keys())
                
                for tf_data in correlation_data.values():
                    common_indicators &= set(tf_data.keys())
                
                analysis['available_indicators'] = list(common_indicators)
                
                # Analyze trend alignment for RSI
                if 'rsi_14' in common_indicators:
                    rsi_values = {tf: data.get('rsi_14', 50) for tf, data in correlation_data.items()}
                    analysis['trend_alignment']['rsi'] = {
                        'values': rsi_values,
                        'all_bullish': all(v < 30 for v in rsi_values.values()),
                        'all_bearish': all(v > 70 for v in rsi_values.values()),
                        'conflicting': len(set(v > 50 for v in rsi_values.values())) > 1
                    }
                
                # Analyze MACD alignment
                if 'macd' in common_indicators:
                    macd_bullish = {}
                    for tf, data in correlation_data.items():
                        macd_data = data.get('macd', {})
                        if isinstance(macd_data, dict):
                            macd_bullish[tf] = macd_data.get('macd_line', 0) > macd_data.get('signal_line', 0)
                    
                    analysis['trend_alignment']['macd'] = {
                        'bullish_count': sum(macd_bullish.values()),
                        'bearish_count': len(macd_bullish) - sum(macd_bullish.values()),
                        'alignment_strength': abs(sum(macd_bullish.values()) - len(macd_bullish)/2) / (len(macd_bullish)/2) if macd_bullish else 0
                    }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting cross-timeframe correlation for {symbol}: {e}")
            return {}
    
    def cleanup_old_data(self, max_age_hours: int = 24):
        """
        Clean up old data to manage memory usage.
        
        Args:
            max_age_hours: Maximum age of data to keep in hours
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            cleaned_count = 0
            
            with self._lock:
                for symbol in self._states:
                    for timeframe in self._states[symbol]:
                        tf_state = self._states[symbol][timeframe]
                        
                        # Clean old candles
                        while tf_state.candles and tf_state.candles[0].timestamp < cutoff_time:
                            tf_state.candles.popleft()
                            cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(f"Cleaned {cleaned_count} old candles from memory")
                
        except Exception as e:
            logger.error(f"Error cleaning old data: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get timeframe manager statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        with self._lock:
            stats = self._stats.copy()
            stats['symbols_processed'] = list(self._stats['symbols_processed'])
            stats['total_timeframes'] = sum(len(tf_states) for tf_states in self._states.values())
            stats['timeframe_states'] = {
                symbol: list(tf_states.keys()) for symbol, tf_states in self._states.items()
            }
            
            return stats
    
    def reset_symbol_timeframe(self, symbol: str, timeframe: str):
        """
        Reset a specific symbol-timeframe combination.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
        """
        try:
            with self._lock:
                if symbol in self._states and timeframe in self._states[symbol]:
                    del self._states[symbol][timeframe]
                    logger.info(f"Reset timeframe {timeframe} for {symbol}")
                    
        except Exception as e:
            logger.error(f"Error resetting {symbol}@{timeframe}: {e}")
    
    def reset_symbol(self, symbol: str):
        """
        Reset all timeframes for a symbol.
        
        Args:
            symbol: Trading symbol
        """
        try:
            with self._lock:
                if symbol in self._states:
                    del self._states[symbol]
                    logger.info(f"Reset all timeframes for {symbol}")
                    
        except Exception as e:
            logger.error(f"Error resetting symbol {symbol}: {e}")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory usage information
        """
        try:
            with self._lock:
                usage = {
                    'total_symbols': len(self._states),
                    'total_timeframes': 0,
                    'total_candles': 0,
                    'symbol_breakdown': {}
                }
                
                for symbol, tf_states in self._states.items():
                    symbol_data = {
                        'timeframes': len(tf_states),
                        'candles': 0,
                        'memory_estimate_mb': 0
                    }
                    
                    for timeframe, tf_state in tf_states.items():
                        candle_count = len(tf_state.candles)
                        symbol_data['candles'] += candle_count
                        # Rough estimate: 50 bytes per candle + indicator states
                        symbol_data['memory_estimate_mb'] += candle_count * 50 / (1024 * 1024)
                    
                    usage['symbol_breakdown'][symbol] = symbol_data
                    usage['total_timeframes'] += symbol_data['timeframes']
                    usage['total_candles'] += symbol_data['candles']
                
                return usage
                
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {}