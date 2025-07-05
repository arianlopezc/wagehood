"""
Indicator Calculator

This module provides the main IndicatorCalculator class that serves as a unified interface
for calculating technical indicators with caching and batch processing capabilities.
"""

import numpy as np
import logging
from typing import Union, Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from functools import lru_cache
from ..core.constants import (
    RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD_DEV,
    MA_FAST, MA_SLOW,
    CACHE_TTL_SECONDS
)
from ..storage.cache import cache_manager, cached

logger = logging.getLogger(__name__)
from .moving_averages import (
    calculate_sma, calculate_ema, calculate_wma, calculate_vwma,
    calculate_ma_crossover, calculate_ma_envelope
)
from .momentum import (
    calculate_rsi, calculate_macd, calculate_stochastic,
    calculate_williams_r, calculate_cci, calculate_momentum, calculate_roc
)
from .volatility import (
    calculate_bollinger_bands, calculate_atr, calculate_keltner_channels,
    calculate_donchian_channels, calculate_volatility, calculate_bb_width,
    calculate_bb_percent, calculate_price_channels
)
from .levels import (
    calculate_support_resistance, calculate_pivot_points,
    calculate_fibonacci_retracements, calculate_fibonacci_extensions,
    detect_breakouts, calculate_trend_lines
)


class IndicatorCalculator:
    """
    Main calculator class for technical indicators with caching and batch processing.
    
    This class provides a unified interface for calculating various technical indicators
    with performance optimizations including caching and vectorized operations.
    """
    
    def __init__(self, enable_cache: bool = True, cache_ttl: int = CACHE_TTL_SECONDS):
        """
        Initialize the IndicatorCalculator.
        
        Args:
            enable_cache: Whether to enable caching (default True)
            cache_ttl: Cache time-to-live in seconds (default from constants)
        """
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        
        # Cache TTL settings for different indicator types
        self._cache_ttl_settings = {
            'moving_averages': 3600,  # 1 hour for moving averages
            'momentum': 1800,         # 30 minutes for momentum indicators
            'volatility': 1800,       # 30 minutes for volatility indicators
            'levels': 7200,           # 2 hours for support/resistance levels
            'signals': 300,           # 5 minutes for trading signals
        }
        
    def _get_cache_key(self, func_name: str, data: Union[np.ndarray, list], 
                       **kwargs) -> str:
        """
        Generate a cache key for the given function and parameters.
        
        Args:
            func_name: Name of the indicator function
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            str: Cache key
        """
        # Convert data to hash for better performance
        if isinstance(data, (list, tuple)):
            data_hash = hash(str(data))
        else:
            data_hash = hash(str(data.tolist() if hasattr(data, 'tolist') else data))
        
        # Create cache key using hash
        return cache_manager.cache_key_hash(func_name, data_hash, **kwargs)
    
    def _get_from_cache(self, cache_key: str) -> Any:
        """
        Get a value from cache if it exists and is valid.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached value or None if not found/invalid
        """
        if not self.enable_cache:
            return None
        
        try:
            return cache_manager.get("indicators", cache_key)
        except Exception as e:
            logger.warning(f"Failed to get from cache: {e}")
            return None
    
    def _set_cache(self, cache_key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in cache.
        
        Args:
            cache_key: Cache key
            value: Value to cache
            ttl: Time to live for cache entry
        """
        if not self.enable_cache:
            return
        
        try:
            cache_manager.set("indicators", cache_key, value, ttl or self.cache_ttl)
        except Exception as e:
            logger.warning(f"Failed to set cache: {e}")
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        if not self.enable_cache:
            return
        
        try:
            cache_manager.clear_namespace("indicators")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
    
    def calculate_sma(self, data: Union[np.ndarray, list], period: int = MA_FAST) -> np.ndarray:
        """
        Calculate Simple Moving Average with caching.
        
        Args:
            data: Price data
            period: Period for the moving average
            
        Returns:
            np.ndarray: SMA values
        """
        cache_key = self._get_cache_key('sma', data, period=period)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            result = calculate_sma(data, period)
            self._set_cache(cache_key, result, self._cache_ttl_settings['moving_averages'])
            return result
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            # Fallback to direct computation without caching
            return calculate_sma(data, period)
    
    def calculate_ema(self, data: Union[np.ndarray, list], period: int = MA_FAST) -> np.ndarray:
        """
        Calculate Exponential Moving Average with caching.
        
        Args:
            data: Price data
            period: Period for the moving average
            
        Returns:
            np.ndarray: EMA values
        """
        cache_key = self._get_cache_key('ema', data, period=period)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            result = calculate_ema(data, period)
            self._set_cache(cache_key, result, self._cache_ttl_settings['moving_averages'])
            return result
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            # Fallback to direct computation without caching
            return calculate_ema(data, period)
    
    def calculate_rsi(self, data: Union[np.ndarray, list], period: int = RSI_PERIOD) -> np.ndarray:
        """
        Calculate Relative Strength Index with caching.
        
        Args:
            data: Price data
            period: Period for RSI calculation
            
        Returns:
            np.ndarray: RSI values
        """
        cache_key = self._get_cache_key('rsi', data, period=period)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            result = calculate_rsi(data, period)
            self._set_cache(cache_key, result, self._cache_ttl_settings['momentum'])
            return result
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            # Fallback to direct computation without caching
            return calculate_rsi(data, period)
    
    def calculate_macd(self, data: Union[np.ndarray, list], fast: int = MACD_FAST, 
                      slow: int = MACD_SLOW, signal: int = MACD_SIGNAL) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate MACD with caching.
        
        Args:
            data: Price data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            tuple: (macd_line, signal_line, histogram)
        """
        cache_key = self._get_cache_key('macd', data, fast=fast, slow=slow, signal=signal)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            result = calculate_macd(data, fast, slow, signal)
            self._set_cache(cache_key, result, self._cache_ttl_settings['momentum'])
            return result
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            # Fallback to direct computation without caching
            return calculate_macd(data, fast, slow, signal)
    
    def calculate_bollinger_bands(self, data: Union[np.ndarray, list], period: int = BB_PERIOD, 
                                 std_dev: float = BB_STD_DEV) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Bollinger Bands with caching.
        
        Args:
            data: Price data
            period: Period for moving average and standard deviation
            std_dev: Standard deviation multiplier
            
        Returns:
            tuple: (upper_band, middle_band, lower_band)
        """
        cache_key = self._get_cache_key('bollinger_bands', data, period=period, std_dev=std_dev)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            result = calculate_bollinger_bands(data, period, std_dev)
            self._set_cache(cache_key, result, self._cache_ttl_settings['volatility'])
            return result
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            # Fallback to direct computation without caching
            return calculate_bollinger_bands(data, period, std_dev)
    
    def calculate_support_resistance(self, data: Union[np.ndarray, list], lookback: int = 20, 
                                   min_touches: int = 3) -> Dict[str, List[float]]:
        """
        Calculate Support and Resistance levels with caching.
        
        Args:
            data: Price data
            lookback: Period for finding local extrema
            min_touches: Minimum number of touches required for a level
            
        Returns:
            dict: Dictionary containing 'support' and 'resistance' lists
        """
        cache_key = self._get_cache_key('support_resistance', data, 
                                       lookback=lookback, min_touches=min_touches)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            result = calculate_support_resistance(data, lookback, min_touches)
            self._set_cache(cache_key, result, self._cache_ttl_settings['levels'])
            return result
        except Exception as e:
            logger.error(f"Error calculating Support/Resistance: {e}")
            # Fallback to direct computation without caching
            return calculate_support_resistance(data, lookback, min_touches)
    
    def calculate_multiple_indicators(self, data: Union[np.ndarray, list], 
                                    indicators: List[str], **kwargs) -> Dict[str, Any]:
        """
        Calculate multiple indicators in a single batch operation.
        
        Args:
            data: Price data
            indicators: List of indicator names to calculate
            **kwargs: Additional parameters for specific indicators
            
        Returns:
            dict: Dictionary containing all calculated indicators
            
        Raises:
            ValueError: If an unsupported indicator is requested
        """
        supported_indicators = {
            'sma': self.calculate_sma,
            'ema': self.calculate_ema,
            'rsi': self.calculate_rsi,
            'macd': self.calculate_macd,
            'bollinger_bands': self.calculate_bollinger_bands,
            'support_resistance': self.calculate_support_resistance,
            'atr': self._calculate_atr_batch,
            'stochastic': self._calculate_stochastic_batch,
            'williams_r': self._calculate_williams_r_batch,
            'cci': self._calculate_cci_batch
        }
        
        results = {}
        
        for indicator in indicators:
            if indicator not in supported_indicators:
                raise ValueError(f"Unsupported indicator: {indicator}")
            
            # Get indicator-specific parameters
            indicator_params = kwargs.get(f'{indicator}_params', {})
            
            try:
                if indicator in ['atr', 'stochastic', 'williams_r', 'cci']:
                    # These indicators require OHLC data
                    if not isinstance(data, dict) or not all(k in data for k in ['high', 'low', 'close']):
                        raise ValueError(f"Indicator {indicator} requires OHLC data (dict with 'high', 'low', 'close')")
                    results[indicator] = supported_indicators[indicator](data, **indicator_params)
                else:
                    results[indicator] = supported_indicators[indicator](data, **indicator_params)
            except Exception as e:
                # Log error but continue with other indicators
                results[indicator] = {'error': str(e)}
        
        return results
    
    def _calculate_atr_batch(self, data: Dict[str, Union[np.ndarray, list]], 
                            period: int = 14) -> np.ndarray:
        """Calculate ATR for batch processing."""
        return calculate_atr(data['high'], data['low'], data['close'], period)
    
    def _calculate_stochastic_batch(self, data: Dict[str, Union[np.ndarray, list]], 
                                   k_period: int = 14, d_period: int = 3, 
                                   smooth_k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Stochastic for batch processing."""
        return calculate_stochastic(data['high'], data['low'], data['close'], 
                                   k_period, d_period, smooth_k)
    
    def _calculate_williams_r_batch(self, data: Dict[str, Union[np.ndarray, list]], 
                                   period: int = 14) -> np.ndarray:
        """Calculate Williams %R for batch processing."""
        return calculate_williams_r(data['high'], data['low'], data['close'], period)
    
    def _calculate_cci_batch(self, data: Dict[str, Union[np.ndarray, list]], 
                            period: int = 20) -> np.ndarray:
        """Calculate CCI for batch processing."""
        return calculate_cci(data['high'], data['low'], data['close'], period)
    
    def get_indicator_signals(self, data: Union[np.ndarray, list], 
                             indicator_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on indicator values.
        
        Args:
            data: Price data
            indicator_results: Dictionary of calculated indicators
            
        Returns:
            dict: Dictionary containing trading signals
        """
        signals = {}
        
        # RSI signals
        if 'rsi' in indicator_results:
            rsi = indicator_results['rsi']
            signals['rsi_overbought'] = rsi > RSI_OVERBOUGHT
            signals['rsi_oversold'] = rsi < RSI_OVERSOLD
        
        # MACD signals
        if 'macd' in indicator_results:
            macd_line, signal_line, histogram = indicator_results['macd']
            signals['macd_bullish'] = macd_line > signal_line
            signals['macd_bearish'] = macd_line < signal_line
            signals['macd_crossover'] = self._detect_crossover(macd_line, signal_line)
        
        # Bollinger Bands signals
        if 'bollinger_bands' in indicator_results:
            upper_band, middle_band, lower_band = indicator_results['bollinger_bands']
            data_array = np.asarray(data)
            signals['bb_upper_breach'] = data_array > upper_band
            signals['bb_lower_breach'] = data_array < lower_band
            signals['bb_squeeze'] = calculate_bb_width(upper_band, lower_band, middle_band) < 0.1
        
        # Moving Average signals
        if 'sma' in indicator_results and 'ema' in indicator_results:
            sma = indicator_results['sma']
            ema = indicator_results['ema']
            signals['ma_crossover'] = calculate_ma_crossover(ema, sma)
        
        return signals
    
    def _detect_crossover(self, fast_line: np.ndarray, slow_line: np.ndarray) -> np.ndarray:
        """
        Detect crossover points between two lines.
        
        Args:
            fast_line: Fast-moving line
            slow_line: Slow-moving line
            
        Returns:
            np.ndarray: Crossover signals (1 for bullish, -1 for bearish, 0 for none)
        """
        return calculate_ma_crossover(fast_line, slow_line)
    
    def validate_data(self, data: Union[np.ndarray, list, Dict], 
                     min_length: int = 50) -> bool:
        """
        Validate input data for indicator calculations.
        
        Args:
            data: Input data to validate
            min_length: Minimum required data length
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        try:
            if isinstance(data, dict):
                # OHLC data validation
                required_keys = ['high', 'low', 'close']
                if not all(key in data for key in required_keys):
                    return False
                
                # Check all arrays have same length
                lengths = [len(data[key]) for key in required_keys]
                if len(set(lengths)) > 1:
                    return False
                
                # Check minimum length
                if min(lengths) < min_length:
                    return False
                
                # Check for NaN values
                for key in required_keys:
                    array = np.asarray(data[key])
                    if np.isnan(array).any():
                        return False
                
            else:
                # Single array validation
                data_array = np.asarray(data)
                
                if len(data_array) < min_length:
                    return False
                
                if np.isnan(data_array).any():
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            dict: Cache statistics
        """
        if not self.enable_cache:
            return {"caching_enabled": False}
        
        try:
            stats = cache_manager.get_stats()
            stats["caching_enabled"] = True
            stats["cache_ttl_settings"] = self._cache_ttl_settings
            return stats
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            return {"caching_enabled": True, "error": str(e)}
    
    # Cached versions of some methods using the decorator
    @cached("indicators", ttl=3600)  # 1 hour cache for optimization results
    def _cached_optimize_parameters(self, data_hash: str, indicator: str, 
                                   param_ranges: tuple, optimization_metric: str) -> Dict[str, Any]:
        """Cached version of parameter optimization."""
        # This is a placeholder - actual optimization would happen here
        best_params = {}
        for param, (min_val, max_val) in dict(param_ranges).items():
            best_params[param] = (min_val + max_val) // 2
        
        return {
            'best_params': best_params,
            'best_score': 0.0,
            'optimization_metric': optimization_metric
        }
    
    def optimize_parameters(self, data: Union[np.ndarray, list], 
                           indicator: str, param_ranges: Dict[str, Tuple[int, int]], 
                           optimization_metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """
        Optimize indicator parameters using grid search.
        
        Args:
            data: Price data
            indicator: Indicator name
            param_ranges: Dictionary of parameter ranges to test
            optimization_metric: Metric to optimize
            
        Returns:
            dict: Best parameters and performance metrics
            
        Note:
            This is a placeholder implementation. Full optimization would require
            backtesting framework integration.
        """
        if self.enable_cache:
            try:
                # Create hash for data
                data_hash = str(hash(str(data)))
                # Convert param_ranges to tuple for hashing
                param_ranges_tuple = tuple(sorted(param_ranges.items()))
                
                return self._cached_optimize_parameters(
                    data_hash, indicator, param_ranges_tuple, optimization_metric
                )
            except Exception as e:
                logger.warning(f"Failed to use cached optimization: {e}")
        
        # Fallback to direct computation
        best_params = {}
        for param, (min_val, max_val) in param_ranges.items():
            best_params[param] = (min_val + max_val) // 2
        
        return {
            'best_params': best_params,
            'best_score': 0.0,
            'optimization_metric': optimization_metric
        }