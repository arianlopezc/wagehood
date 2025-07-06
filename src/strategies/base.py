"""
Base strategy class for all trading strategies
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple, TYPE_CHECKING
import logging
from datetime import datetime
import itertools
from functools import lru_cache

if TYPE_CHECKING:
    import numpy as np

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from ..core.models import Signal, SignalType, MarketData, OHLCV
from ..indicators import IndicatorCalculator

logger = logging.getLogger(__name__)


class TradingStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        """
        Initialize the trading strategy
        
        Args:
            name: Strategy name
            parameters: Strategy parameters dictionary
        """
        self.name = name
        self.parameters = parameters or {}
        self.indicator_calculator = IndicatorCalculator()
        self._last_signal_time = None
        self._current_position = None
        # Cache for indicator calculations
        self._indicator_cache: Dict[str, Any] = {}
        self._cache_timestamp: Optional[datetime] = None
        
    @abstractmethod
    def generate_signals(self, data: MarketData, indicators: Dict[str, Any]) -> List[Signal]:
        """
        Generate trading signals based on market data and indicators
        
        Args:
            data: Market data including OHLCV
            indicators: Pre-calculated indicators
            
        Returns:
            List of trading signals
        """
        pass
    
    @abstractmethod
    def get_required_indicators(self) -> List[str]:
        """
        Get list of required indicators for this strategy
        
        Returns:
            List of indicator names
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get strategy parameters with descriptions and defaults
        
        Returns:
            Dictionary of parameters with metadata
        """
        pass
    
    def validate_signals(self, signals: List[Signal]) -> List[Signal]:
        """
        Validate and filter signals based on strategy rules
        
        Args:
            signals: List of raw signals
            
        Returns:
            List of validated signals
        """
        # Use list comprehension for better performance
        return [
            signal for signal in signals
            if self._validate_signal_basic(signal) and self._validate_signal_strategy(signal)
        ]
    
    def _validate_signal_basic(self, signal: Signal) -> bool:
        """Basic signal validation"""
        # Combined condition for better performance
        return signal.confidence >= 0.1 and signal.price > 0
    
    def _validate_signal_strategy(self, signal: Signal) -> bool:
        """Strategy-specific signal validation - override in subclasses"""
        return True
    
    def optimize_parameters(self, data: MarketData, metric: str = 'sharpe') -> Dict[str, Any]:
        """
        Optimize strategy parameters using historical data
        
        Args:
            data: Historical market data
            metric: Optimization metric ('sharpe', 'return', 'win_rate', 'profit_factor')
            
        Returns:
            Optimized parameters dictionary
        """
        logger.info(f"Optimizing parameters for {self.name} using {metric} metric")
        
        # Get parameter ranges for optimization
        param_ranges = self._get_parameter_ranges()
        
        if not param_ranges:
            logger.warning(f"No parameter ranges defined for {self.name}")
            return self.parameters
        
        # Simple grid search optimization
        best_params = self.parameters.copy()
        best_score = float('-inf')
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(param_ranges)
        
        for params in param_combinations:
            try:
                # Test strategy with these parameters
                old_params = self.parameters.copy()
                self.parameters.update(params)
                
                # Calculate performance score
                score = self._calculate_performance_score(data, metric)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
                # Restore original parameters
                self.parameters = old_params
                
            except Exception as e:
                logger.error(f"Error testing parameters {params}: {e}")
                continue
        
        logger.info(f"Best parameters for {self.name}: {best_params} (score: {best_score:.4f})")
        return best_params
    
    def _get_parameter_ranges(self) -> Dict[str, List]:
        """Get parameter ranges for optimization - override in subclasses"""
        return {}
    
    def _generate_parameter_combinations(self, param_ranges: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for optimization"""
        if not param_ranges:
            return [{}]
        
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())
        
        # Use list comprehension for better performance
        return [
            dict(zip(keys, combination))
            for combination in itertools.product(*values)
        ]
    
    def _calculate_performance_score(self, data: MarketData, metric: str) -> float:
        """Calculate performance score for optimization"""
        try:
            # Generate signals for the data
            indicators = self._calculate_indicators(data)
            signals = self.generate_signals(data, indicators)
            
            if not signals:
                return float('-inf')
            
            # Simple performance calculation
            if metric == 'sharpe':
                return self._calculate_sharpe_ratio(signals, data)
            elif metric == 'return':
                return self._calculate_total_return(signals, data)
            elif metric == 'win_rate':
                return self._calculate_win_rate(signals, data)
            elif metric == 'profit_factor':
                return self._calculate_profit_factor(signals, data)
            else:
                return float('-inf')
                
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return float('-inf')
    
    def _calculate_indicators(self, data: MarketData) -> Dict[str, Any]:
        """Calculate all required indicators for the strategy"""
        # Check cache first
        if (self._cache_timestamp and 
            data.last_updated == self._cache_timestamp and 
            self._indicator_cache):
            return self._indicator_cache
        
        indicators = {}
        required_indicators = self.get_required_indicators()
        
        # Convert data to arrays once
        arrays = data.to_arrays()
        close_prices = arrays['close']
        
        # Define indicator mapping for cleaner code
        indicator_calculators = {
            'sma': lambda: self._calculate_sma(close_prices),
            'ema': lambda: self._calculate_ema(close_prices),
            'rsi': lambda: self._calculate_rsi(close_prices),
            'macd': lambda: self._calculate_macd(close_prices),
            'bollinger': lambda: self._calculate_bollinger_bands(close_prices),
            'support_resistance': lambda: self._calculate_support_resistance(arrays)
        }
        
        for indicator in required_indicators:
            if indicator in indicator_calculators:
                try:
                    indicators[indicator] = indicator_calculators[indicator]()
                except Exception as e:
                    logger.error(f"Error calculating {indicator}: {e}")
                    continue
        
        # Update cache
        self._indicator_cache = indicators
        self._cache_timestamp = data.last_updated
        
        return indicators
    
    @lru_cache(maxsize=32)
    def _get_common_periods(self, indicator_type: str) -> Tuple[int, ...]:
        """Get common periods for indicators"""
        if indicator_type == 'sma':
            return (20, 50, 100, 200)
        elif indicator_type == 'ema':
            return (12, 26, 50, 200)
        elif indicator_type == 'rsi':
            return (14, 21)
        elif indicator_type == 'bollinger':
            return (20, 50)
        return ()
    
    def _calculate_sma(self, close_prices: Union[List[float], 'np.ndarray']) -> Dict[str, Any]:
        """Calculate Simple Moving Average"""
        from ..indicators.moving_averages import calculate_sma
        
        results = {}
        data_len = len(close_prices)
        
        # Only calculate for periods that have enough data
        for period in self._get_common_periods('sma'):
            if period <= data_len:
                results[f'sma_{period}'] = calculate_sma(close_prices, period)
        
        return results
    
    def _calculate_ema(self, close_prices: Union[List[float], 'np.ndarray']) -> Dict[str, Any]:
        """Calculate Exponential Moving Average"""
        from ..indicators.moving_averages import calculate_ema
        
        results = {}
        data_len = len(close_prices)
        
        for period in self._get_common_periods('ema'):
            if period <= data_len:
                results[f'ema_{period}'] = calculate_ema(close_prices, period)
        
        return results
    
    def _calculate_rsi(self, close_prices: Union[List[float], 'np.ndarray']) -> Dict[str, Any]:
        """Calculate RSI"""
        from ..indicators.momentum import calculate_rsi
        
        results = {}
        data_len = len(close_prices)
        
        for period in self._get_common_periods('rsi'):
            if period <= data_len:
                results[f'rsi_{period}'] = calculate_rsi(close_prices, period)
        
        return results
    
    def _calculate_macd(self, close_prices: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """Calculate MACD"""
        from ..indicators.momentum import calculate_macd
        
        if len(close_prices) >= 26:
            macd_line, signal_line, histogram = calculate_macd(close_prices, 12, 26, 9)
            # Structure the data as expected by strategies
            return {
                'macd': {
                    'macd': macd_line,
                    'signal': signal_line,
                    'histogram': histogram
                }
            }
        
        return {}
    
    def _calculate_bollinger_bands(self, close_prices: Union[List[float], 'np.ndarray']) -> Dict[str, Any]:
        """Calculate Bollinger Bands"""
        from ..indicators.volatility import calculate_bollinger_bands
        
        results = {}
        data_len = len(close_prices)
        
        for period in self._get_common_periods('bollinger'):
            if period <= data_len:
                upper_band, middle_band, lower_band = calculate_bollinger_bands(close_prices, period, 2.0)
                # Structure the data as expected by strategies
                results[f'bollinger_{period}'] = {
                    'upper': upper_band,
                    'middle': middle_band,
                    'lower': lower_band
                }
        
        return results
    
    def _calculate_support_resistance(self, arrays: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Support and Resistance levels"""
        from ..indicators.levels import calculate_support_resistance
        
        if len(arrays['close']) >= 20:
            # Use close prices for support/resistance calculation
            return calculate_support_resistance(arrays['close'], lookback=20, min_touches=3)
        
        return {}
    
    def _calculate_sharpe_ratio(self, signals: List[Signal], data: MarketData) -> float:
        """Calculate Sharpe ratio from signals"""
        # Simplified Sharpe ratio calculation
        returns = self._calculate_signal_returns(signals, data)
        if not returns or len(returns) < 2:
            return 0.0
        
        if HAS_NUMPY:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
        else:
            # Pure Python fallback
            avg_return = sum(returns) / len(returns)
            variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
            std_return = variance ** 0.5
        
        if std_return == 0:
            return 0.0
        
        return avg_return / std_return
    
    def _calculate_total_return(self, signals: List[Signal], data: MarketData) -> float:
        """Calculate total return from signals"""
        returns = self._calculate_signal_returns(signals, data)
        if not returns:
            return 0.0
        
        return sum(returns)
    
    def _calculate_win_rate(self, signals: List[Signal], data: MarketData) -> float:
        """Calculate win rate from signals"""
        returns = self._calculate_signal_returns(signals, data)
        if not returns:
            return 0.0
        
        winning_trades = sum(1 for r in returns if r > 0)
        return winning_trades / len(returns)
    
    def _calculate_profit_factor(self, signals: List[Signal], data: MarketData) -> float:
        """Calculate profit factor from signals"""
        returns = self._calculate_signal_returns(signals, data)
        if not returns:
            return 0.0
        
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def _calculate_signal_returns(self, signals: List[Signal], data: MarketData) -> List[float]:
        """Calculate returns from signals - simplified implementation"""
        if not signals:
            return []
        
        returns = []
        entry_price = None
        
        # Process signals efficiently
        for signal in signals:
            if signal.signal_type == SignalType.BUY and entry_price is None:
                entry_price = signal.price
            elif entry_price is not None and signal.signal_type in (SignalType.SELL, SignalType.CLOSE_LONG):
                # Calculate return percentage
                returns.append((signal.price - entry_price) / entry_price)
                entry_price = None
        
        return returns
    
    def calculate_confidence(self, conditions: Dict[str, float], weights: Dict[str, float] = None) -> float:
        """
        Calculate signal confidence based on multiple conditions
        
        Args:
            conditions: Dictionary of condition names and their strength (0.0 to 1.0)
            weights: Optional weights for each condition
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not conditions:
            return 0.0
        
        # Use default weights if not provided
        if weights is None:
            total_weight = len(conditions)
            weighted_sum = sum(conditions.values())
        else:
            # Calculate weighted sum and total weight in single pass
            weighted_sum = 0.0
            total_weight = 0.0
            for key, value in conditions.items():
                weight = weights.get(key, 1.0)
                weighted_sum += value * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        # Calculate and clamp confidence
        confidence = weighted_sum / total_weight
        return max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
    
    def get_signal_metadata(self, **kwargs) -> Dict[str, Any]:
        """
        Generate metadata for signals
        
        Args:
            **kwargs: Additional metadata fields
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            'strategy': self.name,
            'parameters': self.parameters.copy(),
            'timestamp': datetime.now().isoformat(),
        }
        metadata.update(kwargs)
        return metadata
    
    def __str__(self) -> str:
        return f"{self.name}({self.parameters})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"