"""
Mock data generator for creating realistic market data.

This module provides the MockDataGenerator class for generating
various types of market data patterns for testing and backtesting.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import random
import math
from dataclasses import dataclass

# Optional numpy import for performance
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from ..core.models import OHLCV, TimeFrame


@dataclass
class MarketScenario:
    """Configuration for a market scenario."""
    name: str
    periods: int
    trend_strength: float
    volatility: float
    volume_profile: str
    noise_level: float
    description: str


class MockDataGenerator:
    """
    Advanced mock data generator for creating realistic market data.
    
    This class generates various types of market data patterns including
    trending, ranging, and complex realistic patterns with proper OHLC
    relationships and volume profiles.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the mock data generator.
        
        Args:
            seed: Random seed for reproducible data generation
        """
        if HAS_NUMPY:
            self.rng = np.random.RandomState(seed)
        else:
            if seed is not None:
                random.seed(seed)
            self.rng = random
        self._base_volume = 1000000  # Base volume for calculations
    
    def _random_normal(self, mean: float, std: float, size: int = 1):
        """Generate normal random numbers, with or without numpy."""
        # Ensure std is always positive
        std = max(abs(std), 1e-8)
        if HAS_NUMPY:
            return self.rng.normal(mean, std, size)
        else:
            return [random.gauss(mean, std) for _ in range(size)]
    
    def _random_uniform(self, low: float, high: float, size: int = 1):
        """Generate uniform random numbers, with or without numpy."""
        if HAS_NUMPY:
            return self.rng.uniform(low, high, size)
        else:
            return [random.uniform(low, high) for _ in range(size)]
    
    def _random_exponential(self, scale: float, size: int = 1):
        """Generate exponential random numbers, with or without numpy."""
        if HAS_NUMPY:
            return self.rng.exponential(scale, size)
        else:
            # Avoid division by zero
            if scale <= 0:
                scale = 1e-10
            return [random.expovariate(1.0 / scale) for _ in range(size)]
        
    def generate_trending_data(self, periods: int, trend_strength: float = 0.02,
                             volatility: float = 0.15, start_price: float = 100.0,
                             start_date: Optional[datetime] = None,
                             timeframe: TimeFrame = TimeFrame.DAILY) -> List[OHLCV]:
        """
        Generate trending market data with realistic OHLC relationships.
        
        Args:
            periods: Number of periods to generate
            trend_strength: Daily trend strength (0.02 = 2% daily trend)
            volatility: Daily volatility (0.15 = 15% daily volatility)
            start_price: Starting price for the data
            start_date: Starting date (defaults to 30 days ago)
            timeframe: Timeframe for the data
            
        Returns:
            List of OHLCV data points with trending pattern
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        
        data = []
        current_price = start_price
        
        # Generate base price movements
        trend_component = self._random_normal(trend_strength, trend_strength * 0.2, periods)
        volatility_component = self._random_normal(0, volatility, periods)
        
        for i in range(periods):
            # Calculate timestamp based on timeframe
            timestamp = self._get_timestamp(start_date, i, timeframe)
            
            # Calculate price movement
            price_change = trend_component[i] + volatility_component[i]
            new_price = current_price * (1 + price_change)
            
            # Generate OHLC values with realistic relationships
            ohlc = self._generate_ohlc(current_price, new_price, volatility)
            
            # Generate volume with trend correlation
            volume = self._generate_volume(price_change, volatility, i, periods)
            
            data.append(OHLCV(
                timestamp=timestamp,
                open=ohlc['open'],
                high=ohlc['high'],
                low=ohlc['low'],
                close=ohlc['close'],
                volume=volume
            ))
            
            current_price = new_price
        
        return data
    
    def generate_ranging_data(self, periods: int, range_width: float = 0.1,
                            volatility: float = 0.08, center_price: float = 100.0,
                            start_date: Optional[datetime] = None,
                            timeframe: TimeFrame = TimeFrame.DAILY) -> List[OHLCV]:
        """
        Generate ranging (sideways) market data.
        
        Args:
            periods: Number of periods to generate
            range_width: Width of the range (0.1 = 10% range)
            volatility: Intraday volatility
            center_price: Center price of the range
            start_date: Starting date (defaults to 30 days ago)
            timeframe: Timeframe for the data
            
        Returns:
            List of OHLCV data points with ranging pattern
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        
        data = []
        current_price = center_price
        
        # Calculate range boundaries
        upper_bound = center_price * (1 + range_width / 2)
        lower_bound = center_price * (1 - range_width / 2)
        
        for i in range(periods):
            timestamp = self._get_timestamp(start_date, i, timeframe)
            
            # Mean reversion towards center
            distance_from_center = (current_price - center_price) / center_price
            mean_reversion = -distance_from_center * 0.1  # 10% mean reversion strength
            
            # Add some randomness
            random_component = self._random_normal(0, volatility * 0.5)[0]
            
            # Calculate new price with boundaries
            price_change = mean_reversion + random_component
            new_price = current_price * (1 + price_change)
            
            # Keep price within bounds
            new_price = max(lower_bound, min(upper_bound, new_price))
            
            # Generate OHLC values
            ohlc = self._generate_ohlc(current_price, new_price, volatility)
            
            # Generate volume (higher volume near boundaries)
            boundary_factor = min(
                abs(current_price - lower_bound) / (center_price * range_width),
                abs(upper_bound - current_price) / (center_price * range_width)
            )
            volume = self._generate_volume(price_change, volatility, i, periods, boundary_factor)
            
            data.append(OHLCV(
                timestamp=timestamp,
                open=ohlc['open'],
                high=ohlc['high'],
                low=ohlc['low'],
                close=ohlc['close'],
                volume=volume
            ))
            
            current_price = new_price
        
        return data
    
    def generate_realistic_data(self, symbol: str, periods: int,
                              patterns_dict: Optional[Dict[str, float]] = None,
                              start_date: Optional[datetime] = None,
                              timeframe: TimeFrame = TimeFrame.DAILY) -> List[OHLCV]:
        """
        Generate realistic market data with mixed patterns.
        
        Args:
            symbol: Trading symbol (affects base price and volatility)
            periods: Number of periods to generate
            patterns_dict: Dictionary of pattern weights (trend, range, breakout, etc.)
            start_date: Starting date (defaults to periods ago)
            timeframe: Timeframe for the data
            
        Returns:
            List of OHLCV data points with realistic mixed patterns
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=periods)
        
        if patterns_dict is None:
            patterns_dict = {
                'trend': 0.3,
                'range': 0.4,
                'breakout': 0.2,
                'gap': 0.1
            }
        
        # Symbol-specific parameters
        base_price = self._get_symbol_base_price(symbol)
        base_volatility = self._get_symbol_volatility(symbol)
        
        data = []
        current_price = base_price
        
        # Divide periods into segments with different patterns
        segments = self._create_pattern_segments(periods, patterns_dict)
        
        period_idx = 0
        for segment in segments:
            pattern_type = segment['pattern']
            segment_periods = segment['periods']
            
            if pattern_type == 'trend':
                trend_strength = self._random_uniform(-0.03, 0.03)[0]  # -3% to +3% daily
                segment_data = self._generate_trend_segment(
                    segment_periods, current_price, trend_strength, 
                    base_volatility, start_date, period_idx, timeframe
                )
            elif pattern_type == 'range':
                range_width = self._random_uniform(0.05, 0.15)[0]  # 5% to 15% range
                segment_data = self._generate_range_segment(
                    segment_periods, current_price, range_width,
                    base_volatility, start_date, period_idx, timeframe
                )
            elif pattern_type == 'breakout':
                breakout_strength = self._random_uniform(0.05, 0.15)[0]  # 5% to 15% breakout
                segment_data = self._generate_breakout_segment(
                    segment_periods, current_price, breakout_strength,
                    base_volatility, start_date, period_idx, timeframe
                )
            else:  # gap
                gap_size = self._random_uniform(-0.05, 0.05)[0]  # -5% to +5% gap
                segment_data = self._generate_gap_segment(
                    segment_periods, current_price, gap_size,
                    base_volatility, start_date, period_idx, timeframe
                )
            
            data.extend(segment_data)
            period_idx += segment_periods
            
            # Update current price for next segment
            if segment_data:
                current_price = segment_data[-1].close
        
        return data
    
    def add_volume_profile(self, ohlcv_data: List[OHLCV], 
                          profile_type: str = 'normal') -> List[OHLCV]:
        """
        Add realistic volume profile to existing OHLCV data.
        
        Args:
            ohlcv_data: List of OHLCV data points
            profile_type: Type of volume profile ('normal', 'accumulation', 'distribution')
            
        Returns:
            List of OHLCV data with updated volume values
        """
        if not ohlcv_data:
            return ohlcv_data
        
        updated_data = []
        
        for i, candle in enumerate(ohlcv_data):
            # Calculate price movement and volatility
            price_change = abs(candle.close - candle.open) / candle.open
            range_ratio = (candle.high - candle.low) / candle.open
            
            # Base volume calculation
            base_volume = self._base_volume
            
            if profile_type == 'normal':
                # Higher volume on larger price movements
                volume_multiplier = 1 + (price_change * 5) + (range_ratio * 3)
            elif profile_type == 'accumulation':
                # Increasing volume over time during accumulation
                progress = i / len(ohlcv_data)
                volume_multiplier = 0.5 + (progress * 1.5) + (price_change * 2)
            elif profile_type == 'distribution':
                # Decreasing volume over time during distribution
                progress = i / len(ohlcv_data)
                volume_multiplier = 1.5 - (progress * 1.0) + (price_change * 2)
            else:
                volume_multiplier = 1.0
            
            # Add some randomness
            volume_multiplier *= self._random_uniform(0.7, 1.3)[0]
            
            # Calculate final volume
            volume = max(base_volume * volume_multiplier * 0.1, 1000)  # Minimum 1000
            
            updated_data.append(OHLCV(
                timestamp=candle.timestamp,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=volume
            ))
        
        return updated_data
    
    def create_market_scenarios(self) -> Dict[str, MarketScenario]:
        """
        Create predefined market scenarios for testing.
        
        Returns:
            Dictionary of market scenarios with different characteristics
        """
        scenarios = {
            'bull_market': MarketScenario(
                name='Bull Market',
                periods=252,  # 1 year daily
                trend_strength=0.08,  # 8% annual trend
                volatility=0.12,
                volume_profile='accumulation',
                noise_level=0.05,
                description='Strong upward trending market with low volatility'
            ),
            
            'bear_market': MarketScenario(
                name='Bear Market',
                periods=252,
                trend_strength=-0.12,  # -12% annual trend
                volatility=0.18,
                volume_profile='distribution',
                noise_level=0.08,
                description='Downward trending market with high volatility'
            ),
            
            'sideways_market': MarketScenario(
                name='Sideways Market',
                periods=252,
                trend_strength=0.0,
                volatility=0.10,
                volume_profile='normal',
                noise_level=0.06,
                description='Range-bound market with moderate volatility'
            ),
            
            'high_volatility': MarketScenario(
                name='High Volatility',
                periods=100,
                trend_strength=0.02,
                volatility=0.25,
                volume_profile='normal',
                noise_level=0.12,
                description='Highly volatile market conditions'
            ),
            
            'low_volatility': MarketScenario(
                name='Low Volatility',
                periods=100,
                trend_strength=0.01,
                volatility=0.05,
                volume_profile='normal',
                noise_level=0.02,
                description='Very low volatility market conditions'
            ),
            
            'flash_crash': MarketScenario(
                name='Flash Crash',
                periods=50,
                trend_strength=-0.30,
                volatility=0.35,
                volume_profile='distribution',
                noise_level=0.15,
                description='Rapid market decline with extreme volatility'
            )
        }
        
        return scenarios
    
    def _get_timestamp(self, start_date: datetime, period_idx: int, timeframe: TimeFrame) -> datetime:
        """Calculate timestamp for a given period index and timeframe."""
        if timeframe == TimeFrame.MINUTE_1:
            return start_date + timedelta(minutes=period_idx)
        elif timeframe == TimeFrame.MINUTE_5:
            return start_date + timedelta(minutes=period_idx * 5)
        elif timeframe == TimeFrame.MINUTE_15:
            return start_date + timedelta(minutes=period_idx * 15)
        elif timeframe == TimeFrame.MINUTE_30:
            return start_date + timedelta(minutes=period_idx * 30)
        elif timeframe == TimeFrame.HOUR_1:
            return start_date + timedelta(hours=period_idx)
        elif timeframe == TimeFrame.HOUR_4:
            return start_date + timedelta(hours=period_idx * 4)
        elif timeframe == TimeFrame.DAILY:
            return start_date + timedelta(days=period_idx)
        elif timeframe == TimeFrame.WEEKLY:
            return start_date + timedelta(weeks=period_idx)
        elif timeframe == TimeFrame.MONTHLY:
            return start_date + timedelta(days=period_idx * 30)  # Approximate
        else:
            return start_date + timedelta(days=period_idx)
    
    def _generate_ohlc(self, open_price: float, close_price: float, volatility: float) -> Dict[str, float]:
        """Generate realistic OHLC values given open and close prices."""
        # Determine the range of the candle
        body_size = abs(close_price - open_price)
        typical_range = open_price * volatility * 0.5
        
        # Generate high and low with some randomness
        high_extension = self._random_exponential(typical_range * 0.3)[0]
        low_extension = self._random_exponential(typical_range * 0.3)[0]
        
        # Calculate high and low
        high = max(open_price, close_price) + high_extension
        low = min(open_price, close_price) - low_extension
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        return {
            'open': round(open_price, 4),
            'high': round(high, 4),
            'low': round(low, 4),
            'close': round(close_price, 4)
        }
    
    def _generate_volume(self, price_change: float, volatility: float, 
                        period_idx: int, total_periods: int, 
                        boundary_factor: float = 1.0) -> float:
        """Generate realistic volume based on price action."""
        # Base volume calculation
        base_volume = self._base_volume
        
        # Volume increases with price movement
        price_impact = (abs(price_change) / volatility) * 0.5 + 0.5
        
        # Add some time-based patterns (higher volume at market open/close)
        time_factor = 1.0
        if total_periods > 20:  # Only apply for longer periods
            cycle_position = (period_idx % 20) / 20  # 20-period cycle
            time_factor = 1.0 + 0.3 * math.sin(cycle_position * 2 * math.pi)
        
        # Apply boundary factor (for ranging markets)
        boundary_impact = 2.0 - boundary_factor  # Higher volume near boundaries
        
        # Calculate final volume
        volume_multiplier = price_impact * time_factor * boundary_impact
        volume_multiplier *= self._random_uniform(0.7, 1.3)[0]  # Add randomness
        
        volume = base_volume * volume_multiplier
        return max(volume, 1000)  # Minimum volume of 1000
    
    def _get_symbol_base_price(self, symbol: str) -> float:
        """Get base price for a symbol."""
        # Common symbol price mappings
        symbol_prices = {
            'AAPL': 150.0,
            'GOOGL': 2800.0,
            'MSFT': 350.0,
            'TSLA': 800.0,
            'SPY': 450.0,
            'QQQ': 380.0,
            'BTC': 45000.0,
            'ETH': 3200.0,
            'EURUSD': 1.0800,
            'GBPUSD': 1.2500,
            'USDJPY': 110.0,
            'GOLD': 1800.0,
            'SILVER': 25.0,
            'OIL': 70.0
        }
        
        return symbol_prices.get(symbol.upper(), 100.0)
    
    def _get_symbol_volatility(self, symbol: str) -> float:
        """Get base volatility for a symbol."""
        # Common symbol volatility mappings
        symbol_volatilities = {
            'SPY': 0.12,
            'QQQ': 0.15,
            'AAPL': 0.18,
            'GOOGL': 0.20,
            'MSFT': 0.16,
            'TSLA': 0.35,
            'BTC': 0.60,
            'ETH': 0.55,
            'EURUSD': 0.08,
            'GBPUSD': 0.10,
            'USDJPY': 0.07,
            'GOLD': 0.15,
            'SILVER': 0.25,
            'OIL': 0.30
        }
        
        return symbol_volatilities.get(symbol.upper(), 0.15)
    
    def _create_pattern_segments(self, periods: int, patterns_dict: Dict[str, float]) -> List[Dict]:
        """Create segments with different patterns."""
        segments = []
        remaining_periods = periods
        
        # Normalize pattern weights
        total_weight = sum(patterns_dict.values())
        normalized_patterns = {k: v / total_weight for k, v in patterns_dict.items()}
        
        # Create segments
        for pattern, weight in normalized_patterns.items():
            if remaining_periods <= 0:
                break
            
            segment_periods = max(1, int(periods * weight))
            segment_periods = min(segment_periods, remaining_periods)
            
            segments.append({
                'pattern': pattern,
                'periods': segment_periods
            })
            
            remaining_periods -= segment_periods
        
        # Add any remaining periods to the last segment
        if remaining_periods > 0 and segments:
            segments[-1]['periods'] += remaining_periods
        
        return segments
    
    def _generate_trend_segment(self, periods: int, start_price: float, 
                              trend_strength: float, volatility: float,
                              start_date: datetime, period_offset: int, 
                              timeframe: TimeFrame) -> List[OHLCV]:
        """Generate a trending segment."""
        return self.generate_trending_data(
            periods=periods,
            trend_strength=trend_strength,
            volatility=volatility,
            start_price=start_price,
            start_date=self._get_timestamp(start_date, period_offset, timeframe),
            timeframe=timeframe
        )
    
    def _generate_range_segment(self, periods: int, start_price: float,
                              range_width: float, volatility: float,
                              start_date: datetime, period_offset: int,
                              timeframe: TimeFrame) -> List[OHLCV]:
        """Generate a ranging segment."""
        return self.generate_ranging_data(
            periods=periods,
            range_width=range_width,
            volatility=volatility,
            center_price=start_price,
            start_date=self._get_timestamp(start_date, period_offset, timeframe),
            timeframe=timeframe
        )
    
    def _generate_breakout_segment(self, periods: int, start_price: float,
                                 breakout_strength: float, volatility: float,
                                 start_date: datetime, period_offset: int,
                                 timeframe: TimeFrame) -> List[OHLCV]:
        """Generate a breakout segment."""
        # Breakout is just a strong trend
        return self.generate_trending_data(
            periods=periods,
            trend_strength=breakout_strength,
            volatility=volatility * 1.5,  # Higher volatility during breakout
            start_price=start_price,
            start_date=self._get_timestamp(start_date, period_offset, timeframe),
            timeframe=timeframe
        )
    
    def _generate_gap_segment(self, periods: int, start_price: float,
                            gap_size: float, volatility: float,
                            start_date: datetime, period_offset: int,
                            timeframe: TimeFrame) -> List[OHLCV]:
        """Generate a segment with a gap."""
        gap_price = start_price * (1 + gap_size)
        
        # Generate normal trending data after the gap
        return self.generate_trending_data(
            periods=periods,
            trend_strength=0.005,  # Small trend after gap
            volatility=volatility,
            start_price=gap_price,
            start_date=self._get_timestamp(start_date, period_offset, timeframe),
            timeframe=timeframe
        )