"""
Advanced mock data generation utilities for comprehensive testing.
Provides realistic market data, indicators, and trading scenarios.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import random
from pathlib import Path


class MarketCondition(Enum):
    """Market condition types."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CRASH = "crash"
    RECOVERY = "recovery"


class DataPattern(Enum):
    """Data pattern types."""
    TREND = "trend"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"
    GAP = "gap"
    SPIKE = "spike"


@dataclass
class MockDataConfig:
    """Configuration for mock data generation."""
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    frequency: str = '1min'
    market_condition: MarketCondition = MarketCondition.SIDEWAYS
    volatility: float = 0.02
    trend_strength: float = 0.0
    noise_level: float = 0.01
    patterns: List[DataPattern] = None
    anomaly_probability: float = 0.01
    missing_data_probability: float = 0.005
    seed: int = 42


class AdvancedMockDataGenerator:
    """Advanced mock data generator with realistic market behavior."""
    
    def __init__(self, config: MockDataConfig):
        """Initialize the mock data generator."""
        self.config = config
        np.random.seed(config.seed)
        random.seed(config.seed)
        
        # Load reference data if available
        self.reference_data = self._load_reference_data()
        
        # Market parameters
        self.market_params = self._get_market_parameters()
    
    def _load_reference_data(self) -> Dict[str, Any]:
        """Load reference market data for realistic patterns."""
        reference_file = Path(__file__).parent / "reference_data.json"
        
        if reference_file.exists():
            with open(reference_file, 'r') as f:
                return json.load(f)
        
        return {
            'sector_correlations': {
                'TECH': ['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
                'FINANCE': ['JPM', 'BAC', 'GS', 'WFC'],
                'ENERGY': ['XOM', 'CVX', 'COP', 'EOG']
            },
            'volatility_profiles': {
                'low': 0.01,
                'medium': 0.02,
                'high': 0.04,
                'extreme': 0.08
            }
        }
    
    def _get_market_parameters(self) -> Dict[str, Any]:
        """Get market parameters based on condition."""
        params = {
            MarketCondition.BULL: {
                'drift': 0.1,
                'volatility_multiplier': 0.8,
                'trend_persistence': 0.7
            },
            MarketCondition.BEAR: {
                'drift': -0.15,
                'volatility_multiplier': 1.2,
                'trend_persistence': 0.6
            },
            MarketCondition.SIDEWAYS: {
                'drift': 0.0,
                'volatility_multiplier': 1.0,
                'trend_persistence': 0.3
            },
            MarketCondition.VOLATILE: {
                'drift': 0.02,
                'volatility_multiplier': 2.0,
                'trend_persistence': 0.2
            },
            MarketCondition.CRASH: {
                'drift': -0.3,
                'volatility_multiplier': 3.0,
                'trend_persistence': 0.8
            },
            MarketCondition.RECOVERY: {
                'drift': 0.2,
                'volatility_multiplier': 1.5,
                'trend_persistence': 0.5
            }
        }
        
        return params.get(self.config.market_condition, params[MarketCondition.SIDEWAYS])
    
    def generate_market_data(self) -> pd.DataFrame:
        """Generate comprehensive market data."""
        # Generate time series
        date_range = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq=self.config.frequency
        )
        
        all_data = []
        
        for symbol in self.config.symbols:
            symbol_data = self._generate_symbol_data(symbol, date_range)
            all_data.append(symbol_data)
        
        # Combine all symbol data
        df = pd.concat(all_data, ignore_index=True)
        
        # Apply correlations
        df = self._apply_correlations(df)
        
        # Apply patterns
        if self.config.patterns:
            df = self._apply_patterns(df)
        
        # Add anomalies
        df = self._add_anomalies(df)
        
        # Add missing data
        df = self._add_missing_data(df)
        
        return df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
    
    def _generate_symbol_data(self, symbol: str, date_range: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate data for a single symbol."""
        n_points = len(date_range)
        
        # Base price
        base_price = np.random.uniform(10, 500)
        
        # Generate price movements
        returns = self._generate_returns(n_points)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = []
        for i, timestamp in enumerate(date_range):
            ohlcv = self._generate_ohlcv(prices[i], returns[i])
            
            data.append({
                'symbol': symbol,
                'timestamp': timestamp,
                'open': ohlcv['open'],
                'high': ohlcv['high'],
                'low': ohlcv['low'],
                'close': ohlcv['close'],
                'volume': ohlcv['volume'],
                'adj_close': ohlcv['close'],  # Simplified
                'returns': returns[i],
                'log_returns': np.log(ohlcv['close'] / ohlcv['open'])
            })
        
        return pd.DataFrame(data)
    
    def _generate_returns(self, n_points: int) -> np.ndarray:
        """Generate realistic return series."""
        # Base random walk
        base_volatility = self.config.volatility
        market_vol = base_volatility * self.market_params['volatility_multiplier']
        
        # Generate returns with clustering volatility (GARCH-like)
        returns = np.zeros(n_points)
        vol = np.ones(n_points) * market_vol
        
        for i in range(1, n_points):
            # Volatility clustering
            vol[i] = 0.1 * market_vol + 0.8 * vol[i-1] + 0.1 * market_vol * abs(returns[i-1])
            
            # Trend component
            trend = self.market_params['drift'] / 252  # Daily drift
            
            # Random shock
            shock = np.random.normal(0, vol[i])
            
            # Persistence
            if i > 1:
                persistence = self.market_params['trend_persistence']
                momentum = persistence * returns[i-1]
            else:
                momentum = 0
            
            returns[i] = trend + momentum + shock
        
        return returns
    
    def _generate_ohlcv(self, price: float, return_val: float) -> Dict[str, float]:
        """Generate OHLCV data for a single period."""
        # Open price (close of previous period)
        open_price = price / (1 + return_val)
        close_price = price
        
        # High and low based on intraday volatility
        intraday_vol = abs(return_val) * np.random.uniform(1.0, 3.0)
        high_price = max(open_price, close_price) * (1 + intraday_vol * np.random.uniform(0, 1))
        low_price = min(open_price, close_price) * (1 - intraday_vol * np.random.uniform(0, 1))
        
        # Volume based on price movement
        base_volume = np.random.uniform(10000, 100000)
        volume_multiplier = 1 + 5 * abs(return_val)  # Higher volume on big moves
        volume = int(base_volume * volume_multiplier)
        
        return {
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        }
    
    def _apply_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply sector correlations to the data."""
        sectors = self.reference_data.get('sector_correlations', {})
        
        for sector, symbols in sectors.items():
            sector_symbols = [s for s in symbols if s in df['symbol'].unique()]
            
            if len(sector_symbols) > 1:
                # Create correlation matrix
                correlation = 0.3 + 0.4 * np.random.random()
                
                # Apply correlation to returns
                for i, symbol1 in enumerate(sector_symbols):
                    for j, symbol2 in enumerate(sector_symbols[i+1:], i+1):
                        # Get returns for both symbols
                        mask1 = df['symbol'] == symbol1
                        mask2 = df['symbol'] == symbol2
                        
                        if mask1.sum() > 0 and mask2.sum() > 0:
                            returns1 = df.loc[mask1, 'returns'].values
                            returns2 = df.loc[mask2, 'returns'].values
                            
                            min_len = min(len(returns1), len(returns2))
                            if min_len > 0:
                                # Apply correlation
                                corr_component = correlation * returns1[:min_len]
                                df.loc[mask2, 'returns'] = returns2[:min_len] + corr_component
        
        return df
    
    def _apply_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply specific patterns to the data."""
        for pattern in self.config.patterns:
            if pattern == DataPattern.TREND:
                df = self._apply_trend_pattern(df)
            elif pattern == DataPattern.MEAN_REVERSION:
                df = self._apply_mean_reversion_pattern(df)
            elif pattern == DataPattern.BREAKOUT:
                df = self._apply_breakout_pattern(df)
            elif pattern == DataPattern.GAP:
                df = self._apply_gap_pattern(df)
            elif pattern == DataPattern.SPIKE:
                df = self._apply_spike_pattern(df)
        
        return df
    
    def _apply_trend_pattern(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply trend pattern to data."""
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy()
            
            # Add trend component
            trend_strength = self.config.trend_strength
            trend_component = np.linspace(0, trend_strength, len(symbol_data))
            
            # Apply trend to prices
            df.loc[mask, 'close'] *= (1 + trend_component)
            df.loc[mask, 'open'] *= (1 + trend_component)
            df.loc[mask, 'high'] *= (1 + trend_component)
            df.loc[mask, 'low'] *= (1 + trend_component)
        
        return df
    
    def _apply_mean_reversion_pattern(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply mean reversion pattern to data."""
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy()
            
            # Calculate moving average
            ma_window = min(20, len(symbol_data) // 4)
            if ma_window > 0:
                ma = symbol_data['close'].rolling(window=ma_window).mean()
                
                # Apply mean reversion
                reversion_strength = 0.1
                deviation = (symbol_data['close'] - ma) / ma
                reversion = -reversion_strength * deviation
                
                df.loc[mask, 'close'] *= (1 + reversion.fillna(0))
        
        return df
    
    def _apply_breakout_pattern(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply breakout pattern to data."""
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy()
            
            # Create consolidation period followed by breakout
            consolidation_length = len(symbol_data) // 3
            breakout_start = consolidation_length
            
            if breakout_start < len(symbol_data):
                # Reduce volatility in consolidation
                consolidation_mask = mask & (df.index < breakout_start)
                df.loc[consolidation_mask, 'high'] = df.loc[consolidation_mask, 'close'] * 1.005
                df.loc[consolidation_mask, 'low'] = df.loc[consolidation_mask, 'close'] * 0.995
                
                # Increase volatility after breakout
                breakout_mask = mask & (df.index >= breakout_start)
                volatility_increase = 2.0
                df.loc[breakout_mask, 'high'] *= volatility_increase
                df.loc[breakout_mask, 'low'] /= volatility_increase
        
        return df
    
    def _apply_gap_pattern(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply gap pattern to data."""
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy()
            
            # Create random gaps
            n_gaps = max(1, len(symbol_data) // 100)
            gap_indices = np.random.choice(range(1, len(symbol_data)), n_gaps, replace=False)
            
            for gap_idx in gap_indices:
                gap_size = np.random.uniform(0.02, 0.05)
                gap_direction = np.random.choice([-1, 1])
                gap_multiplier = 1 + gap_direction * gap_size
                
                # Apply gap to all prices after the gap
                post_gap_mask = mask & (df.index >= gap_idx)
                df.loc[post_gap_mask, 'open'] *= gap_multiplier
                df.loc[post_gap_mask, 'high'] *= gap_multiplier
                df.loc[post_gap_mask, 'low'] *= gap_multiplier
                df.loc[post_gap_mask, 'close'] *= gap_multiplier
        
        return df
    
    def _apply_spike_pattern(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply spike pattern to data."""
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy()
            
            # Create random spikes
            n_spikes = max(1, len(symbol_data) // 50)
            spike_indices = np.random.choice(range(len(symbol_data)), n_spikes, replace=False)
            
            for spike_idx in spike_indices:
                spike_size = np.random.uniform(0.05, 0.15)
                spike_direction = np.random.choice([-1, 1])
                
                # Apply spike to high/low only
                if spike_direction > 0:
                    df.loc[mask & (df.index == spike_idx), 'high'] *= (1 + spike_size)
                else:
                    df.loc[mask & (df.index == spike_idx), 'low'] *= (1 - spike_size)
        
        return df
    
    def _add_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add anomalies to the data."""
        n_anomalies = int(len(df) * self.config.anomaly_probability)
        
        if n_anomalies > 0:
            anomaly_indices = np.random.choice(range(len(df)), n_anomalies, replace=False)
            
            for idx in anomaly_indices:
                anomaly_type = np.random.choice(['price_spike', 'volume_spike', 'zero_volume'])
                
                if anomaly_type == 'price_spike':
                    multiplier = np.random.uniform(2.0, 5.0)
                    df.loc[idx, 'high'] *= multiplier
                elif anomaly_type == 'volume_spike':
                    multiplier = np.random.uniform(10.0, 100.0)
                    df.loc[idx, 'volume'] *= multiplier
                elif anomaly_type == 'zero_volume':
                    df.loc[idx, 'volume'] = 0
        
        return df
    
    def _add_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add missing data to simulate real-world conditions."""
        n_missing = int(len(df) * self.config.missing_data_probability)
        
        if n_missing > 0:
            missing_indices = np.random.choice(range(len(df)), n_missing, replace=False)
            
            for idx in missing_indices:
                # Randomly choose which fields to make missing
                fields = ['open', 'high', 'low', 'close', 'volume']
                n_fields = np.random.randint(1, len(fields) + 1)
                missing_fields = np.random.choice(fields, n_fields, replace=False)
                
                for field in missing_fields:
                    df.loc[idx, field] = np.nan
        
        return df
    
    def generate_indicators(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate technical indicators for the market data."""
        result = market_data.copy()
        
        # Group by symbol for calculations
        for symbol in result['symbol'].unique():
            mask = result['symbol'] == symbol
            symbol_data = result[mask].copy()
            
            # Simple Moving Averages
            for window in [5, 10, 20, 50, 200]:
                if len(symbol_data) >= window:
                    result.loc[mask, f'sma_{window}'] = symbol_data['close'].rolling(window=window).mean()
            
            # Exponential Moving Averages
            for window in [12, 26]:
                if len(symbol_data) >= window:
                    result.loc[mask, f'ema_{window}'] = symbol_data['close'].ewm(span=window).mean()
            
            # RSI
            if len(symbol_data) >= 14:
                result.loc[mask, 'rsi'] = self._calculate_rsi(symbol_data['close'])
            
            # MACD
            if len(symbol_data) >= 26:
                macd_line, signal_line, histogram = self._calculate_macd(symbol_data['close'])
                result.loc[mask, 'macd'] = macd_line
                result.loc[mask, 'macd_signal'] = signal_line
                result.loc[mask, 'macd_histogram'] = histogram
            
            # Bollinger Bands
            if len(symbol_data) >= 20:
                bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(symbol_data['close'])
                result.loc[mask, 'bb_upper'] = bb_upper
                result.loc[mask, 'bb_middle'] = bb_middle
                result.loc[mask, 'bb_lower'] = bb_lower
            
            # Volume indicators
            if len(symbol_data) >= 10:
                result.loc[mask, 'volume_sma'] = symbol_data['volume'].rolling(window=10).mean()
                result.loc[mask, 'volume_ratio'] = symbol_data['volume'] / symbol_data['volume'].rolling(window=10).mean()
        
        return result
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, middle, lower
    
    def generate_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on the market data."""
        result = market_data.copy()
        
        # Group by symbol for calculations
        for symbol in result['symbol'].unique():
            mask = result['symbol'] == symbol
            symbol_data = result[mask].copy()
            
            # Moving Average Crossover Strategy
            if 'sma_5' in symbol_data.columns and 'sma_20' in symbol_data.columns:
                ma_signals = self._generate_ma_crossover_signals(symbol_data)
                result.loc[mask, 'ma_crossover_signal'] = ma_signals['signal']
                result.loc[mask, 'ma_crossover_confidence'] = ma_signals['confidence']
            
            # RSI Strategy
            if 'rsi' in symbol_data.columns:
                rsi_signals = self._generate_rsi_signals(symbol_data)
                result.loc[mask, 'rsi_signal'] = rsi_signals['signal']
                result.loc[mask, 'rsi_confidence'] = rsi_signals['confidence']
            
            # MACD Strategy
            if 'macd' in symbol_data.columns and 'macd_signal' in symbol_data.columns:
                macd_signals = self._generate_macd_signals(symbol_data)
                result.loc[mask, 'macd_signal'] = macd_signals['signal']
                result.loc[mask, 'macd_confidence'] = macd_signals['confidence']
            
            # Bollinger Bands Strategy
            if all(col in symbol_data.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
                bb_signals = self._generate_bollinger_signals(symbol_data)
                result.loc[mask, 'bollinger_signal'] = bb_signals['signal']
                result.loc[mask, 'bollinger_confidence'] = bb_signals['confidence']
        
        return result
    
    def _generate_ma_crossover_signals(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate moving average crossover signals."""
        signals = pd.Series('HOLD', index=data.index)
        confidence = pd.Series(0.0, index=data.index)
        
        sma_5 = data['sma_5']
        sma_20 = data['sma_20']
        
        # Generate signals
        signals.loc[sma_5 > sma_20] = 'BUY'
        signals.loc[sma_5 < sma_20] = 'SELL'
        
        # Calculate confidence based on separation
        separation = abs(sma_5 - sma_20) / sma_20
        confidence = np.clip(separation * 10, 0.1, 1.0)
        
        return {'signal': signals, 'confidence': confidence}
    
    def _generate_rsi_signals(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate RSI signals."""
        signals = pd.Series('HOLD', index=data.index)
        confidence = pd.Series(0.0, index=data.index)
        
        rsi = data['rsi']
        
        # Generate signals
        signals.loc[rsi < 30] = 'BUY'
        signals.loc[rsi > 70] = 'SELL'
        
        # Calculate confidence based on RSI extremes
        confidence.loc[rsi < 30] = np.clip((30 - rsi.loc[rsi < 30]) / 30, 0.1, 1.0)
        confidence.loc[rsi > 70] = np.clip((rsi.loc[rsi > 70] - 70) / 30, 0.1, 1.0)
        confidence.loc[signals == 'HOLD'] = 0.1
        
        return {'signal': signals, 'confidence': confidence}
    
    def _generate_macd_signals(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate MACD signals."""
        signals = pd.Series('HOLD', index=data.index)
        confidence = pd.Series(0.0, index=data.index)
        
        macd = data['macd']
        macd_signal = data['macd_signal']
        
        # Generate signals
        signals.loc[macd > macd_signal] = 'BUY'
        signals.loc[macd < macd_signal] = 'SELL'
        
        # Calculate confidence based on MACD separation
        separation = abs(macd - macd_signal)
        confidence = np.clip(separation * 100, 0.1, 1.0)
        
        return {'signal': signals, 'confidence': confidence}
    
    def _generate_bollinger_signals(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generate Bollinger Bands signals."""
        signals = pd.Series('HOLD', index=data.index)
        confidence = pd.Series(0.0, index=data.index)
        
        close = data['close']
        bb_upper = data['bb_upper']
        bb_lower = data['bb_lower']
        bb_middle = data['bb_middle']
        
        # Generate signals
        signals.loc[close < bb_lower] = 'BUY'
        signals.loc[close > bb_upper] = 'SELL'
        
        # Calculate confidence based on distance from bands
        confidence.loc[close < bb_lower] = np.clip((bb_lower - close) / bb_lower, 0.1, 1.0)
        confidence.loc[close > bb_upper] = np.clip((close - bb_upper) / bb_upper, 0.1, 1.0)
        confidence.loc[signals == 'HOLD'] = 0.1
        
        return {'signal': signals, 'confidence': confidence}


class ScenarioGenerator:
    """Generate specific testing scenarios."""
    
    def __init__(self, seed: int = 42):
        """Initialize scenario generator."""
        self.seed = seed
        np.random.seed(seed)
    
    def generate_stress_test_scenario(self, symbols: List[str], duration_days: int = 30) -> pd.DataFrame:
        """Generate stress test scenario with extreme market conditions."""
        config = MockDataConfig(
            symbols=symbols,
            start_date=datetime.now() - timedelta(days=duration_days),
            end_date=datetime.now(),
            frequency='1min',
            market_condition=MarketCondition.CRASH,
            volatility=0.05,
            trend_strength=-0.3,
            patterns=[DataPattern.GAP, DataPattern.SPIKE],
            anomaly_probability=0.05,
            seed=self.seed
        )
        
        generator = AdvancedMockDataGenerator(config)
        return generator.generate_market_data()
    
    def generate_bull_market_scenario(self, symbols: List[str], duration_days: int = 90) -> pd.DataFrame:
        """Generate bull market scenario."""
        config = MockDataConfig(
            symbols=symbols,
            start_date=datetime.now() - timedelta(days=duration_days),
            end_date=datetime.now(),
            frequency='5min',
            market_condition=MarketCondition.BULL,
            volatility=0.015,
            trend_strength=0.2,
            patterns=[DataPattern.TREND, DataPattern.BREAKOUT],
            anomaly_probability=0.01,
            seed=self.seed
        )
        
        generator = AdvancedMockDataGenerator(config)
        return generator.generate_market_data()
    
    def generate_sideways_market_scenario(self, symbols: List[str], duration_days: int = 60) -> pd.DataFrame:
        """Generate sideways market scenario."""
        config = MockDataConfig(
            symbols=symbols,
            start_date=datetime.now() - timedelta(days=duration_days),
            end_date=datetime.now(),
            frequency='1min',
            market_condition=MarketCondition.SIDEWAYS,
            volatility=0.02,
            trend_strength=0.0,
            patterns=[DataPattern.CONSOLIDATION, DataPattern.MEAN_REVERSION],
            anomaly_probability=0.005,
            seed=self.seed
        )
        
        generator = AdvancedMockDataGenerator(config)
        return generator.generate_market_data()
    
    def generate_high_frequency_scenario(self, symbols: List[str], duration_hours: int = 24) -> pd.DataFrame:
        """Generate high-frequency trading scenario."""
        config = MockDataConfig(
            symbols=symbols,
            start_date=datetime.now() - timedelta(hours=duration_hours),
            end_date=datetime.now(),
            frequency='1s',
            market_condition=MarketCondition.VOLATILE,
            volatility=0.001,
            patterns=[DataPattern.SPIKE],
            anomaly_probability=0.001,
            seed=self.seed
        )
        
        generator = AdvancedMockDataGenerator(config)
        return generator.generate_market_data()
    
    def generate_earnings_announcement_scenario(self, symbol: str) -> pd.DataFrame:
        """Generate earnings announcement scenario with gap and volatility."""
        config = MockDataConfig(
            symbols=[symbol],
            start_date=datetime.now() - timedelta(days=5),
            end_date=datetime.now(),
            frequency='1min',
            market_condition=MarketCondition.VOLATILE,
            volatility=0.03,
            patterns=[DataPattern.GAP, DataPattern.SPIKE],
            anomaly_probability=0.02,
            seed=self.seed
        )
        
        generator = AdvancedMockDataGenerator(config)
        return generator.generate_market_data()
    
    def generate_market_open_scenario(self, symbols: List[str]) -> pd.DataFrame:
        """Generate market open scenario with gaps and high volume."""
        config = MockDataConfig(
            symbols=symbols,
            start_date=datetime.now().replace(hour=9, minute=30, second=0, microsecond=0),
            end_date=datetime.now().replace(hour=10, minute=30, second=0, microsecond=0),
            frequency='1min',
            market_condition=MarketCondition.VOLATILE,
            volatility=0.025,
            patterns=[DataPattern.GAP, DataPattern.BREAKOUT],
            anomaly_probability=0.02,
            seed=self.seed
        )
        
        generator = AdvancedMockDataGenerator(config)
        return generator.generate_market_data()