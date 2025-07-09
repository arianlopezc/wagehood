"""
Mathematical validation tests for trading strategies.

This module tests the mathematical correctness and logical consistency
of all 5 trading strategies implemented in the system.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.strategies.ma_crossover import MACrossover
from src.strategies.rsi_trend import RSITrend
from src.strategies.macd_rsi import MACDRSIStrategy
from src.strategies.bollinger_breakout import BollingerBreakout
from src.strategies.sr_breakout import SRBreakout
from src.strategies.base import BaseStrategy
from src.core.constants import *
from ..utils.data_validator import DataValidator
from ..utils.performance_monitor import PerformanceMonitor, PerformanceProfile
from . import MATHEMATICAL_TOLERANCE, TEST_DATA_POINTS


class TestMACrossoverStrategy:
    """Test MA Crossover strategy mathematical correctness."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataValidator(tolerance=MATHEMATICAL_TOLERANCE)
        self.performance_monitor = PerformanceMonitor()
        self.strategy = MACrossover()
        
        # Create test data
        np.random.seed(42)
        self.test_data = self._generate_trending_data()
        
        # Create validation suite
        self.validation_suite = self.validator.create_suite("MA Crossover Mathematical Validation")
    
    def _generate_trending_data(self) -> pd.DataFrame:
        """Generate trending price data for testing."""
        # Create uptrend followed by downtrend
        uptrend = np.linspace(100, 120, 500)
        downtrend = np.linspace(120, 90, 500)
        
        # Add some noise
        noise = np.random.normal(0, 1, 1000)
        prices = np.concatenate([uptrend, downtrend]) + noise
        
        return pd.DataFrame({
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 1000)
        }, index=pd.date_range('2023-01-01', periods=1000))
    
    @pytest.mark.mathematical
    def test_ma_crossover_signal_generation(self):
        """Test MA crossover signal generation logic."""
        with PerformanceProfile(self.performance_monitor, "MA Crossover Signals"):
            signals = self.strategy.generate_signals(self.test_data)
        
        # Validate signal properties
        result = self.validator.validate_trading_signals(
            signals=signals,
            name="MA Crossover Signals",
            suite_name=self.validation_suite.name
        )
        
        assert result.passed, f"MA Crossover signals failed validation: {result.message}"
        
        # Test signal logic
        fast_ma = self.test_data['close'].rolling(window=50).mean()
        slow_ma = self.test_data['close'].rolling(window=200).mean()
        
        # Check that buy signals occur when fast MA crosses above slow MA
        buy_signals = signals[signals == 1]
        for idx in buy_signals.index:
            if idx > 200:  # After both MAs are calculated
                prev_idx = idx - 1
                assert fast_ma.loc[prev_idx] <= slow_ma.loc[prev_idx], \
                    f"Buy signal without proper crossover at {idx}"
                assert fast_ma.loc[idx] > slow_ma.loc[idx], \
                    f"Buy signal but fast MA not above slow MA at {idx}"
    
    @pytest.mark.mathematical
    def test_ma_crossover_edge_cases(self):
        """Test MA crossover with edge cases."""
        # Test with flat data (no trend)
        flat_data = pd.DataFrame({
            'open': [100] * 300,
            'high': [101] * 300,
            'low': [99] * 300,
            'close': [100] * 300,
            'volume': [1000] * 300
        })
        
        signals = self.strategy.generate_signals(flat_data)
        
        # Should have very few signals in flat market
        signal_count = (signals != 0).sum()
        assert signal_count < len(flat_data) * 0.1, "Too many signals in flat market"
        
        # Test with insufficient data
        small_data = self.test_data.head(100)
        signals_small = self.strategy.generate_signals(small_data)
        
        # Should handle gracefully
        assert len(signals_small) == len(small_data), "Signal length mismatch"
    
    @pytest.mark.mathematical
    def test_ma_crossover_consistency(self):
        """Test MA crossover consistency across multiple runs."""
        # Run the same strategy multiple times
        results = []
        for i in range(5):
            signals = self.strategy.generate_signals(self.test_data)
            results.append(signals)
        
        # All results should be identical
        for i in range(1, len(results)):
            pd.testing.assert_series_equal(results[0], results[i])


class TestRSITrendStrategy:
    """Test RSI Trend strategy mathematical correctness."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataValidator(tolerance=MATHEMATICAL_TOLERANCE)
        self.performance_monitor = PerformanceMonitor()
        self.strategy = RSITrend()
        
        # Create test data with RSI extremes
        np.random.seed(42)
        self.test_data = self._generate_rsi_extreme_data()
        
        # Create validation suite
        self.validation_suite = self.validator.create_suite("RSI Trend Mathematical Validation")
    
    def _generate_rsi_extreme_data(self) -> pd.DataFrame:
        """Generate data that will create RSI extremes."""
        # Create oversold and overbought conditions
        oversold_trend = np.linspace(100, 80, 200)  # Downtrend for oversold
        overbought_trend = np.linspace(80, 100, 200)  # Uptrend for overbought
        sideways = np.random.normal(100, 2, 600)  # Sideways movement
        
        prices = np.concatenate([oversold_trend, overbought_trend, sideways])
        
        return pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 1000)
        }, index=pd.date_range('2023-01-01', periods=1000))
    
    @pytest.mark.mathematical
    def test_rsi_trend_signal_logic(self):
        """Test RSI trend signal generation logic."""
        with PerformanceProfile(self.performance_monitor, "RSI Trend Signals"):
            signals = self.strategy.generate_signals(self.test_data)
        
        # Validate signal properties
        result = self.validator.validate_trading_signals(
            signals=signals,
            name="RSI Trend Signals",
            suite_name=self.validation_suite.name
        )
        
        assert result.passed, f"RSI Trend signals failed validation: {result.message}"
        
        # Calculate RSI for validation
        from src.indicators.momentum import RSICalculator
        rsi_calc = RSICalculator()
        rsi = rsi_calc.calculate(self.test_data['close'], period=14)
        
        # Test signal logic
        buy_signals = signals[signals == 1]
        sell_signals = signals[signals == -1]
        
        # Buy signals should occur when RSI is oversold and turning up
        for idx in buy_signals.index:
            if idx > 14:  # After RSI is calculated
                current_rsi = rsi.loc[idx]
                assert current_rsi < 40, f"Buy signal but RSI not oversold at {idx}: {current_rsi}"
        
        # Sell signals should occur when RSI is overbought and turning down
        for idx in sell_signals.index:
            if idx > 14:  # After RSI is calculated
                current_rsi = rsi.loc[idx]
                assert current_rsi > 60, f"Sell signal but RSI not overbought at {idx}: {current_rsi}"
    
    @pytest.mark.mathematical
    def test_rsi_trend_boundary_conditions(self):
        """Test RSI trend strategy with boundary conditions."""
        # Test with extreme RSI values
        extreme_up = pd.DataFrame({
            'close': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200] * 10,
            'open': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200] * 10,
            'high': [105, 115, 125, 135, 145, 155, 165, 175, 185, 195, 205] * 10,
            'low': [95, 105, 115, 125, 135, 145, 155, 165, 175, 185, 195] * 10,
            'volume': [1000] * 110
        })
        
        signals = self.strategy.generate_signals(extreme_up)
        
        # Should generate some sell signals due to overbought conditions
        sell_count = (signals == -1).sum()
        assert sell_count > 0, "No sell signals generated in extreme uptrend"


class TestMACDRSIStrategy:
    """Test MACD RSI combined strategy mathematical correctness."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataValidator(tolerance=MATHEMATICAL_TOLERANCE)
        self.performance_monitor = PerformanceMonitor()
        self.strategy = MACDRSIStrategy()
        
        # Create test data
        np.random.seed(42)
        self.test_data = self._generate_complex_data()
        
        # Create validation suite
        self.validation_suite = self.validator.create_suite("MACD RSI Mathematical Validation")
    
    def _generate_complex_data(self) -> pd.DataFrame:
        """Generate complex market data for testing."""
        # Create multiple market phases
        trend_up = np.linspace(100, 130, 250)
        sideways = np.random.normal(130, 3, 250)
        trend_down = np.linspace(130, 90, 250)
        recovery = np.linspace(90, 110, 250)
        
        prices = np.concatenate([trend_up, sideways, trend_down, recovery])
        
        return pd.DataFrame({
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 1000)
        }, index=pd.date_range('2023-01-01', periods=1000))
    
    @pytest.mark.mathematical
    def test_macd_rsi_signal_confluence(self):
        """Test MACD RSI signal confluence logic."""
        with PerformanceProfile(self.performance_monitor, "MACD RSI Signals"):
            signals = self.strategy.generate_signals(self.test_data)
        
        # Validate signal properties
        result = self.validator.validate_trading_signals(
            signals=signals,
            name="MACD RSI Signals",
            suite_name=self.validation_suite.name
        )
        
        assert result.passed, f"MACD RSI signals failed validation: {result.message}"
        
        # Test that signals are less frequent than individual strategies
        # (due to confluence requirement)
        from src.strategies.rsi_trend import RSITrend
        from src.strategies.ma_crossover import MACrossover
        
        rsi_strategy = RSITrend()
        ma_strategy = MACrossover()
        
        rsi_signals = rsi_strategy.generate_signals(self.test_data)
        ma_signals = ma_strategy.generate_signals(self.test_data)
        
        combined_signal_count = (signals != 0).sum()
        rsi_signal_count = (rsi_signals != 0).sum()
        ma_signal_count = (ma_signals != 0).sum()
        
        # Combined strategy should have fewer signals
        assert combined_signal_count <= min(rsi_signal_count, ma_signal_count), \
            "Combined strategy should have fewer signals than individual strategies"
    
    @pytest.mark.mathematical
    def test_macd_rsi_signal_quality(self):
        """Test MACD RSI signal quality."""
        signals = self.strategy.generate_signals(self.test_data)
        
        # Calculate performance metrics
        returns = self.test_data['close'].pct_change()
        signal_returns = []
        
        for i in range(1, len(signals)):
            if signals.iloc[i-1] != 0:  # If there was a signal
                signal_returns.append(returns.iloc[i] * signals.iloc[i-1])
        
        if signal_returns:
            avg_signal_return = np.mean(signal_returns)
            # Signal returns should be positive on average for a good strategy
            assert avg_signal_return > -0.01, f"Poor signal quality: {avg_signal_return}"


class TestBollingerBreakoutStrategy:
    """Test Bollinger Breakout strategy mathematical correctness."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataValidator(tolerance=MATHEMATICAL_TOLERANCE)
        self.performance_monitor = PerformanceMonitor()
        self.strategy = BollingerBreakout()
        
        # Create test data with volatility periods
        np.random.seed(42)
        self.test_data = self._generate_volatility_data()
        
        # Create validation suite
        self.validation_suite = self.validator.create_suite("Bollinger Breakout Mathematical Validation")
    
    def _generate_volatility_data(self) -> pd.DataFrame:
        """Generate data with varying volatility periods."""
        # Low volatility period
        low_vol = np.random.normal(100, 1, 300)
        # High volatility period
        high_vol = np.random.normal(100, 5, 300)
        # Breakout period
        breakout = np.linspace(100, 120, 200)
        # Breakdown period
        breakdown = np.linspace(120, 80, 200)
        
        prices = np.concatenate([low_vol, high_vol, breakout, breakdown])
        
        return pd.DataFrame({
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 1000)
        }, index=pd.date_range('2023-01-01', periods=1000))
    
    @pytest.mark.mathematical
    def test_bollinger_breakout_logic(self):
        """Test Bollinger Breakout signal generation logic."""
        with PerformanceProfile(self.performance_monitor, "Bollinger Breakout Signals"):
            signals = self.strategy.generate_signals(self.test_data)
        
        # Validate signal properties
        result = self.validator.validate_trading_signals(
            signals=signals,
            name="Bollinger Breakout Signals",
            suite_name=self.validation_suite.name
        )
        
        assert result.passed, f"Bollinger Breakout signals failed validation: {result.message}"
        
        # Test breakout logic
        from src.indicators.volatility import BollingerBands
        bb_calc = BollingerBands()
        bb = bb_calc.calculate(self.test_data['close'], period=20, std_dev=2.0)
        
        buy_signals = signals[signals == 1]
        sell_signals = signals[signals == -1]
        
        # Buy signals should occur when price breaks above upper band
        for idx in buy_signals.index:
            if idx > 20:  # After BB is calculated
                price = self.test_data['close'].loc[idx]
                upper_band = bb['upper'].loc[idx]
                assert price >= upper_band * 0.999, \
                    f"Buy signal but price not above upper band at {idx}: {price} vs {upper_band}"
        
        # Sell signals should occur when price breaks below lower band
        for idx in sell_signals.index:
            if idx > 20:  # After BB is calculated
                price = self.test_data['close'].loc[idx]
                lower_band = bb['lower'].loc[idx]
                assert price <= lower_band * 1.001, \
                    f"Sell signal but price not below lower band at {idx}: {price} vs {lower_band}"


class TestSRBreakoutStrategy:
    """Test Support/Resistance Breakout strategy mathematical correctness."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataValidator(tolerance=MATHEMATICAL_TOLERANCE)
        self.performance_monitor = PerformanceMonitor()
        self.strategy = SRBreakout()
        
        # Create test data with clear support/resistance levels
        np.random.seed(42)
        self.test_data = self._generate_sr_data()
        
        # Create validation suite
        self.validation_suite = self.validator.create_suite("SR Breakout Mathematical Validation")
    
    def _generate_sr_data(self) -> pd.DataFrame:
        """Generate data with clear support/resistance levels."""
        # Create price action with clear levels
        base_price = 100
        prices = []
        
        # Test resistance at 110
        for i in range(200):
            if i < 50:
                prices.append(base_price + np.random.normal(0, 2))
            elif i < 100:
                prices.append(min(109, base_price + np.random.normal(5, 2)))
            elif i < 150:
                prices.append(base_price + np.random.normal(8, 2))
            else:
                prices.append(base_price + np.random.normal(12, 2))  # Breakout
        
        # Test support at 90
        for i in range(200):
            if i < 50:
                prices.append(base_price + np.random.normal(0, 2))
            elif i < 100:
                prices.append(max(91, base_price + np.random.normal(-5, 2)))
            elif i < 150:
                prices.append(base_price + np.random.normal(-8, 2))
            else:
                prices.append(base_price + np.random.normal(-12, 2))  # Breakdown
        
        # Add some normal trading
        for i in range(600):
            prices.append(base_price + np.random.normal(0, 5))
        
        return pd.DataFrame({
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, 1000)
        }, index=pd.date_range('2023-01-01', periods=1000))
    
    @pytest.mark.mathematical
    def test_sr_breakout_logic(self):
        """Test Support/Resistance Breakout signal generation logic."""
        with PerformanceProfile(self.performance_monitor, "SR Breakout Signals"):
            signals = self.strategy.generate_signals(self.test_data)
        
        # Validate signal properties
        result = self.validator.validate_trading_signals(
            signals=signals,
            name="SR Breakout Signals",
            suite_name=self.validation_suite.name
        )
        
        assert result.passed, f"SR Breakout signals failed validation: {result.message}"
        
        # Test that signals are generated at reasonable frequency
        signal_count = (signals != 0).sum()
        assert signal_count > 0, "No signals generated"
        assert signal_count < len(signals) * 0.2, "Too many signals generated"


class TestStrategyComparison:
    """Test mathematical consistency across all strategies."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataValidator(tolerance=MATHEMATICAL_TOLERANCE)
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize all strategies
        self.strategies = {
            'MA_Crossover': MACrossover(),
            'RSI_Trend': RSITrend(),
            'MACD_RSI': MACDRSIStrategy(),
            'Bollinger_Breakout': BollingerBreakout(),
            'SR_Breakout': SRBreakout()
        }
        
        # Create test data
        np.random.seed(42)
        self.test_data = self._generate_comprehensive_data()
        
        # Create validation suite
        self.validation_suite = self.validator.create_suite("Strategy Comparison Mathematical Validation")
    
    def _generate_comprehensive_data(self) -> pd.DataFrame:
        """Generate comprehensive test data covering multiple market scenarios."""
        # Different market phases
        uptrend = np.linspace(100, 130, 200)
        sideways = np.random.normal(130, 2, 200)
        downtrend = np.linspace(130, 90, 200)
        volatile = np.random.normal(90, 8, 200)
        recovery = np.linspace(90, 110, 200)
        
        prices = np.concatenate([uptrend, sideways, downtrend, volatile, recovery])
        
        return pd.DataFrame({
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 1000)
        }, index=pd.date_range('2023-01-01', periods=1000))
    
    @pytest.mark.mathematical
    def test_all_strategies_signal_generation(self):
        """Test that all strategies generate valid signals."""
        strategy_results = {}
        
        for name, strategy in self.strategies.items():
            with PerformanceProfile(self.performance_monitor, f"{name} Strategy"):
                signals = strategy.generate_signals(self.test_data)
                strategy_results[name] = signals
            
            # Validate each strategy's signals
            result = self.validator.validate_trading_signals(
                signals=signals,
                name=f"{name} Strategy Signals",
                suite_name=self.validation_suite.name
            )
            
            assert result.passed, f"{name} strategy signals failed validation: {result.message}"
    
    @pytest.mark.mathematical
    def test_strategy_performance_consistency(self):
        """Test performance consistency across strategies."""
        performance_metrics = {}
        
        for name, strategy in self.strategies.items():
            signals = strategy.generate_signals(self.test_data)
            
            # Calculate basic performance metrics
            returns = self.test_data['close'].pct_change()
            signal_returns = []
            
            for i in range(1, len(signals)):
                if signals.iloc[i-1] != 0:
                    signal_returns.append(returns.iloc[i] * signals.iloc[i-1])
            
            if signal_returns:
                performance_metrics[name] = {
                    'avg_return': np.mean(signal_returns),
                    'std_return': np.std(signal_returns),
                    'signal_count': len(signal_returns),
                    'win_rate': np.mean([r > 0 for r in signal_returns])
                }
        
        # Test that all strategies have reasonable performance characteristics
        for name, metrics in performance_metrics.items():
            assert metrics['signal_count'] > 0, f"{name} generated no signals"
            assert metrics['avg_return'] > -0.05, f"{name} has very poor returns: {metrics['avg_return']}"
            assert metrics['win_rate'] > 0.2, f"{name} has very low win rate: {metrics['win_rate']}"
    
    @pytest.mark.mathematical
    @pytest.mark.performance
    def test_strategy_computational_efficiency(self):
        """Test computational efficiency of all strategies."""
        import time
        
        # Test with larger dataset
        large_data = pd.DataFrame({
            'open': np.random.normal(100, 10, 5000),
            'high': np.random.normal(105, 10, 5000),
            'low': np.random.normal(95, 10, 5000),
            'close': np.random.normal(100, 10, 5000),
            'volume': np.random.randint(1000, 10000, 5000)
        })
        
        for name, strategy in self.strategies.items():
            start_time = time.time()
            
            with PerformanceProfile(self.performance_monitor, f"{name} Large Dataset"):
                signals = strategy.generate_signals(large_data)
            
            calculation_time = time.time() - start_time
            
            # Performance assertions
            assert calculation_time < 30.0, f"{name} too slow: {calculation_time:.2f}s"
            assert len(signals) == len(large_data), f"{name} signal length mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])