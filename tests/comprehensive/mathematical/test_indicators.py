"""
Mathematical validation tests for technical indicators.

This module tests the mathematical correctness of all technical indicators
used in the trading system, including RSI, MACD, Bollinger Bands, and moving averages.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import talib  # For cross-validation

from src.indicators.momentum import RSICalculator
from src.indicators.moving_averages import SimpleMovingAverage, ExponentialMovingAverage
from src.indicators.volatility import BollingerBands
from src.indicators.calculator import IndicatorCalculator
from src.core.constants import *
from ..utils.data_validator import DataValidator, ValidationSeverity
from ..utils.performance_monitor import PerformanceMonitor, PerformanceProfile
from . import MATHEMATICAL_TOLERANCE, INDICATOR_TOLERANCE, INDICATOR_RANGES


class TestRSICalculator:
    """Test RSI calculation mathematical correctness."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataValidator(tolerance=INDICATOR_TOLERANCE)
        self.performance_monitor = PerformanceMonitor()
        self.rsi_calculator = RSICalculator()
        
        # Create test data
        np.random.seed(42)
        self.test_data = self._generate_test_data()
        
        # Create validation suite
        self.validation_suite = self.validator.create_suite("RSI Mathematical Validation")
    
    def _generate_test_data(self) -> pd.Series:
        """Generate realistic price data for testing."""
        # Generate correlated price movements
        returns = np.random.normal(0.001, 0.02, 1000)  # 0.1% daily return, 2% volatility
        prices = [100]  # Starting price
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        return pd.Series(prices, index=pd.date_range('2023-01-01', periods=len(prices)))
    
    @pytest.mark.mathematical
    def test_rsi_calculation_accuracy(self):
        """Test RSI calculation accuracy against TA-Lib."""
        with PerformanceProfile(self.performance_monitor, "RSI Calculation"):
            # Calculate RSI using our implementation
            our_rsi = self.rsi_calculator.calculate(self.test_data, period=14)
            
            # Calculate RSI using TA-Lib for validation
            talib_rsi = talib.RSI(self.test_data.values, timeperiod=14)
            talib_rsi_series = pd.Series(talib_rsi, index=self.test_data.index)
            
            # Remove NaN values for comparison
            valid_indices = ~(our_rsi.isna() | talib_rsi_series.isna())
            our_rsi_clean = our_rsi[valid_indices]
            talib_rsi_clean = talib_rsi_series[valid_indices]
        
        # Validate numerical accuracy
        result = self.validator.validate_numerical_equality(
            expected=talib_rsi_clean.values,
            actual=our_rsi_clean.values,
            name="RSI vs TA-Lib",
            tolerance=INDICATOR_TOLERANCE,
            suite_name=self.validation_suite.name
        )
        
        assert result.passed, f"RSI calculation failed: {result.message}"
    
    @pytest.mark.mathematical
    def test_rsi_boundary_conditions(self):
        """Test RSI boundary conditions and properties."""
        # Test with extreme data
        extreme_up = pd.Series([100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200])
        extreme_down = pd.Series([200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100])
        
        rsi_up = self.rsi_calculator.calculate(extreme_up, period=5)
        rsi_down = self.rsi_calculator.calculate(extreme_down, period=5)
        
        # Validate RSI properties
        result_up = self.validator.validate_indicator_properties(
            indicator_values=rsi_up,
            name="RSI Extreme Up",
            min_value=0,
            max_value=100,
            should_be_bounded=True,
            suite_name=self.validation_suite.name
        )
        
        result_down = self.validator.validate_indicator_properties(
            indicator_values=rsi_down,
            name="RSI Extreme Down",
            min_value=0,
            max_value=100,
            should_be_bounded=True,
            suite_name=self.validation_suite.name
        )
        
        assert result_up.passed, f"RSI extreme up failed: {result_up.message}"
        assert result_down.passed, f"RSI extreme down failed: {result_down.message}"
        
        # RSI should be high for consistently rising prices
        assert rsi_up.iloc[-1] > 80, "RSI should be high for consistently rising prices"
        # RSI should be low for consistently falling prices
        assert rsi_down.iloc[-1] < 20, "RSI should be low for consistently falling prices"
    
    @pytest.mark.mathematical
    def test_rsi_period_variations(self):
        """Test RSI calculation with different periods."""
        periods = [5, 14, 21, 30]
        
        for period in periods:
            with PerformanceProfile(self.performance_monitor, f"RSI Period {period}"):
                rsi = self.rsi_calculator.calculate(self.test_data, period=period)
            
            # Validate RSI properties for each period
            result = self.validator.validate_indicator_properties(
                indicator_values=rsi,
                name=f"RSI Period {period}",
                min_value=0,
                max_value=100,
                should_be_bounded=True,
                suite_name=self.validation_suite.name
            )
            
            assert result.passed, f"RSI period {period} failed: {result.message}"
            
            # Check that we have the expected number of valid values
            expected_valid = len(self.test_data) - period
            actual_valid = rsi.notna().sum()
            assert actual_valid >= expected_valid * 0.9, f"Too few valid RSI values for period {period}"
    
    @pytest.mark.mathematical
    def test_rsi_mathematical_properties(self):
        """Test mathematical properties of RSI."""
        rsi = self.rsi_calculator.calculate(self.test_data, period=14)
        
        # Test that RSI is bounded between 0 and 100
        assert rsi.min() >= 0, "RSI should not be below 0"
        assert rsi.max() <= 100, "RSI should not be above 100"
        
        # Test that RSI responds to price changes
        # Create a simple up-trend and down-trend
        up_trend = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
        down_trend = pd.Series([110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100])
        
        rsi_up = self.rsi_calculator.calculate(up_trend, period=5)
        rsi_down = self.rsi_calculator.calculate(down_trend, period=5)
        
        # RSI should be higher for uptrend than downtrend
        assert rsi_up.iloc[-1] > rsi_down.iloc[-1], "RSI should be higher for uptrend"
        
        # Test RSI continuity (no sudden jumps)
        rsi_diff = rsi.diff().abs()
        max_jump = rsi_diff.max()
        assert max_jump < 50, f"RSI has too large jumps: {max_jump}"


class TestMovingAverages:
    """Test moving average calculations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataValidator(tolerance=MATHEMATICAL_TOLERANCE)
        self.performance_monitor = PerformanceMonitor()
        self.sma_calculator = SimpleMovingAverage()
        self.ema_calculator = ExponentialMovingAverage()
        
        # Create test data
        np.random.seed(42)
        self.test_data = pd.Series(
            np.random.normal(100, 10, 1000),
            index=pd.date_range('2023-01-01', periods=1000)
        )
        
        # Create validation suite
        self.validation_suite = self.validator.create_suite("Moving Averages Mathematical Validation")
    
    @pytest.mark.mathematical
    def test_sma_calculation_accuracy(self):
        """Test SMA calculation accuracy."""
        period = 20
        
        with PerformanceProfile(self.performance_monitor, "SMA Calculation"):
            our_sma = self.sma_calculator.calculate(self.test_data, period=period)
            
            # Calculate SMA using pandas rolling for validation
            pandas_sma = self.test_data.rolling(window=period).mean()
        
        # Validate numerical accuracy
        result = self.validator.validate_numerical_equality(
            expected=pandas_sma.values,
            actual=our_sma.values,
            name="SMA vs Pandas",
            tolerance=MATHEMATICAL_TOLERANCE,
            suite_name=self.validation_suite.name
        )
        
        assert result.passed, f"SMA calculation failed: {result.message}"
    
    @pytest.mark.mathematical
    def test_ema_calculation_accuracy(self):
        """Test EMA calculation accuracy."""
        period = 20
        
        with PerformanceProfile(self.performance_monitor, "EMA Calculation"):
            our_ema = self.ema_calculator.calculate(self.test_data, period=period)
            
            # Calculate EMA using pandas ewm for validation
            pandas_ema = self.test_data.ewm(span=period).mean()
        
        # Validate numerical accuracy
        result = self.validator.validate_numerical_equality(
            expected=pandas_ema.values,
            actual=our_ema.values,
            name="EMA vs Pandas",
            tolerance=MATHEMATICAL_TOLERANCE,
            suite_name=self.validation_suite.name
        )
        
        assert result.passed, f"EMA calculation failed: {result.message}"
    
    @pytest.mark.mathematical
    def test_moving_average_properties(self):
        """Test mathematical properties of moving averages."""
        period = 20
        
        sma = self.sma_calculator.calculate(self.test_data, period=period)
        ema = self.ema_calculator.calculate(self.test_data, period=period)
        
        # Test that moving averages smooth the data
        sma_volatility = sma.std()
        ema_volatility = ema.std()
        data_volatility = self.test_data.std()
        
        assert sma_volatility < data_volatility, "SMA should smooth the data"
        assert ema_volatility < data_volatility, "EMA should smooth the data"
        
        # Test that EMA responds faster than SMA to changes
        # Create a step function in the data
        step_data = pd.Series([100] * 50 + [110] * 50)
        sma_step = self.sma_calculator.calculate(step_data, period=10)
        ema_step = self.ema_calculator.calculate(step_data, period=10)
        
        # After the step, EMA should reach the new level faster
        sma_final = sma_step.iloc[-1]
        ema_final = ema_step.iloc[-1]
        
        assert ema_final > sma_final, "EMA should respond faster to changes than SMA"


class TestBollingerBands:
    """Test Bollinger Bands calculations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataValidator(tolerance=MATHEMATICAL_TOLERANCE)
        self.performance_monitor = PerformanceMonitor()
        self.bb_calculator = BollingerBands()
        
        # Create test data
        np.random.seed(42)
        self.test_data = pd.Series(
            np.random.normal(100, 10, 1000),
            index=pd.date_range('2023-01-01', periods=1000)
        )
        
        # Create validation suite
        self.validation_suite = self.validator.create_suite("Bollinger Bands Mathematical Validation")
    
    @pytest.mark.mathematical
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation accuracy."""
        period = 20
        std_dev = 2.0
        
        with PerformanceProfile(self.performance_monitor, "Bollinger Bands Calculation"):
            bb_result = self.bb_calculator.calculate(self.test_data, period=period, std_dev=std_dev)
            
            # Calculate using pandas for validation
            sma = self.test_data.rolling(window=period).mean()
            std = self.test_data.rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
        
        # Validate each band
        for band_name, expected, actual in [
            ("Middle Band", sma.values, bb_result['middle'].values),
            ("Upper Band", upper_band.values, bb_result['upper'].values),
            ("Lower Band", lower_band.values, bb_result['lower'].values)
        ]:
            result = self.validator.validate_numerical_equality(
                expected=expected,
                actual=actual,
                name=f"Bollinger {band_name}",
                tolerance=MATHEMATICAL_TOLERANCE,
                suite_name=self.validation_suite.name
            )
            
            assert result.passed, f"Bollinger {band_name} calculation failed: {result.message}"
    
    @pytest.mark.mathematical
    def test_bollinger_bands_properties(self):
        """Test mathematical properties of Bollinger Bands."""
        period = 20
        std_dev = 2.0
        
        bb_result = self.bb_calculator.calculate(self.test_data, period=period, std_dev=std_dev)
        
        # Test that upper band is always above middle band
        assert (bb_result['upper'] >= bb_result['middle']).all(), "Upper band should be above middle band"
        
        # Test that lower band is always below middle band
        assert (bb_result['lower'] <= bb_result['middle']).all(), "Lower band should be below middle band"
        
        # Test that bands widen during high volatility
        volatility = self.test_data.rolling(window=period).std()
        band_width = bb_result['upper'] - bb_result['lower']
        
        # Calculate correlation between volatility and band width
        correlation = volatility.corr(band_width)
        assert correlation > 0.8, f"Band width should correlate with volatility: {correlation}"


class TestIndicatorCalculator:
    """Test the main indicator calculator orchestration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataValidator(tolerance=MATHEMATICAL_TOLERANCE)
        self.performance_monitor = PerformanceMonitor()
        self.calculator = IndicatorCalculator()
        
        # Create test data
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'open': np.random.normal(100, 10, 1000),
            'high': np.random.normal(105, 10, 1000),
            'low': np.random.normal(95, 10, 1000),
            'close': np.random.normal(100, 10, 1000),
            'volume': np.random.randint(1000, 10000, 1000)
        })
        
        # Ensure OHLC relationships
        self.test_data['high'] = np.maximum(
            self.test_data[['open', 'close']].max(axis=1),
            self.test_data['high']
        )
        self.test_data['low'] = np.minimum(
            self.test_data[['open', 'close']].min(axis=1),
            self.test_data['low']
        )
        
        # Create validation suite
        self.validation_suite = self.validator.create_suite("Indicator Calculator Mathematical Validation")
    
    @pytest.mark.mathematical
    def test_all_indicators_calculation(self):
        """Test calculation of all indicators together."""
        with PerformanceProfile(self.performance_monitor, "All Indicators Calculation"):
            indicators = self.calculator.calculate_all(self.test_data)
        
        # Validate each indicator
        for indicator_name, values in indicators.items():
            if indicator_name in INDICATOR_RANGES:
                min_val, max_val = INDICATOR_RANGES[indicator_name]
                
                result = self.validator.validate_indicator_properties(
                    indicator_values=values,
                    name=f"Indicator {indicator_name}",
                    min_value=min_val if min_val != 0 else None,
                    max_value=max_val if max_val != float('inf') else None,
                    should_be_bounded=min_val != 0 or max_val != float('inf'),
                    suite_name=self.validation_suite.name
                )
                
                assert result.passed, f"Indicator {indicator_name} validation failed: {result.message}"
    
    @pytest.mark.mathematical
    @pytest.mark.performance
    def test_indicator_performance(self):
        """Test indicator calculation performance."""
        import time
        
        # Test with larger dataset
        large_data = pd.DataFrame({
            'open': np.random.normal(100, 10, 10000),
            'high': np.random.normal(105, 10, 10000),
            'low': np.random.normal(95, 10, 10000),
            'close': np.random.normal(100, 10, 10000),
            'volume': np.random.randint(1000, 10000, 10000)
        })
        
        start_time = time.time()
        
        with PerformanceProfile(self.performance_monitor, "Large Dataset Indicators"):
            indicators = self.calculator.calculate_all(large_data)
        
        calculation_time = time.time() - start_time
        
        # Performance assertions
        assert calculation_time < 10.0, f"Indicator calculation too slow: {calculation_time:.2f}s"
        assert len(indicators) > 0, "No indicators calculated"
        
        # Check that all indicators have reasonable number of values
        for indicator_name, values in indicators.items():
            assert len(values) > len(large_data) * 0.8, f"Too few values for {indicator_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])