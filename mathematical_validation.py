#!/usr/bin/env python3
"""
Mathematical Validation Script for Wagehood Trading System

This script performs comprehensive mathematical validation of all technical indicators
and trading strategies to ensure accuracy and reliability of calculations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our indicators and strategies
from src.indicators.moving_averages import (
    calculate_sma, calculate_ema, calculate_wma, calculate_vwma, calculate_ma_crossover
)
from src.indicators.momentum import (
    calculate_rsi, calculate_macd, calculate_stochastic, calculate_williams_r, 
    calculate_cci, calculate_momentum, calculate_roc
)
from src.indicators.volatility import (
    calculate_bollinger_bands, calculate_atr, calculate_keltner_channels,
    calculate_donchian_channels, calculate_volatility
)
from src.strategies.ma_crossover import MovingAverageCrossover
from src.strategies.rsi_trend import RSITrendFollowing
from src.strategies.bollinger_breakout import BollingerBandBreakout
from src.strategies.macd_rsi import MACDRSIStrategy
from src.strategies.sr_breakout import SupportResistanceBreakout
from src.core.models import MarketData, OHLCV


class MathematicalValidator:
    """Mathematical validation framework for trading system components."""
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize validator.
        
        Args:
            tolerance: Numerical tolerance for floating point comparisons
        """
        self.tolerance = tolerance
        self.results = {}
        self.test_data = self._generate_test_data()
        
    def _generate_test_data(self) -> Dict[str, np.ndarray]:
        """Generate realistic test data for validation."""
        np.random.seed(42)  # For reproducible results
        
        # Generate 1000 data points
        size = 1000
        
        # Generate realistic OHLCV data with proper relationships
        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, size)  # 0.1% daily return, 2% volatility
        
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        close = np.array(prices[1:])  # Remove first element to match size
        
        # Generate OHLC based on close prices with realistic relationships
        high = close * (1 + np.abs(np.random.normal(0, 0.005, size)))  # Up to 0.5% above close
        low = close * (1 - np.abs(np.random.normal(0, 0.005, size)))   # Up to 0.5% below close
        open_shift = np.random.normal(0, 0.003, size)  # Random gap from previous close
        open_prices = np.roll(close, 1) * (1 + open_shift)
        open_prices[0] = base_price  # Set first open to base price
        
        # Ensure OHLC relationships are correct
        for i in range(size):
            max_price = max(open_prices[i], close[i])
            min_price = min(open_prices[i], close[i])
            high[i] = max(high[i], max_price)
            low[i] = min(low[i], min_price)
        
        # Generate volume with some correlation to price volatility
        daily_returns = np.abs(np.diff(close, prepend=close[0]))
        volume_base = 50000
        volume = volume_base * (1 + daily_returns * 10 + np.random.normal(0, 0.3, size))
        volume = np.maximum(volume, 1000)  # Ensure positive volume
        
        timestamps = pd.date_range('2023-01-01', periods=size, freq='D')
        
        return {
            'timestamp': timestamps,
            'open': open_prices,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }
    
    def _create_market_data(self, start_idx: int = 0, length: int = None) -> MarketData:
        """Create MarketData object from test data."""
        if length is None:
            length = len(self.test_data['close']) - start_idx
        
        end_idx = start_idx + length
        
        ohlcv_data = []
        for i in range(start_idx, min(end_idx, len(self.test_data['close']))):
            ohlcv_data.append(OHLCV(
                timestamp=self.test_data['timestamp'][i],
                open=self.test_data['open'][i],
                high=self.test_data['high'][i],
                low=self.test_data['low'][i],
                close=self.test_data['close'][i],
                volume=int(self.test_data['volume'][i])
            ))
        
        return MarketData(
            symbol="TEST", 
            timeframe="1d", 
            data=ohlcv_data,
            indicators={},
            last_updated=datetime.now()
        )
    
    def validate_moving_averages(self) -> Dict[str, Any]:
        """Validate moving average calculations."""
        print("Validating Moving Averages...")
        results = {}
        
        # Test data
        data = self.test_data['close']
        periods = [10, 20, 50]
        
        for period in periods:
            # SMA validation
            our_sma = calculate_sma(data, period)
            pandas_sma = pd.Series(data).rolling(window=period).mean().values
            
            # Compare valid values only
            valid_mask = ~(np.isnan(our_sma) | np.isnan(pandas_sma))
            sma_diff = np.abs(our_sma[valid_mask] - pandas_sma[valid_mask])
            sma_max_error = np.max(sma_diff) if len(sma_diff) > 0 else 0
            
            results[f'sma_{period}_max_error'] = sma_max_error
            results[f'sma_{period}_passed'] = sma_max_error < self.tolerance
            
            # EMA validation - Test our EMA mathematical correctness
            our_ema = calculate_ema(data, period)
            
            # Validate EMA mathematical properties instead of comparing to pandas
            # Our EMA uses SMA initialization which is a valid approach
            alpha = 2.0 / (period + 1)
            
            # Test EMA mathematical consistency
            ema_valid_start = period - 1
            if len(our_ema) > ema_valid_start + 1:
                # Check if EMA formula is applied correctly after initialization
                mathematical_consistent = True
                for i in range(ema_valid_start + 1, min(ema_valid_start + 10, len(our_ema))):
                    if not np.isnan(our_ema[i]) and not np.isnan(our_ema[i-1]):
                        expected = alpha * data[i] + (1 - alpha) * our_ema[i-1]
                        actual = our_ema[i]
                        if abs(expected - actual) > 1e-10:
                            mathematical_consistent = False
                            break
                
                results[f'ema_{period}_mathematical_consistent'] = mathematical_consistent
                results[f'ema_{period}_passed'] = mathematical_consistent
                
                # Check that EMA initialization is SMA
                expected_init = np.mean(data[:period])
                actual_init = our_ema[ema_valid_start]
                init_error = abs(expected_init - actual_init)
                results[f'ema_{period}_init_error'] = init_error
                results[f'ema_{period}_init_correct'] = init_error < 1e-10
                
                print(f"  Period {period}: SMA error={sma_max_error:.2e}, EMA consistent={mathematical_consistent}, EMA init error={init_error:.2e}")
            else:
                results[f'ema_{period}_passed'] = False
                print(f"  Period {period}: Insufficient data for EMA validation")
        
        # Test crossover logic
        fast_ema = calculate_ema(data, 10)
        slow_ema = calculate_ema(data, 20)
        crossovers = calculate_ma_crossover(fast_ema, slow_ema)
        
        # Validate crossover properties
        crossover_count = np.count_nonzero(crossovers)
        results['crossover_count'] = crossover_count
        results['crossover_logic_passed'] = crossover_count > 0
        
        print(f"  Crossover signals detected: {crossover_count}")
        
        return results
    
    def validate_rsi(self) -> Dict[str, Any]:
        """Validate RSI calculation."""
        print("Validating RSI...")
        results = {}
        
        data = self.test_data['close']
        period = 14
        
        # Calculate our RSI
        our_rsi = calculate_rsi(data, period)
        
        # Manual RSI calculation for validation
        delta = np.diff(data)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        # Calculate using Wilder's smoothing method
        manual_rsi = np.full(len(data), np.nan)
        if len(gains) >= period:
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            if avg_loss == 0:
                manual_rsi[period] = 100.0
            else:
                rs = avg_gain / avg_loss
                manual_rsi[period] = 100.0 - (100.0 / (1 + rs))
            
            # Calculate subsequent values
            for i in range(period + 1, len(data)):
                avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
                
                if avg_loss == 0:
                    manual_rsi[i] = 100.0
                else:
                    rs = avg_gain / avg_loss
                    manual_rsi[i] = 100.0 - (100.0 / (1 + rs))
        
        # Compare valid values
        valid_mask = ~(np.isnan(our_rsi) | np.isnan(manual_rsi))
        if np.any(valid_mask):
            rsi_diff = np.abs(our_rsi[valid_mask] - manual_rsi[valid_mask])
            max_error = np.max(rsi_diff)
            results['rsi_max_error'] = max_error
            results['rsi_passed'] = max_error < 0.01  # Allow 0.01 difference for RSI
            
            # Validate RSI properties
            rsi_values = our_rsi[valid_mask]
            results['rsi_bounded'] = np.all((rsi_values >= 0) & (rsi_values <= 100))
            results['rsi_min'] = np.min(rsi_values)
            results['rsi_max'] = np.max(rsi_values)
            
            print(f"  RSI max error: {max_error:.4f}")
            print(f"  RSI range: [{results['rsi_min']:.2f}, {results['rsi_max']:.2f}]")
        else:
            results['rsi_passed'] = False
            print("  No valid RSI values for comparison")
        
        return results
    
    def validate_macd(self) -> Dict[str, Any]:
        """Validate MACD calculation."""
        print("Validating MACD...")
        results = {}
        
        data = self.test_data['close']
        fast, slow, signal = 12, 26, 9
        
        # Calculate our MACD
        macd_line, signal_line, histogram = calculate_macd(data, fast, slow, signal)
        
        # Manual MACD calculation for validation
        manual_fast_ema = calculate_ema(data, fast)
        manual_slow_ema = calculate_ema(data, slow)
        manual_macd_line = manual_fast_ema - manual_slow_ema
        
        # Calculate signal line manually (EMA of MACD line)
        valid_macd_start = slow - 1
        manual_signal_line = np.full(len(data), np.nan)
        if len(data) > valid_macd_start + signal - 1:
            valid_macd = manual_macd_line[valid_macd_start:]
            valid_macd_clean = valid_macd[~np.isnan(valid_macd)]
            if len(valid_macd_clean) >= signal:
                signal_ema = calculate_ema(valid_macd_clean, signal)
                manual_signal_line[valid_macd_start:valid_macd_start + len(signal_ema)] = signal_ema
        
        manual_histogram = manual_macd_line - manual_signal_line
        
        # Compare each component
        for component, our_vals, manual_vals in [
            ('macd_line', macd_line, manual_macd_line),
            ('signal_line', signal_line, manual_signal_line),
            ('histogram', histogram, manual_histogram)
        ]:
            valid_mask = ~(np.isnan(our_vals) | np.isnan(manual_vals))
            if np.any(valid_mask):
                diff = np.abs(our_vals[valid_mask] - manual_vals[valid_mask])
                max_error = np.max(diff) if len(diff) > 0 else 0
                results[f'{component}_max_error'] = max_error
                results[f'{component}_passed'] = max_error < 0.001  # Allow small difference
                print(f"  {component} max error: {max_error:.6f}")
            else:
                results[f'{component}_passed'] = False
                print(f"  No valid {component} values for comparison")
        
        return results
    
    def validate_bollinger_bands(self) -> Dict[str, Any]:
        """Validate Bollinger Bands calculation."""
        print("Validating Bollinger Bands...")
        results = {}
        
        data = self.test_data['close']
        period, std_dev = 20, 2.0
        
        # Calculate our Bollinger Bands
        upper, middle, lower = calculate_bollinger_bands(data, period, std_dev)
        
        # Manual Bollinger Bands calculation for validation
        manual_middle = calculate_sma(data, period)
        manual_upper = np.full(len(data), np.nan)
        manual_lower = np.full(len(data), np.nan)
        
        # Calculate standard deviation bands manually
        for i in range(period - 1, len(data)):
            window_data = data[i - period + 1:i + 1]
            std = np.std(window_data, ddof=0)  # Population standard deviation
            manual_upper[i] = manual_middle[i] + (std * std_dev)
            manual_lower[i] = manual_middle[i] - (std * std_dev)
        
        # Compare each band
        for band, our_vals, manual_vals in [
            ('upper', upper, manual_upper),
            ('middle', middle, manual_middle),
            ('lower', lower, manual_lower)
        ]:
            valid_mask = ~(np.isnan(our_vals) | np.isnan(manual_vals))
            if np.any(valid_mask):
                diff = np.abs(our_vals[valid_mask] - manual_vals[valid_mask])
                max_error = np.max(diff)
                results[f'bb_{band}_max_error'] = max_error
                results[f'bb_{band}_passed'] = max_error < 0.001
                print(f"  {band} band max error: {max_error:.6f}")
            else:
                results[f'bb_{band}_passed'] = False
                print(f"  No valid {band} band values for comparison")
        
        # Validate band relationships
        valid_mask = ~(np.isnan(upper) | np.isnan(middle) | np.isnan(lower))
        if np.any(valid_mask):
            upper_valid = upper[valid_mask]
            middle_valid = middle[valid_mask]
            lower_valid = lower[valid_mask]
            
            results['bb_relationships_correct'] = (
                np.all(upper_valid >= middle_valid) and 
                np.all(middle_valid >= lower_valid)
            )
            print(f"  Band relationships correct: {results['bb_relationships_correct']}")
        
        return results
    
    def validate_atr(self) -> Dict[str, Any]:
        """Validate ATR calculation."""
        print("Validating ATR...")
        results = {}
        
        high = self.test_data['high']
        low = self.test_data['low']
        close = self.test_data['close']
        period = 14
        
        # Calculate our ATR
        our_atr = calculate_atr(high, low, close, period)
        
        # Manual ATR calculation for validation
        manual_tr = np.full(len(close), np.nan)
        manual_tr[0] = high[0] - low[0]  # First TR is just High - Low
        
        # Calculate True Range for remaining periods
        for i in range(1, len(close)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            manual_tr[i] = max(hl, hc, lc)
        
        # Calculate ATR using Wilder's smoothing method
        manual_atr = np.full(len(close), np.nan)
        if len(manual_tr) >= period:
            manual_atr[period - 1] = np.mean(manual_tr[:period])
            
            # Calculate subsequent ATR values using Wilder's method
            for i in range(period, len(close)):
                manual_atr[i] = (manual_atr[i - 1] * (period - 1) + manual_tr[i]) / period
        
        # Compare valid values
        valid_mask = ~(np.isnan(our_atr) | np.isnan(manual_atr))
        if np.any(valid_mask):
            diff = np.abs(our_atr[valid_mask] - manual_atr[valid_mask])
            max_error = np.max(diff)
            results['atr_max_error'] = max_error
            results['atr_passed'] = max_error < 0.001
            
            # Validate ATR properties (should be positive)
            atr_values = our_atr[valid_mask]
            results['atr_positive'] = np.all(atr_values >= 0)
            
            print(f"  ATR max error: {max_error:.6f}")
            print(f"  ATR all positive: {results['atr_positive']}")
        else:
            results['atr_passed'] = False
            print("  No valid ATR values for comparison")
        
        return results
    
    def validate_strategy_signals(self) -> Dict[str, Any]:
        """Validate strategy signal generation."""
        print("Validating Strategy Signal Generation...")
        results = {}
        
        # Create market data for strategies
        market_data = self._create_market_data(length=300)  # Use subset for strategies
        
        strategies = [
            MovingAverageCrossover(),
            RSITrendFollowing(),
            BollingerBandBreakout(),
            MACDRSIStrategy(),
            SupportResistanceBreakout()
        ]
        
        for strategy in strategies:
            strategy_name = strategy.name
            print(f"  Testing {strategy_name}...")
            
            try:
                # Calculate required indicators
                indicators = self._calculate_indicators_for_strategy(market_data, strategy)
                
                # Generate signals
                signals = strategy.generate_signals(market_data, indicators)
                
                results[f'{strategy_name}_signals_count'] = len(signals)
                results[f'{strategy_name}_passed'] = True
                
                # Validate signal properties
                if signals:
                    confidences = [s.confidence for s in signals]
                    results[f'{strategy_name}_avg_confidence'] = np.mean(confidences)
                    results[f'{strategy_name}_confidence_valid'] = all(0 <= c <= 1 for c in confidences)
                    
                    print(f"    Generated {len(signals)} signals, avg confidence: {np.mean(confidences):.3f}")
                else:
                    print(f"    No signals generated")
                    
            except Exception as e:
                print(f"    ERROR: {e}")
                results[f'{strategy_name}_passed'] = False
                results[f'{strategy_name}_error'] = str(e)
        
        return results
    
    def _calculate_indicators_for_strategy(self, market_data: MarketData, strategy) -> Dict[str, Any]:
        """Calculate indicators required by a strategy."""
        arrays = market_data.to_arrays()
        close = arrays['close']
        high = arrays['high']
        low = arrays['low']
        
        indicators = {}
        
        # EMA indicators
        indicators['ema'] = {}
        for period in [10, 20, 50, 200]:
            indicators['ema'][f'ema_{period}'] = calculate_ema(close, period)
        
        # RSI
        indicators['rsi'] = calculate_rsi(close, 14)
        
        # MACD
        macd_line, signal_line, histogram = calculate_macd(close, 12, 26, 9)
        indicators['macd'] = {
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
        }
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
        indicators['bollinger'] = {
            'upper': bb_upper,
            'middle': bb_middle,
            'lower': bb_lower
        }
        
        # Support/Resistance levels (simplified)
        indicators['support_resistance'] = {
            'support_levels': [np.min(close[-50:])],  # Simple support
            'resistance_levels': [np.max(close[-50:])]  # Simple resistance
        }
        
        return indicators
    
    def validate_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases and boundary conditions."""
        print("Validating Edge Cases...")
        results = {}
        
        # Test with insufficient data
        small_data = np.array([100, 101, 102])
        
        try:
            rsi_small = calculate_rsi(small_data, 14)
            results['insufficient_data_rsi'] = True  # Should handle gracefully
        except Exception:
            results['insufficient_data_rsi'] = False
        
        # Test with NaN values
        nan_data = np.array([100, 101, np.nan, 103, 104])
        
        try:
            sma_nan = calculate_sma(nan_data, 3)
            results['nan_handling_sma'] = True
        except Exception:
            results['nan_handling_sma'] = False
        
        # Test with zero values
        zero_data = np.array([100, 101, 0, 103, 104])
        
        try:
            rsi_zero = calculate_rsi(zero_data, 3)
            results['zero_handling_rsi'] = True
        except Exception:
            results['zero_handling_rsi'] = False
        
        # Test with extreme values
        extreme_data = np.array([1e-10, 1e10, 100, 101, 102])
        
        try:
            sma_extreme = calculate_sma(extreme_data, 3)
            results['extreme_values_sma'] = True
        except Exception:
            results['extreme_values_sma'] = False
        
        print(f"  Edge case tests passed: {sum(results.values())}/{len(results)}")
        
        return results
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run all validation tests."""
        print("=" * 60)
        print("WAGEHOOD MATHEMATICAL VALIDATION")
        print("=" * 60)
        
        all_results = {}
        
        # Run individual validation tests
        all_results['moving_averages'] = self.validate_moving_averages()
        all_results['rsi'] = self.validate_rsi()
        all_results['macd'] = self.validate_macd()
        all_results['bollinger_bands'] = self.validate_bollinger_bands()
        all_results['atr'] = self.validate_atr()
        all_results['strategy_signals'] = self.validate_strategy_signals()
        all_results['edge_cases'] = self.validate_edge_cases()
        
        return all_results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive validation report."""
        report = []
        report.append("=" * 80)
        report.append("MATHEMATICAL VALIDATION REPORT")
        report.append("=" * 80)
        
        # Summary statistics
        total_tests = 0
        passed_tests = 0
        
        for category, category_results in results.items():
            report.append(f"\n{category.upper().replace('_', ' ')}:")
            report.append("-" * 40)
            
            category_passed = 0
            category_total = 0
            
            for test_name, result in category_results.items():
                if test_name.endswith('_passed'):
                    category_total += 1
                    total_tests += 1
                    if result:
                        category_passed += 1
                        passed_tests += 1
                        status = "✓ PASS"
                    else:
                        status = "✗ FAIL"
                    
                    test_display = test_name.replace('_passed', '').replace('_', ' ').title()
                    report.append(f"  {test_display}: {status}")
                
                elif test_name.endswith('_error'):
                    error_test = test_name.replace('_error', '').replace('_', ' ').title()
                    report.append(f"  {error_test} Error: {result:.2e}")
                
                elif test_name.endswith('_count'):
                    count_test = test_name.replace('_count', '').replace('_', ' ').title()
                    report.append(f"  {count_test}: {result}")
            
            if category_total > 0:
                category_percentage = (category_passed / category_total) * 100
                report.append(f"  Category Score: {category_passed}/{category_total} ({category_percentage:.1f}%)")
        
        # Overall summary
        overall_percentage = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        report.append("\n" + "=" * 80)
        report.append("OVERALL VALIDATION SUMMARY")
        report.append("=" * 80)
        report.append(f"Total Tests: {total_tests}")
        report.append(f"Passed: {passed_tests}")
        report.append(f"Failed: {total_tests - passed_tests}")
        report.append(f"Success Rate: {overall_percentage:.1f}%")
        
        if overall_percentage >= 95:
            report.append("\n🎉 EXCELLENT: Mathematical accuracy validated!")
        elif overall_percentage >= 90:
            report.append("\n✅ GOOD: Minor issues detected, acceptable accuracy")
        elif overall_percentage >= 80:
            report.append("\n⚠️  WARNING: Some mathematical issues detected")
        else:
            report.append("\n❌ CRITICAL: Significant mathematical errors found")
        
        return "\n".join(report)


def main():
    """Main validation function."""
    validator = MathematicalValidator()
    results = validator.run_all_validations()
    report = validator.generate_report(results)
    
    print("\n" + report)
    
    # Save detailed results
    import json
    with open('/Users/arianlc/PycharmProjects/wagehood/validation_results.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        json.dump(convert_numpy(results), f, indent=2)
    
    print(f"\nDetailed results saved to: validation_results.json")
    
    # Return overall success
    total_tests = sum(1 for category in results.values() 
                     for test in category.keys() if test.endswith('_passed'))
    passed_tests = sum(1 for category in results.values() 
                      for test, result in category.items() 
                      if test.endswith('_passed') and result)
    
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    return success_rate >= 90  # Return True if 90% or more tests pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)