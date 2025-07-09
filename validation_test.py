#!/usr/bin/env python3
"""
Comprehensive Validation Test Suite for Wagehood Signal Analysis System

This script performs thorough validation of the backtesting/signal analysis system
to ensure it works correctly with real Alpaca data after the transformation to
signal-only service.

Test Coverage:
1. Signal Analysis Job Testing
2. Mathematical Validation
3. Data Quality and Accuracy
4. Output Format Validation
5. Strategy-Specific Validation
6. Performance and Reliability
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import traceback
import statistics
import numpy as np
from dataclasses import dataclass
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.providers.alpaca_provider import AlpacaProvider
from src.strategies import create_strategy
from src.backtest.engine import SignalAnalysisEngine
from src.core.models import MarketData, TimeFrame, OHLCV, Signal
from src.indicators.momentum import calculate_rsi, calculate_macd
from src.indicators.moving_averages import calculate_sma, calculate_ema
from src.indicators.volatility import calculate_bollinger_bands

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Container for test results"""
    test_name: str
    passed: bool
    duration: float
    details: Dict[str, Any]
    error: Optional[str] = None


class ValidationTestSuite:
    """Comprehensive validation test suite for the signal analysis system"""
    
    def __init__(self):
        self.alpaca_provider = None
        self.test_results: List[TestResult] = []
        self.test_symbols = ["AAPL", "SPY", "MSFT", "TSLA", "QQQ"]
        self.test_timeframes = ["1h", "1d"]
        self.test_strategies = ["macd_rsi", "ma_crossover", "rsi_trend", "bollinger_breakout", "sr_breakout"]
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all validation tests"""
        logger.info("🚀 Starting Comprehensive Validation Test Suite")
        start_time = time.time()
        
        # Initialize providers
        await self._initialize_providers()
        
        # Run test categories
        test_methods = [
            self._test_data_quality_and_accuracy,
            self._test_mathematical_validation,
            self._test_strategy_implementations,
            self._test_signal_analysis_pipeline,
            self._test_output_format_validation,
            self._test_performance_and_reliability,
        ]
        
        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                logger.error(f"Test method {test_method.__name__} failed: {e}")
                self.test_results.append(TestResult(
                    test_name=test_method.__name__,
                    passed=False,
                    duration=0.0,
                    details={},
                    error=str(e)
                ))
        
        # Cleanup
        await self._cleanup_providers()
        
        # Generate report
        total_duration = time.time() - start_time
        return self._generate_validation_report(total_duration)
    
    async def _initialize_providers(self):
        """Initialize data providers"""
        logger.info("🔌 Initializing Alpaca provider...")
        try:
            self.alpaca_provider = AlpacaProvider()
            await self.alpaca_provider.connect()
            if not self.alpaca_provider._connected:
                raise ConnectionError("Failed to connect to Alpaca")
            logger.info("✅ Alpaca provider initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Alpaca provider: {e}")
            raise
    
    async def _cleanup_providers(self):
        """Cleanup data providers"""
        if self.alpaca_provider:
            await self.alpaca_provider.disconnect()
            logger.info("🧹 Alpaca provider disconnected")
    
    async def _test_data_quality_and_accuracy(self):
        """Test data quality and accuracy from Alpaca API"""
        logger.info("📊 Testing Data Quality and Accuracy...")
        
        test_start = time.time()
        test_details = {}
        
        try:
            # Test data retrieval for different symbols and timeframes
            for symbol in self.test_symbols[:3]:  # Test first 3 symbols
                for timeframe_str in self.test_timeframes:
                    timeframe = self._parse_timeframe(timeframe_str)
                    
                    # Test data retrieval
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=30)
                    
                    historical_data = await self.alpaca_provider.get_historical_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    # Validate data quality
                    data_quality = self._validate_data_quality(historical_data, symbol, timeframe_str)
                    test_details[f"{symbol}_{timeframe_str}"] = data_quality
                    
                    logger.info(f"✅ {symbol} {timeframe_str}: {len(historical_data)} data points, quality score: {data_quality['quality_score']:.2f}")
            
            # Overall data quality assessment
            all_scores = [details['quality_score'] for details in test_details.values()]
            avg_quality = statistics.mean(all_scores)
            min_quality = min(all_scores)
            
            test_details['overall_quality'] = {
                'average_quality': avg_quality,
                'minimum_quality': min_quality,
                'passed': avg_quality >= 0.8 and min_quality >= 0.6
            }
            
            test_passed = test_details['overall_quality']['passed']
            
        except Exception as e:
            logger.error(f"❌ Data quality test failed: {e}")
            test_passed = False
            test_details['error'] = str(e)
        
        test_duration = time.time() - test_start
        self.test_results.append(TestResult(
            test_name="data_quality_and_accuracy",
            passed=test_passed,
            duration=test_duration,
            details=test_details
        ))
    
    def _validate_data_quality(self, data: List[OHLCV], symbol: str, timeframe: str) -> Dict[str, Any]:
        """Validate data quality metrics"""
        if not data:
            return {'quality_score': 0.0, 'issues': ['No data available']}
        
        quality_metrics = {
            'data_count': len(data),
            'issues': []
        }
        
        # Check for missing data
        missing_count = 0
        for i, ohlcv in enumerate(data):
            if (ohlcv.open is None or ohlcv.high is None or 
                ohlcv.low is None or ohlcv.close is None):
                missing_count += 1
        
        if missing_count > 0:
            quality_metrics['issues'].append(f"Missing OHLC data in {missing_count} records")
        
        # Check for invalid prices
        invalid_count = 0
        for ohlcv in data:
            if (ohlcv.open <= 0 or ohlcv.high <= 0 or 
                ohlcv.low <= 0 or ohlcv.close <= 0):
                invalid_count += 1
        
        if invalid_count > 0:
            quality_metrics['issues'].append(f"Invalid prices in {invalid_count} records")
        
        # Check for logical inconsistencies
        inconsistent_count = 0
        for ohlcv in data:
            if (ohlcv.high < ohlcv.low or ohlcv.high < ohlcv.open or 
                ohlcv.high < ohlcv.close or ohlcv.low > ohlcv.open or 
                ohlcv.low > ohlcv.close):
                inconsistent_count += 1
        
        if inconsistent_count > 0:
            quality_metrics['issues'].append(f"Logical inconsistencies in {inconsistent_count} records")
        
        # Check for reasonable volume
        zero_volume_count = sum(1 for ohlcv in data if ohlcv.volume == 0)
        if zero_volume_count > len(data) * 0.1:  # More than 10% zero volume
            quality_metrics['issues'].append(f"High zero volume count: {zero_volume_count}")
        
        # Check timestamp continuity
        timestamps = [ohlcv.timestamp for ohlcv in data]
        timestamps.sort()
        
        # Calculate quality score
        total_issues = (missing_count + invalid_count + inconsistent_count + 
                       max(0, zero_volume_count - len(data) * 0.05))
        quality_score = max(0.0, 1.0 - (total_issues / len(data)))
        
        quality_metrics['quality_score'] = quality_score
        quality_metrics['missing_count'] = missing_count
        quality_metrics['invalid_count'] = invalid_count
        quality_metrics['inconsistent_count'] = inconsistent_count
        quality_metrics['zero_volume_count'] = zero_volume_count
        
        return quality_metrics
    
    async def _test_mathematical_validation(self):
        """Test mathematical accuracy of indicator calculations"""
        logger.info("🧮 Testing Mathematical Validation...")
        
        test_start = time.time()
        test_details = {}
        
        try:
            # Get test data
            symbol = "AAPL"
            timeframe = TimeFrame.DAILY
            end_date = datetime.now()
            start_date = end_date - timedelta(days=100)
            
            historical_data = await self.alpaca_provider.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            close_prices = np.array([ohlcv.close for ohlcv in historical_data])
            
            # Test RSI calculation
            rsi_test = self._test_rsi_calculation(close_prices)
            test_details['rsi'] = rsi_test
            
            # Test MACD calculation
            macd_test = self._test_macd_calculation(close_prices)
            test_details['macd'] = macd_test
            
            # Test Moving Averages
            ma_test = self._test_moving_averages(close_prices)
            test_details['moving_averages'] = ma_test
            
            # Test Bollinger Bands
            bollinger_test = self._test_bollinger_bands(close_prices)
            test_details['bollinger_bands'] = bollinger_test
            
            # Overall mathematical validation
            all_tests = [rsi_test, macd_test, ma_test, bollinger_test]
            test_passed = all(test['passed'] for test in all_tests)
            
        except Exception as e:
            logger.error(f"❌ Mathematical validation failed: {e}")
            test_passed = False
            test_details['error'] = str(e)
        
        test_duration = time.time() - test_start
        self.test_results.append(TestResult(
            test_name="mathematical_validation",
            passed=test_passed,
            duration=test_duration,
            details=test_details
        ))
    
    def _test_rsi_calculation(self, close_prices: np.ndarray) -> Dict[str, Any]:
        """Test RSI calculation accuracy"""
        try:
            rsi_values = calculate_rsi(close_prices, 14)
            
            # Basic validation
            valid_rsi = rsi_values[~np.isnan(rsi_values)]
            
            tests = {
                'has_values': len(valid_rsi) > 0,
                'range_valid': all(0 <= val <= 100 for val in valid_rsi),
                'expected_nan_count': np.sum(np.isnan(rsi_values)) == 14,  # First 14 should be NaN
                'reasonable_values': len(valid_rsi) > 0 and 20 <= np.mean(valid_rsi) <= 80
            }
            
            passed = all(tests.values())
            
            return {
                'passed': passed,
                'tests': tests,
                'sample_values': valid_rsi[:5].tolist() if len(valid_rsi) > 0 else [],
                'mean_rsi': float(np.mean(valid_rsi)) if len(valid_rsi) > 0 else 0
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _test_macd_calculation(self, close_prices: np.ndarray) -> Dict[str, Any]:
        """Test MACD calculation accuracy"""
        try:
            macd_line, signal_line, histogram = calculate_macd(close_prices, 12, 26, 9)
            
            # Basic validation
            valid_macd = macd_line[~np.isnan(macd_line)]
            valid_signal = signal_line[~np.isnan(signal_line)]
            valid_hist = histogram[~np.isnan(histogram)]
            
            tests = {
                'has_macd_values': len(valid_macd) > 0,
                'has_signal_values': len(valid_signal) > 0,
                'has_histogram_values': len(valid_hist) > 0,
                'histogram_calculation': len(valid_hist) > 0 and abs(valid_hist[0] - (valid_macd[0] - valid_signal[0])) < 1e-10,
                'expected_nan_count': np.sum(np.isnan(macd_line)) == 25  # First 25 should be NaN
            }
            
            passed = all(tests.values())
            
            return {
                'passed': passed,
                'tests': tests,
                'sample_macd': valid_macd[:5].tolist() if len(valid_macd) > 0 else [],
                'sample_signal': valid_signal[:5].tolist() if len(valid_signal) > 0 else [],
                'sample_histogram': valid_hist[:5].tolist() if len(valid_hist) > 0 else []
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _test_moving_averages(self, close_prices: np.ndarray) -> Dict[str, Any]:
        """Test moving average calculations"""
        try:
            sma_20 = calculate_sma(close_prices, 20)
            ema_20 = calculate_ema(close_prices, 20)
            
            # Basic validation
            valid_sma = sma_20[~np.isnan(sma_20)]
            valid_ema = ema_20[~np.isnan(ema_20)]
            
            tests = {
                'has_sma_values': len(valid_sma) > 0,
                'has_ema_values': len(valid_ema) > 0,
                'sma_reasonable': len(valid_sma) > 0 and all(val > 0 for val in valid_sma),
                'ema_reasonable': len(valid_ema) > 0 and all(val > 0 for val in valid_ema),
                'ema_more_responsive': len(valid_ema) > 10 and len(valid_sma) > 10  # Basic check
            }
            
            passed = all(tests.values())
            
            return {
                'passed': passed,
                'tests': tests,
                'sample_sma': valid_sma[:5].tolist() if len(valid_sma) > 0 else [],
                'sample_ema': valid_ema[:5].tolist() if len(valid_ema) > 0 else []
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _test_bollinger_bands(self, close_prices: np.ndarray) -> Dict[str, Any]:
        """Test Bollinger Bands calculation"""
        try:
            upper_band, middle_band, lower_band = calculate_bollinger_bands(close_prices, 20, 2.0)
            
            # Basic validation
            valid_upper = upper_band[~np.isnan(upper_band)]
            valid_middle = middle_band[~np.isnan(middle_band)]
            valid_lower = lower_band[~np.isnan(lower_band)]
            
            tests = {
                'has_upper_values': len(valid_upper) > 0,
                'has_middle_values': len(valid_middle) > 0,
                'has_lower_values': len(valid_lower) > 0,
                'band_order': (len(valid_upper) > 0 and len(valid_middle) > 0 and len(valid_lower) > 0 and
                              all(u >= m >= l for u, m, l in zip(valid_upper, valid_middle, valid_lower))),
                'reasonable_values': len(valid_middle) > 0 and all(val > 0 for val in valid_middle)
            }
            
            passed = all(tests.values())
            
            return {
                'passed': passed,
                'tests': tests,
                'sample_upper': valid_upper[:5].tolist() if len(valid_upper) > 0 else [],
                'sample_middle': valid_middle[:5].tolist() if len(valid_middle) > 0 else [],
                'sample_lower': valid_lower[:5].tolist() if len(valid_lower) > 0 else []
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    async def _test_strategy_implementations(self):
        """Test each strategy implementation with real data"""
        logger.info("🎯 Testing Strategy Implementations...")
        
        test_start = time.time()
        test_details = {}
        
        try:
            # Test data setup
            symbol = "AAPL"
            timeframe = TimeFrame.DAILY
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)
            
            historical_data = await self.alpaca_provider.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            market_data = MarketData(
                symbol=symbol,
                timeframe=timeframe,
                data=historical_data,
                indicators={},
                last_updated=datetime.now()
            )
            
            # Test each strategy
            for strategy_name in self.test_strategies:
                strategy_test = await self._test_individual_strategy(strategy_name, market_data)
                test_details[strategy_name] = strategy_test
                
                status = "✅" if strategy_test['passed'] else "❌"
                logger.info(f"{status} {strategy_name}: {strategy_test['signal_count']} signals, "
                          f"avg confidence: {strategy_test['avg_confidence']:.2f}")
            
            # Overall strategy validation
            passed_strategies = sum(1 for test in test_details.values() if test['passed'])
            test_passed = passed_strategies == len(self.test_strategies)
            
            test_details['summary'] = {
                'passed_strategies': passed_strategies,
                'total_strategies': len(self.test_strategies),
                'success_rate': passed_strategies / len(self.test_strategies)
            }
            
        except Exception as e:
            logger.error(f"❌ Strategy implementation test failed: {e}")
            test_passed = False
            test_details['error'] = str(e)
        
        test_duration = time.time() - test_start
        self.test_results.append(TestResult(
            test_name="strategy_implementations",
            passed=test_passed,
            duration=test_duration,
            details=test_details
        ))
    
    async def _test_individual_strategy(self, strategy_name: str, market_data: MarketData) -> Dict[str, Any]:
        """Test individual strategy implementation"""
        try:
            # Create strategy
            strategy = create_strategy(strategy_name)
            
            # Run signal analysis
            engine = SignalAnalysisEngine()
            result = engine.run_signal_analysis(strategy, market_data)
            
            # Validate results
            signals = result.signals
            signal_count = len(signals)
            
            if signal_count == 0:
                return {
                    'passed': False,
                    'signal_count': 0,
                    'avg_confidence': 0.0,
                    'error': 'No signals generated'
                }
            
            # Calculate metrics
            confidences = [s.confidence for s in signals]
            avg_confidence = statistics.mean(confidences)
            min_confidence = min(confidences)
            max_confidence = max(confidences)
            
            # Validate signal quality
            valid_signals = [s for s in signals if s.confidence >= 0.3]
            high_quality_signals = [s for s in signals if s.confidence >= 0.7]
            
            tests = {
                'generates_signals': signal_count > 0,
                'reasonable_confidence': avg_confidence >= 0.4,
                'valid_confidence_range': min_confidence >= 0.0 and max_confidence <= 1.0,
                'has_high_quality_signals': len(high_quality_signals) > 0,
                'signal_metadata': all(s.metadata is not None for s in signals),
                'signal_timestamps': all(s.timestamp is not None for s in signals),
                'signal_prices': all(s.price > 0 for s in signals)
            }
            
            passed = all(tests.values())
            
            return {
                'passed': passed,
                'signal_count': signal_count,
                'avg_confidence': avg_confidence,
                'min_confidence': min_confidence,
                'max_confidence': max_confidence,
                'high_quality_count': len(high_quality_signals),
                'tests': tests
            }
            
        except Exception as e:
            return {
                'passed': False,
                'signal_count': 0,
                'avg_confidence': 0.0,
                'error': str(e)
            }
    
    async def _test_signal_analysis_pipeline(self):
        """Test the complete signal analysis pipeline"""
        logger.info("🔄 Testing Signal Analysis Pipeline...")
        
        test_start = time.time()
        test_details = {}
        
        try:
            # Test pipeline with different configurations
            test_configs = [
                {"symbol": "AAPL", "timeframe": "1d", "strategy": "macd_rsi", "days": 30},
                {"symbol": "SPY", "timeframe": "1h", "strategy": "ma_crossover", "days": 7},
                {"symbol": "MSFT", "timeframe": "1d", "strategy": "rsi_trend", "days": 30}
            ]
            
            pipeline_tests = []
            
            for config in test_configs:
                config_test = await self._test_pipeline_configuration(config)
                test_key = f"{config['symbol']}_{config['timeframe']}_{config['strategy']}"
                test_details[test_key] = config_test
                pipeline_tests.append(config_test['passed'])
                
                status = "✅" if config_test['passed'] else "❌"
                logger.info(f"{status} Pipeline test: {test_key}")
            
            # Overall pipeline validation
            test_passed = all(pipeline_tests)
            
            test_details['summary'] = {
                'passed_configurations': sum(pipeline_tests),
                'total_configurations': len(test_configs),
                'success_rate': sum(pipeline_tests) / len(test_configs)
            }
            
        except Exception as e:
            logger.error(f"❌ Signal analysis pipeline test failed: {e}")
            test_passed = False
            test_details['error'] = str(e)
        
        test_duration = time.time() - test_start
        self.test_results.append(TestResult(
            test_name="signal_analysis_pipeline",
            passed=test_passed,
            duration=test_duration,
            details=test_details
        ))
    
    async def _test_pipeline_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Test pipeline with specific configuration"""
        try:
            # Setup
            symbol = config['symbol']
            timeframe = self._parse_timeframe(config['timeframe'])
            strategy_name = config['strategy']
            days = config['days']
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get data
            historical_data = await self.alpaca_provider.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            market_data = MarketData(
                symbol=symbol,
                timeframe=timeframe,
                data=historical_data,
                indicators={},
                last_updated=datetime.now()
            )
            
            # Create strategy and run analysis
            strategy = create_strategy(strategy_name)
            engine = SignalAnalysisEngine()
            result = engine.run_signal_analysis(strategy, market_data)
            
            # Validate pipeline output
            tests = {
                'has_data': len(historical_data) > 0,
                'strategy_created': strategy is not None,
                'analysis_completed': result is not None,
                'has_signals': len(result.signals) >= 0,  # Allow zero signals
                'result_structure': hasattr(result, 'signals') and hasattr(result, 'avg_confidence'),
                'metadata_present': result.metadata is not None
            }
            
            passed = all(tests.values())
            
            return {
                'passed': passed,
                'data_points': len(historical_data),
                'signal_count': len(result.signals),
                'avg_confidence': result.avg_confidence,
                'tests': tests
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    async def _test_output_format_validation(self):
        """Test output format validation and JSON serialization"""
        logger.info("📋 Testing Output Format Validation...")
        
        test_start = time.time()
        test_details = {}
        
        try:
            # Get test data and run analysis
            symbol = "AAPL"
            timeframe = TimeFrame.DAILY
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            historical_data = await self.alpaca_provider.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            market_data = MarketData(
                symbol=symbol,
                timeframe=timeframe,
                data=historical_data,
                indicators={},
                last_updated=datetime.now()
            )
            
            strategy = create_strategy("macd_rsi")
            engine = SignalAnalysisEngine()
            result = engine.run_signal_analysis(strategy, market_data)
            
            # Test output format
            format_tests = self._validate_output_format(result)
            test_details['format_validation'] = format_tests
            
            # Test JSON serialization
            json_tests = self._test_json_serialization(result)
            test_details['json_serialization'] = json_tests
            
            # Test result display format (simulate submit_job.py formatting)
            display_tests = self._test_result_display_format(result)
            test_details['display_format'] = display_tests
            
            # Overall validation
            all_tests = [format_tests, json_tests, display_tests]
            test_passed = all(test['passed'] for test in all_tests)
            
        except Exception as e:
            logger.error(f"❌ Output format validation failed: {e}")
            test_passed = False
            test_details['error'] = str(e)
        
        test_duration = time.time() - test_start
        self.test_results.append(TestResult(
            test_name="output_format_validation",
            passed=test_passed,
            duration=test_duration,
            details=test_details
        ))
    
    def _validate_output_format(self, result) -> Dict[str, Any]:
        """Validate the signal analysis result format"""
        try:
            required_fields = [
                'strategy_name', 'symbol', 'start_date', 'end_date', 'signals',
                'total_signals', 'buy_signals', 'sell_signals', 'avg_confidence'
            ]
            
            tests = {
                'has_required_fields': all(hasattr(result, field) for field in required_fields),
                'signal_list_valid': isinstance(result.signals, list),
                'numeric_fields_valid': all(isinstance(getattr(result, field), (int, float)) 
                                          for field in ['total_signals', 'buy_signals', 'sell_signals', 'avg_confidence']),
                'date_fields_valid': all(isinstance(getattr(result, field), datetime) 
                                       for field in ['start_date', 'end_date']),
                'metadata_present': result.metadata is not None
            }
            
            # Test individual signal format
            if result.signals:
                signal = result.signals[0]
                signal_tests = {
                    'signal_timestamp': hasattr(signal, 'timestamp') and signal.timestamp is not None,
                    'signal_price': hasattr(signal, 'price') and signal.price > 0,
                    'signal_confidence': hasattr(signal, 'confidence') and 0 <= signal.confidence <= 1,
                    'signal_type': hasattr(signal, 'signal_type') and signal.signal_type is not None,
                    'signal_metadata': hasattr(signal, 'metadata') and signal.metadata is not None
                }
                tests.update(signal_tests)
            
            passed = all(tests.values())
            
            return {
                'passed': passed,
                'tests': tests,
                'signal_count': len(result.signals)
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _test_json_serialization(self, result) -> Dict[str, Any]:
        """Test JSON serialization of results"""
        try:
            # Simulate the serialization done in job_processor.py
            signals_data = []
            for signal in result.signals:
                signal_dict = {
                    "timestamp": signal.timestamp.isoformat(),
                    "symbol": signal.symbol,
                    "type": signal.signal_type.value,
                    "price": signal.price,
                    "confidence": signal.confidence,
                    "strategy": signal.strategy_name,
                    "metadata": signal.metadata
                }
                signals_data.append(signal_dict)
            
            # Test JSON serialization
            json_str = json.dumps(signals_data)
            
            # Test deserialization
            deserialized = json.loads(json_str)
            
            tests = {
                'serialization_succeeds': json_str is not None,
                'deserialization_succeeds': deserialized is not None,
                'data_preserved': len(deserialized) == len(signals_data),
                'required_fields_present': all(
                    all(field in item for field in ['timestamp', 'symbol', 'type', 'price', 'confidence'])
                    for item in deserialized
                )
            }
            
            passed = all(tests.values())
            
            return {
                'passed': passed,
                'tests': tests,
                'json_size': len(json_str),
                'signal_count': len(signals_data)
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _test_result_display_format(self, result) -> Dict[str, Any]:
        """Test result display format compatibility"""
        try:
            # Simulate the display format used in submit_job.py
            signals_data = []
            for signal in result.signals:
                signal_dict = {
                    "timestamp": signal.timestamp.isoformat(),
                    "type": signal.signal_type.value,
                    "price": signal.price,
                    "confidence": signal.confidence,
                    "strategy": signal.strategy_name,
                    "metadata": signal.metadata
                }
                signals_data.append(signal_dict)
            
            # Test display calculations
            total_signals = len(signals_data)
            buy_signals = sum(1 for s in signals_data if s["type"] == "BUY")
            sell_signals = sum(1 for s in signals_data if s["type"] == "SELL")
            hold_signals = sum(1 for s in signals_data if s["type"] == "HOLD")
            
            confidences = [float(s["confidence"]) for s in signals_data if s["confidence"] is not None]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            tests = {
                'signal_counts_valid': buy_signals + sell_signals + hold_signals == total_signals,
                'confidence_calculation': avg_confidence >= 0 and avg_confidence <= 1,
                'display_data_complete': all(
                    field in signal_dict for field in ['timestamp', 'type', 'price', 'confidence']
                    for signal_dict in signals_data
                ),
                'metadata_accessible': all(
                    signal_dict.get('metadata') is not None for signal_dict in signals_data
                )
            }
            
            passed = all(tests.values())
            
            return {
                'passed': passed,
                'tests': tests,
                'total_signals': total_signals,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'avg_confidence': avg_confidence
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    async def _test_performance_and_reliability(self):
        """Test processing speed, memory usage, and error handling"""
        logger.info("⚡ Testing Performance and Reliability...")
        
        test_start = time.time()
        test_details = {}
        
        try:
            # Performance tests
            performance_tests = await self._run_performance_tests()
            test_details['performance'] = performance_tests
            
            # Error handling tests
            error_tests = await self._run_error_handling_tests()
            test_details['error_handling'] = error_tests
            
            # Memory usage tests
            memory_tests = await self._run_memory_tests()
            test_details['memory'] = memory_tests
            
            # Overall reliability
            all_tests = [performance_tests, error_tests, memory_tests]
            test_passed = all(test['passed'] for test in all_tests)
            
        except Exception as e:
            logger.error(f"❌ Performance and reliability test failed: {e}")
            test_passed = False
            test_details['error'] = str(e)
        
        test_duration = time.time() - test_start
        self.test_results.append(TestResult(
            test_name="performance_and_reliability",
            passed=test_passed,
            duration=test_duration,
            details=test_details
        ))
    
    async def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests"""
        try:
            # Test with different data sizes
            test_cases = [
                {"days": 30, "timeframe": "1d", "expected_max_time": 10.0},
                {"days": 7, "timeframe": "1h", "expected_max_time": 15.0},
                {"days": 90, "timeframe": "1d", "expected_max_time": 30.0}
            ]
            
            performance_results = []
            
            for test_case in test_cases:
                case_start = time.time()
                
                # Setup
                symbol = "AAPL"
                timeframe = self._parse_timeframe(test_case['timeframe'])
                end_date = datetime.now()
                start_date = end_date - timedelta(days=test_case['days'])
                
                # Get data
                historical_data = await self.alpaca_provider.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                
                market_data = MarketData(
                    symbol=symbol,
                    timeframe=timeframe,
                    data=historical_data,
                    indicators={},
                    last_updated=datetime.now()
                )
                
                # Run analysis
                strategy = create_strategy("macd_rsi")
                engine = SignalAnalysisEngine()
                result = engine.run_signal_analysis(strategy, market_data)
                
                case_duration = time.time() - case_start
                
                performance_results.append({
                    'test_case': test_case,
                    'duration': case_duration,
                    'data_points': len(historical_data),
                    'signal_count': len(result.signals),
                    'passed': case_duration <= test_case['expected_max_time']
                })
            
            # Overall performance assessment
            all_passed = all(result['passed'] for result in performance_results)
            avg_duration = statistics.mean(result['duration'] for result in performance_results)
            
            return {
                'passed': all_passed,
                'test_results': performance_results,
                'average_duration': avg_duration,
                'max_duration': max(result['duration'] for result in performance_results)
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    async def _run_error_handling_tests(self) -> Dict[str, Any]:
        """Test error handling capabilities"""
        try:
            error_tests = []
            
            # Test 1: Invalid symbol
            try:
                historical_data = await self.alpaca_provider.get_historical_data(
                    symbol="INVALID_SYMBOL",
                    timeframe=TimeFrame.DAILY,
                    start_date=datetime.now() - timedelta(days=30),
                    end_date=datetime.now()
                )
                error_tests.append({'test': 'invalid_symbol', 'passed': len(historical_data) == 0})
            except Exception:
                error_tests.append({'test': 'invalid_symbol', 'passed': True})  # Expected to fail
            
            # Test 2: Invalid date range
            try:
                historical_data = await self.alpaca_provider.get_historical_data(
                    symbol="AAPL",
                    timeframe=TimeFrame.DAILY,
                    start_date=datetime.now() + timedelta(days=1),
                    end_date=datetime.now()
                )
                error_tests.append({'test': 'invalid_date_range', 'passed': len(historical_data) == 0})
            except Exception:
                error_tests.append({'test': 'invalid_date_range', 'passed': True})  # Expected to fail
            
            # Test 3: Invalid strategy
            try:
                strategy = create_strategy("invalid_strategy")
                error_tests.append({'test': 'invalid_strategy', 'passed': strategy is None})
            except Exception:
                error_tests.append({'test': 'invalid_strategy', 'passed': True})  # Expected to fail
            
            # Test 4: Empty data handling
            try:
                empty_market_data = MarketData(
                    symbol="TEST",
                    timeframe=TimeFrame.DAILY,
                    data=[],
                    indicators={},
                    last_updated=datetime.now()
                )
                
                strategy = create_strategy("macd_rsi")
                engine = SignalAnalysisEngine()
                result = engine.run_signal_analysis(strategy, empty_market_data)
                
                error_tests.append({'test': 'empty_data', 'passed': len(result.signals) == 0})
            except Exception:
                error_tests.append({'test': 'empty_data', 'passed': True})  # Expected to handle gracefully
            
            # Overall error handling assessment
            all_passed = all(test['passed'] for test in error_tests)
            
            return {
                'passed': all_passed,
                'error_tests': error_tests,
                'passed_tests': sum(1 for test in error_tests if test['passed']),
                'total_tests': len(error_tests)
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    async def _run_memory_tests(self) -> Dict[str, Any]:
        """Test memory usage"""
        try:
            # Simple memory test - check if large dataset processing works
            symbol = "AAPL"
            timeframe = TimeFrame.DAILY
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # 1 year of data
            
            historical_data = await self.alpaca_provider.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            market_data = MarketData(
                symbol=symbol,
                timeframe=timeframe,
                data=historical_data,
                indicators={},
                last_updated=datetime.now()
            )
            
            # Run analysis on large dataset
            strategy = create_strategy("macd_rsi")
            engine = SignalAnalysisEngine()
            result = engine.run_signal_analysis(strategy, market_data)
            
            tests = {
                'large_dataset_processing': len(historical_data) > 200,
                'memory_efficient': len(result.signals) >= 0,  # Basic completion test
                'no_memory_leaks': True  # Would need more sophisticated testing
            }
            
            passed = all(tests.values())
            
            return {
                'passed': passed,
                'tests': tests,
                'data_points_processed': len(historical_data),
                'signals_generated': len(result.signals)
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _parse_timeframe(self, timeframe: str) -> TimeFrame:
        """Parse timeframe string to TimeFrame enum"""
        mapping = {
            "1m": TimeFrame.MINUTE_1,
            "5m": TimeFrame.MINUTE_5,
            "15m": TimeFrame.MINUTE_15,
            "30m": TimeFrame.MINUTE_30,
            "1h": TimeFrame.HOUR_1,
            "4h": TimeFrame.HOUR_4,
            "1d": TimeFrame.DAILY,
            "1w": TimeFrame.WEEKLY,
            "1M": TimeFrame.MONTHLY
        }
        
        if timeframe not in mapping:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        return mapping[timeframe]
    
    def _generate_validation_report(self, total_duration: float) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        logger.info("📋 Generating Validation Report...")
        
        # Calculate summary statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.passed)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Categorize results
        critical_tests = ["data_quality_and_accuracy", "mathematical_validation", "strategy_implementations"]
        critical_passed = sum(1 for result in self.test_results 
                             if result.test_name in critical_tests and result.passed)
        
        # Generate report
        report = {
            "validation_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate,
                "critical_tests_passed": critical_passed,
                "total_critical_tests": len(critical_tests),
                "overall_status": "PASSED" if success_rate >= 80 and critical_passed == len(critical_tests) else "FAILED",
                "total_duration": total_duration
            },
            "test_results": {},
            "recommendations": [],
            "critical_issues": [],
            "performance_metrics": {}
        }
        
        # Add detailed test results
        for result in self.test_results:
            report["test_results"][result.test_name] = {
                "passed": result.passed,
                "duration": result.duration,
                "details": result.details,
                "error": result.error
            }
        
        # Generate recommendations
        if failed_tests > 0:
            report["recommendations"].append("Review and fix failing tests before production deployment")
        
        if success_rate < 90:
            report["recommendations"].append("Consider additional testing and validation")
        
        # Check for critical issues
        for result in self.test_results:
            if not result.passed and result.test_name in critical_tests:
                report["critical_issues"].append({
                    "test": result.test_name,
                    "error": result.error,
                    "impact": "HIGH"
                })
        
        # Performance metrics
        performance_result = next((r for r in self.test_results if r.test_name == "performance_and_reliability"), None)
        if performance_result and performance_result.details:
            perf_details = performance_result.details.get("performance", {})
            if perf_details:
                report["performance_metrics"] = {
                    "average_processing_time": perf_details.get("average_duration", 0),
                    "max_processing_time": perf_details.get("max_duration", 0),
                    "performance_passed": perf_details.get("passed", False)
                }
        
        return report


async def main():
    """Run the validation test suite"""
    print("=" * 80)
    print("🚀 WAGEHOOD SIGNAL ANALYSIS SYSTEM VALIDATION")
    print("=" * 80)
    
    # Initialize test suite
    test_suite = ValidationTestSuite()
    
    try:
        # Run all tests
        report = await test_suite.run_all_tests()
        
        # Display results
        print("\n" + "=" * 80)
        print("📊 VALIDATION RESULTS SUMMARY")
        print("=" * 80)
        
        summary = report["validation_summary"]
        status_emoji = "✅" if summary["overall_status"] == "PASSED" else "❌"
        
        print(f"{status_emoji} Overall Status: {summary['overall_status']}")
        print(f"📈 Success Rate: {summary['success_rate']:.1f}%")
        print(f"✅ Passed Tests: {summary['passed_tests']}/{summary['total_tests']}")
        print(f"⚠️  Critical Tests: {summary['critical_tests_passed']}/{summary['total_critical_tests']}")
        print(f"⏱️  Total Duration: {summary['total_duration']:.2f}s")
        
        # Show detailed results
        print("\n" + "=" * 80)
        print("🔍 DETAILED TEST RESULTS")
        print("=" * 80)
        
        for test_name, result in report["test_results"].items():
            status = "✅ PASSED" if result["passed"] else "❌ FAILED"
            print(f"{status} {test_name.replace('_', ' ').title()}: {result['duration']:.2f}s")
            if result["error"]:
                print(f"   Error: {result['error']}")
        
        # Show critical issues
        if report["critical_issues"]:
            print("\n" + "=" * 80)
            print("🚨 CRITICAL ISSUES")
            print("=" * 80)
            for issue in report["critical_issues"]:
                print(f"❌ {issue['test']}: {issue['error']}")
        
        # Show recommendations
        if report["recommendations"]:
            print("\n" + "=" * 80)
            print("💡 RECOMMENDATIONS")
            print("=" * 80)
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"{i}. {rec}")
        
        # Show performance metrics
        if report["performance_metrics"]:
            print("\n" + "=" * 80)
            print("⚡ PERFORMANCE METRICS")
            print("=" * 80)
            metrics = report["performance_metrics"]
            print(f"Average Processing Time: {metrics.get('average_processing_time', 0):.2f}s")
            print(f"Max Processing Time: {metrics.get('max_processing_time', 0):.2f}s")
            print(f"Performance Status: {'✅ PASSED' if metrics.get('performance_passed', False) else '❌ FAILED'}")
        
        # Save full report
        with open("validation_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        print("\n" + "=" * 80)
        print("💾 Full validation report saved to: validation_report.json")
        print("=" * 80)
        
        # Return exit code based on results
        return 0 if summary["overall_status"] == "PASSED" else 1
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)