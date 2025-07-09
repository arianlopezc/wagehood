"""
Comprehensive Mathematical Tests for Moving Average Crossover Strategy

This module provides comprehensive mathematical validation tests for the
Moving Average Crossover trading strategy including signal generation,
confidence calculations, and edge case handling.
"""

import pytest
import numpy as np
from typing import Dict, List, Tuple
import sys
import os
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from src.strategies.ma_crossover import MovingAverageCrossover
from src.core.models import MarketData, Signal, SignalType, OHLCV

from ..fixtures.test_data import (
    TestDataGenerator,
    PrecisionTestVectors,
    EdgeCaseTestData,
    ValidationHelpers,
    TEST_CONFIG
)


class TestMovingAverageCrossoverBasics:
    """Basic tests for Moving Average Crossover strategy"""
    
    def test_strategy_initialization(self):
        """Test strategy initialization with default and custom parameters"""
        # Test default initialization
        strategy = MovingAverageCrossover()
        
        assert strategy.name == "MovingAverageCrossover"
        assert strategy.parameters['short_period'] == 50
        assert strategy.parameters['long_period'] == 200
        assert strategy.parameters['min_confidence'] == 0.6
        assert strategy.parameters['volume_confirmation'] == True
        
        # Test custom parameters
        custom_params = {
            'short_period': 20,
            'long_period': 100,
            'min_confidence': 0.7,
            'volume_confirmation': False
        }
        
        strategy = MovingAverageCrossover(custom_params)
        
        assert strategy.parameters['short_period'] == 20
        assert strategy.parameters['long_period'] == 100
        assert strategy.parameters['min_confidence'] == 0.7
        assert strategy.parameters['volume_confirmation'] == False
    
    def test_required_indicators(self):
        """Test required indicators"""
        strategy = MovingAverageCrossover()
        required = strategy.get_required_indicators()
        
        assert 'ema' in required
        assert len(required) == 1
    
    def test_parameter_descriptions(self):
        """Test parameter descriptions and metadata"""
        strategy = MovingAverageCrossover()
        params = strategy.get_parameters()
        
        # Check that all parameters have required metadata
        for param_name, param_info in params.items():
            assert 'value' in param_info
            assert 'description' in param_info
            assert 'type' in param_info
            assert 'default' in param_info
            
            if param_info['type'] in ['int', 'float']:
                assert 'min' in param_info
                assert 'max' in param_info


class TestGoldenCrossSignalGeneration:
    """Tests for Golden Cross (bullish) signal generation"""
    
    def test_basic_golden_cross_detection(self):
        """Test basic Golden Cross signal detection"""
        # Create test data with a clear Golden Cross
        base_prices = np.full(100, 100.0)
        
        # Add uptrend that will cause short EMA to cross above long EMA
        trend_prices = np.concatenate([
            np.full(50, 100.0),  # Flat period
            np.linspace(100.0, 120.0, 50)  # Uptrend
        ])
        
        market_data = self._create_market_data(trend_prices)
        strategy = MovingAverageCrossover({
            'short_period': 10,
            'long_period': 20,
            'min_confidence': 0.1,  # Low threshold for testing
            'volume_confirmation': False
        })
        
        # Calculate indicators
        indicators = strategy._calculate_indicators(market_data)
        
        # Generate signals
        signals = strategy.generate_signals(market_data, indicators)
        
        # Should have at least one buy signal
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buy_signals) > 0
        
        # Check signal properties
        for signal in buy_signals:
            assert signal.strategy_name == "MovingAverageCrossover"
            assert signal.confidence > 0.0
            assert signal.price > 0.0
            assert 'Golden Cross' in signal.metadata.get('signal_name', '')
    
    def test_golden_cross_mathematical_accuracy(self):
        """Test mathematical accuracy of Golden Cross detection"""
        # Create precise test data
        prices = np.array([
            100.0, 100.0, 100.0, 100.0, 100.0,  # Period 0-4: Flat
            101.0, 102.0, 103.0, 104.0, 105.0,  # Period 5-9: Gradual rise
            106.0, 107.0, 108.0, 109.0, 110.0   # Period 10-14: Continued rise
        ])
        
        market_data = self._create_market_data(prices)
        strategy = MovingAverageCrossover({
            'short_period': 3,
            'long_period': 5,
            'min_confidence': 0.1,
            'volume_confirmation': False
        })
        
        # Calculate EMAs manually for verification
        from src.indicators.moving_averages import calculate_ema
        short_ema = calculate_ema(prices, 3)
        long_ema = calculate_ema(prices, 5)
        
        # Find the crossover point
        crossover_index = None
        for i in range(1, len(short_ema)):
            if (not np.isnan(short_ema[i-1]) and not np.isnan(long_ema[i-1]) and
                not np.isnan(short_ema[i]) and not np.isnan(long_ema[i])):
                
                if short_ema[i-1] <= long_ema[i-1] and short_ema[i] > long_ema[i]:
                    crossover_index = i
                    break
        
        # Generate signals using strategy
        indicators = strategy._calculate_indicators(market_data)
        signals = strategy.generate_signals(market_data, indicators)
        
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        
        if crossover_index is not None and len(buy_signals) > 0:
            # Verify that the signal occurs at or near the mathematical crossover
            signal_indices = []
            arrays = market_data.to_arrays()
            for signal in buy_signals:
                for i, timestamp in enumerate(arrays['timestamp']):
                    if timestamp == signal.timestamp:
                        signal_indices.append(i)
                        break
            
            # Should have signal near the crossover point
            assert any(abs(idx - crossover_index) <= 1 for idx in signal_indices)
    
    def test_golden_cross_confidence_calculation(self):
        """Test Golden Cross confidence calculation factors"""
        # Create test data with varying characteristics
        prices = TestDataGenerator.generate_linear_trend(50, slope=0.5)
        market_data = self._create_market_data(prices)
        
        strategy = MovingAverageCrossover({
            'short_period': 10,
            'long_period': 20,
            'min_confidence': 0.1,
            'volume_confirmation': False
        })
        
        indicators = strategy._calculate_indicators(market_data)
        signals = strategy.generate_signals(market_data, indicators)
        
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        
        for signal in buy_signals:
            # Confidence should be within valid range
            assert 0.0 <= signal.confidence <= 1.0
            
            # Check metadata contains confidence factors
            metadata = signal.metadata
            assert 'ema_separation' in metadata
            assert 'trend_strength' in metadata
            
            # EMA separation should be positive for Golden Cross
            assert metadata['ema_separation'] > 0.0
    
    def test_golden_cross_volume_confirmation(self):
        """Test volume confirmation in Golden Cross signals"""
        prices = TestDataGenerator.generate_linear_trend(50)
        
        # Create market data with varying volume
        ohlcv_data = TestDataGenerator.generate_ohlcv_data(prices)
        
        # High volume scenario
        high_volume = ohlcv_data['volume'] * 2.0
        market_data_high_vol = self._create_market_data_with_volume(prices, high_volume)
        
        # Low volume scenario  
        low_volume = ohlcv_data['volume'] * 0.5
        market_data_low_vol = self._create_market_data_with_volume(prices, low_volume)
        
        strategy = MovingAverageCrossover({
            'short_period': 10,
            'long_period': 20,
            'min_confidence': 0.1,
            'volume_confirmation': True,
            'volume_threshold': 1.2
        })
        
        # Generate signals for both scenarios
        indicators_high = strategy._calculate_indicators(market_data_high_vol)
        signals_high = strategy.generate_signals(market_data_high_vol, indicators_high)
        
        indicators_low = strategy._calculate_indicators(market_data_low_vol)
        signals_low = strategy.generate_signals(market_data_low_vol, indicators_low)
        
        # High volume signals should generally have higher confidence
        high_vol_confidences = [s.confidence for s in signals_high if s.signal_type == SignalType.BUY]
        low_vol_confidences = [s.confidence for s in signals_low if s.signal_type == SignalType.BUY]
        
        if high_vol_confidences and low_vol_confidences:
            avg_high_conf = np.mean(high_vol_confidences)
            avg_low_conf = np.mean(low_vol_confidences)
            assert avg_high_conf >= avg_low_conf


class TestDeathCrossSignalGeneration:
    """Tests for Death Cross (bearish) signal generation"""
    
    def test_basic_death_cross_detection(self):
        """Test basic Death Cross signal detection"""
        # Create test data with a clear Death Cross
        trend_prices = np.concatenate([
            np.linspace(120.0, 100.0, 50),  # Downtrend
            np.full(50, 100.0)  # Flat period
        ])
        
        market_data = self._create_market_data(trend_prices)
        strategy = MovingAverageCrossover({
            'short_period': 10,
            'long_period': 20,
            'min_confidence': 0.1,
            'volume_confirmation': False
        })
        
        indicators = strategy._calculate_indicators(market_data)
        signals = strategy.generate_signals(market_data, indicators)
        
        # Should have at least one sell signal
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        assert len(sell_signals) > 0
        
        # Check signal properties
        for signal in sell_signals:
            assert signal.strategy_name == "MovingAverageCrossover"
            assert signal.confidence > 0.0
            assert signal.price > 0.0
            assert 'Death Cross' in signal.metadata.get('signal_name', '')
    
    def test_death_cross_mathematical_accuracy(self):
        """Test mathematical accuracy of Death Cross detection"""
        # Create precise test data for downtrend
        prices = np.array([
            110.0, 109.0, 108.0, 107.0, 106.0,  # Period 0-4: Decline
            105.0, 104.0, 103.0, 102.0, 101.0,  # Period 5-9: Continued decline
            100.0, 100.0, 100.0, 100.0, 100.0   # Period 10-14: Flat
        ])
        
        market_data = self._create_market_data(prices)
        strategy = MovingAverageCrossover({
            'short_period': 3,
            'long_period': 5,
            'min_confidence': 0.1,
            'volume_confirmation': False
        })
        
        # Calculate EMAs manually for verification
        from src.indicators.moving_averages import calculate_ema
        short_ema = calculate_ema(prices, 3)
        long_ema = calculate_ema(prices, 5)
        
        # Find the crossover point
        crossover_index = None
        for i in range(1, len(short_ema)):
            if (not np.isnan(short_ema[i-1]) and not np.isnan(long_ema[i-1]) and
                not np.isnan(short_ema[i]) and not np.isnan(long_ema[i])):
                
                if short_ema[i-1] >= long_ema[i-1] and short_ema[i] < long_ema[i]:
                    crossover_index = i
                    break
        
        # Generate signals using strategy
        indicators = strategy._calculate_indicators(market_data)
        signals = strategy.generate_signals(market_data, indicators)
        
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        
        if crossover_index is not None and len(sell_signals) > 0:
            # Verify that the signal occurs at or near the mathematical crossover
            signal_indices = []
            arrays = market_data.to_arrays()
            for signal in sell_signals:
                for i, timestamp in enumerate(arrays['timestamp']):
                    if timestamp == signal.timestamp:
                        signal_indices.append(i)
                        break
            
            # Should have signal near the crossover point
            assert any(abs(idx - crossover_index) <= 1 for idx in signal_indices)


class TestStrategyEdgeCases:
    """Test strategy behavior with edge cases"""
    
    def test_insufficient_data(self):
        """Test strategy with insufficient data"""
        # Very short data series
        prices = np.array([100.0, 101.0, 102.0])
        market_data = self._create_market_data(prices)
        
        strategy = MovingAverageCrossover({
            'short_period': 10,
            'long_period': 20
        })
        
        indicators = strategy._calculate_indicators(market_data)
        signals = strategy.generate_signals(market_data, indicators)
        
        # Should not crash and should return empty signals
        assert isinstance(signals, list)
        assert len(signals) == 0
    
    def test_constant_prices(self):
        """Test strategy with constant prices"""
        prices = EdgeCaseTestData.get_constant_prices()
        market_data = self._create_market_data(prices)
        
        strategy = MovingAverageCrossover({
            'short_period': 5,
            'long_period': 10,
            'min_confidence': 0.1
        })
        
        indicators = strategy._calculate_indicators(market_data)
        signals = strategy.generate_signals(market_data, indicators)
        
        # Should not generate signals for constant prices
        assert len(signals) == 0
    
    def test_extreme_volatility(self):
        """Test strategy with extremely volatile data"""
        prices = EdgeCaseTestData.get_extreme_volatility()
        market_data = self._create_market_data(prices)
        
        strategy = MovingAverageCrossover({
            'short_period': 3,
            'long_period': 5,
            'min_confidence': 0.1
        })
        
        indicators = strategy._calculate_indicators(market_data)
        signals = strategy.generate_signals(market_data, indicators)
        
        # Should handle extreme volatility without crashing
        assert isinstance(signals, list)
        
        # All signals should have valid properties
        for signal in signals:
            assert signal.confidence >= 0.0
            assert signal.confidence <= 1.0
            assert signal.price > 0.0
            assert signal.signal_type in [SignalType.BUY, SignalType.SELL]
    
    def test_nan_values_in_data(self):
        """Test strategy with NaN values in price data"""
        prices = EdgeCaseTestData.get_nan_prices()
        # Extend the array to have enough data
        extended_prices = np.concatenate([prices, np.full(15, 100.0)])
        
        market_data = self._create_market_data(extended_prices)
        
        strategy = MovingAverageCrossover({
            'short_period': 3,
            'long_period': 5,
            'min_confidence': 0.1
        })
        
        indicators = strategy._calculate_indicators(market_data)
        signals = strategy.generate_signals(market_data, indicators)
        
        # Should handle NaN values gracefully
        assert isinstance(signals, list)
        
        # All signals should have finite values
        for signal in signals:
            assert np.isfinite(signal.confidence)
            assert np.isfinite(signal.price)


class TestStrategyOptimization:
    """Test strategy parameter optimization"""
    
    def test_parameter_ranges(self):
        """Test parameter ranges for optimization"""
        strategy = MovingAverageCrossover()
        ranges = strategy._get_parameter_ranges()
        
        # Should have ranges for key parameters
        assert 'short_period' in ranges
        assert 'long_period' in ranges
        assert 'min_confidence' in ranges
        
        # Ranges should be lists
        for param, range_values in ranges.items():
            assert isinstance(range_values, list)
            assert len(range_values) > 1
            
            # Short period should be less than long period
            if param == 'short_period':
                assert all(v < 200 for v in range_values)
            elif param == 'long_period':
                assert all(v > 50 for v in range_values)
    
    def test_signal_validation(self):
        """Test signal validation logic"""
        strategy = MovingAverageCrossover()
        
        # Create valid signal
        valid_signal = Signal(
            timestamp=datetime.now(),
            symbol="TEST",
            signal_type=SignalType.BUY,
            price=100.0,
            confidence=0.8,
            strategy_name="MovingAverageCrossover",
            metadata={
                'signal_name': 'Golden Cross',
                'short_ema': 105.0,
                'long_ema': 100.0,
                'ema_separation': 0.05
            }
        )
        
        # Create invalid signals
        invalid_signals = [
            # Low confidence
            Signal(
                timestamp=datetime.now(),
                symbol="TEST",
                signal_type=SignalType.BUY,
                price=100.0,
                confidence=0.3,  # Below default threshold
                strategy_name="MovingAverageCrossover",
                metadata={'signal_name': 'Golden Cross', 'short_ema': 105.0, 'long_ema': 100.0}
            ),
            # Missing EMA data
            Signal(
                timestamp=datetime.now(),
                symbol="TEST",
                signal_type=SignalType.BUY,
                price=100.0,
                confidence=0.8,
                strategy_name="MovingAverageCrossover",
                metadata={'signal_name': 'Golden Cross'}
            ),
            # Very small EMA separation
            Signal(
                timestamp=datetime.now(),
                symbol="TEST",
                signal_type=SignalType.BUY,
                price=100.0,
                confidence=0.8,
                strategy_name="MovingAverageCrossover",
                metadata={
                    'signal_name': 'Golden Cross',
                    'short_ema': 100.001,
                    'long_ema': 100.0,
                    'ema_separation': 0.00001  # Too small
                }
            )
        ]
        
        # Test validation
        all_signals = [valid_signal] + invalid_signals
        validated_signals = strategy.validate_signals(all_signals)
        
        # Should only keep the valid signal
        assert len(validated_signals) == 1
        assert validated_signals[0] == valid_signal


class TestStrategyPerformanceMetrics:
    """Test strategy performance calculation methods"""
    
    def test_confidence_calculation(self):
        """Test confidence calculation with various inputs"""
        strategy = MovingAverageCrossover()
        
        # Test with equal weights
        conditions = {
            'ema_separation': 0.8,
            'volume': 0.6,
            'trend_strength': 0.7
        }
        
        confidence = strategy.calculate_confidence(conditions)
        expected = (0.8 + 0.6 + 0.7) / 3
        assert abs(confidence - expected) < 1e-10
        
        # Test with custom weights
        weights = {
            'ema_separation': 0.5,
            'volume': 0.3,
            'trend_strength': 0.2
        }
        
        confidence = strategy.calculate_confidence(conditions, weights)
        expected = (0.8 * 0.5 + 0.6 * 0.3 + 0.7 * 0.2) / (0.5 + 0.3 + 0.2)
        assert abs(confidence - expected) < 1e-10
        
        # Test edge cases
        assert strategy.calculate_confidence({}) == 0.0
        assert strategy.calculate_confidence({'test': 1.5}) == 1.0  # Should clamp to 1.0
        assert strategy.calculate_confidence({'test': -0.5}) == 0.0  # Should clamp to 0.0
    
    def test_trend_strength_calculation(self):
        """Test trend strength calculation"""
        # Create data with clear uptrend
        prices = np.linspace(100, 110, 20)
        market_data = self._create_market_data(prices)
        
        strategy = MovingAverageCrossover()
        
        # Test bullish trend strength
        bullish_strength = strategy._calculate_trend_strength(market_data, 15, 'bullish')
        assert bullish_strength > 0.5  # Should be high for consistent uptrend
        
        # Test bearish trend strength on same data
        bearish_strength = strategy._calculate_trend_strength(market_data, 15, 'bearish')
        assert bearish_strength < 0.5  # Should be low for uptrend
        
        # Create data with clear downtrend
        down_prices = np.linspace(110, 100, 20)
        down_market_data = self._create_market_data(down_prices)
        
        # Test bearish trend strength
        bearish_strength = strategy._calculate_trend_strength(down_market_data, 15, 'bearish')
        assert bearish_strength > 0.5  # Should be high for consistent downtrend


class TestStrategyIntegration:
    """Integration tests for the complete strategy"""
    
    def test_complete_signal_generation_cycle(self):
        """Test complete signal generation cycle with realistic data"""
        # Create realistic market data with trend changes
        prices = np.concatenate([
            np.linspace(100, 95, 30),    # Initial downtrend
            np.linspace(95, 105, 40),    # Recovery and uptrend  
            np.linspace(105, 98, 30)     # Final decline
        ])
        
        market_data = self._create_market_data(prices)
        strategy = MovingAverageCrossover({
            'short_period': 10,
            'long_period': 20,
            'min_confidence': 0.5,
            'volume_confirmation': False
        })
        
        # Generate signals
        indicators = strategy._calculate_indicators(market_data)
        signals = strategy.generate_signals(market_data, indicators)
        
        # Should have both buy and sell signals
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        
        assert len(buy_signals) > 0
        assert len(sell_signals) > 0
        
        # Signals should be chronologically ordered
        all_timestamps = [s.timestamp for s in signals]
        assert all_timestamps == sorted(all_timestamps)
        
        # All signals should pass validation
        validated_signals = strategy.validate_signals(signals)
        assert len(validated_signals) == len(signals)
    
    def test_strategy_with_different_timeframes(self):
        """Test strategy with different EMA periods"""
        prices = TestDataGenerator.generate_volatile_data(200)
        market_data = self._create_market_data(prices)
        
        # Test different period combinations
        period_combinations = [
            (5, 10),
            (20, 50),
            (50, 100),
            (10, 30)
        ]
        
        for short, long in period_combinations:
            strategy = MovingAverageCrossover({
                'short_period': short,
                'long_period': long,
                'min_confidence': 0.1
            })
            
            indicators = strategy._calculate_indicators(market_data)
            signals = strategy.generate_signals(market_data, indicators)
            
            # Should generate some signals for each combination
            assert isinstance(signals, list)
            
            # All signals should be valid
            for signal in signals:
                assert signal.confidence >= 0.0
                assert signal.confidence <= 1.0
                assert signal.price > 0.0
                assert signal.signal_type in [SignalType.BUY, SignalType.SELL]
    
    def _create_market_data(self, prices: np.ndarray) -> MarketData:
        """Helper method to create MarketData from prices"""
        ohlcv_data = TestDataGenerator.generate_ohlcv_data(prices)
        timestamps = ValidationHelpers.generate_timestamps(len(prices))
        
        ohlcv_list = []
        for i in range(len(prices)):
            ohlcv_list.append(OHLCV(
                timestamp=timestamps[i],
                open=ohlcv_data['open'][i],
                high=ohlcv_data['high'][i],
                low=ohlcv_data['low'][i],
                close=ohlcv_data['close'][i],
                volume=ohlcv_data['volume'][i]
            ))
        
        return MarketData(symbol="TEST", data=ohlcv_list)
    
    def _create_market_data_with_volume(self, prices: np.ndarray, 
                                      volumes: np.ndarray) -> MarketData:
        """Helper method to create MarketData with custom volume"""
        ohlcv_data = TestDataGenerator.generate_ohlcv_data(prices)
        ohlcv_data['volume'] = volumes
        timestamps = ValidationHelpers.generate_timestamps(len(prices))
        
        ohlcv_list = []
        for i in range(len(prices)):
            ohlcv_list.append(OHLCV(
                timestamp=timestamps[i],
                open=ohlcv_data['open'][i],
                high=ohlcv_data['high'][i],
                low=ohlcv_data['low'][i],
                close=ohlcv_data['close'][i],
                volume=ohlcv_data['volume'][i]
            ))
        
        return MarketData(symbol="TEST", data=ohlcv_list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])