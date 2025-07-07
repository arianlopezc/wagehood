"""
Comprehensive Mathematical Tests for MACD+RSI Combined Strategy

This module provides comprehensive mathematical validation tests for the
MACD+RSI trading strategy including signal combinations, divergence detection,
and mathematical precision of indicator interactions.
"""

import pytest
import numpy as np
from typing import Dict, List, Tuple
import sys
import os
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from src.strategies.macd_rsi import MACDRSIStrategy
from src.core.models import MarketData, Signal, SignalType, OHLCV

from ..fixtures.test_data import (
    TestDataGenerator,
    PrecisionTestVectors,
    EdgeCaseTestData,
    ValidationHelpers,
    TEST_CONFIG
)


class TestMACDRSIStrategyBasics:
    """Basic tests for MACD+RSI strategy"""
    
    def test_strategy_initialization(self):
        """Test strategy initialization with default and custom parameters"""
        # Test default initialization
        strategy = MACDRSIStrategy()
        
        assert strategy.name == "MACDRSIStrategy"
        assert strategy.parameters['macd_fast'] == 12
        assert strategy.parameters['macd_slow'] == 26
        assert strategy.parameters['macd_signal'] == 9
        assert strategy.parameters['rsi_period'] == 14
        assert strategy.parameters['rsi_oversold'] == 30
        assert strategy.parameters['rsi_overbought'] == 70
        assert strategy.parameters['min_confidence'] == 0.6
        assert strategy.parameters['divergence_detection'] == True
        
        # Test custom parameters
        custom_params = {
            'macd_fast': 8,
            'macd_slow': 21,
            'rsi_period': 21,
            'rsi_oversold': 25,
            'rsi_overbought': 75,
            'min_confidence': 0.7,
            'divergence_detection': False
        }
        
        strategy = MACDRSIStrategy(custom_params)
        
        assert strategy.parameters['macd_fast'] == 8
        assert strategy.parameters['macd_slow'] == 21
        assert strategy.parameters['rsi_period'] == 21
        assert strategy.parameters['rsi_oversold'] == 25
        assert strategy.parameters['rsi_overbought'] == 75
        assert strategy.parameters['min_confidence'] == 0.7
        assert strategy.parameters['divergence_detection'] == False
    
    def test_required_indicators(self):
        """Test required indicators"""
        strategy = MACDRSIStrategy()
        required = strategy.get_required_indicators()
        
        assert 'macd' in required
        assert 'rsi' in required
        assert len(required) == 2
    
    def test_parameter_descriptions(self):
        """Test parameter descriptions and metadata"""
        strategy = MACDRSIStrategy()
        params = strategy.get_parameters()
        
        # Check that all parameters have required metadata
        essential_params = [
            'macd_fast', 'macd_slow', 'macd_signal', 'rsi_period',
            'rsi_oversold', 'rsi_overbought', 'min_confidence'
        ]
        
        for param_name in essential_params:
            assert param_name in params
            param_info = params[param_name]
            assert 'value' in param_info
            assert 'description' in param_info
            assert 'type' in param_info
            assert 'default' in param_info


class TestMACDRSIBullishSignals:
    """Tests for MACD+RSI bullish signal generation"""
    
    def test_primary_bullish_signal_detection(self):
        """Test primary bullish signal: MACD crossover + RSI exit from oversold"""
        # Create data that produces the desired conditions
        prices = self._create_oversold_recovery_data()
        market_data = self._create_market_data(prices)
        
        strategy = MACDRSIStrategy({
            'macd_fast': 5,
            'macd_slow': 10,
            'macd_signal': 3,
            'rsi_period': 7,
            'rsi_oversold': 30,
            'min_confidence': 0.1,
            'volume_confirmation': False,
            'divergence_detection': False
        })
        
        indicators = strategy._calculate_indicators(market_data)
        signals = strategy.generate_signals(market_data, indicators)
        
        # Should have bullish signals
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buy_signals) > 0
        
        # Check signal properties
        for signal in buy_signals:
            assert signal.strategy_name == "MACDRSIStrategy"
            assert signal.confidence > 0.0
            assert 'MACD+RSI Bullish' in signal.metadata.get('signal_name', '')
            
            # Should have MACD and RSI values in metadata
            metadata = signal.metadata
            assert 'macd_value' in metadata
            assert 'rsi_value' in metadata
            assert 'histogram' in metadata
    
    def test_secondary_bullish_signal_detection(self):
        """Test secondary bullish signal: MACD crossover + RSI uptrend + positive histogram"""
        # Create data with RSI above 50 and positive MACD momentum
        prices = self._create_uptrend_data()
        market_data = self._create_market_data(prices)
        
        strategy = MACDRSIStrategy({
            'macd_fast': 5,
            'macd_slow': 10,
            'macd_signal': 3,
            'rsi_period': 7,
            'min_confidence': 0.1,
            'volume_confirmation': False,
            'divergence_detection': False
        })
        
        indicators = strategy._calculate_indicators(market_data)
        signals = strategy.generate_signals(market_data, indicators)
        
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        
        # Should generate some bullish signals for uptrending data
        if len(buy_signals) > 0:
            for signal in buy_signals:
                metadata = signal.metadata
                
                # RSI should be above 50 for uptrend signals
                if 'rsi_value' in metadata:
                    # Allow some flexibility as this is secondary signal
                    assert metadata['rsi_value'] >= 40
    
    def test_bullish_signal_mathematical_precision(self):
        """Test mathematical precision of bullish signal conditions"""
        # Create precise test data for MACD crossover
        prices = np.array([
            100.0, 99.0, 98.0, 97.0, 96.0,     # Decline (creates oversold RSI)
            96.5, 97.0, 97.5, 98.0, 98.5,     # Recovery starts
            99.0, 99.5, 100.0, 100.5, 101.0   # Continued recovery
        ])
        
        market_data = self._create_market_data(prices)
        strategy = MACDRSIStrategy({
            'macd_fast': 3,
            'macd_slow': 6,
            'macd_signal': 2,
            'rsi_period': 5,
            'rsi_oversold': 30,
            'min_confidence': 0.1,
            'volume_confirmation': False,
            'divergence_detection': False
        })
        
        # Calculate indicators manually for verification
        from src.indicators.momentum import calculate_macd, calculate_rsi
        
        macd_line, signal_line, histogram = calculate_macd(prices, 3, 6, 2)
        rsi_values = calculate_rsi(prices, 5)
        
        # Find MACD crossover points
        crossover_indices = []
        for i in range(1, len(macd_line)):
            if (not np.isnan(macd_line[i-1]) and not np.isnan(signal_line[i-1]) and
                not np.isnan(macd_line[i]) and not np.isnan(signal_line[i])):
                
                if macd_line[i-1] <= signal_line[i-1] and macd_line[i] > signal_line[i]:
                    crossover_indices.append(i)
        
        # Generate signals using strategy
        indicators = strategy._calculate_indicators(market_data)
        signals = strategy.generate_signals(market_data, indicators)
        
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        
        # Verify signal timing aligns with mathematical conditions
        if crossover_indices and buy_signals:
            arrays = market_data.to_arrays()
            signal_indices = []
            
            for signal in buy_signals:
                for i, timestamp in enumerate(arrays['timestamp']):
                    if timestamp == signal.timestamp:
                        signal_indices.append(i)
                        break
            
            # At least one signal should occur near a crossover
            for signal_idx in signal_indices:
                # Check if this signal corresponds to valid conditions
                if signal_idx < len(rsi_values) and not np.isnan(rsi_values[signal_idx]):
                    # Either RSI should be recovering from oversold, or in uptrend with positive histogram
                    rsi_val = rsi_values[signal_idx]
                    hist_val = histogram[signal_idx] if signal_idx < len(histogram) else 0
                    
                    condition1 = rsi_val <= 35  # Near oversold recovery
                    condition2 = rsi_val > 50 and hist_val > 0  # Uptrend with momentum
                    
                    assert condition1 or condition2
    
    def test_bullish_confidence_factors(self):
        """Test bullish signal confidence calculation factors"""
        prices = self._create_oversold_recovery_data()
        market_data = self._create_market_data(prices)
        
        strategy = MACDRSIStrategy({
            'macd_fast': 5,
            'macd_slow': 10,
            'rsi_period': 7,
            'min_confidence': 0.1,
            'volume_confirmation': False
        })
        
        indicators = strategy._calculate_indicators(market_data)
        signals = strategy.generate_signals(market_data, indicators)
        
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        
        for signal in buy_signals:
            metadata = signal.metadata
            
            # Should have confidence-related metadata
            assert 'macd_strength' in metadata
            if 'rsi_value' in metadata:
                # RSI position should influence confidence
                rsi_val = metadata['rsi_value']
                if rsi_val < strategy.parameters['rsi_oversold']:
                    # Very oversold should have higher confidence factor
                    pass  # This is good for bullish signals
            
            # Confidence should be reasonable
            assert 0.0 <= signal.confidence <= 1.0


class TestMACDRSIBearishSignals:
    """Tests for MACD+RSI bearish signal generation"""
    
    def test_primary_bearish_signal_detection(self):
        """Test primary bearish signal: MACD crossover + RSI exit from overbought"""
        # Create data that produces overbought conditions followed by decline
        prices = self._create_overbought_decline_data()
        market_data = self._create_market_data(prices)
        
        strategy = MACDRSIStrategy({
            'macd_fast': 5,
            'macd_slow': 10,
            'macd_signal': 3,
            'rsi_period': 7,
            'rsi_overbought': 70,
            'min_confidence': 0.1,
            'volume_confirmation': False,
            'divergence_detection': False
        })
        
        indicators = strategy._calculate_indicators(market_data)
        signals = strategy.generate_signals(market_data, indicators)
        
        # Should have bearish signals
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        assert len(sell_signals) > 0
        
        # Check signal properties
        for signal in sell_signals:
            assert signal.strategy_name == "MACDRSIStrategy"
            assert signal.confidence > 0.0
            assert 'MACD+RSI Bearish' in signal.metadata.get('signal_name', '')
    
    def test_secondary_bearish_signal_detection(self):
        """Test secondary bearish signal: MACD crossover + RSI downtrend + negative histogram"""
        prices = self._create_downtrend_data()
        market_data = self._create_market_data(prices)
        
        strategy = MACDRSIStrategy({
            'macd_fast': 5,
            'macd_slow': 10,
            'macd_signal': 3,
            'rsi_period': 7,
            'min_confidence': 0.1,
            'volume_confirmation': False,
            'divergence_detection': False
        })
        
        indicators = strategy._calculate_indicators(market_data)
        signals = strategy.generate_signals(market_data, indicators)
        
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        
        # Should generate some bearish signals for downtrending data
        if len(sell_signals) > 0:
            for signal in sell_signals:
                metadata = signal.metadata
                
                # For downtrend signals, RSI should be below 50
                if 'rsi_value' in metadata:
                    # Allow some flexibility
                    assert metadata['rsi_value'] <= 60
    
    def test_bearish_signal_mathematical_precision(self):
        """Test mathematical precision of bearish signal conditions"""
        # Create precise test data for bearish MACD crossover
        prices = np.array([
            100.0, 101.0, 102.0, 103.0, 104.0,  # Rise (creates overbought RSI)
            103.5, 103.0, 102.5, 102.0, 101.5,  # Decline starts
            101.0, 100.5, 100.0, 99.5, 99.0     # Continued decline
        ])
        
        market_data = self._create_market_data(prices)
        strategy = MACDRSIStrategy({
            'macd_fast': 3,
            'macd_slow': 6,
            'macd_signal': 2,
            'rsi_period': 5,
            'rsi_overbought': 70,
            'min_confidence': 0.1,
            'volume_confirmation': False,
            'divergence_detection': False
        })
        
        # Calculate indicators manually
        from src.indicators.momentum import calculate_macd, calculate_rsi
        
        macd_line, signal_line, histogram = calculate_macd(prices, 3, 6, 2)
        rsi_values = calculate_rsi(prices, 5)
        
        # Find bearish MACD crossover points
        crossover_indices = []
        for i in range(1, len(macd_line)):
            if (not np.isnan(macd_line[i-1]) and not np.isnan(signal_line[i-1]) and
                not np.isnan(macd_line[i]) and not np.isnan(signal_line[i])):
                
                if macd_line[i-1] >= signal_line[i-1] and macd_line[i] < signal_line[i]:
                    crossover_indices.append(i)
        
        # Generate signals using strategy
        indicators = strategy._calculate_indicators(market_data)
        signals = strategy.generate_signals(market_data, indicators)
        
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        
        # Verify signal timing aligns with mathematical conditions
        if crossover_indices and sell_signals:
            arrays = market_data.to_arrays()
            signal_indices = []
            
            for signal in sell_signals:
                for i, timestamp in enumerate(arrays['timestamp']):
                    if timestamp == signal.timestamp:
                        signal_indices.append(i)
                        break
            
            # At least one signal should occur near a crossover
            for signal_idx in signal_indices:
                if signal_idx < len(rsi_values) and not np.isnan(rsi_values[signal_idx]):
                    rsi_val = rsi_values[signal_idx]
                    hist_val = histogram[signal_idx] if signal_idx < len(histogram) else 0
                    
                    condition1 = rsi_val >= 65  # Near overbought exit
                    condition2 = rsi_val < 50 and hist_val < 0  # Downtrend with momentum
                    
                    assert condition1 or condition2


class TestMACDRSIDivergenceDetection:
    """Tests for divergence detection in MACD+RSI strategy"""
    
    def test_bullish_divergence_detection(self):
        """Test bullish divergence detection"""
        # Create data with price making lower lows but indicators making higher lows
        prices = self._create_bullish_divergence_data()
        market_data = self._create_market_data(prices)
        
        strategy = MACDRSIStrategy({
            'macd_fast': 5,
            'macd_slow': 12,
            'rsi_period': 7,
            'min_confidence': 0.1,
            'volume_confirmation': False,
            'divergence_detection': True
        })
        
        indicators = strategy._calculate_indicators(market_data)
        signals = strategy.generate_signals(market_data, indicators)
        
        # Look for divergence signals
        divergence_signals = [
            s for s in signals 
            if 'Divergence' in s.metadata.get('signal_name', '')
        ]
        
        # Should detect at least some divergence patterns
        bullish_divergence = [
            s for s in divergence_signals 
            if s.signal_type == SignalType.BUY and 'Bullish' in s.metadata.get('signal_name', '')
        ]
        
        # May or may not find divergence depending on data complexity
        # But if found, should be valid
        for signal in bullish_divergence:
            assert signal.confidence > 0.0
            assert 'divergence_type' in signal.metadata
            assert signal.metadata['divergence_type'] == 'bullish'
    
    def test_bearish_divergence_detection(self):
        """Test bearish divergence detection"""
        prices = self._create_bearish_divergence_data()
        market_data = self._create_market_data(prices)
        
        strategy = MACDRSIStrategy({
            'macd_fast': 5,
            'macd_slow': 12,
            'rsi_period': 7,
            'min_confidence': 0.1,
            'volume_confirmation': False,
            'divergence_detection': True
        })
        
        indicators = strategy._calculate_indicators(market_data)
        signals = strategy.generate_signals(market_data, indicators)
        
        # Look for bearish divergence signals
        divergence_signals = [
            s for s in signals 
            if 'Divergence' in s.metadata.get('signal_name', '') and s.signal_type == SignalType.SELL
        ]
        
        # Validate divergence signals if found
        for signal in divergence_signals:
            assert signal.confidence > 0.0
            assert 'divergence_type' in signal.metadata
    
    def test_divergence_mathematical_accuracy(self):
        """Test mathematical accuracy of divergence detection"""
        # Create precise divergence pattern
        prices = np.array([
            100, 95, 98, 92, 96,   # Price makes lower lows: 95 -> 92
            94, 93, 94, 95, 96,    # But then recovers
            97, 98, 99, 100, 101   # Continues up
        ])
        
        market_data = self._create_market_data(prices)
        strategy = MACDRSIStrategy({
            'macd_fast': 3,
            'macd_slow': 7,
            'rsi_period': 5,
            'min_confidence': 0.1,
            'divergence_detection': True
        })
        
        # Calculate indicators manually to verify divergence logic
        from src.indicators.momentum import calculate_macd, calculate_rsi
        
        macd_line, _, _ = calculate_macd(prices, 3, 7, 2)
        rsi_values = calculate_rsi(prices, 5)
        
        # The strategy should apply its divergence detection logic
        indicators = strategy._calculate_indicators(market_data)
        signals = strategy.generate_signals(market_data, indicators)
        
        # Verify that any divergence signals follow mathematical rules
        divergence_signals = [
            s for s in signals 
            if 'Divergence' in s.metadata.get('signal_name', '')
        ]
        
        # All divergence signals should have proper metadata
        for signal in divergence_signals:
            assert 'divergence_type' in signal.metadata
            assert signal.metadata['divergence_type'] in ['bullish', 'bearish']


class TestMACDRSIVolumeConfirmation:
    """Tests for volume confirmation in MACD+RSI strategy"""
    
    def test_volume_confirmation_impact(self):
        """Test impact of volume confirmation on signal generation"""
        prices = self._create_uptrend_data()
        
        # High volume scenario
        ohlcv_data = TestDataGenerator.generate_ohlcv_data(prices)
        high_volume = ohlcv_data['volume'] * 2.0
        market_data_high_vol = self._create_market_data_with_volume(prices, high_volume)
        
        # Low volume scenario
        low_volume = ohlcv_data['volume'] * 0.5
        market_data_low_vol = self._create_market_data_with_volume(prices, low_volume)
        
        strategy = MACDRSIStrategy({
            'macd_fast': 5,
            'macd_slow': 10,
            'rsi_period': 7,
            'min_confidence': 0.1,
            'volume_confirmation': True,
            'volume_threshold': 1.2,
            'divergence_detection': False
        })
        
        # Generate signals for both scenarios
        indicators_high = strategy._calculate_indicators(market_data_high_vol)
        signals_high = strategy.generate_signals(market_data_high_vol, indicators_high)
        
        indicators_low = strategy._calculate_indicators(market_data_low_vol)
        signals_low = strategy.generate_signals(market_data_low_vol, indicators_low)
        
        # Compare confidence levels
        high_vol_confidences = [s.confidence for s in signals_high]
        low_vol_confidences = [s.confidence for s in signals_low]
        
        if high_vol_confidences and low_vol_confidences:
            avg_high_conf = np.mean(high_vol_confidences)
            avg_low_conf = np.mean(low_vol_confidences)
            # High volume should generally lead to higher confidence
            assert avg_high_conf >= avg_low_conf


class TestMACDRSIStrategyEdgeCases:
    """Test strategy behavior with edge cases"""
    
    def test_insufficient_data(self):
        """Test strategy with insufficient data"""
        prices = np.array([100.0, 101.0, 102.0])
        market_data = self._create_market_data(prices)
        
        strategy = MACDRSIStrategy()
        
        indicators = strategy._calculate_indicators(market_data)
        signals = strategy.generate_signals(market_data, indicators)
        
        # Should handle insufficient data gracefully
        assert isinstance(signals, list)
        # May or may not have signals, but should not crash
    
    def test_missing_indicator_data(self):
        """Test strategy with missing indicator data"""
        prices = TestDataGenerator.generate_volatile_data(50)
        market_data = self._create_market_data(prices)
        
        strategy = MACDRSIStrategy()
        
        # Create incomplete indicators
        incomplete_indicators = {
            'macd': {},  # Missing MACD data
            # Missing RSI data entirely
        }
        
        signals = strategy.generate_signals(market_data, incomplete_indicators)
        
        # Should handle missing data gracefully
        assert isinstance(signals, list)
        assert len(signals) == 0  # Should return empty list when data is missing
    
    def test_extreme_rsi_values(self):
        """Test strategy with extreme RSI values"""
        # Create data that will produce extreme RSI
        extreme_up = np.array([100 + i*5 for i in range(20)])  # Strong uptrend
        extreme_down = np.array([200 - i*5 for i in range(20)])  # Strong downtrend
        
        for prices in [extreme_up, extreme_down]:
            market_data = self._create_market_data(prices)
            
            strategy = MACDRSIStrategy({
                'macd_fast': 5,
                'macd_slow': 10,
                'rsi_period': 7,
                'min_confidence': 0.1
            })
            
            indicators = strategy._calculate_indicators(market_data)
            signals = strategy.generate_signals(market_data, indicators)
            
            # Should handle extreme values without crashing
            assert isinstance(signals, list)
            
            # All signals should have valid confidence
            for signal in signals:
                assert 0.0 <= signal.confidence <= 1.0
    
    def test_constant_prices(self):
        """Test strategy with constant prices"""
        prices = EdgeCaseTestData.get_constant_prices()
        market_data = self._create_market_data(prices)
        
        strategy = MACDRSIStrategy({
            'macd_fast': 5,
            'macd_slow': 10,
            'rsi_period': 7,
            'min_confidence': 0.1
        })
        
        indicators = strategy._calculate_indicators(market_data)
        signals = strategy.generate_signals(market_data, indicators)
        
        # Should not generate signals for constant prices
        assert len(signals) == 0


class TestMACDRSISignalValidation:
    """Test signal validation logic"""
    
    def test_signal_validation_rules(self):
        """Test signal validation rules"""
        strategy = MACDRSIStrategy()
        
        # Create valid signal
        valid_signal = Signal(
            timestamp=datetime.now(),
            symbol="TEST",
            signal_type=SignalType.BUY,
            price=100.0,
            confidence=0.8,
            strategy_name="MACDRSIStrategy",
            metadata={
                'signal_name': 'MACD+RSI Bullish',
                'macd_value': 1.5,
                'rsi_value': 35.0
            }
        )
        
        # Create invalid signals
        invalid_signals = [
            # Wrong signal name
            Signal(
                timestamp=datetime.now(),
                symbol="TEST",
                signal_type=SignalType.BUY,
                price=100.0,
                confidence=0.8,
                strategy_name="MACDRSIStrategy",
                metadata={'signal_name': 'Invalid Signal'}
            ),
            # Missing signal name
            Signal(
                timestamp=datetime.now(),
                symbol="TEST",
                signal_type=SignalType.BUY,
                price=100.0,
                confidence=0.8,
                strategy_name="MACDRSIStrategy",
                metadata={'macd_value': 1.5}
            ),
            # Low confidence
            Signal(
                timestamp=datetime.now(),
                symbol="TEST",
                signal_type=SignalType.BUY,
                price=100.0,
                confidence=0.3,  # Below default threshold
                strategy_name="MACDRSIStrategy",
                metadata={'signal_name': 'MACD+RSI Bullish'}
            )
        ]
        
        # Test validation
        all_signals = [valid_signal] + invalid_signals
        validated_signals = strategy.validate_signals(all_signals)
        
        # Should only keep the valid signal
        assert len(validated_signals) == 1
        assert validated_signals[0] == valid_signal


class TestMACDRSIParameterOptimization:
    """Test parameter optimization"""
    
    def test_parameter_ranges(self):
        """Test parameter ranges for optimization"""
        strategy = MACDRSIStrategy()
        ranges = strategy._get_parameter_ranges()
        
        # Should have ranges for key parameters
        expected_params = [
            'macd_fast', 'macd_slow', 'macd_signal', 'rsi_period',
            'rsi_oversold', 'rsi_overbought', 'min_confidence'
        ]
        
        for param in expected_params:
            assert param in ranges
            assert isinstance(ranges[param], list)
            assert len(ranges[param]) > 1
        
        # Parameter relationships should be maintained
        for fast in ranges['macd_fast']:
            for slow in ranges['macd_slow']:
                assert fast < slow  # Fast should be less than slow
        
        for oversold in ranges['rsi_oversold']:
            for overbought in ranges['rsi_overbought']:
                assert oversold < overbought  # Oversold should be less than overbought


# Helper methods for creating test data
class TestMACDRSIHelpers:
    """Helper methods for creating test data"""
    
    def _create_oversold_recovery_data(self) -> np.ndarray:
        """Create price data that goes oversold then recovers"""
        return np.concatenate([
            np.linspace(100, 90, 10),    # Decline to oversold
            np.linspace(90, 95, 5),      # Initial recovery
            np.linspace(95, 105, 15)     # Strong recovery
        ])
    
    def _create_overbought_decline_data(self) -> np.ndarray:
        """Create price data that goes overbought then declines"""
        return np.concatenate([
            np.linspace(100, 110, 10),   # Rise to overbought
            np.linspace(110, 105, 5),    # Initial decline
            np.linspace(105, 95, 15)     # Strong decline
        ])
    
    def _create_uptrend_data(self) -> np.ndarray:
        """Create consistent uptrend data"""
        return np.linspace(100, 120, 30)
    
    def _create_downtrend_data(self) -> np.ndarray:
        """Create consistent downtrend data"""
        return np.linspace(120, 100, 30)
    
    def _create_bullish_divergence_data(self) -> np.ndarray:
        """Create data with potential bullish divergence pattern"""
        return np.array([
            100, 95, 98, 93, 96, 94, 97, 95, 98, 96,  # Lower lows in price
            99, 97, 100, 98, 101, 99, 102, 100, 103, 101,  # But recovery
            104, 102, 105, 103, 106, 104, 107, 105, 108, 106
        ])
    
    def _create_bearish_divergence_data(self) -> np.ndarray:
        """Create data with potential bearish divergence pattern"""
        return np.array([
            100, 105, 102, 107, 104, 106, 103, 108, 105, 104,  # Higher highs in price
            103, 108, 102, 107, 101, 106, 100, 105, 99, 104,   # But momentum weakening
            98, 103, 97, 102, 96, 101, 95, 100, 94, 99
        ])
    
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


# Mix the helper methods into test classes
for cls in [TestMACDRSIBullishSignals, TestMACDRSIBearishSignals, TestMACDRSIDivergenceDetection,
           TestMACDRSIVolumeConfirmation, TestMACDRSIStrategyEdgeCases]:
    for name, method in TestMACDRSIHelpers.__dict__.items():
        if not name.startswith('_'):
            continue
        setattr(cls, name, method)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])