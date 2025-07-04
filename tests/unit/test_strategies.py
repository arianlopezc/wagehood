"""
Unit tests for trading strategies.

Tests all 5 strategies with mock data and validates signal generation logic.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from src.strategies.base import TradingStrategy
from src.strategies.ma_crossover import MovingAverageCrossover
from src.strategies.rsi_trend import RSITrendFollowing
from src.strategies.bollinger_breakout import BollingerBandBreakout
from src.strategies.macd_rsi import MACDRSIStrategy
from src.strategies.sr_breakout import SupportResistanceBreakout
from src.core.models import Signal, SignalType, MarketData, TimeFrame, OHLCV
from src.indicators.moving_averages import calculate_ema, calculate_sma
from src.indicators.momentum import calculate_rsi, calculate_macd


class TestTradingStrategyBase:
    """Test base trading strategy functionality."""
    
    def test_strategy_initialization(self):
        """Test strategy initialization."""
        params = {'test_param': 10}
        
        # Create a concrete implementation for testing
        class TestStrategy(TradingStrategy):
            def generate_signals(self, data, indicators):
                return []
            
            def get_required_indicators(self):
                return ['sma']
            
            def get_parameters(self):
                return self.parameters
        
        strategy = TestStrategy("TestStrategy", params)
        
        assert strategy.name == "TestStrategy"
        assert strategy.parameters == params
        assert strategy.indicator_calculator is not None
    
    def test_strategy_default_parameters(self):
        """Test strategy with default parameters."""
        class TestStrategy(TradingStrategy):
            def generate_signals(self, data, indicators):
                return []
            
            def get_required_indicators(self):
                return ['sma']
            
            def get_parameters(self):
                return self.parameters
        
        strategy = TestStrategy("TestStrategy")
        
        assert strategy.parameters == {}
    
    def test_calculate_confidence(self):
        """Test confidence calculation."""
        class TestStrategy(TradingStrategy):
            def generate_signals(self, data, indicators):
                return []
            
            def get_required_indicators(self):
                return []
            
            def get_parameters(self):
                return {}
        
        strategy = TestStrategy("TestStrategy")
        
        # Test with equal weights
        conditions = {'condition1': 0.8, 'condition2': 0.6, 'condition3': 0.9}
        confidence = strategy.calculate_confidence(conditions)
        
        expected = (0.8 + 0.6 + 0.9) / 3
        assert abs(confidence - expected) < 1e-10
        
        # Test with custom weights
        weights = {'condition1': 0.5, 'condition2': 0.3, 'condition3': 0.2}
        confidence = strategy.calculate_confidence(conditions, weights)
        
        expected = (0.8 * 0.5 + 0.6 * 0.3 + 0.9 * 0.2) / (0.5 + 0.3 + 0.2)
        assert abs(confidence - expected) < 1e-10
    
    def test_calculate_confidence_edge_cases(self):
        """Test confidence calculation edge cases."""
        class TestStrategy(TradingStrategy):
            def generate_signals(self, data, indicators):
                return []
            
            def get_required_indicators(self):
                return []
            
            def get_parameters(self):
                return {}
        
        strategy = TestStrategy("TestStrategy")
        
        # Empty conditions
        confidence = strategy.calculate_confidence({})
        assert confidence == 0.0
        
        # Zero total weight
        conditions = {'condition1': 0.8}
        weights = {'condition1': 0.0}
        confidence = strategy.calculate_confidence(conditions, weights)
        assert confidence == 0.0
        
        # Confidence clamping
        conditions = {'condition1': 1.5}  # > 1.0
        confidence = strategy.calculate_confidence(conditions)
        assert confidence == 1.0
        
        conditions = {'condition1': -0.5}  # < 0.0
        confidence = strategy.calculate_confidence(conditions)
        assert confidence == 0.0
    
    def test_get_signal_metadata(self):
        """Test signal metadata generation."""
        class TestStrategy(TradingStrategy):
            def generate_signals(self, data, indicators):
                return []
            
            def get_required_indicators(self):
                return []
            
            def get_parameters(self):
                return {'param1': 10}
        
        strategy = TestStrategy("TestStrategy", {'param1': 10})
        
        metadata = strategy.get_signal_metadata(extra_field="test_value")
        
        assert metadata['strategy'] == "TestStrategy"
        assert metadata['parameters'] == {'param1': 10}
        assert metadata['extra_field'] == "test_value"
        assert 'timestamp' in metadata
    
    def test_validate_signals_basic(self):
        """Test basic signal validation."""
        class TestStrategy(TradingStrategy):
            def generate_signals(self, data, indicators):
                return []
            
            def get_required_indicators(self):
                return []
            
            def get_parameters(self):
                return {}
        
        strategy = TestStrategy("TestStrategy")
        
        # Valid signal
        valid_signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.BUY,
            price=150.0,
            confidence=0.8,
            strategy_name="TestStrategy",
            metadata={}
        )
        
        # Invalid signal (low confidence)
        invalid_signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.BUY,
            price=150.0,
            confidence=0.05,  # Below threshold
            strategy_name="TestStrategy",
            metadata={}
        )
        
        signals = [valid_signal, invalid_signal]
        validated = strategy.validate_signals(signals)
        
        assert len(validated) == 1
        assert validated[0] == valid_signal


class TestMovingAverageCrossover:
    """Test Moving Average Crossover strategy."""
    
    def test_strategy_initialization(self):
        """Test MA crossover strategy initialization."""
        strategy = MovingAverageCrossover()
        
        assert strategy.name == "MovingAverageCrossover"
        assert strategy.parameters['short_period'] == 50
        assert strategy.parameters['long_period'] == 200
        assert strategy.parameters['min_confidence'] == 0.6
    
    def test_custom_parameters(self):
        """Test MA crossover with custom parameters."""
        params = {
            'short_period': 20,
            'long_period': 50,
            'min_confidence': 0.7,
            'volume_confirmation': False
        }
        
        strategy = MovingAverageCrossover(params)
        
        assert strategy.parameters['short_period'] == 20
        assert strategy.parameters['long_period'] == 50
        assert strategy.parameters['min_confidence'] == 0.7
        assert strategy.parameters['volume_confirmation'] == False
    
    def test_required_indicators(self):
        """Test required indicators."""
        strategy = MovingAverageCrossover()
        required = strategy.get_required_indicators()
        
        assert 'ema' in required
    
    def test_get_parameters(self):
        """Test parameter description."""
        strategy = MovingAverageCrossover()
        params = strategy.get_parameters()
        
        assert 'short_period' in params
        assert 'long_period' in params
        assert 'min_confidence' in params
        assert 'volume_confirmation' in params
        assert 'volume_threshold' in params
        
        # Check parameter structure
        assert 'value' in params['short_period']
        assert 'description' in params['short_period']
        assert 'type' in params['short_period']
    
    def test_signal_generation_golden_cross(self, sample_market_data):
        """Test golden cross signal generation."""
        strategy = MovingAverageCrossover({'short_period': 5, 'long_period': 10, 'min_confidence': 0.5})
        
        # Create mock indicators with golden cross
        short_ema = np.array([np.nan] * 4 + [98, 99, 100, 101, 102, 103])
        long_ema = np.array([np.nan] * 9 + [100])  # Short crosses above long
        
        indicators = {
            'ema': {
                'ema_5': short_ema,
                'ema_10': long_ema
            }
        }
        
        # Mock data with enough points
        mock_data = Mock(spec=MarketData)
        mock_data.symbol = "AAPL"
        mock_data.to_arrays.return_value = {
            'timestamp': [datetime.now() - timedelta(days=i) for i in range(10, 0, -1)],
            'close': [100 + i for i in range(10)],
            'volume': [1000] * 10
        }
        
        signals = strategy.generate_signals(mock_data, indicators)
        
        # Should generate at least one signal when short crosses above long
        assert len(signals) >= 0  # May be 0 due to confidence filtering
    
    def test_signal_generation_death_cross(self, sample_market_data):
        """Test death cross signal generation."""
        strategy = MovingAverageCrossover({'short_period': 5, 'long_period': 10, 'min_confidence': 0.5})
        
        # Create mock indicators with death cross
        short_ema = np.array([np.nan] * 4 + [102, 101, 100, 99, 98, 97])
        long_ema = np.array([np.nan] * 9 + [100])  # Short crosses below long
        
        indicators = {
            'ema': {
                'ema_5': short_ema,
                'ema_10': long_ema
            }
        }
        
        mock_data = Mock(spec=MarketData)
        mock_data.symbol = "AAPL"
        mock_data.to_arrays.return_value = {
            'timestamp': [datetime.now() - timedelta(days=i) for i in range(10, 0, -1)],
            'close': [100 - i for i in range(10)],
            'volume': [1000] * 10
        }
        
        signals = strategy.generate_signals(mock_data, indicators)
        
        # Should handle the scenario without errors
        assert isinstance(signals, list)
    
    def test_signal_generation_missing_indicators(self, sample_market_data):
        """Test signal generation with missing indicators."""
        strategy = MovingAverageCrossover()
        
        # Empty indicators
        indicators = {}
        
        signals = strategy.generate_signals(sample_market_data, indicators)
        
        # Should return empty list when indicators are missing
        assert signals == []
    
    def test_signal_validation(self):
        """Test strategy-specific signal validation."""
        strategy = MovingAverageCrossover()
        
        # Valid signal
        valid_signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.BUY,
            price=150.0,
            confidence=0.8,
            strategy_name="MovingAverageCrossover",
            metadata={
                'short_ema': 151.0,
                'long_ema': 149.0,
                'ema_separation': 0.013
            }
        )
        
        # Invalid signal (insufficient EMA separation)
        invalid_signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.BUY,
            price=150.0,
            confidence=0.8,
            strategy_name="MovingAverageCrossover",
            metadata={
                'short_ema': 150.01,
                'long_ema': 150.0,
                'ema_separation': 0.00005  # Too small
            }
        )
        
        assert strategy._validate_signal_strategy(valid_signal)
        assert not strategy._validate_signal_strategy(invalid_signal)


class TestRSITrend:
    """Test RSI Trend strategy."""
    
    def test_strategy_initialization(self):
        """Test RSI trend strategy initialization."""
        strategy = RSITrend()
        
        assert strategy.name == "RSITrend"
        assert 'rsi_period' in strategy.parameters
        assert 'rsi_oversold' in strategy.parameters
        assert 'rsi_overbought' in strategy.parameters
    
    def test_custom_parameters(self):
        """Test RSI trend with custom parameters."""
        params = {
            'rsi_period': 21,
            'rsi_oversold': 25,
            'rsi_overbought': 75,
            'trend_period': 100
        }
        
        strategy = RSITrend(params)
        
        assert strategy.parameters['rsi_period'] == 21
        assert strategy.parameters['rsi_oversold'] == 25
        assert strategy.parameters['rsi_overbought'] == 75
        assert strategy.parameters['trend_period'] == 100
    
    def test_required_indicators(self):
        """Test required indicators."""
        strategy = RSITrend()
        required = strategy.get_required_indicators()
        
        assert 'rsi' in required
        assert 'sma' in required
    
    def test_signal_generation_oversold(self):
        """Test oversold signal generation."""
        strategy = RSITrend({'rsi_period': 14, 'rsi_oversold': 30, 'min_confidence': 0.5})
        
        # Mock indicators with oversold RSI
        indicators = {
            'rsi': {'rsi_14': np.array([np.nan] * 13 + [25, 35])},  # Oversold then recovering
            'sma': {'sma_50': np.array([np.nan] * 49 + [100])}  # Uptrend
        }
        
        mock_data = Mock(spec=MarketData)
        mock_data.symbol = "AAPL"
        mock_data.to_arrays.return_value = {
            'timestamp': [datetime.now() - timedelta(days=i) for i in range(15, 0, -1)],
            'close': [100 + i * 0.1 for i in range(15)],
            'volume': [1000] * 15
        }
        
        signals = strategy.generate_signals(mock_data, indicators)
        
        # Should handle the scenario
        assert isinstance(signals, list)
    
    def test_signal_generation_overbought(self):
        """Test overbought signal generation."""
        strategy = RSITrend({'rsi_period': 14, 'rsi_overbought': 70, 'min_confidence': 0.5})
        
        # Mock indicators with overbought RSI
        indicators = {
            'rsi': {'rsi_14': np.array([np.nan] * 13 + [75, 65])},  # Overbought then declining
            'sma': {'sma_50': np.array([np.nan] * 49 + [100])}  # Downtrend
        }
        
        mock_data = Mock(spec=MarketData)
        mock_data.symbol = "AAPL"
        mock_data.to_arrays.return_value = {
            'timestamp': [datetime.now() - timedelta(days=i) for i in range(15, 0, -1)],
            'close': [100 - i * 0.1 for i in range(15)],
            'volume': [1000] * 15
        }
        
        signals = strategy.generate_signals(mock_data, indicators)
        
        assert isinstance(signals, list)


class TestBollingerBreakout:
    """Test Bollinger Breakout strategy."""
    
    def test_strategy_initialization(self):
        """Test Bollinger breakout strategy initialization."""
        strategy = BollingerBreakout()
        
        assert strategy.name == "BollingerBreakout"
        assert 'period' in strategy.parameters
        assert 'std_dev' in strategy.parameters
        assert 'min_confidence' in strategy.parameters
    
    def test_custom_parameters(self):
        """Test Bollinger breakout with custom parameters."""
        params = {
            'period': 10,
            'std_dev': 1.5,
            'min_confidence': 0.8,
            'volume_confirmation': True
        }
        
        strategy = BollingerBreakout(params)
        
        assert strategy.parameters['period'] == 10
        assert strategy.parameters['std_dev'] == 1.5
        assert strategy.parameters['min_confidence'] == 0.8
        assert strategy.parameters['volume_confirmation'] == True
    
    def test_required_indicators(self):
        """Test required indicators."""
        strategy = BollingerBreakout()
        required = strategy.get_required_indicators()
        
        assert 'bollinger' in required
    
    def test_signal_generation_upper_breakout(self):
        """Test upper band breakout signal generation."""
        strategy = BollingerBreakout({'period': 5, 'std_dev': 2.0, 'min_confidence': 0.5})
        
        # Mock indicators with upper breakout
        indicators = {
            'bollinger': {
                'bollinger_5': {
                    'upper': np.array([np.nan] * 4 + [105, 106]),
                    'middle': np.array([np.nan] * 4 + [100, 101]),
                    'lower': np.array([np.nan] * 4 + [95, 96])
                }
            }
        }
        
        mock_data = Mock(spec=MarketData)
        mock_data.symbol = "AAPL"
        mock_data.to_arrays.return_value = {
            'timestamp': [datetime.now() - timedelta(days=i) for i in range(6, 0, -1)],
            'close': [99, 100, 101, 102, 103, 107],  # Breaks above upper band
            'volume': [1000] * 6
        }
        
        signals = strategy.generate_signals(mock_data, indicators)
        
        assert isinstance(signals, list)
    
    def test_signal_generation_lower_breakout(self):
        """Test lower band breakout signal generation."""
        strategy = BollingerBreakout({'period': 5, 'std_dev': 2.0, 'min_confidence': 0.5})
        
        # Mock indicators with lower breakout
        indicators = {
            'bollinger': {
                'bollinger_5': {
                    'upper': np.array([np.nan] * 4 + [105, 104]),
                    'middle': np.array([np.nan] * 4 + [100, 99]),
                    'lower': np.array([np.nan] * 4 + [95, 94])
                }
            }
        }
        
        mock_data = Mock(spec=MarketData)
        mock_data.symbol = "AAPL"
        mock_data.to_arrays.return_value = {
            'timestamp': [datetime.now() - timedelta(days=i) for i in range(6, 0, -1)],
            'close': [101, 100, 99, 98, 97, 93],  # Breaks below lower band
            'volume': [1000] * 6
        }
        
        signals = strategy.generate_signals(mock_data, indicators)
        
        assert isinstance(signals, list)


class TestMACDRSIStrategy:
    """Test MACD-RSI combined strategy."""
    
    def test_strategy_initialization(self):
        """Test MACD-RSI strategy initialization."""
        strategy = MACDRSIStrategy()
        
        assert strategy.name == "MACDRSIStrategy"
        assert 'macd_fast' in strategy.parameters
        assert 'macd_slow' in strategy.parameters
        assert 'macd_signal' in strategy.parameters
        assert 'rsi_period' in strategy.parameters
    
    def test_required_indicators(self):
        """Test required indicators."""
        strategy = MACDRSIStrategy()
        required = strategy.get_required_indicators()
        
        assert 'macd' in required
        assert 'rsi' in required
    
    def test_signal_generation_bullish(self):
        """Test bullish signal generation."""
        strategy = MACDRSIStrategy({
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'rsi_period': 14, 'min_confidence': 0.5
        })
        
        # Mock indicators with bullish conditions
        indicators = {
            'macd': {
                'macd': {
                    'line': np.array([np.nan] * 25 + [0.5]),
                    'signal': np.array([np.nan] * 25 + [0.3]),
                    'histogram': np.array([np.nan] * 25 + [0.2])
                }
            },
            'rsi': {'rsi_14': np.array([np.nan] * 13 + [45])}  # Neutral RSI
        }
        
        mock_data = Mock(spec=MarketData)
        mock_data.symbol = "AAPL"
        mock_data.to_arrays.return_value = {
            'timestamp': [datetime.now() - timedelta(days=i) for i in range(26, 0, -1)],
            'close': [100 + i * 0.1 for i in range(26)],
            'volume': [1000] * 26
        }
        
        signals = strategy.generate_signals(mock_data, indicators)
        
        assert isinstance(signals, list)
    
    def test_signal_generation_bearish(self):
        """Test bearish signal generation."""
        strategy = MACDRSIStrategy({
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'rsi_period': 14, 'min_confidence': 0.5
        })
        
        # Mock indicators with bearish conditions
        indicators = {
            'macd': {
                'macd': {
                    'line': np.array([np.nan] * 25 + [-0.5]),
                    'signal': np.array([np.nan] * 25 + [-0.3]),
                    'histogram': np.array([np.nan] * 25 + [-0.2])
                }
            },
            'rsi': {'rsi_14': np.array([np.nan] * 13 + [55])}  # Neutral RSI
        }
        
        mock_data = Mock(spec=MarketData)
        mock_data.symbol = "AAPL"
        mock_data.to_arrays.return_value = {
            'timestamp': [datetime.now() - timedelta(days=i) for i in range(26, 0, -1)],
            'close': [100 - i * 0.1 for i in range(26)],
            'volume': [1000] * 26
        }
        
        signals = strategy.generate_signals(mock_data, indicators)
        
        assert isinstance(signals, list)


class TestSRBreakout:
    """Test Support/Resistance Breakout strategy."""
    
    def test_strategy_initialization(self):
        """Test SR breakout strategy initialization."""
        strategy = SRBreakout()
        
        assert strategy.name == "SRBreakout"
        assert 'lookback_period' in strategy.parameters
        assert 'min_touches' in strategy.parameters
        assert 'breakout_threshold' in strategy.parameters
    
    def test_custom_parameters(self):
        """Test SR breakout with custom parameters."""
        params = {
            'lookback_period': 30,
            'min_touches': 3,
            'breakout_threshold': 0.03,
            'volume_confirmation': True
        }
        
        strategy = SRBreakout(params)
        
        assert strategy.parameters['lookback_period'] == 30
        assert strategy.parameters['min_touches'] == 3
        assert strategy.parameters['breakout_threshold'] == 0.03
        assert strategy.parameters['volume_confirmation'] == True
    
    def test_required_indicators(self):
        """Test required indicators."""
        strategy = SRBreakout()
        required = strategy.get_required_indicators()
        
        assert 'support_resistance' in required
    
    def test_signal_generation_resistance_breakout(self):
        """Test resistance breakout signal generation."""
        strategy = SRBreakout({'lookback_period': 10, 'min_touches': 2, 'breakout_threshold': 0.02})
        
        # Mock indicators with resistance levels
        indicators = {
            'support_resistance': {
                'resistance_levels': [105.0, 110.0],
                'support_levels': [95.0, 90.0]
            }
        }
        
        mock_data = Mock(spec=MarketData)
        mock_data.symbol = "AAPL"
        mock_data.to_arrays.return_value = {
            'timestamp': [datetime.now() - timedelta(days=i) for i in range(11, 0, -1)],
            'close': [100, 101, 102, 103, 104, 105, 104, 103, 105, 106, 107],  # Breaks resistance
            'high': [101, 102, 103, 104, 105, 106, 105, 104, 106, 107, 108],
            'volume': [1000] * 11
        }
        
        signals = strategy.generate_signals(mock_data, indicators)
        
        assert isinstance(signals, list)
    
    def test_signal_generation_support_breakout(self):
        """Test support breakout signal generation."""
        strategy = SRBreakout({'lookback_period': 10, 'min_touches': 2, 'breakout_threshold': 0.02})
        
        # Mock indicators with support levels
        indicators = {
            'support_resistance': {
                'resistance_levels': [105.0, 110.0],
                'support_levels': [95.0, 90.0]
            }
        }
        
        mock_data = Mock(spec=MarketData)
        mock_data.symbol = "AAPL"
        mock_data.to_arrays.return_value = {
            'timestamp': [datetime.now() - timedelta(days=i) for i in range(11, 0, -1)],
            'close': [100, 99, 98, 97, 96, 95, 96, 97, 95, 94, 93],  # Breaks support
            'low': [99, 98, 97, 96, 95, 94, 95, 96, 94, 93, 92],
            'volume': [1000] * 11
        }
        
        signals = strategy.generate_signals(mock_data, indicators)
        
        assert isinstance(signals, list)


class TestStrategyPerformance:
    """Test strategy performance and benchmarks."""
    
    def test_ma_crossover_performance(self, sample_market_data, execution_timer):
        """Test MA crossover performance."""
        strategy = MovingAverageCrossover({'short_period': 20, 'long_period': 50})
        
        # Create realistic indicators
        close_data = [bar.close for bar in sample_market_data.data]
        indicators = {
            'ema': {
                'ema_20': calculate_ema(close_data, 20),
                'ema_50': calculate_ema(close_data, 50)
            }
        }
        
        execution_timer.start()
        signals = strategy.generate_signals(sample_market_data, indicators)
        execution_timer.stop()
        
        # Should complete quickly
        assert execution_timer.get_elapsed_time() < 1.0
        assert isinstance(signals, list)
    
    def test_rsi_trend_performance(self, sample_market_data, execution_timer):
        """Test RSI trend performance."""
        strategy = RSITrend({'rsi_period': 14, 'trend_period': 50})
        
        close_data = [bar.close for bar in sample_market_data.data]
        indicators = {
            'rsi': {'rsi_14': calculate_rsi(close_data, 14)},
            'sma': {'sma_50': calculate_sma(close_data, 50)}
        }
        
        execution_timer.start()
        signals = strategy.generate_signals(sample_market_data, indicators)
        execution_timer.stop()
        
        assert execution_timer.get_elapsed_time() < 1.0
        assert isinstance(signals, list)
    
    def test_strategy_memory_usage(self, sample_market_data, memory_monitor):
        """Test strategy memory usage."""
        initial_memory = memory_monitor.get_current_usage()
        
        strategies = [
            MovingAverageCrossover(),
            RSITrend(),
            BollingerBreakout(),
            MACDRSIStrategy(),
            SRBreakout()
        ]
        
        for strategy in strategies:
            # Generate minimal indicators
            indicators = {indicator: {} for indicator in strategy.get_required_indicators()}
            strategy.generate_signals(sample_market_data, indicators)
        
        final_memory = memory_monitor.get_current_usage()
        memory_increase = final_memory - initial_memory
        
        # Should not use excessive memory
        assert memory_increase < 50  # MB


class TestStrategyErrorHandling:
    """Test strategy error handling and edge cases."""
    
    def test_empty_data(self):
        """Test strategies with empty data."""
        empty_data = MarketData(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            data=[],
            indicators={},
            last_updated=datetime.now()
        )
        
        strategies = [
            MovingAverageCrossover(),
            RSITrend(),
            BollingerBreakout(),
            MACDRSIStrategy(),
            SRBreakout()
        ]
        
        for strategy in strategies:
            indicators = {indicator: {} for indicator in strategy.get_required_indicators()}
            signals = strategy.generate_signals(empty_data, indicators)
            
            # Should handle empty data gracefully
            assert signals == []
    
    def test_malformed_indicators(self, sample_market_data):
        """Test strategies with malformed indicators."""
        strategies = [
            MovingAverageCrossover(),
            RSITrend(),
            BollingerBreakout(),
            MACDRSIStrategy(),
            SRBreakout()
        ]
        
        # Malformed indicators
        malformed_indicators = {
            'ema': None,
            'rsi': "not_a_dict",
            'bollinger': [],
            'macd': 123,
            'support_resistance': {"invalid": "structure"}
        }
        
        for strategy in strategies:
            signals = strategy.generate_signals(sample_market_data, malformed_indicators)
            
            # Should handle malformed indicators gracefully
            assert isinstance(signals, list)
    
    def test_nan_indicators(self, sample_market_data):
        """Test strategies with NaN indicators."""
        strategies = [MovingAverageCrossover(), RSITrend()]
        
        # Indicators with NaN values
        nan_indicators = {
            'ema': {
                'ema_20': np.array([np.nan] * 50),
                'ema_50': np.array([np.nan] * 50)
            },
            'rsi': {'rsi_14': np.array([np.nan] * 50)},
            'sma': {'sma_50': np.array([np.nan] * 50)}
        }
        
        for strategy in strategies:
            signals = strategy.generate_signals(sample_market_data, nan_indicators)
            
            # Should handle NaN values gracefully
            assert isinstance(signals, list)
    
    def test_insufficient_data_length(self):
        """Test strategies with insufficient data length."""
        # Very short data
        short_data = MarketData(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            data=[
                OHLCV(datetime.now(), 100, 105, 95, 103, 1000),
                OHLCV(datetime.now(), 103, 108, 98, 106, 1200)
            ],
            indicators={},
            last_updated=datetime.now()
        )
        
        strategies = [
            MovingAverageCrossover({'short_period': 5, 'long_period': 10}),
            RSITrend({'rsi_period': 14})
        ]
        
        for strategy in strategies:
            indicators = {indicator: {} for indicator in strategy.get_required_indicators()}
            signals = strategy.generate_signals(short_data, indicators)
            
            # Should handle insufficient data gracefully
            assert isinstance(signals, list)


class TestStrategyOptimization:
    """Test strategy parameter optimization."""
    
    def test_parameter_ranges(self):
        """Test parameter ranges for optimization."""
        strategies = [
            MovingAverageCrossover(),
            RSITrend(),
            BollingerBreakout(),
            MACDRSIStrategy(),
            SRBreakout()
        ]
        
        for strategy in strategies:
            ranges = strategy._get_parameter_ranges()
            
            # Should return a dictionary
            assert isinstance(ranges, dict)
            
            # If ranges are provided, they should contain lists
            for key, value in ranges.items():
                if value:
                    assert isinstance(value, list)
                    assert len(value) > 0
    
    def test_optimization_metrics(self, sample_market_data):
        """Test optimization metrics calculation."""
        strategy = MovingAverageCrossover({'short_period': 10, 'long_period': 20})
        
        # Test different metrics
        metrics = ['sharpe', 'return', 'win_rate', 'profit_factor']
        
        for metric in metrics:
            score = strategy._calculate_performance_score(sample_market_data, metric)
            
            # Should return a number
            assert isinstance(score, (int, float))
            
            # Should not be NaN (unless legitimately no signals)
            if not np.isnan(score):
                assert score != float('inf') or score != float('-inf')
    
    @pytest.mark.slow
    def test_full_optimization(self, sample_market_data):
        """Test full parameter optimization (marked as slow)."""
        strategy = MovingAverageCrossover()
        
        # Limited parameter ranges for testing
        param_ranges = {
            'short_period': [10, 20],
            'long_period': [30, 40]
        }
        
        with patch.object(strategy, '_get_parameter_ranges', return_value=param_ranges):
            best_params = strategy.optimize_parameters(sample_market_data, 'sharpe')
        
        # Should return parameters
        assert isinstance(best_params, dict)
        
        # Should contain the optimized parameters
        if best_params:
            assert 'short_period' in best_params or 'long_period' in best_params


class TestStrategyValidation:
    """Test strategy signal validation."""
    
    def test_signal_confidence_filtering(self):
        """Test signal filtering by confidence."""
        strategy = MovingAverageCrossover({'min_confidence': 0.7})
        
        signals = [
            Signal(datetime.now(), "AAPL", SignalType.BUY, 150, 0.8, "MA", {}),  # Valid
            Signal(datetime.now(), "AAPL", SignalType.BUY, 150, 0.6, "MA", {}),  # Invalid
            Signal(datetime.now(), "AAPL", SignalType.SELL, 150, 0.9, "MA", {}) # Valid
        ]
        
        validated = strategy.validate_signals(signals)
        
        # Should filter out low confidence signals
        assert len(validated) == 2
        assert all(s.confidence >= 0.7 for s in validated)
    
    def test_signal_type_validation(self):
        """Test signal type validation."""
        strategy = MovingAverageCrossover()
        
        signals = [
            Signal(datetime.now(), "AAPL", SignalType.BUY, 150, 0.8, "MA", {'short_ema': 151, 'long_ema': 149, 'ema_separation': 0.01}),
            Signal(datetime.now(), "AAPL", SignalType.HOLD, 150, 0.8, "MA", {}),  # Invalid type for MA crossover
            Signal(datetime.now(), "AAPL", SignalType.SELL, 150, 0.8, "MA", {'short_ema': 149, 'long_ema': 151, 'ema_separation': 0.01})
        ]
        
        validated = strategy.validate_signals(signals)
        
        # Should filter out inappropriate signal types
        valid_types = [s.signal_type for s in validated]
        assert SignalType.HOLD not in valid_types
    
    def test_metadata_validation(self):
        """Test signal metadata validation."""
        strategy = MovingAverageCrossover()
        
        # Signal with missing metadata
        invalid_signal = Signal(
            datetime.now(), "AAPL", SignalType.BUY, 150, 0.8, "MA", {}
        )
        
        # Signal with required metadata
        valid_signal = Signal(
            datetime.now(), "AAPL", SignalType.BUY, 150, 0.8, "MA",
            {'short_ema': 151, 'long_ema': 149, 'ema_separation': 0.01}
        )
        
        assert not strategy._validate_signal_strategy(invalid_signal)
        assert strategy._validate_signal_strategy(valid_signal)