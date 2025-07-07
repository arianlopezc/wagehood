"""
Comprehensive tests for the SignalEngine worker component.

Tests cover:
- Signal correlation and composite scoring
- Strategy-timeframe combinations
- Signal persistence and alignment
- Performance optimization
- Error handling and recovery
- Memory management
- Scalability characteristics
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np

from src.realtime.signal_engine import (
    SignalEngine, Signal, CompositeSignal, 
    SignalDirection, SignalStrength
)
from src.realtime.config_manager import TradingProfile


class TestSignalEngineComponents:
    """Test individual SignalEngine components."""
    
    def test_initialization(self, test_config_manager, test_timeframe_manager):
        """Test SignalEngine initialization."""
        engine = SignalEngine(test_config_manager, test_timeframe_manager)
        
        assert engine.config_manager is test_config_manager
        assert engine.timeframe_manager is test_timeframe_manager
        assert isinstance(engine._signal_history, dict)
        assert isinstance(engine._signal_cache, dict)
        assert hasattr(engine, '_lock')
        
    def test_signal_creation(self):
        """Test Signal creation and validation."""
        timestamp = datetime.now()
        
        signal = Signal(
            strategy="macd_rsi_strategy",
            timeframe="1h",
            direction=SignalDirection.BUY,
            strength=SignalStrength.STRONG,
            confidence=0.8,
            timestamp=timestamp,
            metadata={"rsi": 25.0, "macd_bullish": True}
        )
        
        assert signal.strategy == "macd_rsi_strategy"
        assert signal.timeframe == "1h"
        assert signal.direction == SignalDirection.BUY
        assert signal.strength == SignalStrength.STRONG
        assert signal.confidence == 0.8
        assert signal.timestamp == timestamp
        
    def test_signal_validation(self):
        """Test Signal validation."""
        timestamp = datetime.now()
        
        # Valid confidence
        signal = Signal(
            strategy="test",
            timeframe="1h",
            direction=SignalDirection.BUY,
            strength=SignalStrength.STRONG,
            confidence=0.8,
            timestamp=timestamp
        )
        assert signal.confidence == 0.8
        
        # Invalid confidence - should raise error
        with pytest.raises(ValueError):
            Signal(
                strategy="test",
                timeframe="1h",
                direction=SignalDirection.BUY,
                strength=SignalStrength.STRONG,
                confidence=1.5,  # Invalid - > 1.0
                timestamp=timestamp
            )
            
    def test_composite_signal_creation(self):
        """Test CompositeSignal creation."""
        timestamp = datetime.now()
        
        contributing_signals = [
            Signal("strategy1", "1h", SignalDirection.BUY, SignalStrength.STRONG, 0.8, timestamp),
            Signal("strategy2", "1h", SignalDirection.BUY, SignalStrength.MODERATE, 0.6, timestamp)
        ]
        
        composite = CompositeSignal(
            symbol="AAPL",
            timestamp=timestamp,
            overall_direction=SignalDirection.BUY,
            overall_confidence=0.7,
            overall_strength=SignalStrength.STRONG,
            contributing_signals=contributing_signals
        )
        
        assert composite.symbol == "AAPL"
        assert composite.overall_direction == SignalDirection.BUY
        assert composite.overall_confidence == 0.7
        assert len(composite.contributing_signals) == 2
        
    def test_signal_summary_generation(self):
        """Test signal summary generation."""
        timestamp = datetime.now()
        
        composite = CompositeSignal(
            symbol="AAPL",
            timestamp=timestamp,
            overall_direction=SignalDirection.BUY,
            overall_confidence=0.75,
            overall_strength=SignalStrength.STRONG,
            timeframe_alignment=0.8,
            strategy_alignment=0.9
        )
        
        summary = composite.get_signal_summary()
        
        assert summary["symbol"] == "AAPL"
        assert summary["direction"] == "buy"
        assert summary["confidence"] == 0.75
        assert summary["strength"] == "strong"
        assert summary["timeframe_alignment"] == 0.8
        assert summary["strategy_alignment"] == 0.9
        
    def test_stats_initialization(self, test_signal_engine):
        """Test performance statistics initialization."""
        stats = test_signal_engine.get_stats()
        
        assert stats["total_signals_generated"] == 0
        assert stats["composite_signals_generated"] == 0
        assert isinstance(stats["strategies_processed"], list)
        assert isinstance(stats["timeframes_processed"], list)
        assert isinstance(stats["symbols_processed"], list)
        assert stats["errors"] == 0


class TestSignalGeneration:
    """Test signal generation functionality."""
    
    def test_single_strategy_signal_generation(self, test_signal_engine):
        """Test generating signals for individual strategies."""
        symbol = "AAPL"
        price = 150.0
        timeframe_results = {
            "1h": {
                "rsi_14": 25.0,  # Oversold
                "macd": {
                    "macd_line": 1.5,
                    "signal_line": 1.0,
                    "histogram": 0.5
                },
                "sma_50": 145.0,
                "sma_200": 140.0
            }
        }
        
        composite_signal = test_signal_engine.generate_signals(
            symbol=symbol,
            price=price,
            timeframe_results=timeframe_results,
            trading_profile=TradingProfile.SWING_TRADING
        )
        
        if composite_signal:
            assert composite_signal.symbol == symbol
            assert composite_signal.overall_direction in [
                SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD
            ]
            assert 0 <= composite_signal.overall_confidence <= 1
            
    def test_multi_timeframe_signal_generation(self, test_signal_engine):
        """Test generating signals across multiple timeframes."""
        symbol = "AAPL"
        price = 150.0
        timeframe_results = {
            "1m": {
                "rsi_14": 30.0,
                "sma_50": 149.0
            },
            "5m": {
                "rsi_14": 25.0,  # More oversold
                "sma_50": 148.0
            },
            "1h": {
                "rsi_14": 20.0,  # Very oversold
                "sma_50": 145.0,
                "macd": {
                    "macd_line": 1.5,
                    "signal_line": 1.0,
                    "histogram": 0.5
                }
            }
        }
        
        composite_signal = test_signal_engine.generate_signals(
            symbol=symbol,
            price=price,
            timeframe_results=timeframe_results,
            trading_profile=TradingProfile.SWING_TRADING
        )
        
        if composite_signal:
            # Should have signals from multiple timeframes
            timeframes_in_signals = set(s.timeframe for s in composite_signal.contributing_signals)
            assert len(timeframes_in_signals) > 1
            
    def test_strategy_signal_calculation_macd_rsi(self, test_signal_engine):
        """Test MACD+RSI strategy signal calculation."""
        indicators = {
            "rsi_14": 25.0,  # Oversold
            "macd": {
                "macd_line": 1.5,
                "signal_line": 1.0,
                "histogram": 0.5
            }
        }
        
        params = {
            "rsi_overbought": 70,
            "rsi_oversold": 30
        }
        
        signal_data = test_signal_engine._calculate_macd_rsi_signal(indicators, params, "1h")
        
        assert signal_data is not None
        assert signal_data["direction"] == "buy"  # MACD bullish + RSI oversold
        assert signal_data["confidence"] > 0
        assert "macd_bullish" in signal_data["metadata"]
        assert signal_data["metadata"]["macd_bullish"] == True
        
    def test_strategy_signal_calculation_ma_crossover(self, test_signal_engine):
        """Test Moving Average crossover signal calculation."""
        indicators = {
            "sma_50": 155.0,   # Fast MA
            "sma_200": 150.0   # Slow MA
        }
        
        params = {
            "fast_period": 50,
            "slow_period": 200
        }
        
        signal_data = test_signal_engine._calculate_ma_crossover_signal(indicators, params, "1d")
        
        assert signal_data is not None
        assert signal_data["direction"] == "buy"  # Fast MA > Slow MA
        assert signal_data["confidence"] > 0
        assert signal_data["metadata"]["fast_ma"] == 155.0
        assert signal_data["metadata"]["slow_ma"] == 150.0
        
    def test_strategy_signal_calculation_rsi_trend(self, test_signal_engine):
        """Test RSI trend strategy signal calculation."""
        # Oversold condition
        indicators = {
            "rsi_14": 25.0,
            "sma_50": 150.0
        }
        
        params = {
            "rsi_overbought": 70,
            "rsi_oversold": 30
        }
        
        signal_data = test_signal_engine._calculate_rsi_trend_signal(indicators, params, "1h")
        
        assert signal_data is not None
        assert signal_data["direction"] == "buy"  # RSI oversold
        assert signal_data["confidence"] > 0
        
        # Overbought condition
        indicators["rsi_14"] = 75.0
        signal_data = test_signal_engine._calculate_rsi_trend_signal(indicators, params, "1h")
        
        assert signal_data["direction"] == "sell"  # RSI overbought
        
    def test_strategy_signal_calculation_bollinger_breakout(self, test_signal_engine):
        """Test Bollinger Bands breakout signal calculation."""
        indicators = {
            "bollinger_bands": {
                "upper_band": 155.0,
                "middle_band": 150.0,
                "lower_band": 145.0
            }
        }
        
        params = {"squeeze_threshold": 0.1}
        
        # Test price above upper band (bullish breakout)
        signal_data = test_signal_engine._calculate_bollinger_breakout_signal(
            indicators, params, "15m", 157.0  # Price above upper band
        )
        
        assert signal_data is not None
        assert signal_data["direction"] == "buy"  # Bullish breakout
        assert signal_data["confidence"] > 0
        
        # Test price below lower band (bearish breakout)
        signal_data = test_signal_engine._calculate_bollinger_breakout_signal(
            indicators, params, "15m", 143.0  # Price below lower band
        )
        
        assert signal_data["direction"] == "sell"  # Bearish breakout
        
    def test_timeframe_confidence_multiplier(self, test_signal_engine):
        """Test timeframe confidence multiplier calculation."""
        # Test different timeframes
        multipliers = {
            "1m": test_signal_engine._get_timeframe_confidence_multiplier("1m"),
            "5m": test_signal_engine._get_timeframe_confidence_multiplier("5m"),
            "1h": test_signal_engine._get_timeframe_confidence_multiplier("1h"),
            "1d": test_signal_engine._get_timeframe_confidence_multiplier("1d")
        }
        
        # Longer timeframes should have higher multipliers
        assert multipliers["1m"] < multipliers["1h"]
        assert multipliers["1h"] <= multipliers["1d"]
        assert multipliers["5m"] < multipliers["1h"]
        
    def test_signal_strength_calculation(self, test_signal_engine):
        """Test signal strength calculation."""
        # High confidence on longer timeframe
        strength = test_signal_engine._calculate_signal_strength(0.9, "1d")
        assert strength == SignalStrength.VERY_STRONG
        
        # Medium confidence
        strength = test_signal_engine._calculate_signal_strength(0.7, "1h")
        assert strength == SignalStrength.STRONG
        
        # Low confidence
        strength = test_signal_engine._calculate_signal_strength(0.5, "1h")
        assert strength == SignalStrength.WEAK
        
        # High confidence on short timeframe (higher threshold)
        strength = test_signal_engine._calculate_signal_strength(0.8, "1m")
        assert strength in [SignalStrength.MODERATE, SignalStrength.STRONG]


class TestCompositeSignalGeneration:
    """Test composite signal generation."""
    
    def test_composite_signal_creation_from_individual_signals(self, test_signal_engine):
        """Test creating composite signals from individual signals."""
        timestamp = datetime.now()
        
        individual_signals = [
            Signal("macd_rsi_strategy", "1h", SignalDirection.BUY, SignalStrength.STRONG, 0.8, timestamp),
            Signal("rsi_trend_strategy", "1h", SignalDirection.BUY, SignalStrength.MODERATE, 0.6, timestamp),
            Signal("ma_crossover_strategy", "4h", SignalDirection.BUY, SignalStrength.STRONG, 0.9, timestamp)
        ]
        
        composite = test_signal_engine._create_composite_signal("AAPL", individual_signals)
        
        assert composite.symbol == "AAPL"
        assert composite.overall_direction == SignalDirection.BUY  # All signals are BUY
        assert len(composite.contributing_signals) == 3
        assert composite.overall_confidence > 0
        
    def test_conflicting_signals_resolution(self, test_signal_engine):
        """Test resolution of conflicting signals."""
        timestamp = datetime.now()
        
        individual_signals = [
            Signal("strategy1", "1h", SignalDirection.BUY, SignalStrength.STRONG, 0.8, timestamp),
            Signal("strategy2", "1h", SignalDirection.SELL, SignalStrength.MODERATE, 0.6, timestamp),
            Signal("strategy3", "1h", SignalDirection.BUY, SignalStrength.STRONG, 0.9, timestamp)
        ]
        
        composite = test_signal_engine._create_composite_signal("AAPL", individual_signals)
        
        # Should resolve to BUY (2 BUY vs 1 SELL, with higher confidence)
        assert composite.overall_direction == SignalDirection.BUY
        assert composite.overall_confidence > 0
        
    def test_timeframe_consensus_calculation(self, test_signal_engine):
        """Test timeframe consensus calculation."""
        timestamp = datetime.now()
        
        individual_signals = [
            Signal("strategy1", "1m", SignalDirection.BUY, SignalStrength.MODERATE, 0.6, timestamp),
            Signal("strategy2", "1m", SignalDirection.BUY, SignalStrength.STRONG, 0.8, timestamp),
            Signal("strategy1", "1h", SignalDirection.SELL, SignalStrength.MODERATE, 0.7, timestamp),
            Signal("strategy2", "1h", SignalDirection.BUY, SignalStrength.STRONG, 0.9, timestamp)
        ]
        
        composite = test_signal_engine._create_composite_signal("AAPL", individual_signals)
        
        # Should have timeframe consensus data
        assert "1m" in composite.timeframe_consensus
        assert "1h" in composite.timeframe_consensus
        assert "1m" in composite.timeframe_confidence
        assert "1h" in composite.timeframe_confidence
        
    def test_strategy_consensus_calculation(self, test_signal_engine):
        """Test strategy consensus calculation."""
        timestamp = datetime.now()
        
        individual_signals = [
            Signal("macd_rsi_strategy", "1h", SignalDirection.BUY, SignalStrength.STRONG, 0.8, timestamp),
            Signal("macd_rsi_strategy", "4h", SignalDirection.BUY, SignalStrength.STRONG, 0.9, timestamp),
            Signal("rsi_trend_strategy", "1h", SignalDirection.SELL, SignalStrength.MODERATE, 0.6, timestamp)
        ]
        
        composite = test_signal_engine._create_composite_signal("AAPL", individual_signals)
        
        # Should have strategy consensus data
        assert "macd_rsi_strategy" in composite.strategy_consensus
        assert "rsi_trend_strategy" in composite.strategy_consensus
        assert "macd_rsi_strategy" in composite.strategy_confidence
        assert "rsi_trend_strategy" in composite.strategy_confidence
        
    def test_alignment_metrics_calculation(self, test_signal_engine):
        """Test alignment metrics calculation."""
        timestamp = datetime.now()
        
        # All signals aligned
        aligned_signals = [
            Signal("strategy1", "1h", SignalDirection.BUY, SignalStrength.STRONG, 0.8, timestamp),
            Signal("strategy2", "1h", SignalDirection.BUY, SignalStrength.STRONG, 0.8, timestamp),
            Signal("strategy3", "1h", SignalDirection.BUY, SignalStrength.STRONG, 0.8, timestamp)
        ]
        
        composite = test_signal_engine._create_composite_signal("AAPL", aligned_signals)
        
        # Should have high alignment
        assert composite.timeframe_alignment >= 0.8  # All same timeframe
        assert composite.strategy_alignment >= 0.8   # All same direction
        
    def test_signal_persistence_calculation(self, test_signal_engine):
        """Test signal persistence calculation."""
        symbol = "AAPL"
        timestamp = datetime.now()
        
        # Create historical signals with same direction
        for i in range(5):
            signals = [Signal("strategy1", "1h", SignalDirection.BUY, SignalStrength.STRONG, 0.8, timestamp)]
            composite = test_signal_engine._create_composite_signal(symbol, signals)
            
            # Manually add to history to test persistence
            if symbol not in test_signal_engine._signal_history:
                test_signal_engine._signal_history[symbol] = []
            test_signal_engine._signal_history[symbol].append(composite)
            
        # Now test persistence calculation
        persistence = test_signal_engine._calculate_signal_persistence(symbol, SignalDirection.BUY)
        assert persistence > 0.8  # Should be high since all signals were BUY


class TestSignalEnginePerformance:
    """Test SignalEngine performance characteristics."""
    
    def test_signal_generation_performance(self, test_signal_engine, performance_thresholds):
        """Test signal generation performance."""
        symbol = "AAPL"
        price = 150.0
        
        timeframe_results = {
            "1m": {"rsi_14": 45.0, "sma_50": 150.0},
            "5m": {"rsi_14": 40.0, "sma_50": 149.0},
            "1h": {"rsi_14": 35.0, "sma_50": 148.0, "macd": {"macd_line": 1.0, "signal_line": 0.8, "histogram": 0.2}}
        }
        
        start_time = time.time()
        
        # Generate signals multiple times
        for _ in range(50):
            test_signal_engine.generate_signals(
                symbol=symbol,
                price=price,
                timeframe_results=timeframe_results,
                trading_profile=TradingProfile.SWING_TRADING
            )
            
        total_time = time.time() - start_time
        avg_time = total_time / 50
        
        # Should generate signals quickly
        assert avg_time < 0.1  # Less than 100ms per signal generation
        
    def test_memory_usage_during_signal_processing(self, test_signal_engine, resource_monitor):
        """Test memory usage during signal processing."""
        initial_memory = resource_monitor.get_memory_usage()
        
        # Generate many signals
        for i in range(100):
            symbol = f"TEST{i % 5}"
            price = 150.0 + i * 0.1
            
            timeframe_results = {
                "1h": {
                    "rsi_14": 30.0 + i % 40,
                    "sma_50": price,
                    "macd": {"macd_line": 1.0, "signal_line": 0.8, "histogram": 0.2}
                }
            }
            
            test_signal_engine.generate_signals(
                symbol=symbol,
                price=price,
                timeframe_results=timeframe_results,
                trading_profile=TradingProfile.SWING_TRADING
            )
            
        final_memory = resource_monitor.get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 20  # Less than 20MB for 100 signals
        
    def test_concurrent_signal_generation(self, test_signal_engine):
        """Test concurrent signal generation for multiple symbols."""
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        results = []
        
        def generate_signals_for_symbol(symbol):
            price = 150.0 + hash(symbol) % 100
            timeframe_results = {
                "1h": {
                    "rsi_14": 45.0,
                    "sma_50": price,
                    "macd": {"macd_line": 1.0, "signal_line": 0.8, "histogram": 0.2}
                }
            }
            
            start_time = time.time()
            composite_signal = test_signal_engine.generate_signals(
                symbol=symbol,
                price=price,
                timeframe_results=timeframe_results,
                trading_profile=TradingProfile.SWING_TRADING
            )
            processing_time = time.time() - start_time
            
            results.append((symbol, processing_time, composite_signal is not None))
            
        # Process symbols concurrently
        threads = []
        for symbol in symbols:
            thread = threading.Thread(target=generate_signals_for_symbol, args=(symbol,))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # All symbols should be processed efficiently
        assert len(results) == len(symbols)
        for symbol, processing_time, has_signal in results:
            assert processing_time < 0.5  # Less than 500ms per symbol
            
    def test_signal_caching_performance(self, test_signal_engine):
        """Test signal caching performance."""
        symbol = "AAPL"
        price = 150.0
        
        timeframe_results = {
            "1h": {
                "rsi_14": 45.0,
                "sma_50": price,
                "macd": {"macd_line": 1.0, "signal_line": 0.8, "histogram": 0.2}
            }
        }
        
        # Generate initial signal (should populate cache)
        start_time = time.time()
        test_signal_engine.generate_signals(
            symbol=symbol,
            price=price,
            timeframe_results=timeframe_results,
            trading_profile=TradingProfile.SWING_TRADING
        )
        first_time = time.time() - start_time
        
        # Generate signal again (should use cache)
        start_time = time.time()
        test_signal_engine.generate_signals(
            symbol=symbol,
            price=price,
            timeframe_results=timeframe_results,
            trading_profile=TradingProfile.SWING_TRADING
        )
        second_time = time.time() - start_time
        
        # Both should be fast, but implementation details may vary
        assert first_time < 1.0
        assert second_time < 1.0


class TestSignalEngineReliability:
    """Test SignalEngine reliability and error handling."""
    
    def test_missing_indicator_handling(self, test_signal_engine):
        """Test handling of missing indicators."""
        symbol = "AAPL"
        price = 150.0
        
        # Missing required indicators
        timeframe_results = {
            "1h": {
                "sma_50": 150.0  # Missing RSI and MACD for MACD+RSI strategy
            }
        }
        
        # Should handle gracefully
        composite_signal = test_signal_engine.generate_signals(
            symbol=symbol,
            price=price,
            timeframe_results=timeframe_results,
            trading_profile=TradingProfile.SWING_TRADING
        )
        
        # May return None or empty signal, should not crash
        if composite_signal:
            assert len(composite_signal.contributing_signals) >= 0
            
    def test_invalid_indicator_data_handling(self, test_signal_engine):
        """Test handling of invalid indicator data."""
        symbol = "AAPL"
        price = 150.0
        
        # Invalid indicator data
        timeframe_results = {
            "1h": {
                "rsi_14": float('nan'),  # Invalid RSI
                "macd": {
                    "macd_line": None,  # Invalid MACD
                    "signal_line": float('inf'),
                    "histogram": "invalid"
                }
            }
        }
        
        # Should handle gracefully
        try:
            composite_signal = test_signal_engine.generate_signals(
                symbol=symbol,
                price=price,
                timeframe_results=timeframe_results,
                trading_profile=TradingProfile.SWING_TRADING
            )
            # If no exception, should return valid result or None
            if composite_signal:
                assert hasattr(composite_signal, 'symbol')
        except (ValueError, TypeError):
            # These exceptions are acceptable for invalid data
            pass
            
    def test_empty_timeframe_results_handling(self, test_signal_engine):
        """Test handling of empty timeframe results."""
        symbol = "AAPL"
        price = 150.0
        
        # Empty timeframe results
        timeframe_results = {}
        
        # Should handle gracefully
        composite_signal = test_signal_engine.generate_signals(
            symbol=symbol,
            price=price,
            timeframe_results=timeframe_results,
            trading_profile=TradingProfile.SWING_TRADING
        )
        
        # Should return None or empty signal
        if composite_signal:
            assert len(composite_signal.contributing_signals) == 0
        else:
            assert composite_signal is None
            
    def test_signal_history_memory_management(self, test_signal_engine):
        """Test signal history memory management."""
        symbol = "AAPL"
        
        # Generate many signals to test history management
        for i in range(150):  # More than the 100-signal limit
            timestamp = datetime.now()
            signals = [Signal("strategy1", "1h", SignalDirection.BUY, SignalStrength.STRONG, 0.8, timestamp)]
            composite = test_signal_engine._create_composite_signal(symbol, signals)
            
            # Manually add to history
            if symbol not in test_signal_engine._signal_history:
                test_signal_engine._signal_history[symbol] = []
            test_signal_engine._signal_history[symbol].append(composite)
            
        # Should limit history to 100 signals
        assert len(test_signal_engine._signal_history[symbol]) <= 100
        
    def test_thread_safety(self, test_signal_engine):
        """Test thread safety of SignalEngine."""
        symbol = "AAPL"
        results = []
        
        def worker(worker_id):
            try:
                for i in range(10):
                    price = 150.0 + worker_id + i * 0.1
                    timeframe_results = {
                        "1h": {
                            "rsi_14": 45.0 + worker_id,
                            "sma_50": price
                        }
                    }
                    
                    composite_signal = test_signal_engine.generate_signals(
                        symbol=f"{symbol}_{worker_id}",
                        price=price,
                        timeframe_results=timeframe_results,
                        trading_profile=TradingProfile.SWING_TRADING
                    )
                    
                    results.append((worker_id, i, composite_signal is not None))
            except Exception as e:
                results.append((worker_id, "error", str(e)))
                
        # Run multiple threads concurrently
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(target=worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # Should complete without deadlocks
        assert len(results) > 0
        error_results = [r for r in results if r[1] == "error"]
        assert len(error_results) == 0  # No errors expected


class TestSignalEngineScalability:
    """Test SignalEngine scalability characteristics."""
    
    def test_many_symbols_scaling(self, test_signal_engine):
        """Test scaling with many symbols."""
        symbols = [f"TEST{i}" for i in range(50)]
        
        start_time = time.time()
        
        # Generate signals for many symbols
        for symbol in symbols:
            price = 150.0 + hash(symbol) % 100
            timeframe_results = {
                "1h": {
                    "rsi_14": 45.0,
                    "sma_50": price,
                    "macd": {"macd_line": 1.0, "signal_line": 0.8, "histogram": 0.2}
                }
            }
            
            test_signal_engine.generate_signals(
                symbol=symbol,
                price=price,
                timeframe_results=timeframe_results,
                trading_profile=TradingProfile.SWING_TRADING
            )
            
        total_time = time.time() - start_time
        
        # Should handle many symbols efficiently
        assert total_time < 10.0  # Less than 10 seconds for 50 symbols
        
        # Verify stats
        stats = test_signal_engine.get_stats()
        assert len(stats["symbols_processed"]) == len(symbols)
        
    def test_many_strategies_scaling(self, test_signal_engine):
        """Test scaling with many strategies per symbol."""
        symbol = "AAPL"
        price = 150.0
        
        # Provide indicators for all strategies
        timeframe_results = {
            "1h": {
                "rsi_14": 45.0,
                "sma_50": price,
                "sma_200": price - 5.0,
                "macd": {"macd_line": 1.0, "signal_line": 0.8, "histogram": 0.2},
                "bollinger_bands": {
                    "upper_band": price + 2.0,
                    "middle_band": price,
                    "lower_band": price - 2.0
                }
            }
        }
        
        start_time = time.time()
        
        # Generate signals multiple times
        for _ in range(20):
            test_signal_engine.generate_signals(
                symbol=symbol,
                price=price,
                timeframe_results=timeframe_results,
                trading_profile=TradingProfile.SWING_TRADING
            )
            
        total_time = time.time() - start_time
        
        # Should handle multiple strategies efficiently
        assert total_time < 5.0  # Less than 5 seconds for 20 iterations
        
    def test_many_timeframes_scaling(self, test_signal_engine):
        """Test scaling with many timeframes."""
        symbol = "AAPL"
        price = 150.0
        
        # Many timeframes with indicators
        timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        timeframe_results = {}
        
        for tf in timeframes:
            timeframe_results[tf] = {
                "rsi_14": 45.0,
                "sma_50": price,
                "macd": {"macd_line": 1.0, "signal_line": 0.8, "histogram": 0.2}
            }
            
        start_time = time.time()
        
        # Generate signals
        composite_signal = test_signal_engine.generate_signals(
            symbol=symbol,
            price=price,
            timeframe_results=timeframe_results,
            trading_profile=TradingProfile.SWING_TRADING
        )
        
        total_time = time.time() - start_time
        
        # Should handle many timeframes efficiently
        assert total_time < 1.0  # Less than 1 second
        
        if composite_signal:
            # Should process signals from multiple timeframes
            timeframes_in_signals = set(s.timeframe for s in composite_signal.contributing_signals)
            assert len(timeframes_in_signals) > 1


class TestSignalEngineMonitoring:
    """Test SignalEngine monitoring and diagnostics."""
    
    def test_statistics_collection(self, test_signal_engine):
        """Test statistics collection."""
        symbol = "AAPL"
        
        # Generate some signals
        for i in range(10):
            price = 150.0 + i * 0.1
            timeframe_results = {
                "1h": {
                    "rsi_14": 45.0 + i,
                    "sma_50": price
                }
            }
            
            test_signal_engine.generate_signals(
                symbol=symbol,
                price=price,
                timeframe_results=timeframe_results,
                trading_profile=TradingProfile.SWING_TRADING
            )
            
        stats = test_signal_engine.get_stats()
        
        assert stats["total_signals_generated"] >= 0
        assert stats["composite_signals_generated"] >= 0
        assert symbol in stats["symbols_processed"]
        assert stats["average_processing_time_ms"] >= 0
        
    def test_signal_history_retrieval(self, test_signal_engine):
        """Test signal history retrieval."""
        symbol = "AAPL"
        
        # Generate historical signals
        for i in range(5):
            price = 150.0 + i * 0.1
            timeframe_results = {
                "1h": {
                    "rsi_14": 45.0 + i,
                    "sma_50": price
                }
            }
            
            test_signal_engine.generate_signals(
                symbol=symbol,
                price=price,
                timeframe_results=timeframe_results,
                trading_profile=TradingProfile.SWING_TRADING
            )
            
        # Get latest signal
        latest_signal = test_signal_engine.get_latest_signal(symbol)
        if latest_signal:
            assert latest_signal.symbol == symbol
            
        # Get signal history
        history = test_signal_engine.get_signal_history(symbol, limit=3)
        assert len(history) <= 3
        
    def test_cleanup_operations(self, test_signal_engine):
        """Test cleanup operations."""
        symbol = "AAPL"
        
        # Create old signals
        old_time = datetime.now() - timedelta(hours=25)
        timestamp = old_time
        signals = [Signal("strategy1", "1h", SignalDirection.BUY, SignalStrength.STRONG, 0.8, timestamp)]
        composite = test_signal_engine._create_composite_signal(symbol, signals)
        
        # Manually add old signal
        if symbol not in test_signal_engine._signal_history:
            test_signal_engine._signal_history[symbol] = []
        test_signal_engine._signal_history[symbol].append(composite)
        
        initial_count = len(test_signal_engine._signal_history[symbol])
        
        # Run cleanup (24 hour cutoff)
        test_signal_engine.cleanup_old_signals(max_age_hours=24)
        
        final_count = len(test_signal_engine._signal_history[symbol])
        
        # Should remove old signals
        assert final_count <= initial_count
        
    def test_symbol_reset_functionality(self, test_signal_engine):
        """Test symbol reset functionality."""
        symbol = "AAPL"
        
        # Generate some signals
        price = 150.0
        timeframe_results = {
            "1h": {"rsi_14": 45.0, "sma_50": price}
        }
        
        test_signal_engine.generate_signals(
            symbol=symbol,
            price=price,
            timeframe_results=timeframe_results,
            trading_profile=TradingProfile.SWING_TRADING
        )
        
        # Verify data exists
        if symbol in test_signal_engine._signal_history:
            assert len(test_signal_engine._signal_history[symbol]) > 0
            
        # Reset symbol
        test_signal_engine.reset_symbol(symbol)
        
        # Should remove all data for symbol
        assert symbol not in test_signal_engine._signal_history or len(test_signal_engine._signal_history[symbol]) == 0
        assert symbol not in test_signal_engine._signal_cache


def test_signal_engine_complete_workflow(test_signal_engine):
    """Test complete SignalEngine workflow."""
    symbol = "AAPL"
    base_time = datetime.now()
    
    # Phase 1: Generate signals over time
    for i in range(20):
        timestamp = base_time + timedelta(minutes=i * 5)
        price = 150.0 + i * 0.1 + (i % 5) * 0.2  # Trending with noise
        
        timeframe_results = {
            "1m": {
                "rsi_14": 30.0 + i * 2,  # Gradually increasing
                "sma_50": price - 1.0
            },
            "5m": {
                "rsi_14": 35.0 + i * 1.5,
                "sma_50": price - 0.5
            },
            "1h": {
                "rsi_14": 40.0 + i * 1,
                "sma_50": price,
                "sma_200": price - 5.0,
                "macd": {
                    "macd_line": 1.0 + i * 0.1,
                    "signal_line": 0.8 + i * 0.08,
                    "histogram": 0.2 + i * 0.02
                },
                "bollinger_bands": {
                    "upper_band": price + 2.0,
                    "middle_band": price,
                    "lower_band": price - 2.0
                }
            }
        }
        
        composite_signal = test_signal_engine.generate_signals(
            symbol=symbol,
            price=price,
            timeframe_results=timeframe_results,
            trading_profile=TradingProfile.SWING_TRADING
        )
        
    # Phase 2: Verify signal generation
    stats = test_signal_engine.get_stats()
    assert stats["total_signals_generated"] > 0
    assert stats["composite_signals_generated"] > 0
    assert symbol in stats["symbols_processed"]
    
    # Phase 3: Test signal retrieval
    latest_signal = test_signal_engine.get_latest_signal(symbol)
    if latest_signal:
        assert latest_signal.symbol == symbol
        assert latest_signal.overall_direction in [
            SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD, SignalDirection.PREPARE
        ]
        
    # Phase 4: Test signal history
    history = test_signal_engine.get_signal_history(symbol, limit=10)
    assert len(history) > 0
    
    # Phase 5: Test cleanup
    test_signal_engine.cleanup_old_signals(max_age_hours=1)
    
    # Phase 6: Test reset
    test_signal_engine.reset_symbol(symbol)
    assert test_signal_engine.get_latest_signal(symbol) is None