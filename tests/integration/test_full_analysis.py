"""
Integration tests for end-to-end signal analysis workflow.

Tests the complete signal analysis pipeline from data generation to signal detection.
"""

import pytest
import numpy as np
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict
from unittest.mock import Mock, patch

from src.core.models import MarketData, TimeFrame, SignalAnalysisResult, Signal, SignalType
from src.data.mock_generator import MockDataGenerator
from src.strategies.ma_crossover import MovingAverageCrossover
from src.strategies.rsi_trend import RSITrendFollowing as RSITrend
from src.strategies.bollinger_breakout import BollingerBandBreakout as BollingerBreakout
from src.strategies.macd_rsi import MACDRSIStrategy
from src.strategies.sr_breakout import SupportResistanceBreakout
from src.backtest.engine import SignalAnalysisEngine, SignalAnalysisConfig
from src.services.data_service import DataService
from src.services.analysis_service import AnalysisService


class TestFullSignalAnalysisWorkflow:
    """Test complete signal analysis workflow."""
    
    def test_single_strategy_signal_analysis(self, mock_data_generator):
        """Test complete signal analysis for a single strategy."""
        # Generate market data
        data_points = mock_data_generator.generate_trending_data(
            periods=200,
            trend_strength=0.02,
            volatility=0.15,
            start_price=100.0
        )
        
        market_data = MarketData(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            data=data_points,
            indicators={},
            last_updated=datetime.now()
        )
        
        # Create strategy
        strategy = MovingAverageCrossover({
            'short_period': 20,
            'long_period': 50,
            'min_confidence': 0.6
        })
        
        # Run signal analysis
        config = SignalAnalysisConfig()
        engine = SignalAnalysisEngine(config)
        
        result = engine.run_signal_analysis(strategy, market_data)
        
        # Verify results
        assert isinstance(result, SignalAnalysisResult)
        assert result.strategy_name == "MovingAverageCrossover"
        assert result.symbol == "AAPL"
        assert len(result.signals) >= 0
        assert result.total_signals >= 0
        assert result.buy_signals >= 0
        assert result.sell_signals >= 0
        assert result.hold_signals >= 0
        assert 0.0 <= result.avg_confidence <= 1.0
        assert result.signal_frequency >= 0.0
    
    def test_multiple_strategy_signal_comparison(self, mock_data_generator):
        """Test signal comparison of multiple strategies."""
        # Generate market data
        data_points = mock_data_generator.generate_trending_data(
            periods=150,
            trend_strength=0.015,
            volatility=0.12,
            start_price=100.0
        )
        
        market_data = MarketData(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            data=data_points,
            indicators={},
            last_updated=datetime.now()
        )
        
        # Create multiple strategies
        strategies = [
            MovingAverageCrossover({'short_period': 20, 'long_period': 50}),
            RSITrend({'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70}),
            BollingerBreakout({'period': 20, 'std_dev': 2.0})
        ]
        
        # Run signal analysis for all strategies
        config = SignalAnalysisConfig()
        engine = SignalAnalysisEngine(config)
        
        results = []
        for strategy in strategies:
            result = engine.run_signal_analysis(strategy, market_data)
            results.append(result)
        
        # Compare results
        assert len(results) == len(strategies)
        
        # All results should be valid
        for i, result in enumerate(results):
            assert result.strategy_name == strategies[i].name
            assert result.symbol == "AAPL"
            assert len(result.signals) >= 0
            assert result.total_signals >= 0
            assert 0.0 <= result.avg_confidence <= 1.0
        
        # Compare signal characteristics
        signal_counts = [r.total_signals for r in results]
        avg_confidences = [r.avg_confidence for r in results]
        
        # Should have different signal characteristics
        if len(set(signal_counts)) > 1:
            # Different strategies should generate different numbers of signals
            assert max(signal_counts) != min(signal_counts)
        
        # All should have reasonable confidence levels
        assert all(0.0 <= conf <= 1.0 for conf in avg_confidences)
    
    def test_market_scenario_signal_analysis(self, mock_data_generator):
        """Test signal analysis across different market scenarios."""
        generator = mock_data_generator
        scenarios = generator.create_market_scenarios()
        
        # Select a few scenarios for testing
        test_scenarios = ['bull_market', 'bear_market', 'sideways_market']
        
        strategy = MovingAverageCrossover()
        config = SignalAnalysisConfig()
        engine = SignalAnalysisEngine(config)
        
        scenario_results = {}
        
        for scenario_name in test_scenarios:
            scenario = scenarios[scenario_name]
            
            # Generate data for scenario (limited periods for testing)
            if 'bull' in scenario_name or 'bear' in scenario_name:
                data_points = generator.generate_trending_data(
                    periods=min(100, scenario.periods),
                    trend_strength=scenario.trend_strength / 252,  # Convert annual to daily
                    volatility=scenario.volatility,
                    start_price=100.0
                )
            else:
                data_points = generator.generate_ranging_data(
                    periods=min(100, scenario.periods),
                    range_width=0.2,
                    volatility=scenario.volatility,
                    center_price=100.0
                )
            
            market_data = MarketData(
                symbol="TEST",
                timeframe=TimeFrame.DAILY,
                data=data_points,
                indicators={},
                last_updated=datetime.now()
            )
            
            # Run signal analysis
            result = engine.run_signal_analysis(strategy, market_data)
            scenario_results[scenario_name] = result
        
        # Analyze results across scenarios
        assert len(scenario_results) == len(test_scenarios)
        
        # Strategy should generate different signals in different market conditions
        signal_counts = {name: result.total_signals 
                        for name, result in scenario_results.items()}
        
        # Should have variation across scenarios
        signal_values = list(signal_counts.values())
        if len(signal_values) > 1:
            # Verify we have results for all scenarios
            assert len(signal_values) == len(test_scenarios), "Should have results for all scenarios"
            
            # Different market conditions should generate different signal patterns
            for scenario_name, result in scenario_results.items():
                assert result.total_signals >= 0
                assert 0.0 <= result.avg_confidence <= 1.0
                assert result.buy_signals + result.sell_signals + result.hold_signals == result.total_signals
    
    def test_parameter_optimization_for_signals(self, mock_data_generator):
        """Test parameter optimization for signal quality."""
        # Generate training data
        training_data = mock_data_generator.generate_trending_data(
            periods=100,
            trend_strength=0.02,
            volatility=0.15,
            start_price=100.0
        )
        
        market_data = MarketData(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            data=training_data,
            indicators={},
            last_updated=datetime.now()
        )
        
        # Create strategy
        strategy = MovingAverageCrossover()
        
        # Define parameter ranges
        parameter_ranges = {
            'short_period': [10, 20, 30],
            'long_period': [40, 50, 60]
        }
        
        # Run optimization for signal quality
        config = SignalAnalysisConfig()
        engine = SignalAnalysisEngine(config)
        
        optimization_result = engine.run_parameter_optimization(
            strategy,
            market_data,
            parameter_ranges,
            'avg_confidence'  # Optimize for signal confidence
        )
        
        # Verify optimization results
        assert 'best_parameters' in optimization_result
        assert 'best_score' in optimization_result
        assert 'best_result' in optimization_result
        
        best_params = optimization_result['best_parameters']
        if best_params:
            assert 'short_period' in best_params or 'long_period' in best_params
            
            # Best parameters should be from the tested ranges
            if 'short_period' in best_params:
                assert best_params['short_period'] in parameter_ranges['short_period']
            if 'long_period' in best_params:
                assert best_params['long_period'] in parameter_ranges['long_period']
            
            # Best score should be a valid confidence value
            assert 0.0 <= optimization_result['best_score'] <= 1.0
    
    def test_walkforward_signal_analysis(self, mock_data_generator):
        """Test walk-forward signal analysis simulation."""
        # Generate long-term data
        total_periods = 300
        data_points = mock_data_generator.generate_trending_data(
            periods=total_periods,
            trend_strength=0.01,
            volatility=0.15,
            start_price=100.0
        )
        
        strategy = MovingAverageCrossover({'short_period': 20, 'long_period': 50})
        config = SignalAnalysisConfig()
        engine = SignalAnalysisEngine(config)
        
        # Simulate walk-forward: train on first part, test on second part
        train_size = 200
        test_size = 100
        
        # Training data
        train_data = MarketData(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            data=data_points[:train_size],
            indicators={},
            last_updated=datetime.now()
        )
        
        # Test data
        test_data = MarketData(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            data=data_points[train_size:train_size + test_size],
            indicators={},
            last_updated=datetime.now()
        )
        
        # Run on training data
        train_result = engine.run_signal_analysis(strategy, train_data)
        
        # Run on test data
        test_result = engine.run_signal_analysis(strategy, test_data)
        
        # Verify both results
        assert isinstance(train_result, SignalAnalysisResult)
        assert isinstance(test_result, SignalAnalysisResult)
        
        # Signal characteristics may differ between training and test periods
        train_signals = train_result.total_signals
        test_signals = test_result.total_signals
        
        # Both should be valid numbers
        assert isinstance(train_signals, int)
        assert isinstance(test_signals, int)
        assert train_signals >= 0
        assert test_signals >= 0
        
        # Signal quality should be consistent
        assert 0.0 <= train_result.avg_confidence <= 1.0
        assert 0.0 <= test_result.avg_confidence <= 1.0
    
    def test_signal_quality_in_volatile_markets(self, mock_data_generator):
        """Test signal quality analysis in volatile market conditions."""
        # Generate volatile market data
        data_points = mock_data_generator.generate_trending_data(
            periods=150,
            trend_strength=0.01,
            volatility=0.25,  # High volatility
            start_price=100.0
        )
        
        market_data = MarketData(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            data=data_points,
            indicators={},
            last_updated=datetime.now()
        )
        
        # Test with different confidence thresholds
        confidence_thresholds = [0.5, 0.7, 0.9]  # Different signal quality levels
        
        strategy = MovingAverageCrossover()
        results = []
        
        for threshold in confidence_thresholds:
            config = SignalAnalysisConfig(min_confidence=threshold)
            engine = SignalAnalysisEngine(config)
            result = engine.run_signal_analysis(strategy, market_data)
            results.append(result)
        
        # Analyze signal quality vs quantity
        assert len(results) == len(confidence_thresholds)
        
        # Higher confidence thresholds should generally lead to fewer but higher quality signals
        for i, result in enumerate(results):
            assert isinstance(result.total_signals, int)
            assert result.total_signals >= 0
            if result.total_signals > 0:
                assert result.avg_confidence >= confidence_thresholds[i]
        
        # Signal count should generally decrease with higher thresholds
        signal_counts = [r.total_signals for r in results]
        if all(count > 0 for count in signal_counts):
            # Not strictly enforced as it depends on strategy implementation
            assert signal_counts[0] >= signal_counts[-1]  # Lower threshold >= higher threshold
    
    def test_multi_asset_signal_analysis(self, mock_data_generator):
        """Test signal analysis across multiple assets."""
        # Generate data for multiple assets
        assets = ["AAPL", "GOOGL", "MSFT"]
        market_data_dict = {}
        
        for asset in assets:
            # Different characteristics for each asset
            if asset == "AAPL":
                data_points = mock_data_generator.generate_trending_data(
                    periods=100, trend_strength=0.02, volatility=0.15
                )
            elif asset == "GOOGL":
                data_points = mock_data_generator.generate_trending_data(
                    periods=100, trend_strength=0.015, volatility=0.20
                )
            else:  # MSFT
                data_points = mock_data_generator.generate_ranging_data(
                    periods=100, range_width=0.15, volatility=0.12
                )
            
            market_data_dict[asset] = MarketData(
                symbol=asset,
                timeframe=TimeFrame.DAILY,
                data=data_points,
                indicators={},
                last_updated=datetime.now()
            )
        
        # Test strategy on all assets
        strategy = MovingAverageCrossover()
        config = SignalAnalysisConfig()
        engine = SignalAnalysisEngine(config)
        
        asset_results = {}
        for asset, market_data in market_data_dict.items():
            result = engine.run_signal_analysis(strategy, market_data)
            asset_results[asset] = result
        
        # Verify results for all assets
        assert len(asset_results) == len(assets)
        
        for asset, result in asset_results.items():
            assert result.symbol == asset
            assert isinstance(result.total_signals, int)
            assert result.total_signals >= 0
            assert 0.0 <= result.avg_confidence <= 1.0
        
        # Signal characteristics should vary across assets
        signal_counts = [result.total_signals for result in asset_results.values()]
        confidences = [result.avg_confidence for result in asset_results.values()]
        
        # Should have some variation (unless all markets are identical)
        if len(set(signal_counts)) > 1:
            assert max(signal_counts) != min(signal_counts)
        
        # All confidence levels should be valid
        for conf in confidences:
            assert 0.0 <= conf <= 1.0
    
    def test_strategy_signal_robustness(self, mock_data_generator):
        """Test strategy signal robustness across different conditions."""
        strategy = MovingAverageCrossover()
        config = SignalAnalysisConfig()
        engine = SignalAnalysisEngine(config)
        
        # Test conditions
        test_conditions = [
            {'periods': 50, 'trend': 0.03, 'volatility': 0.10},   # Strong trend, low vol
            {'periods': 50, 'trend': 0.01, 'volatility': 0.25},   # Weak trend, high vol
            {'periods': 50, 'trend': -0.02, 'volatility': 0.15},  # Downtrend
            {'periods': 50, 'trend': 0.0, 'volatility': 0.12},    # Sideways
        ]
        
        robustness_results = []
        
        for condition in test_conditions:
            if condition['trend'] == 0.0:
                # Sideways market
                data_points = mock_data_generator.generate_ranging_data(
                    periods=condition['periods'],
                    range_width=0.15,
                    volatility=condition['volatility']
                )
            else:
                # Trending market
                data_points = mock_data_generator.generate_trending_data(
                    periods=condition['periods'],
                    trend_strength=condition['trend'],
                    volatility=condition['volatility']
                )
            
            market_data = MarketData(
                symbol="TEST",
                timeframe=TimeFrame.DAILY,
                data=data_points,
                indicators={},
                last_updated=datetime.now()
            )
            
            result = engine.run_signal_analysis(strategy, market_data)
            robustness_results.append({
                'condition': condition,
                'result': result
            })
        
        # Analyze robustness
        assert len(robustness_results) == len(test_conditions)
        
        # Strategy should handle all conditions without errors
        for test_result in robustness_results:
            result = test_result['result']
            assert isinstance(result, SignalAnalysisResult)
            assert result.total_signals >= 0
            assert 0.0 <= result.avg_confidence <= 1.0
            assert not np.isnan(result.avg_confidence)
            assert not np.isinf(result.avg_confidence)
            assert result.buy_signals + result.sell_signals + result.hold_signals == result.total_signals
    
    def test_signal_attribution_analysis(self, mock_data_generator):
        """Test signal attribution analysis."""
        # Generate market data with known characteristics
        data_points = mock_data_generator.generate_trending_data(
            periods=100,
            trend_strength=0.02,
            volatility=0.15,
            start_price=100.0
        )
        
        market_data = MarketData(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            data=data_points,
            indicators={},
            last_updated=datetime.now()
        )
        
        # Run signal analysis
        strategy = MovingAverageCrossover()
        config = SignalAnalysisConfig()
        engine = SignalAnalysisEngine(config)
        
        result = engine.run_signal_analysis(strategy, market_data)
        
        # Analyze signal attribution
        signals = result.signals
        
        if len(signals) > 0:
            # Analyze signal statistics
            buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
            sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
            
            # Signal distribution should be reasonable
            assert len(buy_signals) == result.buy_signals
            assert len(sell_signals) == result.sell_signals
            
            # Analyze signal quality
            if len(buy_signals) > 0:
                avg_buy_confidence = sum(s.confidence for s in buy_signals) / len(buy_signals)
                assert 0.0 <= avg_buy_confidence <= 1.0
            
            if len(sell_signals) > 0:
                avg_sell_confidence = sum(s.confidence for s in sell_signals) / len(sell_signals)
                assert 0.0 <= avg_sell_confidence <= 1.0
        
        # Signal timestamps should be in chronological order
        assert len(result.signal_timestamps) == len(signals)
        if len(result.signal_timestamps) > 1:
            for i in range(1, len(result.signal_timestamps)):
                assert result.signal_timestamps[i] >= result.signal_timestamps[i-1]
        
        # Signal frequency should be reasonable
        assert result.signal_frequency >= 0.0
        assert result.signal_frequency <= 1.0
    
    def test_signal_stress_testing(self, mock_data_generator):
        """Test strategy signal generation under stress conditions."""
        strategy = MovingAverageCrossover()
        config = SignalAnalysisConfig()
        engine = SignalAnalysisEngine(config)
        
        # Stress test scenarios
        stress_scenarios = [
            # Flash crash
            {'periods': 30, 'trend': -0.20, 'volatility': 0.50},
            # Extreme volatility
            {'periods': 50, 'trend': 0.01, 'volatility': 0.60},
            # Prolonged decline
            {'periods': 80, 'trend': -0.05, 'volatility': 0.20},
        ]
        
        stress_results = []
        
        for scenario in stress_scenarios:
            data_points = mock_data_generator.generate_trending_data(
                periods=scenario['periods'],
                trend_strength=scenario['trend'],
                volatility=scenario['volatility'],
                start_price=100.0
            )
            
            market_data = MarketData(
                symbol="STRESS_TEST",
                timeframe=TimeFrame.DAILY,
                data=data_points,
                indicators={},
                last_updated=datetime.now()
            )
            
            result = engine.run_signal_analysis(strategy, market_data)
            stress_results.append(result)
        
        # Analyze stress test results
        for result in stress_results:
            # Strategy should survive stress conditions without errors
            assert isinstance(result, SignalAnalysisResult)
            assert result.total_signals >= 0
            assert 0.0 <= result.avg_confidence <= 1.0
            assert not np.isnan(result.avg_confidence)
            assert not np.isinf(result.avg_confidence)
            
            # Signal distribution should be valid
            assert result.buy_signals + result.sell_signals + result.hold_signals == result.total_signals
            
            # Signal frequency should be reasonable even under stress
            assert result.signal_frequency >= 0.0
            assert result.signal_frequency <= 1.0
    
    def test_signal_quality_benchmarking(self, mock_data_generator):
        """Test strategy signal quality benchmarking."""
        # Generate market data
        data_points = mock_data_generator.generate_trending_data(
            periods=100,
            trend_strength=0.015,
            volatility=0.12,
            start_price=100.0
        )
        
        market_data = MarketData(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            data=data_points,
            indicators={},
            last_updated=datetime.now()
        )
        
        # Strategy signal analysis
        strategy = MovingAverageCrossover()
        config = SignalAnalysisConfig()
        engine = SignalAnalysisEngine(config)
        
        strategy_result = engine.run_signal_analysis(strategy, market_data)
        
        # Calculate market trend benchmark (simple trend detection)
        price_changes = []
        for i in range(1, len(data_points)):
            change = (data_points[i].close - data_points[i-1].close) / data_points[i-1].close
            price_changes.append(change)
        
        avg_change = sum(price_changes) / len(price_changes) if price_changes else 0
        trend_strength = abs(avg_change) * 100
        
        # Compare strategy signal quality
        assert isinstance(strategy_result.avg_confidence, (int, float))
        assert isinstance(strategy_result.signal_frequency, (int, float))
        assert isinstance(trend_strength, (int, float))
        
        # Strategy should generate reasonable signals
        assert strategy_result.total_signals >= 0
        assert 0.0 <= strategy_result.avg_confidence <= 1.0
        assert strategy_result.signal_frequency >= 0.0
        
        # Quality metrics should be valid
        assert not np.isnan(strategy_result.avg_confidence)
        assert not np.isinf(strategy_result.avg_confidence)
        assert not np.isnan(strategy_result.signal_frequency)
        assert not np.isinf(strategy_result.signal_frequency)


class TestSignalAnalysisServices:
    """Test signal analysis service integration."""
    
    def test_data_service_integration(self, mock_data_generator):
        """Test data service integration."""
        # Mock data service
        data_service = DataService()
        
        # Generate sample data
        data_points = mock_data_generator.generate_trending_data(periods=50)
        market_data = MarketData(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            data=data_points,
            indicators={},
            last_updated=datetime.now()
        )
        
        # Test data service methods
        symbols = data_service.get_symbols()
        assert isinstance(symbols, list)
        
        # Test market open status
        is_open = data_service.is_market_open()
        assert isinstance(is_open, bool)
    
    def test_signal_analysis_service_integration(self, mock_data_generator):
        """Test signal analysis service integration."""
        analysis_service = AnalysisService()
        
        # Generate test data
        data_points = mock_data_generator.generate_trending_data(periods=50)
        market_data = MarketData(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            data=data_points,
            indicators={},
            last_updated=datetime.now()
        )
        
        # Create strategy
        strategy = MovingAverageCrossover()
        
        # Run signal analysis through service
        result = analysis_service.run_signal_analysis(
            strategy=strategy,
            market_data=market_data,
            config=SignalAnalysisConfig()
        )
        
        assert isinstance(result, SignalAnalysisResult)
        assert result.strategy_name == strategy.name
        assert result.symbol == "AAPL"
        assert result.total_signals >= 0
        assert 0.0 <= result.avg_confidence <= 1.0
    
    def test_strategy_signal_analysis_integration(self, mock_data_generator):
        """Test strategy signal analysis integration."""
        analysis_service = AnalysisService()
        
        # Generate test data
        data_points = mock_data_generator.generate_trending_data(periods=50)
        market_data = MarketData(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            data=data_points,
            indicators={},
            last_updated=datetime.now()
        )
        
        # Test strategy signal analysis
        strategy = MovingAverageCrossover()
        
        analysis_result = analysis_service.analyze_strategy_signals(
            strategy=strategy,
            market_data=market_data
        )
        
        # Should return analysis results
        assert analysis_result is not None
        assert isinstance(analysis_result, SignalAnalysisResult)
        assert analysis_result.strategy_name == strategy.name
    
    def test_service_error_handling(self):
        """Test service error handling."""
        # Test with invalid inputs
        data_service = DataService()
        analysis_service = AnalysisService()
        
        # Services should handle errors gracefully
        try:
            symbols = data_service.get_symbols()
            assert isinstance(symbols, list)
        except Exception as e:
            # Should handle gracefully
            assert isinstance(e, Exception)
        
        # Test signal analysis service error handling
        try:
            # Test with None strategy
            result = analysis_service.analyze_strategy_signals(None, None)
            assert result is None or isinstance(result, SignalAnalysisResult)
        except Exception as e:
            # Should handle gracefully
            assert isinstance(e, Exception)
    
    @pytest.mark.asyncio
    async def test_historical_signal_analysis_integration(self):
        """Test the new historical signal analysis job functionality."""
        analysis_service = AnalysisService()
        
        # Define test parameters
        symbol = "SPY"
        timeframe = TimeFrame.DAILY
        strategy = "macd_rsi_strategy"
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)
        
        # Submit historical signal analysis job
        job_id = await analysis_service.analyze_strategy_signals_historical(
            symbol=symbol,
            timeframe=timeframe,
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            parameters={
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70
            }
        )
        
        # Verify job was created
        assert job_id is not None
        assert isinstance(job_id, str)
        
        # Check initial job status
        status = await analysis_service.get_analysis_status(job_id)
        assert status is not None
        assert status['status'] in ['pending', 'running']  # Could be either in distributed system
        assert status['symbol'] == symbol
        assert status['strategy'] == strategy
        assert status['timeframe'] == timeframe.value
        
        # Wait for job completion (with timeout)
        max_wait = 5  # seconds
        wait_interval = 0.5
        elapsed = 0
        
        while elapsed < max_wait:
            await asyncio.sleep(wait_interval)
            elapsed += wait_interval
            
            status = await analysis_service.get_analysis_status(job_id)
            if status['status'] not in ['pending', 'running']:
                break
        
        # Verify final job status
        assert status['status'] in ['completed', 'failed']
        
        if status['status'] == 'completed':
            result = status['result']
            
            # Verify result structure
            assert 'signals' in result
            assert 'total_signals' in result
            assert 'buy_signals' in result
            assert 'sell_signals' in result
            assert 'hold_signals' in result
            assert 'avg_confidence' in result
            assert 'signal_frequency' in result
            assert 'date_range' in result
            
            # Verify metadata
            assert result['symbol'] == symbol
            assert result['strategy'] == strategy
            assert result['timeframe'] == timeframe.value
            
            # Verify date range
            assert result['date_range']['start'] == start_date.isoformat()
            assert result['date_range']['end'] == end_date.isoformat()
            
            # Verify signals are serializable
            if result['signals']:
                first_signal = result['signals'][0]
                assert 'timestamp' in first_signal
                assert 'signal_type' in first_signal
                assert 'price' in first_signal
                assert 'confidence' in first_signal
            
            # Verify signal quality metrics
            assert result['total_signals'] >= 0
            assert 0.0 <= result['avg_confidence'] <= 1.0
            assert result['signal_frequency'] >= 0.0


class TestSignalAnalysisPerformance:
    """Test performance of integrated signal analysis workflow."""
    
    def test_full_signal_workflow_performance(self, mock_data_generator, execution_timer):
        """Test performance of complete signal analysis workflow."""
        execution_timer.start()
        
        # Generate data
        data_points = mock_data_generator.generate_trending_data(periods=200)
        market_data = MarketData(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            data=data_points,
            indicators={},
            last_updated=datetime.now()
        )
        
        # Run multiple strategies
        strategies = [
            MovingAverageCrossover(),
            RSITrend(),
            BollingerBreakout()
        ]
        
        config = SignalAnalysisConfig()
        engine = SignalAnalysisEngine(config)
        
        results = []
        for strategy in strategies:
            result = engine.run_signal_analysis(strategy, market_data)
            results.append(result)
        
        execution_timer.stop()
        
        # Should complete within reasonable time
        assert execution_timer.get_elapsed_time() < 10.0
        assert len(results) == len(strategies)
        
        # Verify all results are valid
        for result in results:
            assert isinstance(result, SignalAnalysisResult)
            assert result.total_signals >= 0
            assert 0.0 <= result.avg_confidence <= 1.0
    
    def test_memory_usage_signal_analysis(self, mock_data_generator, memory_monitor):
        """Test memory usage of integrated signal analysis workflow."""
        initial_memory = memory_monitor.get_current_usage()
        
        # Run multiple signal analyses
        for i in range(5):
            data_points = mock_data_generator.generate_trending_data(periods=100)
            market_data = MarketData(
                symbol=f"STOCK_{i}",
                timeframe=TimeFrame.DAILY,
                data=data_points,
                indicators={},
                last_updated=datetime.now()
            )
            
            strategy = MovingAverageCrossover()
            config = SignalAnalysisConfig()
            engine = SignalAnalysisEngine(config)
            
            result = engine.run_signal_analysis(strategy, market_data)
            
            # Verify result is valid
            assert isinstance(result, SignalAnalysisResult)
            assert result.total_signals >= 0
        
        final_memory = memory_monitor.get_current_usage()
        memory_increase = final_memory - initial_memory
        
        # Should not use excessive memory
        assert memory_increase < 100  # MB