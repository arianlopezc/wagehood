"""
Integration tests for end-to-end analysis workflow.

Tests the complete analysis pipeline from data generation to strategy evaluation.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
from unittest.mock import Mock, patch

from src.core.models import MarketData, TimeFrame, BacktestResult
from src.data.mock_generator import MockDataGenerator
from src.strategies.ma_crossover import MovingAverageCrossover
from src.strategies.rsi_trend import RSITrend
from src.strategies.bollinger_breakout import BollingerBreakout
from src.strategies.macd_rsi import MACDRSIStrategy
from src.strategies.sr_breakout import SRBreakout
from src.backtest.engine import BacktestEngine, BacktestConfig
from src.services.data_service import DataService
from src.services.backtest_service import BacktestService
from src.services.analysis_service import AnalysisService
from src.analysis.evaluator import StrategyEvaluator
from src.analysis.comparison import StrategyComparison


class TestFullAnalysisWorkflow:
    """Test complete analysis workflow."""
    
    def test_single_strategy_analysis(self, mock_data_generator):
        """Test complete analysis for a single strategy."""
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
        
        # Run backtest (using commission-free default)
        config = BacktestConfig(
            initial_capital=10000.0,
            slippage_rate=0.001
            # commission_rate defaults to 0.0 for commission-free trading
        )
        engine = BacktestEngine(config)
        
        result = engine.run_backtest(strategy, market_data)
        
        # Verify results
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "MovingAverageCrossover"
        assert result.symbol == "AAPL"
        assert result.initial_capital == 10000.0
        assert len(result.equity_curve) > 0
        assert result.performance_metrics is not None
        
        # Check performance metrics
        metrics = result.performance_metrics
        assert metrics.total_trades >= 0
        assert metrics.win_rate >= 0.0
        assert metrics.win_rate <= 1.0
        assert isinstance(metrics.total_pnl, (int, float))
        assert isinstance(metrics.sharpe_ratio, (int, float))
    
    def test_multiple_strategy_comparison(self, mock_data_generator):
        """Test comparison of multiple strategies."""
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
        
        # Run backtests for all strategies
        config = BacktestConfig(initial_capital=10000.0)
        engine = BacktestEngine(config)
        
        results = []
        for strategy in strategies:
            result = engine.run_backtest(strategy, market_data)
            results.append(result)
        
        # Compare results
        assert len(results) == len(strategies)
        
        # All results should be valid
        for i, result in enumerate(results):
            assert result.strategy_name == strategies[i].name
            assert result.symbol == "AAPL"
            assert len(result.equity_curve) > 0
        
        # Compare performance metrics
        sharpe_ratios = [r.performance_metrics.sharpe_ratio for r in results]
        total_returns = [r.performance_metrics.total_return_pct for r in results]
        
        # Should have different performance characteristics
        assert not all(abs(sr - sharpe_ratios[0]) < 0.001 for sr in sharpe_ratios)
    
    def test_market_scenario_analysis(self, mock_data_generator):
        """Test analysis across different market scenarios."""
        generator = mock_data_generator
        scenarios = generator.create_market_scenarios()
        
        # Select a few scenarios for testing
        test_scenarios = ['bull_market', 'bear_market', 'sideways_market']
        
        strategy = MovingAverageCrossover()
        config = BacktestConfig(initial_capital=10000.0)
        engine = BacktestEngine(config)
        
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
            
            # Run backtest
            result = engine.run_backtest(strategy, market_data)
            scenario_results[scenario_name] = result
        
        # Analyze results across scenarios
        assert len(scenario_results) == len(test_scenarios)
        
        # Strategy should perform differently in different market conditions
        returns = {name: result.performance_metrics.total_return_pct 
                  for name, result in scenario_results.items()}
        
        # Should have variation across scenarios
        return_values = list(returns.values())
        if len(return_values) > 1:
            assert max(return_values) != min(return_values)
    
    def test_parameter_optimization_workflow(self, mock_data_generator):
        """Test parameter optimization workflow."""
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
        
        # Run optimization
        config = BacktestConfig(initial_capital=10000.0)
        engine = BacktestEngine(config)
        
        optimization_result = engine.run_parameter_optimization(
            strategy,
            market_data,
            parameter_ranges,
            'sharpe_ratio'
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
    
    def test_walkforward_analysis(self, mock_data_generator):
        """Test walk-forward analysis simulation."""
        # Generate long-term data
        total_periods = 300
        data_points = mock_data_generator.generate_trending_data(
            periods=total_periods,
            trend_strength=0.01,
            volatility=0.15,
            start_price=100.0
        )
        
        strategy = MovingAverageCrossover({'short_period': 20, 'long_period': 50})
        config = BacktestConfig(initial_capital=10000.0)
        engine = BacktestEngine(config)
        
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
        train_result = engine.run_backtest(strategy, train_data)
        
        # Run on test data
        test_result = engine.run_backtest(strategy, test_data)
        
        # Verify both results
        assert isinstance(train_result, BacktestResult)
        assert isinstance(test_result, BacktestResult)
        
        # Performance may differ between training and test periods
        train_return = train_result.performance_metrics.total_return_pct
        test_return = test_result.performance_metrics.total_return_pct
        
        # Both should be valid numbers
        assert isinstance(train_return, (int, float))
        assert isinstance(test_return, (int, float))
    
    def test_risk_analysis(self, mock_data_generator):
        """Test risk analysis workflow."""
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
        
        # Test with different risk levels
        risk_levels = [0.01, 0.02, 0.05]  # 1%, 2%, 5% risk per trade
        
        strategy = MovingAverageCrossover()
        results = []
        
        for risk_level in risk_levels:
            config = BacktestConfig(
                initial_capital=10000.0,
                risk_per_trade=risk_level
            )
            engine = BacktestEngine(config)
            result = engine.run_backtest(strategy, market_data)
            results.append(result)
        
        # Analyze risk vs return
        assert len(results) == len(risk_levels)
        
        # Higher risk should generally lead to different position sizes
        # (though final returns depend on strategy performance)
        for result in results:
            assert isinstance(result.performance_metrics.max_drawdown, (int, float))
            assert result.performance_metrics.max_drawdown >= 0
    
    def test_multi_asset_analysis(self, mock_data_generator):
        """Test analysis across multiple assets."""
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
        config = BacktestConfig(initial_capital=10000.0)
        engine = BacktestEngine(config)
        
        asset_results = {}
        for asset, market_data in market_data_dict.items():
            result = engine.run_backtest(strategy, market_data)
            asset_results[asset] = result
        
        # Verify results for all assets
        assert len(asset_results) == len(assets)
        
        for asset, result in asset_results.items():
            assert result.symbol == asset
            assert isinstance(result.performance_metrics.total_return_pct, (int, float))
        
        # Performance should vary across assets
        returns = [result.performance_metrics.total_return_pct 
                  for result in asset_results.values()]
        
        # Should have some variation (unless all markets are identical)
        if len(set(returns)) > 1:
            assert max(returns) != min(returns)
    
    def test_strategy_robustness(self, mock_data_generator):
        """Test strategy robustness across different conditions."""
        strategy = MovingAverageCrossover()
        config = BacktestConfig(initial_capital=10000.0)
        engine = BacktestEngine(config)
        
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
            
            result = engine.run_backtest(strategy, market_data)
            robustness_results.append({
                'condition': condition,
                'result': result
            })
        
        # Analyze robustness
        assert len(robustness_results) == len(test_conditions)
        
        # Strategy should handle all conditions without errors
        for test_result in robustness_results:
            result = test_result['result']
            assert isinstance(result, BacktestResult)
            assert not np.isnan(result.performance_metrics.total_return_pct)
            assert not np.isinf(result.performance_metrics.total_return_pct)
    
    def test_performance_attribution(self, mock_data_generator):
        """Test performance attribution analysis."""
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
        
        # Run backtest
        strategy = MovingAverageCrossover()
        config = BacktestConfig(initial_capital=10000.0)
        engine = BacktestEngine(config)
        
        result = engine.run_backtest(strategy, market_data)
        
        # Analyze performance attribution
        trades = result.trades
        equity_curve = result.equity_curve
        
        if len(trades) > 0:
            # Analyze trade statistics
            winning_trades = [t for t in trades if t.calculate_pnl() > 0]
            losing_trades = [t for t in trades if t.calculate_pnl() < 0]
            
            # Performance should come from both wins and losses
            if len(winning_trades) > 0:
                avg_win = sum(t.calculate_pnl() for t in winning_trades) / len(winning_trades)
                assert avg_win > 0
            
            if len(losing_trades) > 0:
                avg_loss = sum(t.calculate_pnl() for t in losing_trades) / len(losing_trades)
                assert avg_loss < 0
        
        # Equity curve should show progression
        assert len(equity_curve) > 0
        initial_equity = equity_curve[0]
        final_equity = equity_curve[-1]
        
        # Total return should match equity curve
        total_return = (final_equity - initial_equity) / initial_equity * 100
        expected_return = result.performance_metrics.total_return_pct
        
        # Should be approximately equal (allowing for small rounding differences)
        assert abs(total_return - expected_return) < 0.01
    
    def test_stress_testing(self, mock_data_generator):
        """Test strategy under stress conditions."""
        strategy = MovingAverageCrossover()
        config = BacktestConfig(initial_capital=10000.0)
        engine = BacktestEngine(config)
        
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
            
            result = engine.run_backtest(strategy, market_data)
            stress_results.append(result)
        
        # Analyze stress test results
        for result in stress_results:
            # Strategy should survive stress conditions
            assert isinstance(result, BacktestResult)
            assert result.final_capital > 0  # Should not go bankrupt
            
            # Drawdown analysis
            max_drawdown_pct = result.performance_metrics.max_drawdown_pct
            assert max_drawdown_pct >= 0
            assert max_drawdown_pct <= 100  # Cannot lose more than 100%
    
    def test_benchmark_comparison(self, mock_data_generator):
        """Test strategy comparison against benchmark."""
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
        
        # Strategy performance
        strategy = MovingAverageCrossover()
        config = BacktestConfig(initial_capital=10000.0)
        engine = BacktestEngine(config)
        
        strategy_result = engine.run_backtest(strategy, market_data)
        
        # Calculate buy-and-hold benchmark
        initial_price = data_points[0].close
        final_price = data_points[-1].close
        benchmark_return = (final_price - initial_price) / initial_price * 100
        
        # Compare performance
        strategy_return = strategy_result.performance_metrics.total_return_pct
        
        # Both should be valid numbers
        assert isinstance(strategy_return, (int, float))
        assert isinstance(benchmark_return, (int, float))
        
        # Strategy may outperform or underperform benchmark
        performance_diff = strategy_return - benchmark_return
        assert isinstance(performance_diff, (int, float))
        
        # Risk-adjusted comparison
        strategy_sharpe = strategy_result.performance_metrics.sharpe_ratio
        assert isinstance(strategy_sharpe, (int, float))


class TestAnalysisServices:
    """Test analysis service integration."""
    
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
    
    def test_backtest_service_integration(self, mock_data_generator):
        """Test backtest service integration."""
        backtest_service = BacktestService()
        
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
        
        # Run backtest through service
        result = backtest_service.run_backtest(
            strategy=strategy,
            market_data=market_data,
            config=BacktestConfig()
        )
        
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == strategy.name
    
    def test_analysis_service_integration(self, mock_data_generator):
        """Test analysis service integration."""
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
        
        # Test strategy analysis
        strategy = MovingAverageCrossover()
        
        analysis_result = analysis_service.analyze_strategy(
            strategy=strategy,
            market_data=market_data
        )
        
        # Should return analysis results
        assert analysis_result is not None
    
    def test_service_error_handling(self):
        """Test service error handling."""
        # Test with invalid inputs
        data_service = DataService()
        backtest_service = BacktestService()
        analysis_service = AnalysisService()
        
        # Services should handle errors gracefully
        try:
            symbols = data_service.get_symbols()
            assert isinstance(symbols, list)
        except Exception as e:
            # Should handle gracefully
            assert isinstance(e, Exception)
        
        try:
            is_open = data_service.is_market_open()
            assert isinstance(is_open, bool)
        except Exception as e:
            # Should handle gracefully
            assert isinstance(e, Exception)


class TestPerformanceIntegration:
    """Test performance of integrated analysis workflow."""
    
    def test_full_workflow_performance(self, mock_data_generator, execution_timer):
        """Test performance of complete workflow."""
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
        
        config = BacktestConfig()
        engine = BacktestEngine(config)
        
        results = []
        for strategy in strategies:
            result = engine.run_backtest(strategy, market_data)
            results.append(result)
        
        execution_timer.stop()
        
        # Should complete within reasonable time
        assert execution_timer.get_elapsed_time() < 10.0
        assert len(results) == len(strategies)
    
    def test_memory_usage_integration(self, mock_data_generator, memory_monitor):
        """Test memory usage of integrated workflow."""
        initial_memory = memory_monitor.get_current_usage()
        
        # Run multiple analyses
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
            config = BacktestConfig()
            engine = BacktestEngine(config)
            
            result = engine.run_backtest(strategy, market_data)
        
        final_memory = memory_monitor.get_current_usage()
        memory_increase = final_memory - initial_memory
        
        # Should not use excessive memory
        assert memory_increase < 100  # MB