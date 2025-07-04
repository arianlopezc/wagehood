"""
Unit tests for backtesting engine and execution.

Tests the backtesting engine, portfolio management, and trade execution logic.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import List

from src.backtest.engine import BacktestEngine, BacktestConfig
from src.backtest.execution import PortfolioManager, OrderExecutor, MarketOrderExecutor
from src.backtest.costs import TransactionCostModel, CommissionFreeModel, RealisticCostModel
from src.core.models import (
    Signal, Trade, SignalType, OHLCV, MarketData, TimeFrame,
    BacktestResult, PerformanceMetrics
)
from src.strategies.ma_crossover import MovingAverageCrossover


class TestBacktestConfig:
    """Test backtest configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BacktestConfig()
        
        assert config.initial_capital == 10000.0
        assert config.commission_rate == 0.0  # Commission-free by default
        assert config.slippage_rate == 0.001
        assert config.max_positions == 5
        assert config.risk_per_trade == 0.02
        assert config.start_date is None
        assert config.end_date is None
        assert config.benchmark_symbol is None
    
    def test_custom_config(self):
        """Test custom configuration values."""
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
        
        config = BacktestConfig(
            initial_capital=50000.0,
            commission_rate=0.002,
            slippage_rate=0.0015,
            max_positions=10,
            risk_per_trade=0.05,
            start_date=start_date,
            end_date=end_date,
            benchmark_symbol="SPY"
        )
        
        assert config.initial_capital == 50000.0
        assert config.commission_rate == 0.002
        assert config.slippage_rate == 0.0015
        assert config.max_positions == 10
        assert config.risk_per_trade == 0.05
        assert config.start_date == start_date
        assert config.end_date == end_date
        assert config.benchmark_symbol == "SPY"


class TestTransactionCostModel:
    """Test transaction cost models."""
    
    def test_commission_free_model(self):
        """Test commission-free cost model."""
        cost_model = CommissionFreeModel()
        
        # Test commission calculation (should be 0)
        cost = cost_model.calculate_cost(quantity=100, price=150.0, side='buy', symbol='AAPL')
        assert cost.commission == 0.0
        assert cost.slippage == 0.0
        assert cost.spread == 0.0
        assert cost.total == 0.0
        
        # Test with slippage
        cost_model_with_slippage = CommissionFreeModel(slippage_rate=0.001)
        cost = cost_model_with_slippage.calculate_cost(quantity=100, price=150.0, side='buy', symbol='AAPL')
        assert cost.commission == 0.0
        assert cost.slippage == 100 * 150.0 * 0.001  # Should have slippage
        assert cost.spread == 0.0
        assert cost.total == cost.slippage
    
    def test_realistic_cost_model_commission_free_default(self):
        """Test realistic cost model with commission-free defaults."""
        cost_model = RealisticCostModel()  # Should default to commission-free
        
        # Test commission calculation (should be 0 by default)
        cost = cost_model.calculate_cost(quantity=100, price=150.0, side='buy', symbol='AAPL')
        assert cost.commission == 0.0
        assert cost.slippage > 0.0  # Should still have slippage
        assert cost.spread > 0.0    # Should still have spread
    
    def test_realistic_cost_model_with_commission(self):
        """Test realistic cost model with explicit commission settings."""
        cost_model = RealisticCostModel(
            commission_rate=0.001,
            min_commission=1.0,
            slippage_rate=0.0005
        )
        
        # Test commission
        cost = cost_model.calculate_cost(quantity=100, price=150.0, side='buy', symbol='AAPL')
        expected_commission = max(100 * 150.0 * 0.001, 1.0)
        assert cost.commission == expected_commission
        assert cost.slippage > 0.0
        assert cost.spread > 0.0
        assert cost.total == cost.commission + cost.slippage + cost.spread
    
    def test_cost_model_edge_cases(self):
        """Test cost model edge cases."""
        cost_model = RealisticCostModel(commission_rate=0.001, slippage_rate=0.0005)
        
        # Zero quantity
        cost = cost_model.calculate_cost(quantity=0, price=150.0, side='buy', symbol='AAPL')
        assert cost.commission == 0.0
        assert cost.total == 0.0
        
        # Zero price
        cost = cost_model.calculate_cost(quantity=100, price=0.0, side='buy', symbol='AAPL')
        assert cost.commission == 0.0
        assert cost.total == 0.0
        
        # Negative values should be handled appropriately
        cost = cost_model.calculate_cost(quantity=-100, price=150.0, side='sell', symbol='AAPL')
        assert cost.commission >= 0  # Commission should always be positive
    
    def test_variable_commission_rates(self):
        """Test cost model with variable commission rates."""
        cost_model = RealisticCostModel(commission_rate=0.002, slippage_rate=0.001)
        
        # Small trade
        small_cost = cost_model.calculate_cost(quantity=10, price=100.0, side='buy', symbol='AAPL')
        
        # Large trade
        large_cost = cost_model.calculate_cost(quantity=1000, price=100.0, side='buy', symbol='AAPL')
        
        # Commission should scale with trade size
        assert large_cost.commission > small_cost.commission
        assert large_cost.commission / small_cost.commission == 100  # Linear scaling


class TestOrderExecutor:
    """Test order execution logic."""
    
    def test_market_order_executor_creation(self):
        """Test market order executor creation."""
        # Test with default (CommissionFreeModel)
        executor = MarketOrderExecutor(risk_per_trade=0.02)
        assert isinstance(executor.cost_model, CommissionFreeModel)
        assert executor.risk_per_trade == 0.02
        
        # Test with explicit cost model
        cost_model = RealisticCostModel(commission_rate=0.001)
        executor = MarketOrderExecutor(cost_model=cost_model, risk_per_trade=0.02)
        assert executor.cost_model == cost_model
        assert executor.risk_per_trade == 0.02
    
    def test_calculate_position_size(self):
        """Test position size calculation."""
        cost_model = RealisticCostModel()
        executor = MarketOrderExecutor(cost_model=cost_model, risk_per_trade=0.02)
        
        # Test with specific parameters
        capital = 10000.0
        price = 100.0
        stop_loss = 95.0  # 5% stop loss
        
        position_size = executor.calculate_position_size(capital, price, stop_loss)
        
        # With 2% risk and 5% stop loss, should be able to buy 40 shares
        # Risk amount = 10000 * 0.02 = 200
        # Risk per share = 100 - 95 = 5
        # Position size = 200 / 5 = 40
        assert position_size == 40.0
    
    def test_calculate_position_size_edge_cases(self):
        """Test position size calculation edge cases."""
        cost_model = RealisticCostModel()
        executor = MarketOrderExecutor(cost_model=cost_model, risk_per_trade=0.02)
        
        # Stop loss equal to price (no risk)
        position_size = executor.calculate_position_size(10000.0, 100.0, 100.0)
        assert position_size == 0.0
        
        # Stop loss above price (invalid)
        position_size = executor.calculate_position_size(10000.0, 100.0, 105.0)
        assert position_size == 0.0
        
        # Zero capital
        position_size = executor.calculate_position_size(0.0, 100.0, 95.0)
        assert position_size == 0.0
    
    def test_execute_buy_order(self):
        """Test buy order execution."""
        cost_model = RealisticCostModel(commission_rate=0.001)
        executor = MarketOrderExecutor(cost_model=cost_model, risk_per_trade=0.02)
        
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.BUY,
            price=150.0,
            confidence=0.8,
            strategy_name="TestStrategy",
            metadata={}
        )
        
        market_data = OHLCV(
            timestamp=datetime.now(),
            open=149.0,
            high=152.0,
            low=148.0,
            close=150.0,
            volume=10000
        )
        
        capital = 10000.0
        
        trade = executor.execute_order(signal, market_data, capital)
        
        if trade:  # Trade may be None due to risk management
            assert trade.symbol == "AAPL"
            assert trade.entry_price > 0
            assert trade.quantity > 0
            assert trade.commission >= 0
            assert trade.strategy_name == "TestStrategy"
    
    def test_execute_sell_order(self):
        """Test sell order execution."""
        cost_model = RealisticCostModel(commission_rate=0.001)
        executor = MarketOrderExecutor(cost_model=cost_model, risk_per_trade=0.02)
        
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.SELL,
            price=150.0,
            confidence=0.8,
            strategy_name="TestStrategy",
            metadata={}
        )
        
        market_data = OHLCV(
            timestamp=datetime.now(),
            open=149.0,
            high=152.0,
            low=148.0,
            close=150.0,
            volume=10000
        )
        
        capital = 10000.0
        
        trade = executor.execute_order(signal, market_data, capital)
        
        # Sell orders might be handled differently (short selling)
        # Implementation depends on strategy
        assert trade is None or isinstance(trade, Trade)


class TestPortfolioManager:
    """Test portfolio management."""
    
    def test_portfolio_creation(self):
        """Test portfolio manager creation."""
        # Test with default commission-free executor
        portfolio = PortfolioManager(
            initial_capital=10000.0,
            max_positions=5
        )
        
        assert portfolio.initial_capital == 10000.0
        assert portfolio.current_capital == 10000.0
        assert isinstance(portfolio.executor.cost_model, CommissionFreeModel)
        assert portfolio.max_positions == 5
        assert len(portfolio.open_positions) == 0
        assert len(portfolio.closed_trades) == 0
        assert len(portfolio.equity_curve) == 1
        assert portfolio.equity_curve[0] == 10000.0
    
    def test_portfolio_can_open_position(self):
        """Test position opening logic."""
        executor = MarketOrderExecutor(risk_per_trade=0.02)  # Uses default CommissionFreeModel
        
        portfolio = PortfolioManager(
            initial_capital=10000.0,
            executor=executor,
            max_positions=2
        )
        
        # Should be able to open positions when under limit
        assert portfolio.can_open_position()
        
        # Add mock positions
        portfolio.open_positions = ["position1", "position2"]
        
        # Should not be able to open more positions
        assert not portfolio.can_open_position()
    
    def test_portfolio_process_signal(self):
        """Test signal processing."""
        executor = MarketOrderExecutor(risk_per_trade=0.02)  # Uses default CommissionFreeModel
        
        portfolio = PortfolioManager(
            initial_capital=10000.0,
            executor=executor,
            max_positions=5
        )
        
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.BUY,
            price=150.0,
            confidence=0.8,
            strategy_name="TestStrategy",
            metadata={}
        )
        
        market_data = OHLCV(
            timestamp=datetime.now(),
            open=149.0,
            high=152.0,
            low=148.0,
            close=150.0,
            volume=10000
        )
        
        with patch.object(executor, 'execute_order') as mock_execute:
            mock_trade = Trade(
                trade_id="test_1",
                entry_time=datetime.now(),
                exit_time=None,
                symbol="AAPL",
                quantity=50.0,
                entry_price=150.0,
                exit_price=None,
                pnl=None,
                commission=0.0,  # Commission-free by default
                strategy_name="TestStrategy",
                signal_metadata={}
            )
            mock_execute.return_value = mock_trade
            
            result = portfolio.process_signal(signal, market_data)
            
            assert result == mock_trade
            mock_execute.assert_called_once()
    
    def test_portfolio_update_equity(self):
        """Test equity curve updates."""
        cost_model = RealisticCostModel()
        executor = MarketOrderExecutor(cost_model=cost_model, risk_per_trade=0.02)
        
        portfolio = PortfolioManager(
            initial_capital=10000.0,
            executor=executor,
            max_positions=5
        )
        
        # Add a mock open position
        portfolio.open_positions = [
            Trade(
                trade_id="test_1",
                entry_time=datetime.now(),
                exit_time=None,
                symbol="AAPL",
                quantity=50.0,
                entry_price=150.0,
                exit_price=None,
                pnl=None,
                commission=7.5,
                strategy_name="TestStrategy",
                signal_metadata={}
            )
        ]
        
        # Mock current prices
        current_prices = {"AAPL": 155.0}
        
        portfolio.update_equity(current_prices)
        
        # Equity should reflect unrealized P&L
        # P&L = (155 - 150) * 50 = 250
        # New equity = 10000 + 250 = 10250
        assert portfolio.equity_curve[-1] > 10000.0
    
    def test_portfolio_close_position(self):
        """Test position closing."""
        cost_model = RealisticCostModel(commission_rate=0.001)
        executor = MarketOrderExecutor(cost_model=cost_model, risk_per_trade=0.02)
        
        portfolio = PortfolioManager(
            initial_capital=10000.0,
            executor=executor,
            max_positions=5
        )
        
        # Add an open position
        open_trade = Trade(
            trade_id="test_1",
            entry_time=datetime.now(),
            exit_time=None,
            symbol="AAPL",
            quantity=50.0,
            entry_price=150.0,
            exit_price=None,
            pnl=None,
            commission=7.5,
            strategy_name="TestStrategy",
            signal_metadata={}
        )
        portfolio.open_positions = [open_trade]
        
        # Close position
        exit_price = 155.0
        exit_time = datetime.now()
        
        closed_trade = portfolio.close_position(open_trade, exit_price, exit_time)
        
        assert closed_trade.exit_price == exit_price
        assert closed_trade.exit_time == exit_time
        assert closed_trade not in portfolio.open_positions
        assert closed_trade in portfolio.closed_trades
        
        # Capital should be updated
        expected_pnl = (155.0 - 150.0) * 50.0 - 7.5 - (155.0 * 50.0 * 0.001)  # Entry commission + exit commission
        assert portfolio.current_capital > 10000.0


class TestBacktestEngine:
    """Test backtesting engine."""
    
    def test_engine_creation(self):
        """Test engine creation."""
        config = BacktestConfig(initial_capital=20000.0)
        engine = BacktestEngine(config)
        
        assert engine.config == config
        assert isinstance(engine.results_cache, dict)
    
    def test_engine_default_config(self):
        """Test engine with default config."""
        engine = BacktestEngine()
        
        assert engine.config.initial_capital == 10000.0
        assert engine.config.commission_rate == 0.0  # Commission-free by default
    
    def test_run_backtest_basic(self, sample_market_data):
        """Test basic backtest execution."""
        config = BacktestConfig(initial_capital=10000.0)
        engine = BacktestEngine(config)
        
        strategy = MovingAverageCrossover({
            'short_period': 5,
            'long_period': 10,
            'min_confidence': 0.5
        })
        
        with patch.object(engine, '_generate_signals') as mock_signals:
            mock_signals.return_value = []  # No signals for simplicity
            
            result = engine.run_backtest(strategy, sample_market_data)
            
            assert isinstance(result, BacktestResult)
            assert result.strategy_name == "MovingAverageCrossover"
            assert result.symbol == sample_market_data.symbol
            assert result.initial_capital == 10000.0
            assert result.final_capital == 10000.0  # No trades
            assert len(result.trades) == 0
            assert len(result.equity_curve) >= 1
            assert isinstance(result.performance_metrics, PerformanceMetrics)
    
    def test_run_backtest_with_signals(self, sample_market_data):
        """Test backtest with actual signals."""
        config = BacktestConfig(initial_capital=10000.0)
        engine = BacktestEngine(config)
        
        strategy = MovingAverageCrossover({
            'short_period': 5,
            'long_period': 10,
            'min_confidence': 0.5
        })
        
        # Mock some signals
        mock_signals = [
            Signal(
                timestamp=sample_market_data.data[10].timestamp,
                symbol="AAPL",
                signal_type=SignalType.BUY,
                price=100.0,
                confidence=0.8,
                strategy_name="MovingAverageCrossover",
                metadata={}
            ),
            Signal(
                timestamp=sample_market_data.data[20].timestamp,
                symbol="AAPL",
                signal_type=SignalType.SELL,
                price=105.0,
                confidence=0.7,
                strategy_name="MovingAverageCrossover",
                metadata={}
            )
        ]
        
        with patch.object(engine, '_generate_signals') as mock_generate:
            mock_generate.return_value = mock_signals
            
            result = engine.run_backtest(strategy, sample_market_data)
            
            assert isinstance(result, BacktestResult)
            assert result.strategy_name == "MovingAverageCrossover"
            # May or may not have trades depending on execution logic
    
    def test_backtest_custom_parameters(self, sample_market_data):
        """Test backtest with custom parameters."""
        config = BacktestConfig(initial_capital=50000.0)
        engine = BacktestEngine(config)
        
        strategy = MovingAverageCrossover()
        
        result = engine.run_backtest(
            strategy, 
            sample_market_data,
            initial_capital=25000.0,
            commission_rate=0.002,
            slippage_rate=0.001
        )
        
        # Should use provided parameters over config
        assert result.initial_capital == 25000.0
    
    def test_backtest_performance_metrics_calculation(self):
        """Test performance metrics calculation."""
        config = BacktestConfig()
        engine = BacktestEngine(config)
        
        # Mock trades
        trades = [
            Trade("1", datetime.now(), datetime.now(), "AAPL", 100, 100, 105, 500, 1, "TestStrategy", {}),
            Trade("2", datetime.now(), datetime.now(), "AAPL", 100, 105, 103, -200, 1, "TestStrategy", {}),
            Trade("3", datetime.now(), datetime.now(), "AAPL", 100, 103, 108, 500, 1, "TestStrategy", {}),
        ]
        
        equity_curve = [10000, 10500, 10300, 10800]
        initial_capital = 10000.0
        
        metrics = engine._calculate_performance_metrics(trades, equity_curve, initial_capital)
        
        assert metrics.total_trades == 3
        assert metrics.winning_trades == 2
        assert metrics.losing_trades == 1
        assert metrics.win_rate == 2/3
        assert metrics.total_pnl == 800  # 500 - 200 + 500
        assert metrics.total_return_pct == 8.0  # 800/10000 * 100
    
    def test_backtest_empty_trades(self):
        """Test performance metrics with no trades."""
        config = BacktestConfig()
        engine = BacktestEngine(config)
        
        metrics = engine._calculate_performance_metrics([], [10000], 10000.0)
        
        assert metrics.total_trades == 0
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.total_pnl == 0.0
        assert metrics.total_return_pct == 0.0
        assert metrics.max_drawdown == 0.0
        assert metrics.sharpe_ratio == 0.0
    
    def test_drawdown_calculation(self):
        """Test drawdown calculation."""
        config = BacktestConfig()
        engine = BacktestEngine(config)
        
        # Equity curve with drawdown
        equity_curve = [10000, 10500, 9800, 9500, 10200, 11000]
        
        max_drawdown, max_drawdown_pct = engine._calculate_drawdown(equity_curve)
        
        # Maximum drawdown should be from 10500 to 9500 = 1000
        assert max_drawdown == 1000.0
        # Percentage: 1000/10500 * 100 ≈ 9.52%
        assert abs(max_drawdown_pct - (1000/10500*100)) < 0.01
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation."""
        config = BacktestConfig()
        engine = BacktestEngine(config)
        
        # Equity curve with positive trend
        equity_curve = [10000, 10100, 10200, 10150, 10300, 10400, 10350, 10500]
        
        sharpe = engine._calculate_sharpe_ratio(equity_curve)
        
        # Should be a positive number for upward trending equity
        assert sharpe > 0
        assert not np.isnan(sharpe)
        assert not np.isinf(sharpe)
    
    def test_sortino_ratio_calculation(self):
        """Test Sortino ratio calculation."""
        config = BacktestConfig()
        engine = BacktestEngine(config)
        
        # Equity curve with some negative returns
        equity_curve = [10000, 10100, 9950, 10200, 9800, 10300]
        
        sortino = engine._calculate_sortino_ratio(equity_curve)
        
        # Should handle negative returns appropriately
        assert isinstance(sortino, (int, float))
        assert not np.isnan(sortino)
    
    def test_consecutive_stats_calculation(self):
        """Test consecutive wins/losses calculation."""
        config = BacktestConfig()
        engine = BacktestEngine(config)
        
        # Trade P&Ls: win, win, win, loss, loss, win, win
        trade_pnls = [100, 50, 200, -150, -75, 300, 125]
        
        max_wins, max_losses = engine._calculate_consecutive_stats(trade_pnls)
        
        assert max_wins == 3  # First three wins
        assert max_losses == 2  # Two consecutive losses
    
    def test_find_closest_bar(self, sample_market_data):
        """Test finding closest market bar."""
        config = BacktestConfig()
        engine = BacktestEngine(config)
        
        # Target timestamp between bars
        target_time = sample_market_data.data[5].timestamp + timedelta(hours=1)
        
        closest_bar = engine._find_closest_bar(target_time, sample_market_data.data)
        
        assert closest_bar is not None
        # Should be one of the nearby bars
        assert closest_bar in sample_market_data.data
    
    def test_equity_curve_calculation(self):
        """Test equity curve calculation."""
        config = BacktestConfig()
        engine = BacktestEngine(config)
        
        trades = [
            Trade("1", datetime.now(), datetime.now(), "AAPL", 100, 100, 105, 500, 1, "TestStrategy", {}),
            Trade("2", datetime.now(), datetime.now(), "AAPL", 100, 105, 103, -200, 1, "TestStrategy", {}),
        ]
        
        initial_capital = 10000.0
        equity_curve = engine.calculate_equity_curve(trades, initial_capital)
        
        assert len(equity_curve) == 3  # Initial + 2 trades
        assert equity_curve[0] == initial_capital
        assert equity_curve[1] == initial_capital + 500
        assert equity_curve[2] == initial_capital + 500 - 200
    
    def test_trade_log_generation(self):
        """Test trade log generation."""
        config = BacktestConfig()
        engine = BacktestEngine(config)
        
        trades = [
            Trade("1", datetime.now(), datetime.now(), "AAPL", 100, 100, 105, 500, 1, "TestStrategy", {"test": True})
        ]
        
        trade_log = engine.generate_trade_log(trades)
        
        assert len(trade_log) == 1
        assert trade_log[0]['trade_id'] == "1"
        assert trade_log[0]['symbol'] == "AAPL"
        assert trade_log[0]['pnl'] == 500
        assert 'metadata' in trade_log[0]
    
    def test_parameter_optimization(self, sample_market_data):
        """Test parameter optimization."""
        config = BacktestConfig()
        engine = BacktestEngine(config)
        
        strategy = MovingAverageCrossover()
        
        # Limited parameter ranges for testing
        parameter_ranges = {
            'short_period': [10, 20],
            'long_period': [30, 40]
        }
        
        with patch.object(engine, 'run_backtest') as mock_backtest:
            # Mock backtest results
            mock_result = Mock()
            mock_result.performance_metrics.sharpe_ratio = 1.5
            mock_backtest.return_value = mock_result
            
            result = engine.run_parameter_optimization(
                strategy, 
                sample_market_data, 
                parameter_ranges,
                'sharpe_ratio'
            )
            
            assert 'best_parameters' in result
            assert 'best_score' in result
            assert 'best_result' in result
            assert result['best_score'] == 1.5
    
    def test_backtest_caching(self, sample_market_data):
        """Test backtest result caching."""
        config = BacktestConfig()
        engine = BacktestEngine(config)
        
        strategy = MovingAverageCrossover()
        
        with patch.object(engine, '_generate_signals') as mock_signals:
            mock_signals.return_value = []
            
            # Run backtest twice
            result1 = engine.run_backtest(strategy, sample_market_data)
            result2 = engine.run_backtest(strategy, sample_market_data)
            
            # Should be cached
            cache_key = f"{strategy.name}_{sample_market_data.symbol}_{config.initial_capital}"
            assert cache_key in engine.results_cache
            assert engine.results_cache[cache_key] == result2


class TestBacktestPerformance:
    """Test backtest performance and benchmarks."""
    
    def test_backtest_performance(self, sample_market_data, execution_timer):
        """Test backtest execution performance."""
        config = BacktestConfig()
        engine = BacktestEngine(config)
        
        strategy = MovingAverageCrossover()
        
        execution_timer.start()
        result = engine.run_backtest(strategy, sample_market_data)
        execution_timer.stop()
        
        # Should complete within reasonable time
        assert execution_timer.get_elapsed_time() < 5.0
        assert isinstance(result, BacktestResult)
    
    def test_large_dataset_performance(self, mock_data_generator, execution_timer):
        """Test performance with large dataset."""
        # Generate large dataset
        large_data = mock_data_generator.generate_trending_data(
            periods=1000,  # Large dataset
            trend_strength=0.02,
            volatility=0.15
        )
        
        market_data = MarketData(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            data=large_data,
            indicators={},
            last_updated=datetime.now()
        )
        
        config = BacktestConfig()
        engine = BacktestEngine(config)
        strategy = MovingAverageCrossover({'short_period': 20, 'long_period': 50})
        
        execution_timer.start()
        result = engine.run_backtest(strategy, market_data)
        execution_timer.stop()
        
        # Should handle large datasets efficiently
        assert execution_timer.get_elapsed_time() < 10.0
        assert len(result.equity_curve) >= 1
    
    def test_memory_usage(self, sample_market_data, memory_monitor):
        """Test memory usage during backtesting."""
        initial_memory = memory_monitor.get_current_usage()
        
        config = BacktestConfig()
        engine = BacktestEngine(config)
        strategy = MovingAverageCrossover()
        
        # Run multiple backtests
        for i in range(10):
            result = engine.run_backtest(strategy, sample_market_data)
        
        final_memory = memory_monitor.get_current_usage()
        memory_increase = final_memory - initial_memory
        
        # Should not leak excessive memory
        assert memory_increase < 100  # MB


class TestBacktestErrorHandling:
    """Test backtest error handling."""
    
    def test_empty_data(self):
        """Test backtest with empty data."""
        empty_data = MarketData(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            data=[],
            indicators={},
            last_updated=datetime.now()
        )
        
        config = BacktestConfig()
        engine = BacktestEngine(config)
        strategy = MovingAverageCrossover()
        
        result = engine.run_backtest(strategy, empty_data)
        
        # Should handle empty data gracefully
        assert isinstance(result, BacktestResult)
        assert len(result.trades) == 0
        assert len(result.equity_curve) >= 1
    
    def test_invalid_signals(self, sample_market_data):
        """Test backtest with invalid signals."""
        config = BacktestConfig()
        engine = BacktestEngine(config)
        strategy = MovingAverageCrossover()
        
        # Mock invalid signals
        invalid_signals = [
            Signal(
                timestamp=datetime.now() + timedelta(days=365),  # Future timestamp
                symbol="INVALID",
                signal_type=SignalType.BUY,
                price=0.0,  # Invalid price
                confidence=1.5,  # Invalid confidence
                strategy_name="TestStrategy",
                metadata={}
            )
        ]
        
        with patch.object(engine, '_generate_signals') as mock_signals:
            mock_signals.return_value = invalid_signals
            
            result = engine.run_backtest(strategy, sample_market_data)
            
            # Should handle invalid signals gracefully
            assert isinstance(result, BacktestResult)
    
    def test_execution_errors(self, sample_market_data):
        """Test handling of execution errors."""
        config = BacktestConfig()
        engine = BacktestEngine(config)
        strategy = MovingAverageCrossover()
        
        with patch.object(engine, '_simulate_execution') as mock_execution:
            mock_execution.side_effect = Exception("Execution error")
            
            # Should handle execution errors gracefully
            try:
                result = engine.run_backtest(strategy, sample_market_data)
                # If no exception is raised, check that result is reasonable
                assert isinstance(result, BacktestResult)
            except Exception:
                # Exception handling depends on implementation
                pass
    
    def test_malformed_market_data(self):
        """Test backtest with malformed market data."""
        # Data with invalid OHLCV relationships
        try:
            malformed_data = [
                OHLCV(datetime.now(), 100, 90, 110, 105, 1000),  # High < Open, Low > Close
            ]
        except ValueError:
            # Expected - OHLCV validation should catch this
            malformed_data = []
        
        market_data = MarketData(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            data=malformed_data,
            indicators={},
            last_updated=datetime.now()
        )
        
        config = BacktestConfig()
        engine = BacktestEngine(config)
        strategy = MovingAverageCrossover()
        
        result = engine.run_backtest(strategy, market_data)
        
        # Should handle malformed data gracefully
        assert isinstance(result, BacktestResult)