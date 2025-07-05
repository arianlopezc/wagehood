"""
Pytest configuration and fixtures for the trading system tests.

This module provides shared fixtures, test data, and configuration for all tests.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Generator
from unittest.mock import Mock, MagicMock
import tempfile
import os
from pathlib import Path

# Import core modules
from src.core.models import (
    OHLCV, Signal, Trade, SignalType, TimeFrame, 
    MarketData, PerformanceMetrics, BacktestResult
)
from src.data.mock_generator import MockDataGenerator
from src.strategies.ma_crossover import MovingAverageCrossover
from src.strategies.rsi_trend import RSITrendFollowing
from src.strategies.bollinger_breakout import BollingerBandBreakout
from src.strategies.macd_rsi import MACDRSIStrategy
from src.strategies.sr_breakout import SupportResistanceBreakout
from src.backtest.engine import BacktestEngine, BacktestConfig
from src.services.data_service import DataService
from src.services.backtest_service import BacktestService
from src.services.analysis_service import AnalysisService

# Test configuration
TEST_SEED = 42
TEST_PERIODS = 100
TEST_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY"]


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def mock_data_generator():
    """Create a mock data generator with fixed seed."""
    return MockDataGenerator(seed=TEST_SEED)


@pytest.fixture(scope="session")
def sample_ohlcv_data(mock_data_generator) -> List[OHLCV]:
    """Generate sample OHLCV data for testing."""
    return mock_data_generator.generate_trending_data(
        periods=TEST_PERIODS,
        trend_strength=0.02,
        volatility=0.15,
        start_price=100.0
    )


@pytest.fixture(scope="session")
def sample_market_data(sample_ohlcv_data) -> MarketData:
    """Create sample market data structure."""
    return MarketData(
        symbol="AAPL",
        timeframe=TimeFrame.DAILY,
        data=sample_ohlcv_data,
        indicators={},
        last_updated=datetime.now()
    )


@pytest.fixture(scope="session")
def sample_signals() -> List[Signal]:
    """Create sample trading signals."""
    base_time = datetime.now() - timedelta(days=30)
    signals = []
    
    # Buy signals
    for i in range(5):
        signals.append(Signal(
            timestamp=base_time + timedelta(days=i * 3),
            symbol="AAPL",
            signal_type=SignalType.BUY,
            price=100.0 + i * 5,
            confidence=0.7 + i * 0.05,
            strategy_name="TestStrategy",
            metadata={"test": True, "signal_id": i}
        ))
    
    # Sell signals
    for i in range(5):
        signals.append(Signal(
            timestamp=base_time + timedelta(days=i * 3 + 1),
            symbol="AAPL",
            signal_type=SignalType.SELL,
            price=105.0 + i * 5,
            confidence=0.65 + i * 0.05,
            strategy_name="TestStrategy",
            metadata={"test": True, "signal_id": i + 5}
        ))
    
    return signals


@pytest.fixture(scope="session")
def sample_trades() -> List[Trade]:
    """Create sample trade data."""
    base_time = datetime.now() - timedelta(days=30)
    trades = []
    
    for i in range(10):
        entry_time = base_time + timedelta(days=i * 2)
        exit_time = entry_time + timedelta(days=1)
        entry_price = 100.0 + i * 2
        exit_price = entry_price + (-1 if i % 3 == 0 else 1) * (i + 1)
        
        trades.append(Trade(
            trade_id=f"test_trade_{i}",
            entry_time=entry_time,
            exit_time=exit_time,
            symbol="AAPL",
            quantity=100.0,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=(exit_price - entry_price) * 100.0,
            commission=0.0,  # Commission-free by default
            strategy_name="TestStrategy",
            signal_metadata={"test": True, "trade_id": i}
        ))
    
    return trades


@pytest.fixture
def sample_performance_metrics() -> PerformanceMetrics:
    """Create sample performance metrics."""
    return PerformanceMetrics(
        total_trades=10,
        winning_trades=6,
        losing_trades=4,
        win_rate=0.6,
        total_pnl=1500.0,
        total_return_pct=15.0,
        max_drawdown=500.0,
        max_drawdown_pct=5.0,
        sharpe_ratio=1.2,
        sortino_ratio=1.5,
        profit_factor=1.8,
        avg_win=350.0,
        avg_loss=-200.0,
        largest_win=800.0,
        largest_loss=-400.0,
        avg_trade_duration_hours=24.0,
        max_consecutive_wins=3,
        max_consecutive_losses=2
    )


@pytest.fixture
def sample_backtest_result(sample_market_data, sample_trades, sample_signals, sample_performance_metrics) -> BacktestResult:
    """Create sample backtest result."""
    initial_capital = 10000.0
    final_capital = 11500.0
    equity_curve = [initial_capital + i * 150 for i in range(len(sample_trades))]
    
    return BacktestResult(
        strategy_name="TestStrategy",
        symbol="AAPL",
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        initial_capital=initial_capital,
        final_capital=final_capital,
        trades=sample_trades,
        equity_curve=equity_curve,
        performance_metrics=sample_performance_metrics,
        signals=sample_signals
    )


@pytest.fixture
def backtest_config() -> BacktestConfig:
    """Create backtest configuration."""
    return BacktestConfig(
        initial_capital=10000.0,
        # commission_rate defaults to 0.0 for commission-free trading
        slippage_rate=0.001,
        max_positions=5,
        risk_per_trade=0.02
    )


@pytest.fixture
def backtest_engine(backtest_config) -> BacktestEngine:
    """Create backtest engine."""
    return BacktestEngine(backtest_config)


@pytest.fixture
def ma_crossover_strategy() -> MovingAverageCrossover:
    """Create MA crossover strategy."""
    return MovingAverageCrossover({
        'short_period': 20,
        'long_period': 50,
        'min_confidence': 0.6
    })


@pytest.fixture
def rsi_trend_strategy() -> RSITrendFollowing:
    """Create RSI trend strategy."""
    return RSITrendFollowing({
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'trend_period': 50
    })


@pytest.fixture
def bollinger_breakout_strategy() -> BollingerBandBreakout:
    """Create Bollinger Breakout strategy."""
    return BollingerBandBreakout({
        'period': 20,
        'std_dev': 2.0,
        'min_confidence': 0.6
    })


@pytest.fixture
def macd_rsi_strategy() -> MACDRSIStrategy:
    """Create MACD-RSI strategy."""
    return MACDRSIStrategy({
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'rsi_period': 14
    })


@pytest.fixture
def sr_breakout_strategy() -> SupportResistanceBreakout:
    """Create Support/Resistance breakout strategy."""
    return SupportResistanceBreakout({
        'lookback_period': 20,
        'min_touches': 2,
        'breakout_threshold': 0.02
    })


@pytest.fixture
def all_strategies(ma_crossover_strategy, rsi_trend_strategy, bollinger_breakout_strategy, 
                  macd_rsi_strategy, sr_breakout_strategy) -> List:
    """Get all strategy instances."""
    return [
        ma_crossover_strategy,
        rsi_trend_strategy,
        bollinger_breakout_strategy,
        macd_rsi_strategy,
        sr_breakout_strategy
    ]


@pytest.fixture
def mock_data_service() -> Mock:
    """Create mock data service."""
    mock_service = Mock(spec=DataService)
    mock_service.get_market_data.return_value = Mock()
    mock_service.get_symbols.return_value = TEST_SYMBOLS
    mock_service.is_market_open.return_value = True
    return mock_service


@pytest.fixture
def mock_backtest_service() -> Mock:
    """Create mock backtest service."""
    mock_service = Mock(spec=BacktestService)
    mock_service.run_backtest.return_value = Mock()
    mock_service.get_results.return_value = []
    return mock_service


@pytest.fixture
def mock_analysis_service() -> Mock:
    """Create mock analysis service."""
    mock_service = Mock(spec=AnalysisService)
    mock_service.analyze_strategy.return_value = Mock()
    mock_service.compare_strategies.return_value = Mock()
    return mock_service


@pytest.fixture
def temp_directory() -> Generator[str, None, None]:
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def test_data_path() -> Path:
    """Get path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def known_indicator_values() -> Dict[str, Any]:
    """Known indicator values for validation."""
    return {
        "sma": {
            "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "period": 3,
            "expected": [np.nan, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        },
        "ema": {
            "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "period": 3,
            "expected": [np.nan, np.nan, 2.0, 2.5, 3.25, 4.125, 5.0625, 6.03125, 7.015625, 8.0078125]
        },
        "rsi": {
            "data": [44, 44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.85, 46.08, 45.89,
                    46.03, 46.83, 47.69, 46.49, 46.26, 47.09, 47.37, 47.20, 46.21, 46.25],
            "period": 14,
            "expected_last": 51.78  # Approximate RSI for last value
        },
        "macd": {
            "data": [12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18,
                    18.5, 19, 19.5, 20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24, 24.5, 25],
            "fast": 12,
            "slow": 26,
            "signal": 9
        }
    }


@pytest.fixture
def market_scenarios(mock_data_generator) -> Dict[str, List[OHLCV]]:
    """Generate different market scenarios for testing."""
    scenarios = {}
    
    # Bull market
    scenarios["bull_market"] = mock_data_generator.generate_trending_data(
        periods=100, trend_strength=0.05, volatility=0.10, start_price=100.0
    )
    
    # Bear market
    scenarios["bear_market"] = mock_data_generator.generate_trending_data(
        periods=100, trend_strength=-0.05, volatility=0.15, start_price=100.0
    )
    
    # Sideways market
    scenarios["sideways_market"] = mock_data_generator.generate_ranging_data(
        periods=100, range_width=0.15, volatility=0.08, center_price=100.0
    )
    
    # High volatility
    scenarios["high_volatility"] = mock_data_generator.generate_trending_data(
        periods=100, trend_strength=0.02, volatility=0.30, start_price=100.0
    )
    
    # Low volatility
    scenarios["low_volatility"] = mock_data_generator.generate_trending_data(
        periods=100, trend_strength=0.01, volatility=0.05, start_price=100.0
    )
    
    return scenarios


@pytest.fixture
def performance_benchmarks() -> Dict[str, float]:
    """Performance benchmarks for testing."""
    return {
        "max_signal_generation_time": 1.0,  # seconds
        "max_backtest_time": 5.0,  # seconds
        "max_indicator_calculation_time": 0.5,  # seconds
        "max_memory_usage_mb": 100,  # MB
        "min_win_rate": 0.4,  # 40%
        "max_drawdown_pct": 20.0,  # 20%
        "min_profit_factor": 1.0
    }



@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset random state before each test."""
    np.random.seed(TEST_SEED)
    

@pytest.fixture
def mock_market_data_arrays():
    """Mock market data in array format for calculations."""
    np.random.seed(TEST_SEED)
    n_points = 100
    
    # Generate realistic price data
    prices = np.cumsum(np.random.normal(0, 1, n_points)) + 100
    
    return {
        'timestamp': [datetime.now() - timedelta(days=n_points-i) for i in range(n_points)],
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.02, n_points)),
        'low': prices * (1 - np.random.uniform(0, 0.02, n_points)),
        'close': prices + np.random.normal(0, 0.5, n_points),
        'volume': np.random.uniform(1000, 10000, n_points)
    }


# Performance monitoring fixtures
@pytest.fixture
def memory_monitor():
    """Memory usage monitor for performance tests."""
    import psutil
    import os
    
    class MemoryMonitor:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        def get_current_usage(self):
            return self.process.memory_info().rss / 1024 / 1024  # MB
        
        def get_peak_usage(self):
            return self.process.memory_info().peak_wss / 1024 / 1024 if hasattr(self.process.memory_info(), 'peak_wss') else self.get_current_usage()
    
    return MemoryMonitor()


@pytest.fixture
def execution_timer():
    """Execution time monitor for performance tests."""
    import time
    
    class ExecutionTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        def get_elapsed_time(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return 0
    
    return ExecutionTimer()


# Property-based testing fixtures
@pytest.fixture
def property_test_data():
    """Generate property-based test data."""
    from hypothesis import strategies as st
    
    return {
        'prices': st.lists(st.floats(min_value=1.0, max_value=1000.0), min_size=10, max_size=1000),
        'volumes': st.lists(st.floats(min_value=0.0, max_value=1000000.0), min_size=10, max_size=1000),
        'periods': st.integers(min_value=1, max_value=100),
        'percentages': st.floats(min_value=0.0, max_value=1.0),
        'confidence_scores': st.floats(min_value=0.0, max_value=1.0),
        'trend_strengths': st.floats(min_value=-0.1, max_value=0.1),
        'volatilities': st.floats(min_value=0.01, max_value=0.5)
    }


# Error simulation fixtures
@pytest.fixture
def error_scenarios():
    """Error scenarios for testing error handling."""
    return {
        'network_error': ConnectionError("Network connection failed"),
        'data_error': ValueError("Invalid data format"),
        'calculation_error': ZeroDivisionError("Division by zero"),
        'memory_error': MemoryError("Not enough memory"),
        'timeout_error': TimeoutError("Operation timed out")
    }


# Cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test."""
    yield
    # Add any cleanup code here if needed
    pass