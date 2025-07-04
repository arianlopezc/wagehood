"""
Unit tests for core data models.

Tests all core models including OHLCV, Signal, Trade, MarketData, and related classes.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List

from src.core.models import (
    OHLCV, Signal, Trade, SignalType, MarketData, TimeFrame,
    PerformanceMetrics, BacktestResult, StrategyConfig, StrategyStatus
)
from src.core.exceptions import ValidationError


class TestOHLCV:
    """Test OHLCV data model."""
    
    def test_valid_ohlcv_creation(self):
        """Test creation of valid OHLCV data."""
        timestamp = datetime.now()
        ohlcv = OHLCV(
            timestamp=timestamp,
            open=100.0,
            high=105.0,
            low=98.0,
            close=103.0,
            volume=10000.0
        )
        
        assert ohlcv.timestamp == timestamp
        assert ohlcv.open == 100.0
        assert ohlcv.high == 105.0
        assert ohlcv.low == 98.0
        assert ohlcv.close == 103.0
        assert ohlcv.volume == 10000.0
    
    def test_ohlcv_validation_high_too_low(self):
        """Test OHLCV validation when high is too low."""
        with pytest.raises(ValueError, match="High must be >= max"):
            OHLCV(
                timestamp=datetime.now(),
                open=100.0,
                high=99.0,  # High less than open
                low=98.0,
                close=103.0,
                volume=10000.0
            )
    
    def test_ohlcv_validation_low_too_high(self):
        """Test OHLCV validation when low is too high."""
        with pytest.raises(ValueError, match="Low must be <= min"):
            OHLCV(
                timestamp=datetime.now(),
                open=100.0,
                high=105.0,
                low=101.0,  # Low greater than open
                close=103.0,
                volume=10000.0
            )
    
    def test_ohlcv_validation_negative_volume(self):
        """Test OHLCV validation with negative volume."""
        with pytest.raises(ValueError, match="Volume cannot be negative"):
            OHLCV(
                timestamp=datetime.now(),
                open=100.0,
                high=105.0,
                low=98.0,
                close=103.0,
                volume=-1000.0
            )
    
    def test_ohlcv_edge_cases(self):
        """Test OHLCV edge cases."""
        # All prices equal (doji)
        ohlcv = OHLCV(
            timestamp=datetime.now(),
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=0.0
        )
        assert ohlcv.open == ohlcv.high == ohlcv.low == ohlcv.close
        
        # Zero volume
        assert ohlcv.volume == 0.0
    
    @pytest.mark.parametrize("open_price,high_price,low_price,close_price,volume", [
        (100.0, 105.0, 95.0, 102.0, 10000.0),
        (50.0, 50.0, 50.0, 50.0, 0.0),
        (0.001, 0.002, 0.0005, 0.0015, 1000000.0),
        (1000.0, 1050.0, 950.0, 1025.0, 5000.0),
    ])
    def test_ohlcv_parametrized_valid_data(self, open_price, high_price, low_price, close_price, volume):
        """Test OHLCV with various valid parameter combinations."""
        ohlcv = OHLCV(
            timestamp=datetime.now(),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume
        )
        
        assert ohlcv.open == open_price
        assert ohlcv.high == high_price
        assert ohlcv.low == low_price
        assert ohlcv.close == close_price
        assert ohlcv.volume == volume


class TestSignal:
    """Test Signal data model."""
    
    def test_valid_signal_creation(self):
        """Test creation of valid signal."""
        timestamp = datetime.now()
        signal = Signal(
            timestamp=timestamp,
            symbol="AAPL",
            signal_type=SignalType.BUY,
            price=150.0,
            confidence=0.85,
            strategy_name="TestStrategy",
            metadata={"test": True}
        )
        
        assert signal.timestamp == timestamp
        assert signal.symbol == "AAPL"
        assert signal.signal_type == SignalType.BUY
        assert signal.price == 150.0
        assert signal.confidence == 0.85
        assert signal.strategy_name == "TestStrategy"
        assert signal.metadata == {"test": True}
    
    def test_signal_validation_confidence_too_low(self):
        """Test signal validation with confidence too low."""
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                signal_type=SignalType.BUY,
                price=150.0,
                confidence=-0.1,
                strategy_name="TestStrategy",
                metadata={}
            )
    
    def test_signal_validation_confidence_too_high(self):
        """Test signal validation with confidence too high."""
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                signal_type=SignalType.BUY,
                price=150.0,
                confidence=1.1,
                strategy_name="TestStrategy",
                metadata={}
            )
    
    def test_signal_validation_negative_price(self):
        """Test signal validation with negative price."""
        with pytest.raises(ValueError, match="Price must be positive"):
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                signal_type=SignalType.BUY,
                price=-150.0,
                confidence=0.85,
                strategy_name="TestStrategy",
                metadata={}
            )
    
    def test_signal_validation_zero_price(self):
        """Test signal validation with zero price."""
        with pytest.raises(ValueError, match="Price must be positive"):
            Signal(
                timestamp=datetime.now(),
                symbol="AAPL",
                signal_type=SignalType.BUY,
                price=0.0,
                confidence=0.85,
                strategy_name="TestStrategy",
                metadata={}
            )
    
    @pytest.mark.parametrize("signal_type", [
        SignalType.BUY,
        SignalType.SELL,
        SignalType.HOLD,
        SignalType.CLOSE_LONG,
        SignalType.CLOSE_SHORT
    ])
    def test_signal_types(self, signal_type):
        """Test all signal types."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=signal_type,
            price=150.0,
            confidence=0.85,
            strategy_name="TestStrategy",
            metadata={}
        )
        
        assert signal.signal_type == signal_type
    
    def test_signal_edge_cases(self):
        """Test signal edge cases."""
        # Minimum confidence
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.BUY,
            price=150.0,
            confidence=0.0,
            strategy_name="TestStrategy",
            metadata={}
        )
        assert signal.confidence == 0.0
        
        # Maximum confidence
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.BUY,
            price=150.0,
            confidence=1.0,
            strategy_name="TestStrategy",
            metadata={}
        )
        assert signal.confidence == 1.0
        
        # Very small price
        signal = Signal(
            timestamp=datetime.now(),
            symbol="AAPL",
            signal_type=SignalType.BUY,
            price=0.0001,
            confidence=0.5,
            strategy_name="TestStrategy",
            metadata={}
        )
        assert signal.price == 0.0001


class TestTrade:
    """Test Trade data model."""
    
    def test_valid_trade_creation(self):
        """Test creation of valid trade."""
        entry_time = datetime.now()
        exit_time = entry_time + timedelta(hours=2)
        
        trade = Trade(
            trade_id="test_trade_1",
            entry_time=entry_time,
            exit_time=exit_time,
            symbol="AAPL",
            quantity=100.0,
            entry_price=150.0,
            exit_price=155.0,
            pnl=500.0,
            commission=2.0,
            strategy_name="TestStrategy",
            signal_metadata={"test": True}
        )
        
        assert trade.trade_id == "test_trade_1"
        assert trade.entry_time == entry_time
        assert trade.exit_time == exit_time
        assert trade.symbol == "AAPL"
        assert trade.quantity == 100.0
        assert trade.entry_price == 150.0
        assert trade.exit_price == 155.0
        assert trade.pnl == 500.0
        assert trade.commission == 2.0
        assert trade.strategy_name == "TestStrategy"
        assert trade.signal_metadata == {"test": True}
    
    def test_open_trade_creation(self):
        """Test creation of open trade."""
        entry_time = datetime.now()
        
        trade = Trade(
            trade_id="test_trade_open",
            entry_time=entry_time,
            exit_time=None,
            symbol="AAPL",
            quantity=100.0,
            entry_price=150.0,
            exit_price=None,
            pnl=None,
            commission=1.0,
            strategy_name="TestStrategy",
            signal_metadata={}
        )
        
        assert trade.is_open
        assert trade.exit_time is None
        assert trade.exit_price is None
        assert trade.pnl is None
    
    def test_trade_is_open_property(self):
        """Test is_open property."""
        # Open trade
        trade = Trade(
            trade_id="test_open",
            entry_time=datetime.now(),
            exit_time=None,
            symbol="AAPL",
            quantity=100.0,
            entry_price=150.0,
            exit_price=None,
            pnl=None,
            commission=1.0,
            strategy_name="TestStrategy",
            signal_metadata={}
        )
        assert trade.is_open
        
        # Closed trade
        trade.exit_time = datetime.now()
        assert not trade.is_open
    
    def test_trade_duration_calculation(self):
        """Test trade duration calculation."""
        entry_time = datetime.now()
        exit_time = entry_time + timedelta(hours=3, minutes=30)
        
        trade = Trade(
            trade_id="test_duration",
            entry_time=entry_time,
            exit_time=exit_time,
            symbol="AAPL",
            quantity=100.0,
            entry_price=150.0,
            exit_price=155.0,
            pnl=500.0,
            commission=2.0,
            strategy_name="TestStrategy",
            signal_metadata={}
        )
        
        # Should be 3.5 hours
        assert trade.duration_hours == 3.5
    
    def test_trade_duration_open_trade(self):
        """Test duration calculation for open trade."""
        trade = Trade(
            trade_id="test_open_duration",
            entry_time=datetime.now(),
            exit_time=None,
            symbol="AAPL",
            quantity=100.0,
            entry_price=150.0,
            exit_price=None,
            pnl=None,
            commission=1.0,
            strategy_name="TestStrategy",
            signal_metadata={}
        )
        
        assert trade.duration_hours is None
    
    def test_trade_pnl_calculation_closed(self):
        """Test P&L calculation for closed trade."""
        trade = Trade(
            trade_id="test_pnl_closed",
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            symbol="AAPL",
            quantity=100.0,
            entry_price=150.0,
            exit_price=155.0,
            pnl=498.0,  # Already calculated
            commission=2.0,
            strategy_name="TestStrategy",
            signal_metadata={}
        )
        
        # Should calculate: (155 - 150) * 100 - 2 = 498
        calculated_pnl = trade.calculate_pnl()
        assert calculated_pnl == 498.0
    
    def test_trade_pnl_calculation_open_with_current_price(self):
        """Test P&L calculation for open trade with current price."""
        trade = Trade(
            trade_id="test_pnl_open",
            entry_time=datetime.now(),
            exit_time=None,
            symbol="AAPL",
            quantity=100.0,
            entry_price=150.0,
            exit_price=None,
            pnl=None,
            commission=2.0,
            strategy_name="TestStrategy",
            signal_metadata={}
        )
        
        # Should calculate: (152 - 150) * 100 = 200 (no commission for open trades)
        calculated_pnl = trade.calculate_pnl(current_price=152.0)
        assert calculated_pnl == 200.0
    
    def test_trade_pnl_calculation_open_without_current_price(self):
        """Test P&L calculation for open trade without current price."""
        trade = Trade(
            trade_id="test_pnl_open_no_price",
            entry_time=datetime.now(),
            exit_time=None,
            symbol="AAPL",
            quantity=100.0,
            entry_price=150.0,
            exit_price=None,
            pnl=None,
            commission=2.0,
            strategy_name="TestStrategy",
            signal_metadata={}
        )
        
        # Should return 0.0 when no current price provided
        calculated_pnl = trade.calculate_pnl()
        assert calculated_pnl == 0.0
    
    def test_trade_negative_pnl(self):
        """Test trade with negative P&L."""
        trade = Trade(
            trade_id="test_negative_pnl",
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            symbol="AAPL",
            quantity=100.0,
            entry_price=150.0,
            exit_price=145.0,
            pnl=-502.0,
            commission=2.0,
            strategy_name="TestStrategy",
            signal_metadata={}
        )
        
        # Should calculate: (145 - 150) * 100 - 2 = -502
        calculated_pnl = trade.calculate_pnl()
        assert calculated_pnl == -502.0


class TestMarketData:
    """Test MarketData model."""
    
    def test_market_data_creation(self, sample_ohlcv_data):
        """Test creation of market data."""
        market_data = MarketData(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            data=sample_ohlcv_data,
            indicators={},
            last_updated=datetime.now()
        )
        
        assert market_data.symbol == "AAPL"
        assert market_data.timeframe == TimeFrame.DAILY
        assert len(market_data.data) == len(sample_ohlcv_data)
        assert isinstance(market_data.indicators, dict)
        assert isinstance(market_data.last_updated, datetime)
    
    def test_market_data_to_arrays_with_data(self, sample_ohlcv_data):
        """Test to_arrays method with data."""
        market_data = MarketData(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            data=sample_ohlcv_data,
            indicators={},
            last_updated=datetime.now()
        )
        
        arrays = market_data.to_arrays()
        
        assert 'timestamp' in arrays
        assert 'open' in arrays
        assert 'high' in arrays
        assert 'low' in arrays
        assert 'close' in arrays
        assert 'volume' in arrays
        
        assert len(arrays['timestamp']) == len(sample_ohlcv_data)
        assert len(arrays['open']) == len(sample_ohlcv_data)
        assert len(arrays['high']) == len(sample_ohlcv_data)
        assert len(arrays['low']) == len(sample_ohlcv_data)
        assert len(arrays['close']) == len(sample_ohlcv_data)
        assert len(arrays['volume']) == len(sample_ohlcv_data)
    
    def test_market_data_to_arrays_empty(self):
        """Test to_arrays method with empty data."""
        market_data = MarketData(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            data=[],
            indicators={},
            last_updated=datetime.now()
        )
        
        arrays = market_data.to_arrays()
        
        assert 'timestamp' in arrays
        assert 'open' in arrays
        assert 'high' in arrays
        assert 'low' in arrays
        assert 'close' in arrays
        assert 'volume' in arrays
        
        # All arrays should be empty
        for key in arrays:
            assert len(arrays[key]) == 0
    
    def test_market_data_with_indicators(self, sample_ohlcv_data):
        """Test market data with indicators."""
        indicators = {
            'sma_20': [100, 101, 102],
            'rsi': [50, 55, 45],
            'macd': {'line': [0.1, 0.2, 0.3], 'signal': [0.05, 0.15, 0.25]}
        }
        
        market_data = MarketData(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            data=sample_ohlcv_data,
            indicators=indicators,
            last_updated=datetime.now()
        )
        
        assert market_data.indicators == indicators
        assert 'sma_20' in market_data.indicators
        assert 'rsi' in market_data.indicators
        assert 'macd' in market_data.indicators
    
    @pytest.mark.parametrize("timeframe", [
        TimeFrame.MINUTE_1,
        TimeFrame.MINUTE_5,
        TimeFrame.MINUTE_15,
        TimeFrame.MINUTE_30,
        TimeFrame.HOUR_1,
        TimeFrame.HOUR_4,
        TimeFrame.DAILY,
        TimeFrame.WEEKLY,
        TimeFrame.MONTHLY
    ])
    def test_market_data_timeframes(self, timeframe, sample_ohlcv_data):
        """Test market data with different timeframes."""
        market_data = MarketData(
            symbol="AAPL",
            timeframe=timeframe,
            data=sample_ohlcv_data,
            indicators={},
            last_updated=datetime.now()
        )
        
        assert market_data.timeframe == timeframe


class TestPerformanceMetrics:
    """Test PerformanceMetrics model."""
    
    def test_performance_metrics_creation(self):
        """Test creation of performance metrics."""
        metrics = PerformanceMetrics(
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            win_rate=0.6,
            total_pnl=5000.0,
            total_return_pct=50.0,
            max_drawdown=1000.0,
            max_drawdown_pct=10.0,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            profit_factor=2.5,
            avg_win=150.0,
            avg_loss=-100.0,
            largest_win=500.0,
            largest_loss=-300.0,
            avg_trade_duration_hours=24.0,
            max_consecutive_wins=5,
            max_consecutive_losses=3
        )
        
        assert metrics.total_trades == 100
        assert metrics.winning_trades == 60
        assert metrics.losing_trades == 40
        assert metrics.win_rate == 0.6
        assert metrics.total_pnl == 5000.0
        assert metrics.total_return_pct == 50.0
        assert metrics.max_drawdown == 1000.0
        assert metrics.max_drawdown_pct == 10.0
        assert metrics.sharpe_ratio == 1.5
        assert metrics.sortino_ratio == 2.0
        assert metrics.profit_factor == 2.5
        assert metrics.avg_win == 150.0
        assert metrics.avg_loss == -100.0
        assert metrics.largest_win == 500.0
        assert metrics.largest_loss == -300.0
        assert metrics.avg_trade_duration_hours == 24.0
        assert metrics.max_consecutive_wins == 5
        assert metrics.max_consecutive_losses == 3
    
    def test_performance_metrics_validation(self):
        """Test performance metrics validation."""
        # Test with realistic values
        metrics = PerformanceMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            total_return_pct=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            avg_trade_duration_hours=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0
        )
        
        # Should not raise any errors
        assert metrics.total_trades == 0


class TestBacktestResult:
    """Test BacktestResult model."""
    
    def test_backtest_result_creation(self, sample_market_data, sample_trades, 
                                    sample_signals, sample_performance_metrics):
        """Test creation of backtest result."""
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        initial_capital = 10000.0
        final_capital = 15000.0
        equity_curve = [10000.0, 11000.0, 12000.0, 13000.0, 14000.0, 15000.0]
        
        result = BacktestResult(
            strategy_name="TestStrategy",
            symbol="AAPL",
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=final_capital,
            trades=sample_trades,
            equity_curve=equity_curve,
            performance_metrics=sample_performance_metrics,
            signals=sample_signals
        )
        
        assert result.strategy_name == "TestStrategy"
        assert result.symbol == "AAPL"
        assert result.start_date == start_date
        assert result.end_date == end_date
        assert result.initial_capital == initial_capital
        assert result.final_capital == final_capital
        assert result.trades == sample_trades
        assert result.equity_curve == equity_curve
        assert result.performance_metrics == sample_performance_metrics
        assert result.signals == sample_signals
    
    def test_backtest_result_empty_trades(self, sample_performance_metrics):
        """Test backtest result with empty trades."""
        result = BacktestResult(
            strategy_name="TestStrategy",
            symbol="AAPL",
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            initial_capital=10000.0,
            final_capital=10000.0,
            trades=[],
            equity_curve=[10000.0],
            performance_metrics=sample_performance_metrics,
            signals=[]
        )
        
        assert len(result.trades) == 0
        assert len(result.signals) == 0
        assert len(result.equity_curve) == 1


class TestStrategyConfig:
    """Test StrategyConfig model."""
    
    def test_strategy_config_creation(self):
        """Test creation of strategy config."""
        created_at = datetime.now()
        updated_at = created_at + timedelta(minutes=10)
        
        config = StrategyConfig(
            strategy_name="TestStrategy",
            parameters={"period": 20, "threshold": 0.5},
            symbols=["AAPL", "GOOGL"],
            timeframes=[TimeFrame.DAILY, TimeFrame.HOUR_1],
            max_positions=5,
            risk_per_trade=0.02,
            status=StrategyStatus.ACTIVE,
            created_at=created_at,
            updated_at=updated_at
        )
        
        assert config.strategy_name == "TestStrategy"
        assert config.parameters == {"period": 20, "threshold": 0.5}
        assert config.symbols == ["AAPL", "GOOGL"]
        assert config.timeframes == [TimeFrame.DAILY, TimeFrame.HOUR_1]
        assert config.max_positions == 5
        assert config.risk_per_trade == 0.02
        assert config.status == StrategyStatus.ACTIVE
        assert config.created_at == created_at
        assert config.updated_at == updated_at
    
    @pytest.mark.parametrize("status", [
        StrategyStatus.ACTIVE,
        StrategyStatus.PAUSED,
        StrategyStatus.STOPPED,
        StrategyStatus.ERROR
    ])
    def test_strategy_status_values(self, status):
        """Test all strategy status values."""
        config = StrategyConfig(
            strategy_name="TestStrategy",
            parameters={},
            symbols=["AAPL"],
            timeframes=[TimeFrame.DAILY],
            max_positions=5,
            risk_per_trade=0.02,
            status=status,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        assert config.status == status


class TestTimeFrame:
    """Test TimeFrame enum."""
    
    def test_timeframe_values(self):
        """Test all timeframe values."""
        assert TimeFrame.MINUTE_1.value == "1m"
        assert TimeFrame.MINUTE_5.value == "5m"
        assert TimeFrame.MINUTE_15.value == "15m"
        assert TimeFrame.MINUTE_30.value == "30m"
        assert TimeFrame.HOUR_1.value == "1h"
        assert TimeFrame.HOUR_4.value == "4h"
        assert TimeFrame.DAILY.value == "1d"
        assert TimeFrame.WEEKLY.value == "1w"
        assert TimeFrame.MONTHLY.value == "1M"
    
    def test_timeframe_comparison(self):
        """Test timeframe comparison."""
        assert TimeFrame.MINUTE_1 != TimeFrame.MINUTE_5
        assert TimeFrame.DAILY == TimeFrame.DAILY
        assert TimeFrame.HOUR_1 != TimeFrame.DAILY


class TestSignalType:
    """Test SignalType enum."""
    
    def test_signal_type_values(self):
        """Test all signal type values."""
        assert SignalType.BUY.value == "buy"
        assert SignalType.SELL.value == "sell"
        assert SignalType.HOLD.value == "hold"
        assert SignalType.CLOSE_LONG.value == "close_long"
        assert SignalType.CLOSE_SHORT.value == "close_short"
    
    def test_signal_type_comparison(self):
        """Test signal type comparison."""
        assert SignalType.BUY != SignalType.SELL
        assert SignalType.BUY == SignalType.BUY
        assert SignalType.HOLD != SignalType.CLOSE_LONG