"""
Backtesting engine for strategy evaluation
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from ..core.models import (
    Signal, Trade, BacktestResult, PerformanceMetrics, 
    MarketData, OHLCV, StrategyConfig
)
from ..strategies.base import TradingStrategy
from .execution import PortfolioManager, OrderExecutor, MarketOrderExecutor
from .costs import TransactionCostModel, CommissionFreeModel, RealisticCostModel

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest execution"""
    initial_capital: float = 10000.0
    commission_rate: float = 0.0  # Default to commission-free trading
    slippage_rate: float = 0.001
    max_positions: int = 5
    risk_per_trade: float = 0.02
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    benchmark_symbol: Optional[str] = None


class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, config: BacktestConfig = None):
        """
        Initialize backtest engine
        
        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()
        self.results_cache: Dict[str, BacktestResult] = {}
        
    def run_backtest(self, 
                    strategy: TradingStrategy,
                    data: MarketData,
                    initial_capital: float = None,
                    commission_rate: float = None,
                    slippage_rate: float = None) -> BacktestResult:
        """
        Run a complete backtest for a strategy
        
        Args:
            strategy: Trading strategy to test
            data: Market data
            initial_capital: Starting capital (overrides config)
            commission_rate: Commission rate (overrides config, default: 0.0 for commission-free)
            slippage_rate: Slippage rate (overrides config)
            
        Returns:
            BacktestResult with complete results
        """
        logger.info(f"Starting backtest for {strategy.name}")
        
        # Use provided parameters or defaults from config
        capital = initial_capital or self.config.initial_capital
        comm_rate = commission_rate if commission_rate is not None else self.config.commission_rate
        slip_rate = slippage_rate or self.config.slippage_rate
        
        # Create cost model - use CommissionFreeModel for true commission-free trading
        if comm_rate == 0.0:
            cost_model = CommissionFreeModel(slippage_rate=slip_rate)
        else:
            cost_model = RealisticCostModel(
                commission_rate=comm_rate,
                slippage_rate=slip_rate
            )
        
        # Create executor
        executor = MarketOrderExecutor(
            cost_model=cost_model,
            risk_per_trade=self.config.risk_per_trade
        )
        
        # Create portfolio manager
        portfolio = PortfolioManager(
            initial_capital=capital,
            executor=executor,
            max_positions=self.config.max_positions
        )
        
        # Generate signals
        signals = self._generate_signals(strategy, data)
        logger.info(f"Generated {len(signals)} signals")
        
        # Simulate execution
        executed_trades = self._simulate_execution(signals, data, portfolio)
        logger.info(f"Executed {len(executed_trades)} trades")
        
        # Calculate equity curve
        equity_curve = portfolio.equity_curve
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            executed_trades, equity_curve, capital
        )
        
        # Create result
        result = BacktestResult(
            strategy_name=strategy.name,
            symbol=data.symbol,
            start_date=data.data[0].timestamp if data.data else datetime.now(),
            end_date=data.data[-1].timestamp if data.data else datetime.now(),
            initial_capital=capital,
            final_capital=equity_curve[-1] if equity_curve else capital,
            trades=executed_trades,
            equity_curve=equity_curve,
            performance_metrics=performance_metrics,
            signals=signals
        )
        
        # Cache result
        cache_key = f"{strategy.name}_{data.symbol}_{capital}"
        self.results_cache[cache_key] = result
        
        logger.info(f"Backtest completed for {strategy.name}")
        return result
    
    def _generate_signals(self, strategy: TradingStrategy, data: MarketData) -> List[Signal]:
        """Generate trading signals from strategy"""
        signals = []
        
        # Calculate required indicators
        indicators = strategy._calculate_indicators(data)
        
        # Generate signals
        try:
            raw_signals = strategy.generate_signals(data, indicators)
            validated_signals = strategy.validate_signals(raw_signals)
            signals.extend(validated_signals)
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            
        return signals
    
    def _simulate_execution(self, 
                          signals: List[Signal],
                          data: MarketData,
                          portfolio: PortfolioManager) -> List[Trade]:
        """Simulate order execution"""
        executed_trades = []
        
        # Create price lookup for efficient access
        price_lookup = {bar.timestamp: bar for bar in data.data}
        
        for signal in signals:
            # Find corresponding market data
            market_bar = price_lookup.get(signal.timestamp)
            if market_bar is None:
                # Try to find closest bar
                market_bar = self._find_closest_bar(signal.timestamp, data.data)
            
            if market_bar is None:
                logger.warning(f"No market data found for signal at {signal.timestamp}")
                continue
            
            # Execute signal
            trade = portfolio.process_signal(signal, market_bar)
            if trade:
                executed_trades.append(trade)
        
        return executed_trades
    
    def _find_closest_bar(self, timestamp: datetime, bars: List[OHLCV]) -> Optional[OHLCV]:
        """Find the closest market bar to a given timestamp"""
        if not bars:
            return None
        
        closest_bar = None
        min_diff = float('inf')
        
        for bar in bars:
            diff = abs((bar.timestamp - timestamp).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_bar = bar
        
        return closest_bar
    
    def _calculate_performance_metrics(self, 
                                     trades: List[Trade],
                                     equity_curve: List[float],
                                     initial_capital: float) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        if not trades:
            return PerformanceMetrics(
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
        
        # Basic trade statistics
        closed_trades = [t for t in trades if t.exit_time is not None]
        total_trades = len(closed_trades)
        
        if total_trades == 0:
            return PerformanceMetrics(
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
        
        # Calculate P&L for each trade
        trade_pnls = []
        for trade in closed_trades:
            pnl = trade.calculate_pnl()
            trade_pnls.append(pnl)
        
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
        
        # Win/Loss statistics
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)
        win_rate = num_winning / total_trades if total_trades > 0 else 0.0
        
        # P&L statistics
        total_pnl = sum(trade_pnls)
        total_return_pct = (total_pnl / initial_capital) * 100 if initial_capital > 0 else 0.0
        
        avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0.0
        avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0.0
        
        largest_win = max(winning_trades) if winning_trades else 0.0
        largest_loss = min(losing_trades) if losing_trades else 0.0
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0.0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        
        # Drawdown calculation
        max_drawdown, max_drawdown_pct = self._calculate_drawdown(equity_curve)
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(equity_curve)
        sortino_ratio = self._calculate_sortino_ratio(equity_curve)
        
        # Trade duration
        durations = [t.duration_hours for t in closed_trades if t.duration_hours is not None]
        avg_trade_duration = sum(durations) / len(durations) if durations else 0.0
        
        # Consecutive wins/losses
        max_consecutive_wins, max_consecutive_losses = self._calculate_consecutive_stats(trade_pnls)
        
        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=num_winning,
            losing_trades=num_losing,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_duration_hours=avg_trade_duration,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses
        )
    
    def _calculate_drawdown(self, equity_curve: List[float]) -> tuple[float, float]:
        """Calculate maximum drawdown"""
        if len(equity_curve) < 2:
            return 0.0, 0.0
        
        max_drawdown = 0.0
        max_drawdown_pct = 0.0
        peak = equity_curve[0]
        
        for value in equity_curve[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = peak - value
                drawdown_pct = (drawdown / peak) * 100 if peak > 0 else 0.0
                
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    max_drawdown_pct = drawdown_pct
        
        return max_drawdown, max_drawdown_pct
    
    def _calculate_sharpe_ratio(self, equity_curve: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if len(equity_curve) < 2:
            return 0.0
        
        # Calculate returns
        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i-1] > 0:
                ret = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
                returns.append(ret)
        
        if len(returns) < 2:
            return 0.0
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate)
        import statistics
        avg_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize (assuming daily data)
        sharpe = (avg_return / std_return) * (252 ** 0.5)
        return sharpe
    
    def _calculate_sortino_ratio(self, equity_curve: List[float]) -> float:
        """Calculate Sortino ratio"""
        if len(equity_curve) < 2:
            return 0.0
        
        # Calculate returns
        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i-1] > 0:
                ret = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
                returns.append(ret)
        
        if len(returns) < 2:
            return 0.0
        
        # Calculate downside deviation
        import statistics
        avg_return = statistics.mean(returns)
        negative_returns = [r for r in returns if r < 0]
        
        if len(negative_returns) < 2:
            return float('inf') if avg_return > 0 else 0.0
        
        downside_deviation = statistics.stdev(negative_returns)
        
        if downside_deviation == 0:
            return 0.0
        
        # Annualize (assuming daily data)
        sortino = (avg_return / downside_deviation) * (252 ** 0.5)
        return sortino
    
    def _calculate_consecutive_stats(self, trade_pnls: List[float]) -> tuple[int, int]:
        """Calculate maximum consecutive wins and losses"""
        if not trade_pnls:
            return 0, 0
        
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_consecutive_wins = 0
        current_consecutive_losses = 0
        
        for pnl in trade_pnls:
            if pnl > 0:
                current_consecutive_wins += 1
                current_consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_consecutive_wins)
            elif pnl < 0:
                current_consecutive_losses += 1
                current_consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
            else:
                current_consecutive_wins = 0
                current_consecutive_losses = 0
        
        return max_consecutive_wins, max_consecutive_losses
    
    def calculate_equity_curve(self, trades: List[Trade], initial_capital: float) -> List[float]:
        """Calculate equity curve from trades"""
        equity_curve = [initial_capital]
        current_equity = initial_capital
        
        for trade in trades:
            if trade.exit_time is not None:  # Closed trade
                pnl = trade.calculate_pnl()
                current_equity += pnl
                equity_curve.append(current_equity)
        
        return equity_curve
    
    def generate_trade_log(self, trades: List[Trade]) -> List[Dict[str, Any]]:
        """Generate detailed trade log"""
        trade_log = []
        
        for trade in trades:
            log_entry = {
                'trade_id': trade.trade_id,
                'symbol': trade.symbol,
                'strategy': trade.strategy_name,
                'entry_time': trade.entry_time.isoformat(),
                'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                'quantity': trade.quantity,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'pnl': trade.calculate_pnl(),
                'commission': trade.commission,
                'duration_hours': trade.duration_hours,
                'metadata': trade.signal_metadata
            }
            trade_log.append(log_entry)
        
        return trade_log
    
    def run_parameter_optimization(self,
                                 strategy: TradingStrategy,
                                 data: MarketData,
                                 parameter_ranges: Dict[str, List[Any]],
                                 optimization_metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """
        Run parameter optimization for a strategy
        
        Args:
            strategy: Strategy to optimize
            data: Market data
            parameter_ranges: Dictionary of parameter names and their ranges
            optimization_metric: Metric to optimize
            
        Returns:
            Dictionary with best parameters and results
        """
        logger.info(f"Starting parameter optimization for {strategy.name}")
        
        best_params = None
        best_score = float('-inf')
        best_result = None
        
        # Generate parameter combinations
        import itertools
        keys = list(parameter_ranges.keys())
        values = list(parameter_ranges.values())
        
        for combination in itertools.product(*values):
            params = dict(zip(keys, combination))
            
            # Update strategy parameters
            original_params = strategy.parameters.copy()
            strategy.parameters.update(params)
            
            try:
                # Run backtest
                result = self.run_backtest(strategy, data)
                
                # Extract optimization metric
                if optimization_metric == 'sharpe_ratio':
                    score = result.performance_metrics.sharpe_ratio
                elif optimization_metric == 'total_return':
                    score = result.performance_metrics.total_return_pct
                elif optimization_metric == 'profit_factor':
                    score = result.performance_metrics.profit_factor
                elif optimization_metric == 'win_rate':
                    score = result.performance_metrics.win_rate
                else:
                    score = result.performance_metrics.sharpe_ratio
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_result = result
                
            except Exception as e:
                logger.error(f"Error in optimization iteration {params}: {e}")
                continue
            finally:
                # Restore original parameters
                strategy.parameters = original_params
        
        logger.info(f"Optimization completed. Best parameters: {best_params}")
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'best_result': best_result,
            'optimization_metric': optimization_metric
        }