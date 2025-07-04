"""
Performance evaluator for comprehensive strategy analysis
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import statistics
import math

from ..core.models import Trade, BacktestResult, PerformanceMetrics


class PerformanceEvaluator:
    """Comprehensive performance evaluation for trading strategies"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance evaluator
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_all_metrics(self, 
                            trades: List[Trade],
                            equity_curve: List[float],
                            initial_capital: float) -> PerformanceMetrics:
        """
        Calculate all performance metrics
        
        Args:
            trades: List of executed trades
            equity_curve: Equity curve values
            initial_capital: Initial capital
            
        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        if not trades:
            return self._empty_metrics()
        
        # Filter closed trades
        closed_trades = [t for t in trades if t.exit_time is not None and t.pnl is not None]
        
        if not closed_trades:
            return self._empty_metrics()
        
        # Calculate basic metrics
        basic_metrics = self._calculate_basic_metrics(closed_trades, initial_capital)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(equity_curve, initial_capital)
        
        # Calculate advanced metrics
        advanced_metrics = self._calculate_advanced_metrics(closed_trades, equity_curve)
        
        # Combine all metrics
        return PerformanceMetrics(
            **basic_metrics,
            **risk_metrics,
            **advanced_metrics
        )
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty performance metrics"""
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
    
    def _calculate_basic_metrics(self, trades: List[Trade], initial_capital: float) -> Dict[str, Any]:
        """Calculate basic performance metrics"""
        total_trades = len(trades)
        
        # Calculate P&L for each trade
        trade_pnls = [trade.pnl for trade in trades if trade.pnl is not None]
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
        
        # Basic statistics
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)
        win_rate = num_winning / total_trades if total_trades > 0 else 0.0
        
        # P&L statistics
        total_pnl = sum(trade_pnls)
        total_return_pct = (total_pnl / initial_capital) * 100 if initial_capital > 0 else 0.0
        
        # Win/Loss statistics
        avg_win = statistics.mean(winning_trades) if winning_trades else 0.0
        avg_loss = statistics.mean(losing_trades) if losing_trades else 0.0
        largest_win = max(winning_trades) if winning_trades else 0.0
        largest_loss = min(losing_trades) if losing_trades else 0.0
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0.0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        
        # Trade duration
        durations = [t.duration_hours for t in trades if t.duration_hours is not None]
        avg_trade_duration = statistics.mean(durations) if durations else 0.0
        
        return {
            'total_trades': total_trades,
            'winning_trades': num_winning,
            'losing_trades': num_losing,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_trade_duration_hours': avg_trade_duration
        }
    
    def _calculate_risk_metrics(self, equity_curve: List[float], initial_capital: float) -> Dict[str, Any]:
        """Calculate risk-adjusted metrics"""
        if len(equity_curve) < 2:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0
            }
        
        # Drawdown calculation
        max_drawdown, max_drawdown_pct = self._calculate_drawdown(equity_curve)
        
        # Risk ratios
        sharpe_ratio = self._calculate_sharpe_ratio(equity_curve)
        sortino_ratio = self._calculate_sortino_ratio(equity_curve)
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio
        }
    
    def _calculate_advanced_metrics(self, trades: List[Trade], equity_curve: List[float]) -> Dict[str, Any]:
        """Calculate advanced performance metrics"""
        # Consecutive wins/losses
        trade_pnls = [trade.pnl for trade in trades if trade.pnl is not None]
        max_consecutive_wins, max_consecutive_losses = self._calculate_consecutive_stats(trade_pnls)
        
        return {
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses
        }
    
    def _calculate_drawdown(self, equity_curve: List[float]) -> Tuple[float, float]:
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
        
        # Calculate Sharpe ratio
        avg_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)
        
        if std_return == 0:
            return 0.0
        
        # Adjust for risk-free rate (assume daily returns)
        risk_free_daily = (1 + self.risk_free_rate) ** (1/252) - 1
        excess_return = avg_return - risk_free_daily
        
        # Annualize
        sharpe = (excess_return / std_return) * (252 ** 0.5)
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
        avg_return = statistics.mean(returns)
        risk_free_daily = (1 + self.risk_free_rate) ** (1/252) - 1
        
        downside_returns = [r - risk_free_daily for r in returns if r < risk_free_daily]
        
        if len(downside_returns) < 2:
            return float('inf') if avg_return > risk_free_daily else 0.0
        
        downside_deviation = statistics.stdev(downside_returns)
        
        if downside_deviation == 0:
            return 0.0
        
        # Annualize
        excess_return = avg_return - risk_free_daily
        sortino = (excess_return / downside_deviation) * (252 ** 0.5)
        return sortino
    
    def _calculate_consecutive_stats(self, trade_pnls: List[float]) -> Tuple[int, int]:
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
    
    def calculate_calmar_ratio(self, equity_curve: List[float]) -> float:
        """Calculate Calmar ratio (Annual return / Maximum drawdown)"""
        if len(equity_curve) < 2:
            return 0.0
        
        # Calculate annual return
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        days = len(equity_curve) - 1
        annual_return = ((1 + total_return) ** (252 / days)) - 1 if days > 0 else 0.0
        
        # Calculate maximum drawdown
        _, max_drawdown_pct = self._calculate_drawdown(equity_curve)
        
        if max_drawdown_pct == 0:
            return float('inf') if annual_return > 0 else 0.0
        
        return annual_return / (max_drawdown_pct / 100)
    
    def calculate_var(self, equity_curve: List[float], confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk (VaR)"""
        if len(equity_curve) < 2:
            return 0.0
        
        # Calculate returns
        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i-1] > 0:
                ret = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
                returns.append(ret)
        
        if not returns:
            return 0.0
        
        # Calculate VaR
        returns.sort()
        var_index = int(len(returns) * confidence_level)
        return abs(returns[var_index]) if var_index < len(returns) else 0.0
    
    def calculate_expected_shortfall(self, equity_curve: List[float], confidence_level: float = 0.05) -> float:
        """Calculate Expected Shortfall (CVaR)"""
        if len(equity_curve) < 2:
            return 0.0
        
        # Calculate returns
        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i-1] > 0:
                ret = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
                returns.append(ret)
        
        if not returns:
            return 0.0
        
        # Calculate Expected Shortfall
        returns.sort()
        var_index = int(len(returns) * confidence_level)
        tail_returns = returns[:var_index]
        
        return abs(statistics.mean(tail_returns)) if tail_returns else 0.0
    
    def calculate_information_ratio(self, 
                                  strategy_returns: List[float],
                                  benchmark_returns: List[float]) -> float:
        """Calculate Information Ratio"""
        if len(strategy_returns) != len(benchmark_returns) or len(strategy_returns) < 2:
            return 0.0
        
        # Calculate excess returns
        excess_returns = [s - b for s, b in zip(strategy_returns, benchmark_returns)]
        
        if len(excess_returns) < 2:
            return 0.0
        
        # Calculate Information Ratio
        avg_excess_return = statistics.mean(excess_returns)
        tracking_error = statistics.stdev(excess_returns)
        
        if tracking_error == 0:
            return 0.0
        
        return avg_excess_return / tracking_error
    
    def calculate_omega_ratio(self, equity_curve: List[float], threshold: float = 0.0) -> float:
        """Calculate Omega ratio"""
        if len(equity_curve) < 2:
            return 0.0
        
        # Calculate returns
        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i-1] > 0:
                ret = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
                returns.append(ret)
        
        if not returns:
            return 0.0
        
        # Calculate Omega ratio
        gains = sum(max(0, r - threshold) for r in returns)
        losses = sum(max(0, threshold - r) for r in returns)
        
        if losses == 0:
            return float('inf') if gains > 0 else 0.0
        
        return gains / losses
    
    def calculate_trade_metrics(self, trades: List[Trade]) -> Dict[str, Any]:
        """Calculate detailed trade-level metrics"""
        if not trades:
            return {}
        
        closed_trades = [t for t in trades if t.exit_time is not None and t.pnl is not None]
        
        if not closed_trades:
            return {}
        
        # Trade timing analysis
        entry_times = [t.entry_time for t in closed_trades]
        exit_times = [t.exit_time for t in closed_trades]
        
        # Duration analysis
        durations = [t.duration_hours for t in closed_trades if t.duration_hours is not None]
        
        # Size analysis
        sizes = [abs(t.quantity) for t in closed_trades]
        
        # P&L analysis
        pnls = [t.pnl for t in closed_trades]
        
        return {
            'avg_duration_hours': statistics.mean(durations) if durations else 0.0,
            'median_duration_hours': statistics.median(durations) if durations else 0.0,
            'min_duration_hours': min(durations) if durations else 0.0,
            'max_duration_hours': max(durations) if durations else 0.0,
            'avg_trade_size': statistics.mean(sizes) if sizes else 0.0,
            'median_trade_size': statistics.median(sizes) if sizes else 0.0,
            'avg_pnl': statistics.mean(pnls) if pnls else 0.0,
            'median_pnl': statistics.median(pnls) if pnls else 0.0,
            'pnl_std': statistics.stdev(pnls) if len(pnls) > 1 else 0.0,
            'total_commission': sum(t.commission for t in closed_trades),
            'avg_commission': statistics.mean([t.commission for t in closed_trades])
        }
    
    def compare_strategies(self, backtest_results: List[BacktestResult]) -> Dict[str, Any]:
        """Compare multiple strategy results"""
        if not backtest_results:
            return {}
        
        comparison = {}
        
        # Extract metrics for each strategy
        for result in backtest_results:
            strategy_name = result.strategy_name
            metrics = result.performance_metrics
            
            comparison[strategy_name] = {
                'total_return_pct': metrics.total_return_pct,
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'max_drawdown_pct': metrics.max_drawdown_pct,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'total_trades': metrics.total_trades,
                'calmar_ratio': self.calculate_calmar_ratio(result.equity_curve)
            }
        
        # Calculate rankings
        rankings = self._calculate_rankings(comparison)
        
        return {
            'individual_metrics': comparison,
            'rankings': rankings,
            'best_strategy': self._find_best_strategy(comparison)
        }
    
    def _calculate_rankings(self, comparison: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, int]]:
        """Calculate rankings for each metric"""
        if not comparison:
            return {}
        
        metrics = list(next(iter(comparison.values())).keys())
        rankings = {}
        
        for metric in metrics:
            # Get values for this metric
            values = [(name, data[metric]) for name, data in comparison.items()]
            
            # Sort by value (descending for most metrics, ascending for drawdown)
            reverse = metric != 'max_drawdown_pct'
            values.sort(key=lambda x: x[1], reverse=reverse)
            
            # Assign rankings
            for i, (name, value) in enumerate(values):
                if name not in rankings:
                    rankings[name] = {}
                rankings[name][metric] = i + 1
        
        return rankings
    
    def _find_best_strategy(self, comparison: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Find the best strategy based on composite score"""
        if not comparison:
            return {}
        
        # Calculate composite scores
        scores = {}
        
        for name, metrics in comparison.items():
            # Weight different metrics
            score = (
                metrics['sharpe_ratio'] * 0.3 +
                metrics['total_return_pct'] * 0.2 +
                metrics['sortino_ratio'] * 0.2 +
                (100 - metrics['max_drawdown_pct']) * 0.15 +  # Invert drawdown
                metrics['win_rate'] * 0.1 +
                min(metrics['profit_factor'], 5) * 0.05  # Cap profit factor
            )
            scores[name] = score
        
        # Find best strategy
        best_strategy = max(scores, key=scores.get)
        
        return {
            'name': best_strategy,
            'composite_score': scores[best_strategy],
            'all_scores': scores
        }
    
    def rank_strategies_by_metric(self, 
                                backtest_results: List[BacktestResult],
                                metric: str = 'sharpe_ratio') -> List[Tuple[str, float]]:
        """Rank strategies by a specific metric"""
        if not backtest_results:
            return []
        
        # Extract metric values
        strategy_metrics = []
        
        for result in backtest_results:
            metrics = result.performance_metrics
            
            if metric == 'total_return_pct':
                value = metrics.total_return_pct
            elif metric == 'sharpe_ratio':
                value = metrics.sharpe_ratio
            elif metric == 'sortino_ratio':
                value = metrics.sortino_ratio
            elif metric == 'max_drawdown_pct':
                value = metrics.max_drawdown_pct
            elif metric == 'win_rate':
                value = metrics.win_rate
            elif metric == 'profit_factor':
                value = metrics.profit_factor
            elif metric == 'calmar_ratio':
                value = self.calculate_calmar_ratio(result.equity_curve)
            else:
                value = metrics.sharpe_ratio
            
            strategy_metrics.append((result.strategy_name, value))
        
        # Sort by metric (descending for most metrics, ascending for drawdown)
        reverse = metric != 'max_drawdown_pct'
        strategy_metrics.sort(key=lambda x: x[1], reverse=reverse)
        
        return strategy_metrics