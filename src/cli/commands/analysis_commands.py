"""
Analysis Commands Module

This module provides comprehensive analysis commands for the Wagehood CLI,
including strategy effectiveness analysis, performance comparisons, and 
trading recommendations.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns

from ..utils.output import OutputFormatter
from ..utils.logging import CLILogger, log_operation
from ..config import CLIConfig
from ...strategies import (
    STRATEGY_REGISTRY, STRATEGY_METADATA, DEFAULT_STRATEGY_PARAMS,
    create_strategy, get_strategy_by_priority
)
from ...data.providers.alpaca_provider import AlpacaProvider
from ...data.providers.mock_provider import MockProvider
from ...core.models import MarketData, OHLCV, TimeFrame
from ...backtest.engine import BacktestEngine, BacktestConfig
from ...analysis.evaluator import PerformanceEvaluator

logger = logging.getLogger(__name__)


@dataclass
class TradingStyleMetrics:
    """Metrics for a specific trading style"""
    win_rate: float
    avg_return_per_trade: float
    max_drawdown: float
    sharpe_ratio: float
    num_signals: int
    avg_hold_time_hours: float
    profit_factor: float
    total_return: float
    recommendation_score: float
    style_fit: str  # 'Excellent', 'Good', 'Fair', 'Poor'


@dataclass
class StrategyAnalysis:
    """Complete analysis results for a strategy"""
    strategy_name: str
    symbol: str
    analysis_period: str
    day_trading: TradingStyleMetrics
    swing_trading: TradingStyleMetrics
    position_trading: TradingStyleMetrics
    overall_score: float
    best_style: str
    metadata: Dict[str, Any]


class StrategyAnalyzer:
    """Analyzer for strategy effectiveness across trading styles"""
    
    def __init__(self, config: CLIConfig):
        self.config = config
        self.logger = CLILogger("strategy_analyzer")
        self.backtest_engine = BacktestEngine()
        self.evaluator = PerformanceEvaluator()
        
    async def analyze_strategy_effectiveness(self, 
                                           symbol: str,
                                           period: str = "1y",
                                           strategies: Optional[List[str]] = None,
                                           use_mock_data: bool = False) -> List[StrategyAnalysis]:
        """
        Analyze strategy effectiveness for different trading styles
        
        Args:
            symbol: Symbol to analyze
            period: Time period for analysis
            strategies: Specific strategies to test (all if None)
            use_mock_data: Whether to use mock data for testing
            
        Returns:
            List of strategy analyses
        """
        # Get historical data
        market_data = await self._get_historical_data(symbol, period, use_mock_data)
        
        # Get strategies to test
        strategies_to_test = strategies or list(STRATEGY_REGISTRY.keys())
        
        # Analyze each strategy
        analyses = []
        for strategy_name in strategies_to_test:
            try:
                analysis = await self._analyze_single_strategy(
                    strategy_name, symbol, market_data, period
                )
                analyses.append(analysis)
            except Exception as e:
                self.logger.error(f"Error analyzing strategy {strategy_name}: {e}")
                continue
        
        # Sort by overall score
        analyses.sort(key=lambda x: x.overall_score, reverse=True)
        
        return analyses
    
    async def _get_historical_data(self, symbol: str, period: str, use_mock_data: bool) -> MarketData:
        """Fetch historical data for analysis"""
        try:
            if use_mock_data:
                # Use mock data for testing
                provider = MockProvider()
                await provider.connect()
                
                # Parse period
                days = self._parse_period_to_days(period)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                ohlcv_data = await provider.get_historical_data(
                    symbol, TimeFrame.DAILY, start_date, end_date
                )
                
                # Wrap in MarketData object
                return MarketData(
                    symbol=symbol,
                    timeframe=TimeFrame.DAILY,
                    data=ohlcv_data,
                    indicators={},
                    last_updated=datetime.now()
                )
            else:
                # Use Alpaca provider
                provider = AlpacaProvider({
                    'api_key': os.getenv('ALPACA_API_KEY'),
                    'secret_key': os.getenv('ALPACA_SECRET_KEY')
                })
                await provider.connect()
                
                days = self._parse_period_to_days(period)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                ohlcv_data = await provider.get_historical_data(
                    symbol, TimeFrame.DAILY, start_date, end_date
                )
                
                # Wrap in MarketData object
                return MarketData(
                    symbol=symbol,
                    timeframe=TimeFrame.DAILY,
                    data=ohlcv_data,
                    indicators={},
                    last_updated=datetime.now()
                )
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            raise
    
    def _parse_period_to_days(self, period: str) -> int:
        """Parse period string to days"""
        period = period.lower()
        if period.endswith('d'):
            return int(period[:-1])
        elif period.endswith('w'):
            return int(period[:-1]) * 7
        elif period.endswith('m'):
            return int(period[:-1]) * 30
        elif period.endswith('y'):
            return int(period[:-1]) * 365
        else:
            return 365  # Default to 1 year
    
    async def _analyze_single_strategy(self, 
                                     strategy_name: str,
                                     symbol: str,
                                     market_data: MarketData,
                                     period: str) -> StrategyAnalysis:
        """Analyze a single strategy for all trading styles"""
        # Create strategy instance
        strategy = create_strategy(strategy_name)
        
        # Run backtest
        backtest_result = self.backtest_engine.run_backtest(
            strategy, market_data, initial_capital=10000
        )
        
        # Analyze for different trading styles
        day_trading_metrics = self._analyze_for_day_trading(backtest_result)
        swing_trading_metrics = self._analyze_for_swing_trading(backtest_result)
        position_trading_metrics = self._analyze_for_position_trading(backtest_result)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(
            day_trading_metrics, swing_trading_metrics, position_trading_metrics
        )
        
        # Determine best style
        best_style = self._determine_best_style(
            day_trading_metrics, swing_trading_metrics, position_trading_metrics
        )
        
        return StrategyAnalysis(
            strategy_name=strategy_name,
            symbol=symbol,
            analysis_period=period,
            day_trading=day_trading_metrics,
            swing_trading=swing_trading_metrics,
            position_trading=position_trading_metrics,
            overall_score=overall_score,
            best_style=best_style,
            metadata=STRATEGY_METADATA.get(strategy_name, {})
        )
    
    def _analyze_for_day_trading(self, backtest_result) -> TradingStyleMetrics:
        """Analyze strategy for day trading (short-term, high frequency)"""
        metrics = backtest_result.performance_metrics
        
        # Filter trades for day trading (hold time < 1 day)
        day_trades = [
            trade for trade in backtest_result.trades
            if trade.duration_hours and trade.duration_hours < 24
        ]
        
        if not day_trades:
            return TradingStyleMetrics(
                win_rate=0.0, avg_return_per_trade=0.0, max_drawdown=0.0,
                sharpe_ratio=0.0, num_signals=0, avg_hold_time_hours=0.0,
                profit_factor=0.0, total_return=0.0, recommendation_score=0.0,
                style_fit="Poor"
            )
        
        # Calculate day trading specific metrics
        day_pnls = [trade.calculate_pnl() for trade in day_trades]
        winning_trades = [pnl for pnl in day_pnls if pnl > 0]
        losing_trades = [pnl for pnl in day_pnls if pnl < 0]
        
        win_rate = len(winning_trades) / len(day_trades) if day_trades else 0.0
        avg_return_per_trade = sum(day_pnls) / len(day_pnls) if day_pnls else 0.0
        avg_hold_time = sum(t.duration_hours for t in day_trades) / len(day_trades)
        
        # Calculate profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0.0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        
        # Calculate recommendation score for day trading
        recommendation_score = self._calculate_day_trading_score(
            win_rate, avg_return_per_trade, len(day_trades), avg_hold_time
        )
        
        style_fit = self._get_style_fit_rating(recommendation_score)
        
        return TradingStyleMetrics(
            win_rate=win_rate,
            avg_return_per_trade=avg_return_per_trade,
            max_drawdown=metrics.max_drawdown_pct,
            sharpe_ratio=metrics.sharpe_ratio,
            num_signals=len(day_trades),
            avg_hold_time_hours=avg_hold_time,
            profit_factor=profit_factor,
            total_return=sum(day_pnls),
            recommendation_score=recommendation_score,
            style_fit=style_fit
        )
    
    def _analyze_for_swing_trading(self, backtest_result) -> TradingStyleMetrics:
        """Analyze strategy for swing trading (medium-term, 1-10 days)"""
        metrics = backtest_result.performance_metrics
        
        # Filter trades for swing trading (hold time 1-10 days)
        swing_trades = [
            trade for trade in backtest_result.trades
            if trade.duration_hours and 24 <= trade.duration_hours <= 240
        ]
        
        if not swing_trades:
            return TradingStyleMetrics(
                win_rate=0.0, avg_return_per_trade=0.0, max_drawdown=0.0,
                sharpe_ratio=0.0, num_signals=0, avg_hold_time_hours=0.0,
                profit_factor=0.0, total_return=0.0, recommendation_score=0.0,
                style_fit="Poor"
            )
        
        # Calculate swing trading specific metrics
        swing_pnls = [trade.calculate_pnl() for trade in swing_trades]
        winning_trades = [pnl for pnl in swing_pnls if pnl > 0]
        losing_trades = [pnl for pnl in swing_pnls if pnl < 0]
        
        win_rate = len(winning_trades) / len(swing_trades) if swing_trades else 0.0
        avg_return_per_trade = sum(swing_pnls) / len(swing_pnls) if swing_pnls else 0.0
        avg_hold_time = sum(t.duration_hours for t in swing_trades) / len(swing_trades)
        
        # Calculate profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0.0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        
        # Calculate recommendation score for swing trading
        recommendation_score = self._calculate_swing_trading_score(
            win_rate, avg_return_per_trade, len(swing_trades), avg_hold_time
        )
        
        style_fit = self._get_style_fit_rating(recommendation_score)
        
        return TradingStyleMetrics(
            win_rate=win_rate,
            avg_return_per_trade=avg_return_per_trade,
            max_drawdown=metrics.max_drawdown_pct,
            sharpe_ratio=metrics.sharpe_ratio,
            num_signals=len(swing_trades),
            avg_hold_time_hours=avg_hold_time,
            profit_factor=profit_factor,
            total_return=sum(swing_pnls),
            recommendation_score=recommendation_score,
            style_fit=style_fit
        )
    
    def _analyze_for_position_trading(self, backtest_result) -> TradingStyleMetrics:
        """Analyze strategy for position trading (long-term, >10 days)"""
        metrics = backtest_result.performance_metrics
        
        # Filter trades for position trading (hold time > 10 days)
        position_trades = [
            trade for trade in backtest_result.trades
            if trade.duration_hours and trade.duration_hours > 240
        ]
        
        if not position_trades:
            return TradingStyleMetrics(
                win_rate=0.0, avg_return_per_trade=0.0, max_drawdown=0.0,
                sharpe_ratio=0.0, num_signals=0, avg_hold_time_hours=0.0,
                profit_factor=0.0, total_return=0.0, recommendation_score=0.0,
                style_fit="Poor"
            )
        
        # Calculate position trading specific metrics
        position_pnls = [trade.calculate_pnl() for trade in position_trades]
        winning_trades = [pnl for pnl in position_pnls if pnl > 0]
        losing_trades = [pnl for pnl in position_pnls if pnl < 0]
        
        win_rate = len(winning_trades) / len(position_trades) if position_trades else 0.0
        avg_return_per_trade = sum(position_pnls) / len(position_pnls) if position_pnls else 0.0
        avg_hold_time = sum(t.duration_hours for t in position_trades) / len(position_trades)
        
        # Calculate profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0.0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        
        # Calculate recommendation score for position trading
        recommendation_score = self._calculate_position_trading_score(
            win_rate, avg_return_per_trade, len(position_trades), avg_hold_time
        )
        
        style_fit = self._get_style_fit_rating(recommendation_score)
        
        return TradingStyleMetrics(
            win_rate=win_rate,
            avg_return_per_trade=avg_return_per_trade,
            max_drawdown=metrics.max_drawdown_pct,
            sharpe_ratio=metrics.sharpe_ratio,
            num_signals=len(position_trades),
            avg_hold_time_hours=avg_hold_time,
            profit_factor=profit_factor,
            total_return=sum(position_pnls),
            recommendation_score=recommendation_score,
            style_fit=style_fit
        )
    
    def _calculate_day_trading_score(self, win_rate: float, avg_return: float, 
                                   num_signals: int, avg_hold_time: float) -> float:
        """Calculate recommendation score for day trading"""
        # Day trading prefers high frequency, short hold times, consistent returns
        frequency_score = min(num_signals / 100, 1.0)  # Prefer 100+ signals per year
        hold_time_score = max(0, 1.0 - (avg_hold_time / 24))  # Prefer < 24 hours
        consistency_score = win_rate  # Prefer high win rate
        return_score = max(0, min(avg_return / 0.01, 1.0))  # Prefer 1%+ per trade
        
        return (frequency_score * 0.3 + hold_time_score * 0.2 + 
                consistency_score * 0.3 + return_score * 0.2)
    
    def _calculate_swing_trading_score(self, win_rate: float, avg_return: float,
                                     num_signals: int, avg_hold_time: float) -> float:
        """Calculate recommendation score for swing trading"""
        # Swing trading prefers moderate frequency, medium hold times
        frequency_score = min(num_signals / 50, 1.0)  # Prefer 50+ signals per year
        hold_time_score = 1.0 if 24 <= avg_hold_time <= 240 else 0.5  # Prefer 1-10 days
        consistency_score = win_rate
        return_score = max(0, min(avg_return / 0.015, 1.0))  # Prefer 1.5%+ per trade
        
        return (frequency_score * 0.25 + hold_time_score * 0.25 + 
                consistency_score * 0.3 + return_score * 0.2)
    
    def _calculate_position_trading_score(self, win_rate: float, avg_return: float,
                                        num_signals: int, avg_hold_time: float) -> float:
        """Calculate recommendation score for position trading"""
        # Position trading prefers lower frequency, longer hold times, higher returns
        frequency_score = min(num_signals / 20, 1.0)  # Prefer 20+ signals per year
        hold_time_score = min(avg_hold_time / 720, 1.0)  # Prefer 30+ days
        consistency_score = win_rate
        return_score = max(0, min(avg_return / 0.03, 1.0))  # Prefer 3%+ per trade
        
        return (frequency_score * 0.2 + hold_time_score * 0.3 + 
                consistency_score * 0.25 + return_score * 0.25)
    
    def _get_style_fit_rating(self, score: float) -> str:
        """Convert score to style fit rating"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Poor"
    
    def _calculate_overall_score(self, day_trading: TradingStyleMetrics,
                               swing_trading: TradingStyleMetrics,
                               position_trading: TradingStyleMetrics) -> float:
        """Calculate overall strategy score"""
        return max(
            day_trading.recommendation_score,
            swing_trading.recommendation_score,
            position_trading.recommendation_score
        )
    
    def _determine_best_style(self, day_trading: TradingStyleMetrics,
                            swing_trading: TradingStyleMetrics,
                            position_trading: TradingStyleMetrics) -> str:
        """Determine the best trading style for the strategy"""
        scores = {
            "Day Trading": day_trading.recommendation_score,
            "Swing Trading": swing_trading.recommendation_score,
            "Position Trading": position_trading.recommendation_score
        }
        return max(scores, key=scores.get)


class AnalysisFormatter:
    """Formatter for analysis results"""
    
    def __init__(self, formatter: OutputFormatter):
        self.formatter = formatter
        self.console = formatter.console
    
    def format_analysis_results(self, analyses: List[StrategyAnalysis], 
                              output_format: str = "table") -> None:
        """Format and display analysis results"""
        if output_format == "json":
            self._format_json(analyses)
        else:
            self._format_table(analyses)
    
    def _format_json(self, analyses: List[StrategyAnalysis]) -> None:
        """Format results as JSON"""
        output = []
        for analysis in analyses:
            output.append({
                "strategy": analysis.strategy_name,
                "symbol": analysis.symbol,
                "period": analysis.analysis_period,
                "overall_score": analysis.overall_score,
                "best_style": analysis.best_style,
                "day_trading": asdict(analysis.day_trading),
                "swing_trading": asdict(analysis.swing_trading),
                "position_trading": asdict(analysis.position_trading),
                "metadata": analysis.metadata
            })
        
        self.console.print(json.dumps(output, indent=2))
    
    def _format_table(self, analyses: List[StrategyAnalysis]) -> None:
        """Format results as rich table"""
        # Summary table
        summary_table = Table(title="Strategy Effectiveness Analysis Summary")
        summary_table.add_column("Strategy", style="bold")
        summary_table.add_column("Overall Score", justify="right")
        summary_table.add_column("Best Style", style="green")
        summary_table.add_column("Day Trading", justify="center")
        summary_table.add_column("Swing Trading", justify="center")
        summary_table.add_column("Position Trading", justify="center")
        
        for analysis in analyses:
            summary_table.add_row(
                analysis.strategy_name,
                f"{analysis.overall_score:.2f}",
                analysis.best_style,
                self._get_style_indicator(analysis.day_trading.style_fit),
                self._get_style_indicator(analysis.swing_trading.style_fit),
                self._get_style_indicator(analysis.position_trading.style_fit)
            )
        
        self.console.print(summary_table)
        self.console.print()
        
        # Detailed analysis for top strategies
        for i, analysis in enumerate(analyses[:3]):  # Show top 3
            self._format_detailed_analysis(analysis, i + 1)
    
    def _get_style_indicator(self, style_fit: str) -> str:
        """Get colored indicator for style fit"""
        if style_fit == "Excellent":
            return "[green]●[/green]"
        elif style_fit == "Good":
            return "[yellow]●[/yellow]"
        elif style_fit == "Fair":
            return "[orange]●[/orange]"
        else:
            return "[red]●[/red]"
    
    def _format_detailed_analysis(self, analysis: StrategyAnalysis, rank: int) -> None:
        """Format detailed analysis for a single strategy"""
        title = f"#{rank}: {analysis.strategy_name} - {analysis.symbol}"
        
        # Create metrics table
        metrics_table = Table(title=f"Detailed Metrics - {analysis.strategy_name}")
        metrics_table.add_column("Metric", style="bold")
        metrics_table.add_column("Day Trading", justify="right")
        metrics_table.add_column("Swing Trading", justify="right")
        metrics_table.add_column("Position Trading", justify="right")
        
        metrics = [
            ("Win Rate", 
             f"{analysis.day_trading.win_rate:.1%}",
             f"{analysis.swing_trading.win_rate:.1%}",
             f"{analysis.position_trading.win_rate:.1%}"),
            ("Avg Return/Trade",
             f"{analysis.day_trading.avg_return_per_trade:.2%}",
             f"{analysis.swing_trading.avg_return_per_trade:.2%}",
             f"{analysis.position_trading.avg_return_per_trade:.2%}"),
            ("Profit Factor",
             f"{analysis.day_trading.profit_factor:.2f}",
             f"{analysis.swing_trading.profit_factor:.2f}",
             f"{analysis.position_trading.profit_factor:.2f}"),
            ("Num Signals",
             f"{analysis.day_trading.num_signals}",
             f"{analysis.swing_trading.num_signals}",
             f"{analysis.position_trading.num_signals}"),
            ("Avg Hold Time",
             f"{analysis.day_trading.avg_hold_time_hours:.1f}h",
             f"{analysis.swing_trading.avg_hold_time_hours:.1f}h",
             f"{analysis.position_trading.avg_hold_time_hours:.1f}h"),
            ("Recommendation",
             f"{analysis.day_trading.recommendation_score:.2f} ({analysis.day_trading.style_fit})",
             f"{analysis.swing_trading.recommendation_score:.2f} ({analysis.swing_trading.style_fit})",
             f"{analysis.position_trading.recommendation_score:.2f} ({analysis.position_trading.style_fit})")
        ]
        
        for metric in metrics:
            metrics_table.add_row(*metric)
        
        # Create recommendation panel
        best_style = analysis.best_style.lower().replace(" ", "_")
        best_metrics = getattr(analysis, best_style)
        
        recommendation_text = f"""
[bold green]Best Trading Style:[/bold green] {analysis.best_style}
[bold]Style Fit:[/bold] {best_metrics.style_fit}
[bold]Expected Win Rate:[/bold] {best_metrics.win_rate:.1%}
[bold]Expected Return/Trade:[/bold] {best_metrics.avg_return_per_trade:.2%}
[bold]Signals per Year:[/bold] {best_metrics.num_signals}
[bold]Average Hold Time:[/bold] {best_metrics.avg_hold_time_hours:.1f} hours

[bold]Strategy Info:[/bold]
Difficulty: {analysis.metadata.get('difficulty', 'Unknown')}
Priority: {analysis.metadata.get('priority', 'Unknown')}
"""
        
        recommendation_panel = Panel(
            recommendation_text,
            title=f"Recommendation for {analysis.strategy_name}",
            border_style="green"
        )
        
        self.console.print(Panel(metrics_table, title=title))
        self.console.print(recommendation_panel)
        self.console.print()


# CLI Command Group
@click.group()
def analyze():
    """Analysis and strategy evaluation commands"""
    pass


@analyze.command()
@click.argument('symbol', type=str)
@click.option('--period', '-p', default='1y', 
              help='Analysis period (e.g., 1y, 6m, 90d)')
@click.option('--strategies', '-s', multiple=True,
              help='Specific strategies to test (all if not specified)')
@click.option('--format', '-f', 'output_format', 
              type=click.Choice(['table', 'json']), default='table',
              help='Output format')
@click.option('--use-mock-data', is_flag=True,
              help='Use mock data for testing (useful for development)')
@click.pass_context
def strategy_effectiveness(ctx, symbol, period, strategies, output_format, use_mock_data):
    """
    Analyze which trading strategy would be most effective for a given symbol.
    
    This command analyzes all available trading strategies for a symbol and 
    provides recommendations based on different trading styles:
    
    - Day Trading (short-term, high frequency)
    - Swing Trading (medium-term, 1-10 days)  
    - Position Trading (long-term, >10 days)
    
    For each strategy and style combination, it analyzes:
    - Win rate and average return per trade
    - Maximum drawdown and Sharpe ratio
    - Number of signals generated
    - Hold time distribution
    - Overall recommendation score
    
    Examples:
        wagehood analyze strategy-effectiveness SPY
        wagehood analyze strategy-effectiveness AAPL --period 6m --format json
        wagehood analyze strategy-effectiveness TSLA --strategies ma_crossover macd_rsi
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    console = ctx.obj['console']
    
    # Validate symbol
    symbol = symbol.upper()
    
    # Convert strategies tuple to list
    strategies_list = list(strategies) if strategies else None
    
    # Validate strategies
    if strategies_list:
        invalid_strategies = [s for s in strategies_list if s not in STRATEGY_REGISTRY]
        if invalid_strategies:
            console.print(f"[red]Invalid strategies: {', '.join(invalid_strategies)}[/red]")
            console.print(f"Available strategies: {', '.join(STRATEGY_REGISTRY.keys())}")
            return
    
    async def run_analysis():
        try:
            # Create analyzer
            analyzer = StrategyAnalyzer(config)
            
            # Show progress
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                # Add main task
                main_task = progress.add_task(
                    f"Analyzing strategy effectiveness for {symbol}...", 
                    total=100
                )
                
                # Fetch data
                progress.update(main_task, advance=20, 
                              description="Fetching historical data...")
                
                # Run analysis
                progress.update(main_task, advance=30,
                              description="Running strategy analysis...")
                
                analyses = await analyzer.analyze_strategy_effectiveness(
                    symbol, period, strategies_list, use_mock_data
                )
                
                progress.update(main_task, advance=40,
                              description="Calculating recommendations...")
                
                # Complete progress
                progress.update(main_task, advance=10, 
                              description="Formatting results...")
                
                progress.update(main_task, completed=100)
            
            # Format and display results
            analysis_formatter = AnalysisFormatter(formatter)
            
            if not analyses:
                console.print("[yellow]No analysis results generated. Check your data and try again.[/yellow]")
                return
            
            # Show header
            console.print(f"\n[bold]Strategy Effectiveness Analysis for {symbol}[/bold]")
            console.print(f"Analysis Period: {period}")
            console.print(f"Strategies Analyzed: {len(analyses)}")
            console.print(f"Data Source: {'Mock Data' if use_mock_data else 'Alpaca Markets'}")
            console.print()
            
            # Display results
            analysis_formatter.format_analysis_results(analyses, output_format)
            
            # Show summary recommendations
            if output_format == "table" and analyses:
                console.print("\n[bold]Summary Recommendations:[/bold]")
                for i, analysis in enumerate(analyses[:3]):  # Top 3
                    console.print(f"{i+1}. [green]{analysis.strategy_name}[/green] - Best for {analysis.best_style}")
                    
                    # Get best metrics
                    best_style = analysis.best_style.lower().replace(" ", "_")
                    best_metrics = getattr(analysis, best_style)
                    
                    console.print(f"   Win Rate: {best_metrics.win_rate:.1%}, "
                                f"Avg Return: {best_metrics.avg_return_per_trade:.2%}, "
                                f"Signals: {best_metrics.num_signals}")
                console.print()
            
        except Exception as e:
            console.print(f"[red]Error during analysis: {str(e)}[/red]")
            if config.verbose:
                import traceback
                console.print(traceback.format_exc())
    
    # Run the async analysis
    asyncio.run(run_analysis())


@analyze.command()
@click.argument('strategies', nargs=-1, required=True)
@click.option('--symbol', '-s', default='SPY',
              help='Symbol to use for comparison')
@click.option('--period', '-p', default='1y',
              help='Analysis period')
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['table', 'json']), default='table',
              help='Output format')
@click.option('--use-mock-data', is_flag=True,
              help='Use mock data for testing (useful for development)')
@click.pass_context
def compare_strategies(ctx, strategies, symbol, period, output_format, use_mock_data):
    """
    Compare multiple strategies side-by-side.
    
    This command allows you to compare 2-5 specific strategies directly
    to see which performs best for your trading style.
    
    Examples:
        wagehood analyze compare-strategies ma_crossover macd_rsi
        wagehood analyze compare-strategies ma_crossover macd_rsi bollinger_breakout --symbol AAPL
    """
    config = ctx.obj['config']
    formatter = ctx.obj['formatter']
    console = ctx.obj['console']
    
    # Validate strategies
    if len(strategies) < 2:
        console.print("[red]Please provide at least 2 strategies to compare[/red]")
        return
    
    if len(strategies) > 5:
        console.print("[red]Please provide at most 5 strategies to compare[/red]")
        return
    
    invalid_strategies = [s for s in strategies if s not in STRATEGY_REGISTRY]
    if invalid_strategies:
        console.print(f"[red]Invalid strategies: {', '.join(invalid_strategies)}[/red]")
        console.print(f"Available strategies: {', '.join(STRATEGY_REGISTRY.keys())}")
        return
    
    # Run the strategy effectiveness analysis with specific strategies
    ctx.invoke(strategy_effectiveness, 
               symbol=symbol, 
               period=period, 
               strategies=strategies,
               output_format=output_format,
               use_mock_data=use_mock_data)


@analyze.command()
@click.option('--format', '-f', 'output_format',
              type=click.Choice(['table', 'json']), default='table',
              help='Output format')
@click.pass_context
def list_strategies(ctx, output_format):
    """
    List all available strategies with their metadata.
    
    Shows strategy information including difficulty level, priority,
    and documented performance characteristics.
    """
    formatter = ctx.obj['formatter']
    console = ctx.obj['console']
    
    if output_format == 'json':
        console.print(json.dumps(STRATEGY_METADATA, indent=2))
        return
    
    # Create table
    table = Table(title="Available Trading Strategies")
    table.add_column("Strategy", style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Difficulty")
    table.add_column("Priority", justify="right")
    table.add_column("Description")
    
    for strategy_id, metadata in STRATEGY_METADATA.items():
        table.add_row(
            strategy_id,
            metadata.get('name', 'Unknown'),
            metadata.get('difficulty', 'Unknown'),
            str(metadata.get('priority', 'Unknown')),
            metadata.get('description', 'No description')
        )
    
    console.print(table)
    console.print()
    
    # Show usage examples
    console.print("[bold]Usage Examples:[/bold]")
    console.print("wagehood analyze strategy-effectiveness SPY")
    console.print("wagehood analyze strategy-effectiveness AAPL --strategies ma_crossover macd_rsi")
    console.print("wagehood analyze compare-strategies ma_crossover macd_rsi bollinger_breakout")


# Export the command group
analysis_commands = analyze