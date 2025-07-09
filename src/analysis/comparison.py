"""
Strategy comparison tools for detailed analysis
"""

from typing import Dict, Any, List, Tuple
from datetime import datetime
import statistics
import json

from ..core.models import BacktestResult
from .evaluator import PerformanceEvaluator


class StrategyComparator:
    """Advanced strategy comparison and analysis tools"""

    def __init__(self, evaluator: PerformanceEvaluator = None):
        """
        Initialize strategy comparator

        Args:
            evaluator: Performance evaluator instance
        """
        self.evaluator = evaluator or PerformanceEvaluator()

    def compare_strategies(
        self, backtest_results: List[BacktestResult]
    ) -> Dict[str, Any]:
        """
        Comprehensive strategy comparison

        Args:
            backtest_results: List of backtest results to compare

        Returns:
            Dictionary with detailed comparison results
        """
        if not backtest_results:
            return {}

        comparison = {
            "summary": self._create_comparison_summary(backtest_results),
            "detailed_metrics": self._create_detailed_comparison(backtest_results),
            "rankings": self._create_rankings(backtest_results),
            "risk_analysis": self._create_risk_analysis(backtest_results),
            "correlation_analysis": self._create_correlation_analysis(backtest_results),
            "best_strategy": self._find_best_strategy(backtest_results),
            "recommendations": self._create_recommendations(backtest_results),
        }

        return comparison

    def _create_comparison_summary(
        self, results: List[BacktestResult]
    ) -> Dict[str, Any]:
        """Create high-level comparison summary"""
        summary = {
            "total_strategies": len(results),
            "date_range": {
                "start": min(r.start_date for r in results),
                "end": max(r.end_date for r in results),
            },
            "symbols": list(set(r.symbol for r in results)),
            "total_trades": sum(r.performance_metrics.total_trades for r in results),
            "best_return": max(r.performance_metrics.total_return_pct for r in results),
            "worst_return": min(
                r.performance_metrics.total_return_pct for r in results
            ),
            "best_sharpe": max(r.performance_metrics.sharpe_ratio for r in results),
            "worst_sharpe": min(r.performance_metrics.sharpe_ratio for r in results),
        }

        return summary

    def _create_detailed_comparison(
        self, results: List[BacktestResult]
    ) -> Dict[str, Dict[str, Any]]:
        """Create detailed metric comparison"""
        detailed = {}

        for result in results:
            metrics = result.performance_metrics

            detailed[result.strategy_name] = {
                "basic_metrics": {
                    "total_return_pct": metrics.total_return_pct,
                    "total_trades": metrics.total_trades,
                    "win_rate": metrics.win_rate,
                    "profit_factor": metrics.profit_factor,
                    "avg_win": metrics.avg_win,
                    "avg_loss": metrics.avg_loss,
                    "largest_win": metrics.largest_win,
                    "largest_loss": metrics.largest_loss,
                },
                "risk_metrics": {
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "sortino_ratio": metrics.sortino_ratio,
                    "max_drawdown_pct": metrics.max_drawdown_pct,
                    "calmar_ratio": self.evaluator.calculate_calmar_ratio(
                        result.equity_curve
                    ),
                    "var_5pct": self.evaluator.calculate_var(result.equity_curve, 0.05),
                    "expected_shortfall": self.evaluator.calculate_expected_shortfall(
                        result.equity_curve
                    ),
                },
                "trading_metrics": {
                    "avg_trade_duration_hours": metrics.avg_trade_duration_hours,
                    "max_consecutive_wins": metrics.max_consecutive_wins,
                    "max_consecutive_losses": metrics.max_consecutive_losses,
                    "trade_frequency": self._calculate_trade_frequency(result),
                    "total_commission": sum(t.commission for t in result.trades),
                },
                "consistency_metrics": {
                    "monthly_returns": self._calculate_monthly_returns(result),
                    "winning_months": self._calculate_winning_months(result),
                    "return_volatility": self._calculate_return_volatility(
                        result.equity_curve
                    ),
                },
            }

        return detailed

    def _create_rankings(
        self, results: List[BacktestResult]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Create rankings for each metric"""
        rankings = {}

        metrics_to_rank = [
            ("total_return_pct", "Total Return %", False),
            ("sharpe_ratio", "Sharpe Ratio", False),
            ("sortino_ratio", "Sortino Ratio", False),
            ("max_drawdown_pct", "Max Drawdown %", True),  # Lower is better
            ("win_rate", "Win Rate", False),
            ("profit_factor", "Profit Factor", False),
            ("calmar_ratio", "Calmar Ratio", False),
        ]

        for metric_key, metric_name, ascending in metrics_to_rank:
            metric_values = []

            for result in results:
                if metric_key == "calmar_ratio":
                    value = self.evaluator.calculate_calmar_ratio(result.equity_curve)
                else:
                    value = getattr(result.performance_metrics, metric_key)

                metric_values.append((result.strategy_name, value))

            # Sort by metric value
            metric_values.sort(key=lambda x: x[1], reverse=not ascending)
            rankings[metric_name] = metric_values

        return rankings

    def _create_risk_analysis(self, results: List[BacktestResult]) -> Dict[str, Any]:
        """Create comprehensive risk analysis"""
        risk_analysis = {
            "risk_return_scatter": [],
            "drawdown_analysis": {},
            "volatility_analysis": {},
            "tail_risk_analysis": {},
        }

        for result in results:
            metrics = result.performance_metrics

            # Risk-return scatter data
            risk_analysis["risk_return_scatter"].append(
                {
                    "strategy": result.strategy_name,
                    "return": metrics.total_return_pct,
                    "risk": metrics.max_drawdown_pct,
                    "sharpe": metrics.sharpe_ratio,
                }
            )

            # Drawdown analysis
            drawdown_periods = self._calculate_drawdown_periods(result.equity_curve)
            risk_analysis["drawdown_analysis"][result.strategy_name] = {
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "avg_drawdown_duration": (
                    statistics.mean([d["duration"] for d in drawdown_periods])
                    if drawdown_periods
                    else 0
                ),
                "drawdown_periods": len(drawdown_periods),
                "recovery_factor": self._calculate_recovery_factor(result),
            }

            # Volatility analysis
            returns = self._calculate_returns(result.equity_curve)
            risk_analysis["volatility_analysis"][result.strategy_name] = {
                "return_volatility": (
                    statistics.stdev(returns) if len(returns) > 1 else 0
                ),
                "downside_volatility": self._calculate_downside_volatility(returns),
                "upside_volatility": self._calculate_upside_volatility(returns),
            }

            # Tail risk analysis
            risk_analysis["tail_risk_analysis"][result.strategy_name] = {
                "var_1pct": self.evaluator.calculate_var(result.equity_curve, 0.01),
                "var_5pct": self.evaluator.calculate_var(result.equity_curve, 0.05),
                "expected_shortfall_5pct": self.evaluator.calculate_expected_shortfall(
                    result.equity_curve, 0.05
                ),
                "skewness": self._calculate_skewness(returns),
                "kurtosis": self._calculate_kurtosis(returns),
            }

        return risk_analysis

    def _create_correlation_analysis(
        self, results: List[BacktestResult]
    ) -> Dict[str, Any]:
        """Create correlation analysis between strategies"""
        if len(results) < 2:
            return {}

        correlation_matrix = {}

        # Calculate pairwise correlations
        for i, result1 in enumerate(results):
            strategy1 = result1.strategy_name
            correlation_matrix[strategy1] = {}

            returns1 = self._calculate_returns(result1.equity_curve)

            for j, result2 in enumerate(results):
                strategy2 = result2.strategy_name
                returns2 = self._calculate_returns(result2.equity_curve)

                # Calculate correlation
                correlation = self._calculate_correlation(returns1, returns2)
                correlation_matrix[strategy1][strategy2] = correlation

        return {
            "correlation_matrix": correlation_matrix,
            "diversification_benefits": self._analyze_diversification_benefits(results),
            "portfolio_suggestions": self._suggest_portfolio_combinations(results),
        }

    def _find_best_strategy(self, results: List[BacktestResult]) -> Dict[str, Any]:
        """Find the best strategy using composite scoring"""
        if not results:
            return {}

        scores = {}

        for result in results:
            metrics = result.performance_metrics

            # Composite score with different weights
            score = (
                metrics.sharpe_ratio * 0.25
                + (metrics.total_return_pct / 100) * 0.2
                + metrics.sortino_ratio * 0.2
                + (100 - metrics.max_drawdown_pct) / 100 * 0.15
                + metrics.win_rate * 0.1
                + min(metrics.profit_factor, 5) * 0.05
                + (metrics.total_trades / 100) * 0.05  # Normalize trade count
            )

            scores[result.strategy_name] = {
                "composite_score": score,
                "metrics": {
                    "total_return_pct": metrics.total_return_pct,
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "max_drawdown_pct": metrics.max_drawdown_pct,
                    "win_rate": metrics.win_rate,
                },
            }

        # Find best strategy
        best_strategy = max(scores, key=lambda x: scores[x]["composite_score"])

        return {
            "best_strategy": best_strategy,
            "composite_score": scores[best_strategy]["composite_score"],
            "all_scores": scores,
            "selection_criteria": {
                "sharpe_ratio_weight": 0.25,
                "total_return_weight": 0.2,
                "sortino_ratio_weight": 0.2,
                "drawdown_weight": 0.15,
                "win_rate_weight": 0.1,
                "profit_factor_weight": 0.05,
                "trade_frequency_weight": 0.05,
            },
        }

    def _create_recommendations(
        self, results: List[BacktestResult]
    ) -> List[Dict[str, Any]]:
        """Create strategy recommendations"""
        recommendations = []

        if not results:
            return recommendations

        # Sort by composite score
        scores = []
        for result in results:
            metrics = result.performance_metrics
            score = (
                metrics.sharpe_ratio * 0.3
                + (metrics.total_return_pct / 100) * 0.25
                + (100 - metrics.max_drawdown_pct) / 100 * 0.25
                + metrics.win_rate * 0.2
            )
            scores.append((result.strategy_name, score, result))

        scores.sort(key=lambda x: x[1], reverse=True)

        # Top performer
        if scores:
            best_name, best_score, best_result = scores[0]
            recommendations.append(
                {
                    "type": "Best Overall",
                    "strategy": best_name,
                    "reason": f"Highest composite score ({best_score:.3f}) with balanced risk-return profile",
                    "metrics": {
                        "return": best_result.performance_metrics.total_return_pct,
                        "sharpe": best_result.performance_metrics.sharpe_ratio,
                        "drawdown": best_result.performance_metrics.max_drawdown_pct,
                    },
                }
            )

        # Best risk-adjusted
        best_sharpe = max(results, key=lambda x: x.performance_metrics.sharpe_ratio)
        if best_sharpe.performance_metrics.sharpe_ratio > 0:
            recommendations.append(
                {
                    "type": "Best Risk-Adjusted",
                    "strategy": best_sharpe.strategy_name,
                    "reason": f"Highest Sharpe ratio ({best_sharpe.performance_metrics.sharpe_ratio:.3f})",
                    "metrics": {
                        "sharpe": best_sharpe.performance_metrics.sharpe_ratio,
                        "sortino": best_sharpe.performance_metrics.sortino_ratio,
                        "return": best_sharpe.performance_metrics.total_return_pct,
                    },
                }
            )

        # Most consistent
        lowest_drawdown = min(
            results, key=lambda x: x.performance_metrics.max_drawdown_pct
        )
        if lowest_drawdown.performance_metrics.max_drawdown_pct < 20:
            recommendations.append(
                {
                    "type": "Most Consistent",
                    "strategy": lowest_drawdown.strategy_name,
                    "reason": f"Lowest maximum drawdown ({lowest_drawdown.performance_metrics.max_drawdown_pct:.2f}%)",
                    "metrics": {
                        "drawdown": lowest_drawdown.performance_metrics.max_drawdown_pct,
                        "win_rate": lowest_drawdown.performance_metrics.win_rate,
                        "return": lowest_drawdown.performance_metrics.total_return_pct,
                    },
                }
            )

        # High return potential
        highest_return = max(
            results, key=lambda x: x.performance_metrics.total_return_pct
        )
        if highest_return.performance_metrics.total_return_pct > 10:
            recommendations.append(
                {
                    "type": "Highest Return Potential",
                    "strategy": highest_return.strategy_name,
                    "reason": f"Highest total return ({highest_return.performance_metrics.total_return_pct:.2f}%)",
                    "metrics": {
                        "return": highest_return.performance_metrics.total_return_pct,
                        "drawdown": highest_return.performance_metrics.max_drawdown_pct,
                        "sharpe": highest_return.performance_metrics.sharpe_ratio,
                    },
                }
            )

        return recommendations

    def create_comparison_report(self, results: List[BacktestResult]) -> str:
        """Create a comprehensive comparison report"""
        if not results:
            return "No backtest results to compare."

        comparison = self.compare_strategies(results)

        report = []
        report.append("=" * 60)
        report.append("STRATEGY COMPARISON REPORT")
        report.append("=" * 60)
        report.append("")

        # Summary
        summary = comparison["summary"]
        report.append(f"Total Strategies: {summary['total_strategies']}")
        report.append(
            f"Date Range: {summary['date_range']['start'].strftime('%Y-%m-%d')} to {summary['date_range']['end'].strftime('%Y-%m-%d')}"
        )
        report.append(f"Symbols: {', '.join(summary['symbols'])}")
        report.append(f"Total Trades: {summary['total_trades']}")
        report.append("")

        # Rankings
        report.append("TOP PERFORMERS BY METRIC:")
        report.append("-" * 30)
        rankings = comparison["rankings"]

        for metric_name, ranked_strategies in rankings.items():
            if ranked_strategies:
                top_strategy, top_value = ranked_strategies[0]
                report.append(f"{metric_name}: {top_strategy} ({top_value:.3f})")

        report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 15)

        for rec in comparison["recommendations"]:
            report.append(f"\n{rec['type']}: {rec['strategy']}")
            report.append(f"Reason: {rec['reason']}")
            report.append("Key Metrics:")
            for key, value in rec["metrics"].items():
                report.append(f"  {key}: {value:.3f}")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)

    def export_comparison_data(self, results: List[BacktestResult], filename: str):
        """Export comparison data to JSON file"""
        comparison = self.compare_strategies(results)

        # Convert datetime objects to strings for JSON serialization
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        with open(filename, "w") as f:
            json.dump(comparison, f, indent=2, default=convert_datetime)

    # Helper methods
    def _calculate_trade_frequency(self, result: BacktestResult) -> float:
        """Calculate average trades per month"""
        if not result.trades:
            return 0.0

        days = (result.end_date - result.start_date).days
        if days == 0:
            return 0.0

        months = days / 30.44  # Average days per month
        return len(result.trades) / months

    def _calculate_monthly_returns(self, result: BacktestResult) -> List[float]:
        """Calculate monthly returns"""
        # Simplified implementation - would need more sophisticated date handling
        if len(result.equity_curve) < 30:
            return []

        monthly_returns = []
        for i in range(30, len(result.equity_curve), 30):
            start_value = result.equity_curve[i - 30]
            end_value = result.equity_curve[i]
            if start_value > 0:
                monthly_return = (end_value - start_value) / start_value
                monthly_returns.append(monthly_return)

        return monthly_returns

    def _calculate_winning_months(self, result: BacktestResult) -> float:
        """Calculate percentage of winning months"""
        monthly_returns = self._calculate_monthly_returns(result)
        if not monthly_returns:
            return 0.0

        winning_months = sum(1 for r in monthly_returns if r > 0)
        return winning_months / len(monthly_returns)

    def _calculate_return_volatility(self, equity_curve: List[float]) -> float:
        """Calculate return volatility"""
        returns = self._calculate_returns(equity_curve)
        if len(returns) < 2:
            return 0.0

        return statistics.stdev(returns) * (252**0.5)  # Annualized

    def _calculate_returns(self, equity_curve: List[float]) -> List[float]:
        """Calculate period returns"""
        if len(equity_curve) < 2:
            return []

        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i - 1] > 0:
                ret = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
                returns.append(ret)

        return returns

    def _calculate_drawdown_periods(
        self, equity_curve: List[float]
    ) -> List[Dict[str, Any]]:
        """Calculate drawdown periods"""
        if len(equity_curve) < 2:
            return []

        drawdown_periods = []
        peak = equity_curve[0]
        in_drawdown = False
        drawdown_start = 0

        for i, value in enumerate(equity_curve):
            if value > peak:
                if in_drawdown:
                    # End of drawdown period
                    drawdown_periods.append(
                        {
                            "start": drawdown_start,
                            "end": i - 1,
                            "duration": i - drawdown_start,
                            "depth": (peak - min(equity_curve[drawdown_start:i]))
                            / peak,
                        }
                    )
                    in_drawdown = False
                peak = value
            else:
                if not in_drawdown:
                    # Start of drawdown period
                    in_drawdown = True
                    drawdown_start = i

        # Handle ongoing drawdown
        if in_drawdown:
            drawdown_periods.append(
                {
                    "start": drawdown_start,
                    "end": len(equity_curve) - 1,
                    "duration": len(equity_curve) - drawdown_start,
                    "depth": (peak - min(equity_curve[drawdown_start:])) / peak,
                }
            )

        return drawdown_periods

    def _calculate_recovery_factor(self, result: BacktestResult) -> float:
        """Calculate recovery factor (total return / max drawdown)"""
        total_return = result.performance_metrics.total_return_pct
        max_drawdown = result.performance_metrics.max_drawdown_pct

        if max_drawdown == 0:
            return float("inf") if total_return > 0 else 0.0

        return total_return / max_drawdown

    def _calculate_downside_volatility(self, returns: List[float]) -> float:
        """Calculate downside volatility"""
        if len(returns) < 2:
            return 0.0

        negative_returns = [r for r in returns if r < 0]
        if len(negative_returns) < 2:
            return 0.0

        return statistics.stdev(negative_returns) * (252**0.5)

    def _calculate_upside_volatility(self, returns: List[float]) -> float:
        """Calculate upside volatility"""
        if len(returns) < 2:
            return 0.0

        positive_returns = [r for r in returns if r > 0]
        if len(positive_returns) < 2:
            return 0.0

        return statistics.stdev(positive_returns) * (252**0.5)

    def _calculate_correlation(
        self, returns1: List[float], returns2: List[float]
    ) -> float:
        """Calculate correlation between two return series"""
        if len(returns1) != len(returns2) or len(returns1) < 2:
            return 0.0

        try:
            return statistics.correlation(returns1, returns2)
        except:
            return 0.0

    def _calculate_skewness(self, returns: List[float]) -> float:
        """Calculate skewness of returns"""
        if len(returns) < 3:
            return 0.0

        mean = statistics.mean(returns)
        std = statistics.stdev(returns)

        if std == 0:
            return 0.0

        skewness = sum((r - mean) ** 3 for r in returns) / (len(returns) * std**3)
        return skewness

    def _calculate_kurtosis(self, returns: List[float]) -> float:
        """Calculate kurtosis of returns"""
        if len(returns) < 4:
            return 0.0

        mean = statistics.mean(returns)
        std = statistics.stdev(returns)

        if std == 0:
            return 0.0

        kurtosis = sum((r - mean) ** 4 for r in returns) / (len(returns) * std**4) - 3
        return kurtosis

    def _analyze_diversification_benefits(
        self, results: List[BacktestResult]
    ) -> Dict[str, Any]:
        """Analyze diversification benefits"""
        if len(results) < 2:
            return {}

        # Calculate individual strategy statistics
        individual_returns = []
        individual_volatilities = []

        for result in results:
            returns = self._calculate_returns(result.equity_curve)
            if returns:
                individual_returns.append(statistics.mean(returns))
                individual_volatilities.append(
                    statistics.stdev(returns) if len(returns) > 1 else 0
                )

        # Calculate equal-weight portfolio statistics
        if individual_returns:
            avg_return = statistics.mean(individual_returns)
            avg_volatility = statistics.mean(individual_volatilities)

            return {
                "avg_individual_return": avg_return,
                "avg_individual_volatility": avg_volatility,
                "portfolio_return": avg_return,  # Simplified
                "diversification_ratio": avg_volatility
                / (avg_volatility * 0.8),  # Simplified
            }

        return {}

    def _suggest_portfolio_combinations(
        self, results: List[BacktestResult]
    ) -> List[Dict[str, Any]]:
        """Suggest portfolio combinations"""
        if len(results) < 2:
            return []

        suggestions = []

        # Low correlation pairs
        correlation_matrix = {}
        for i, result1 in enumerate(results):
            returns1 = self._calculate_returns(result1.equity_curve)
            for j, result2 in enumerate(results[i + 1 :], i + 1):
                returns2 = self._calculate_returns(result2.equity_curve)
                corr = self._calculate_correlation(returns1, returns2)

                if corr < 0.3:  # Low correlation
                    suggestions.append(
                        {
                            "type": "Low Correlation Pair",
                            "strategies": [
                                result1.strategy_name,
                                result2.strategy_name,
                            ],
                            "correlation": corr,
                            "reason": "Low correlation provides diversification benefits",
                        }
                    )

        # High Sharpe ratio combinations
        high_sharpe_strategies = [
            r for r in results if r.performance_metrics.sharpe_ratio > 1.0
        ]

        if len(high_sharpe_strategies) >= 2:
            strategy_names = [s.strategy_name for s in high_sharpe_strategies[:3]]
            suggestions.append(
                {
                    "type": "High Sharpe Portfolio",
                    "strategies": strategy_names,
                    "reason": "Combination of strategies with high risk-adjusted returns",
                }
            )

        return suggestions[:5]  # Limit to top 5 suggestions
