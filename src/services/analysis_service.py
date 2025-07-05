"""Analysis service for indicators, optimization, and strategy comparison."""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import asyncio

from ..indicators.calculator import IndicatorCalculator
from ..analysis.evaluator import PerformanceEvaluator
from ..analysis.comparison import StrategyComparator
from ..core.models import TimeFrame, StrategyStatus

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service for analysis operations."""
    
    def __init__(self):
        """Initialize the analysis service."""
        self.indicator_calculator = IndicatorCalculator()
        self.evaluator = PerformanceEvaluator()
        self.comparison = StrategyComparator()
        self.running_jobs = {}  # Track async jobs
        logger.info("AnalysisService initialized")
    
    async def calculate_indicators(
        self,
        symbol: str,
        timeframe: TimeFrame,
        indicators: List[str],
        parameters: Dict[str, Any]
    ) -> Dict[str, List[float]]:
        """
        Calculate technical indicators for a symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            indicators: List of indicator names
            parameters: Indicator parameters
            
        Returns:
            Dict mapping indicator names to their values
        """
        try:
            logger.info(f"Calculating {len(indicators)} indicators for {symbol} ({timeframe})")
            
            # Calculate indicators using calculator
            results = self.indicator_calculator.calculate_multiple(
                symbol=symbol,
                timeframe=timeframe,
                indicators=indicators,
                parameters=parameters
            )
            
            logger.info(f"Calculated {len(results)} indicators")
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            raise
    
    async def optimize_strategy(
        self,
        symbol: str,
        timeframe: TimeFrame,
        strategy: str,
        parameter_ranges: Dict[str, List[Any]],
        optimization_metric: str,
        initial_capital: float = 10000.0,
        commission: float = 0.0  # Default to commission-free trading
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            strategy: Strategy type
            parameter_ranges: Parameter ranges to test
            optimization_metric: Metric to optimize
            initial_capital: Initial capital
            commission: Commission rate (default: 0.0 for commission-free trading)
            
        Returns:
            Dict containing optimization results
        """
        try:
            logger.info(f"Optimizing {strategy} for {symbol} ({timeframe})")
            
            # Run optimization using evaluator
            results = self.evaluator.optimize_parameters(
                symbol=symbol,
                timeframe=timeframe,
                strategy=strategy,
                parameter_ranges=parameter_ranges,
                optimization_metric=optimization_metric,
                initial_capital=initial_capital,
                commission=commission
            )
            
            logger.info(f"Optimization completed with {len(results['results'])} parameter combinations")
            
            return results
            
        except Exception as e:
            logger.error(f"Error optimizing strategy: {e}")
            raise
    
    async def optimize_strategy_async(
        self,
        symbol: str,
        timeframe: TimeFrame,
        strategy: str,
        parameter_ranges: Dict[str, List[Any]],
        optimization_metric: str,
        initial_capital: float = 10000.0,
        commission: float = 0.0  # Default to commission-free trading
    ) -> str:
        """
        Run strategy optimization asynchronously.
        
        Returns:
            Job ID for tracking the optimization
        """
        job_id = str(uuid.uuid4())
        
        # Store job info
        self.running_jobs[job_id] = {
            'status': 'running',
            'started_at': datetime.utcnow(),
            'symbol': symbol,
            'timeframe': timeframe.value,
            'strategy': strategy.value,
            'optimization_metric': optimization_metric.value
        }
        
        # Start async task
        asyncio.create_task(self._optimize_strategy_async_task(
            job_id, symbol, timeframe, strategy, parameter_ranges,
            optimization_metric, initial_capital, commission
        ))
        
        return job_id
    
    async def _optimize_strategy_async_task(
        self,
        job_id: str,
        symbol: str,
        timeframe: TimeFrame,
        strategy: str,
        parameter_ranges: Dict[str, List[Any]],
        optimization_metric: str,
        initial_capital: float,
        commission: float
    ):
        """Internal async task for optimization."""
        try:
            # Run the optimization
            result = await self.optimize_strategy(
                symbol, timeframe, strategy, parameter_ranges,
                optimization_metric, initial_capital, commission
            )
            
            # Update job status
            self.running_jobs[job_id].update({
                'status': 'completed',
                'completed_at': datetime.utcnow(),
                'result': result
            })
            
        except Exception as e:
            logger.error(f"Error in async optimization {job_id}: {e}")
            self.running_jobs[job_id].update({
                'status': 'failed',
                'completed_at': datetime.utcnow(),
                'error': str(e)
            })
    
    async def get_optimization_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of an async optimization job."""
        return self.running_jobs.get(job_id)
    
    async def compare_strategies(
        self,
        symbol: str,
        timeframe: TimeFrame,
        strategies: List[Dict[str, Any]],
        initial_capital: float = 10000.0,
        commission: float = 0.001
    ) -> Dict[str, Any]:
        """
        Compare multiple strategies on the same symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            strategies: List of strategy configurations
            initial_capital: Initial capital
            commission: Commission rate
            
        Returns:
            Dict containing comparison results
        """
        try:
            logger.info(f"Comparing {len(strategies)} strategies for {symbol} ({timeframe})")
            
            # Run comparison using comparison service
            results = self.comparison.compare_strategies(
                symbol=symbol,
                timeframe=timeframe,
                strategies=strategies,
                initial_capital=initial_capital,
                commission=commission
            )
            
            logger.info(f"Strategy comparison completed")
            
            return results
            
        except Exception as e:
            logger.error(f"Error comparing strategies: {e}")
            raise
    
    async def get_strategy_rankings(
        self,
        metric: str,
        symbol: Optional[str] = None,
        timeframe: Optional[TimeFrame] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get strategy rankings based on performance metrics.
        
        Args:
            metric: Metric to rank by
            symbol: Optional symbol filter
            timeframe: Optional timeframe filter
            limit: Optional record limit
            
        Returns:
            List of strategy rankings
        """
        try:
            logger.info(f"Getting strategy rankings by {metric}")
            
            # Get rankings using evaluator
            rankings = self.evaluator.get_strategy_rankings(
                metric=metric,
                symbol=symbol,
                timeframe=timeframe,
                limit=limit
            )
            
            logger.info(f"Retrieved {len(rankings)} strategy rankings")
            
            return rankings
            
        except Exception as e:
            logger.error(f"Error getting strategy rankings: {e}")
            raise
    
    async def get_best_strategy(
        self,
        symbol: str,
        timeframe: TimeFrame,
        metric: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best performing strategy for a specific symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            metric: Metric to determine best strategy
            
        Returns:
            Best strategy info or None if not found
        """
        try:
            logger.info(f"Getting best strategy for {symbol} ({timeframe}) by {metric}")
            
            # Get best strategy using evaluator
            best_strategy = self.evaluator.get_best_strategy(
                symbol=symbol,
                timeframe=timeframe,
                metric=metric
            )
            
            if best_strategy:
                logger.info(f"Best strategy: {best_strategy['strategy']}")
            else:
                logger.warning(f"No strategies found for {symbol} ({timeframe})")
            
            return best_strategy
            
        except Exception as e:
            logger.error(f"Error getting best strategy: {e}")
            raise
    
    async def get_correlation_analysis(
        self,
        symbols: List[str],
        timeframe: TimeFrame,
        lookback_days: int = 252
    ) -> Dict[str, Any]:
        """
        Perform correlation analysis between multiple symbols.
        
        Args:
            symbols: List of trading symbols
            timeframe: Data timeframe
            lookback_days: Number of days to look back
            
        Returns:
            Correlation analysis results
        """
        try:
            logger.info(f"Performing correlation analysis for {len(symbols)} symbols")
            
            # Run correlation analysis using evaluator
            results = self.evaluator.analyze_correlations(
                symbols=symbols,
                timeframe=timeframe,
                lookback_days=lookback_days
            )
            
            logger.info("Correlation analysis completed")
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing correlation analysis: {e}")
            raise
    
    async def get_risk_metrics(
        self,
        symbol: str,
        timeframe: TimeFrame,
        strategy: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate risk metrics for a strategy.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            strategy: Strategy type
            parameters: Strategy parameters
            
        Returns:
            Risk metrics
        """
        try:
            logger.info(f"Calculating risk metrics for {strategy} on {symbol} ({timeframe})")
            
            # Calculate risk metrics using evaluator
            metrics = self.evaluator.calculate_risk_metrics(
                symbol=symbol,
                timeframe=timeframe,
                strategy=strategy,
                parameters=parameters
            )
            
            logger.info("Risk metrics calculated")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            raise
    
    async def get_performance_attribution(
        self,
        backtest_id: str
    ) -> Dict[str, Any]:
        """
        Perform performance attribution analysis for a backtest.
        
        Args:
            backtest_id: Backtest ID
            
        Returns:
            Performance attribution results
        """
        try:
            logger.info(f"Performing performance attribution for backtest {backtest_id}")
            
            # Get performance attribution using evaluator
            attribution = self.evaluator.analyze_performance_attribution(backtest_id)
            
            logger.info("Performance attribution completed")
            
            return attribution
            
        except Exception as e:
            logger.error(f"Error performing performance attribution: {e}")
            raise