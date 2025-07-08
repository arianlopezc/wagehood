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
from ..jobs.distributed import DistributedJobManager
from ..jobs.models import JobType

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service for analysis operations."""
    
    def __init__(self):
        """Initialize the analysis service."""
        self.indicator_calculator = IndicatorCalculator()
        self.evaluator = PerformanceEvaluator()
        self.comparison = StrategyComparator()
        self.running_jobs = {}  # Legacy in-memory jobs (for backward compatibility)
        
        # Distributed job management
        self.distributed_job_manager = DistributedJobManager()
        
        logger.info("AnalysisService initialized with distributed job support")
    
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
        """Get the status of an async optimization job from distributed system with legacy fallback."""
        # Try distributed job system first
        try:
            status = await self.distributed_job_manager.get_job_status(job_id)
            if status:
                return status
        except Exception as e:
            logger.warning(f"Failed to get distributed job status for {job_id}: {e}")
        
        # Fallback to legacy in-memory jobs
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
    
    async def analyze_strategy(self, strategy, market_data):
        """
        Analyze a strategy with market data for backward compatibility.
        
        Args:
            strategy: Strategy instance
            market_data: MarketData instance
            
        Returns:
            Analysis results dictionary
        """
        try:
            logger.info(f"Analyzing strategy {strategy.name} on {market_data.symbol}")
            
            # Extract strategy parameters
            strategy_params = strategy.get_parameters() if hasattr(strategy, 'get_parameters') else {}
            
            # Calculate risk metrics
            risk_metrics = await self.get_risk_metrics(
                symbol=market_data.symbol,
                timeframe=market_data.timeframe,
                strategy=strategy.name,
                parameters=strategy_params
            )
            
            # Return analysis results
            return {
                'strategy_name': strategy.name,
                'symbol': market_data.symbol,
                'timeframe': market_data.timeframe.value,
                'risk_metrics': risk_metrics,
                'data_points': len(market_data.data),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing strategy: {e}")
            raise
    
    async def analyze_strategy_historical(
        self,
        symbol: str,
        timeframe: TimeFrame,
        strategy: str,
        start_date: datetime,
        end_date: datetime,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Analyze a strategy for a specific symbol over a historical date range.
        
        This method now uses the distributed job system with deduplication and Redis-based coordination.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            strategy: Strategy type (e.g., 'macd_rsi_strategy', 'ma_crossover')
            start_date: Start date for analysis
            end_date: End date for analysis
            parameters: Optional strategy parameters (uses defaults if not provided)
            
        Returns:
            Job ID for tracking the analysis
        """
        # Validate date range
        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")
        
        if end_date > datetime.utcnow():
            raise ValueError("end_date cannot be in the future")
        
        try:
            # Create distributed job with deduplication
            job_id = await self.distributed_job_manager.create_job(
                job_type=JobType.HISTORICAL_ANALYSIS,
                symbol=symbol,
                timeframe=timeframe.value,
                strategy=strategy,
                start_date=start_date,
                end_date=end_date,
                parameters=parameters,
                priority=0
            )
            
            # Start background worker for this job
            asyncio.create_task(self._process_distributed_historical_job(job_id))
            
            logger.info(f"Created distributed historical analysis job {job_id} for {symbol} {strategy}")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to create distributed job: {e}")
            # Fallback to legacy system
            return await self._create_legacy_historical_job(
                symbol, timeframe, strategy, start_date, end_date, parameters
            )
    
    async def _process_historical_analysis_job(
        self,
        job_id: str,
        symbol: str,
        timeframe: TimeFrame,
        strategy: str,
        start_date: datetime,
        end_date: datetime,
        parameters: Optional[Dict[str, Any]]
    ):
        """
        Process historical analysis job by calculating all signals in the date range.
        
        This worker method:
        1. Fetches historical data for the date range
        2. Instantiates the requested strategy
        3. Calculates indicators and generates signals
        4. Stores results in the job tracking system
        """
        try:
            logger.info(f"Processing historical analysis job {job_id}")
            
            # Import strategy registry to get strategy classes
            from ..strategies import STRATEGY_REGISTRY
            
            # Get strategy class
            if strategy not in STRATEGY_REGISTRY:
                raise ValueError(f"Unknown strategy: {strategy}. Available: {list(STRATEGY_REGISTRY.keys())}")
            
            strategy_class = STRATEGY_REGISTRY[strategy]
            strategy_instance = strategy_class(parameters) if parameters else strategy_class()
            
            # Import necessary services
            from ..services.data_service import DataService
            from ..services.backtest_service import BacktestService
            from ..core.models import MarketData
            
            # Use BacktestService which already handles date ranges
            backtest_service = BacktestService()
            
            # Run backtest to get all signals
            result = backtest_service.run_backtest(
                symbol=symbol,
                timeframe=timeframe,
                strategy=strategy_instance,
                parameters=parameters,
                start_date=start_date,
                end_date=end_date,
                initial_capital=10000.0,  # Not used for signal generation
                commission=0.0  # Not used for signal generation
            )
            
            # Extract signals from backtest result
            signals = []
            if result and hasattr(result, 'signals'):
                signals = result.signals
            elif isinstance(result, dict) and 'signals' in result:
                signals = result['signals']
            
            # Convert signals to serializable format
            signal_data = []
            for signal in signals:
                signal_data.append({
                    'timestamp': signal.timestamp.isoformat() if hasattr(signal, 'timestamp') else None,
                    'signal_type': signal.signal_type.value if hasattr(signal, 'signal_type') else str(signal),
                    'price': getattr(signal, 'price', 0),
                    'confidence': getattr(signal, 'confidence', 0),
                    'metadata': getattr(signal, 'metadata', {})
                })
            
            # Calculate summary statistics
            buy_signals = sum(1 for s in signal_data if s['signal_type'] == 'buy')
            sell_signals = sum(1 for s in signal_data if s['signal_type'] == 'sell')
            
            # Update job status with results
            self.running_jobs[job_id].update({
                'status': 'completed',
                'completed_at': datetime.utcnow(),
                'result': {
                    'signals': signal_data,
                    'total_signals': len(signal_data),
                    'buy_signals': buy_signals,
                    'sell_signals': sell_signals,
                    'date_range': {
                        'start': start_date.isoformat(),
                        'end': end_date.isoformat()
                    },
                    'symbol': symbol,
                    'timeframe': timeframe.value,
                    'strategy': strategy,
                    'parameters': parameters or {}
                }
            })
            
            logger.info(f"Completed historical analysis job {job_id}: {len(signal_data)} signals generated")
            
        except Exception as e:
            logger.error(f"Error in historical analysis job {job_id}: {e}")
            self.running_jobs[job_id].update({
                'status': 'failed',
                'completed_at': datetime.utcnow(),
                'error': str(e)
            })
    
    async def _create_legacy_historical_job(
        self,
        symbol: str,
        timeframe: TimeFrame,
        strategy: str,
        start_date: datetime,
        end_date: datetime,
        parameters: Optional[Dict[str, Any]]
    ) -> str:
        """Fallback to legacy in-memory job system."""
        job_id = str(uuid.uuid4())
        
        # Store job info in memory
        self.running_jobs[job_id] = {
            'status': 'running',
            'started_at': datetime.utcnow(),
            'symbol': symbol,
            'timeframe': timeframe.value,
            'strategy': strategy,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'parameters': parameters or {}
        }
        
        # Start async task
        asyncio.create_task(self._process_historical_analysis_job(
            job_id, symbol, timeframe, strategy, start_date, end_date, parameters
        ))
        
        logger.info(f"Started legacy historical analysis job {job_id}")
        return job_id
    
    async def _process_distributed_historical_job(self, job_id: str):
        """Process a distributed historical analysis job."""
        try:
            # Get job details from distributed system
            job_status = await self.distributed_job_manager.get_job_status(job_id)
            if not job_status:
                logger.error(f"Distributed job {job_id} not found")
                return
            
            # Extract parameters
            symbol = job_status.get('symbol')
            timeframe_str = job_status.get('timeframe')
            strategy = job_status.get('strategy')
            start_date_str = job_status.get('start_date')
            end_date_str = job_status.get('end_date')
            parameters = job_status.get('parameters', {})
            
            if not all([symbol, timeframe_str, strategy, start_date_str, end_date_str]):
                from ..jobs.models import JobStatus
                await self.distributed_job_manager.update_job_status(
                    job_id, 
                    JobStatus.FAILED,
                    error_message="Missing required job parameters"
                )
                return
            
            # Convert parameters to proper types
            timeframe = TimeFrame(timeframe_str)
            start_date = datetime.fromisoformat(start_date_str)
            end_date = datetime.fromisoformat(end_date_str)
            
            # Update status to processing
            from ..jobs.models import JobStatus
            await self.distributed_job_manager.update_job_status(
                job_id, 
                JobStatus.PROCESSING
            )
            
            # Execute the actual analysis (reuse existing logic)
            result = await self._execute_historical_analysis(
                symbol, timeframe, strategy, start_date, end_date, parameters
            )
            
            if result:
                # Store result in distributed system
                await self.distributed_job_manager.store_job_result(job_id, result)
                logger.info(f"Completed distributed job {job_id}")
            else:
                await self.distributed_job_manager.update_job_status(
                    job_id,
                    JobStatus.FAILED,
                    error_message="Job execution failed"
                )
                
        except Exception as e:
            logger.error(f"Error processing distributed job {job_id}: {e}")
            from ..jobs.models import JobStatus
            await self.distributed_job_manager.update_job_status(
                job_id,
                JobStatus.FAILED,
                error_message=str(e)
            )
    
    async def _execute_historical_analysis(
        self,
        symbol: str,
        timeframe: TimeFrame,
        strategy: str,
        start_date: datetime,
        end_date: datetime,
        parameters: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Execute historical analysis logic (extracted from original method)."""
        try:
            logger.info(f"Executing historical analysis for {symbol} {strategy}")
            
            # Import strategy registry to get strategy classes
            from ..strategies import STRATEGY_REGISTRY
            
            # Get strategy class
            if strategy not in STRATEGY_REGISTRY:
                raise ValueError(f"Unknown strategy: {strategy}. Available: {list(STRATEGY_REGISTRY.keys())}")
            
            strategy_class = STRATEGY_REGISTRY[strategy]
            strategy_instance = strategy_class(parameters) if parameters else strategy_class()
            
            # Import necessary services
            from ..services.data_service import DataService
            from ..services.backtest_service import BacktestService
            from ..core.models import MarketData
            
            # Use BacktestService which already handles date ranges
            backtest_service = BacktestService()
            
            # Run backtest to get all signals
            result = backtest_service.run_backtest(
                symbol=symbol,
                timeframe=timeframe,
                strategy=strategy_instance,
                parameters=parameters,
                start_date=start_date,
                end_date=end_date,
                initial_capital=10000.0,  # Not used for signal generation
                commission=0.0  # Not used for signal generation
            )
            
            # Extract signals from backtest result
            signals = []
            if result and hasattr(result, 'signals'):
                signals = result.signals
            elif isinstance(result, dict) and 'signals' in result:
                signals = result['signals']
            
            # Convert signals to serializable format
            signal_data = []
            for signal in signals:
                signal_data.append({
                    'timestamp': signal.timestamp.isoformat() if hasattr(signal, 'timestamp') else None,
                    'signal_type': signal.signal_type.value if hasattr(signal, 'signal_type') else str(signal),
                    'price': getattr(signal, 'price', 0),
                    'confidence': getattr(signal, 'confidence', 0),
                    'metadata': getattr(signal, 'metadata', {})
                })
            
            # Calculate summary statistics
            buy_signals = sum(1 for s in signal_data if s['signal_type'] == 'buy')
            sell_signals = sum(1 for s in signal_data if s['signal_type'] == 'sell')
            
            # Return structured result
            return {
                'signals': signal_data,
                'total_signals': len(signal_data),
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'symbol': symbol,
                'timeframe': timeframe.value,
                'strategy': strategy,
                'parameters': parameters or {}
            }
            
        except Exception as e:
            logger.error(f"Error executing historical analysis: {e}")
            raise