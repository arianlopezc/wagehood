"""
Parallel Summary Processor for End-of-Day Signal Generation

This module provides safe concurrent processing of multiple symbols for EOD summaries
without using the DailyDataFetcher, implementing its own rate limiting and error handling.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class SymbolAnalysisTask:
    """Represents a single symbol analysis task."""
    symbol: str
    start_date: datetime
    end_date: datetime
    strategies: List[str]
    result: Optional[Any] = None
    error: Optional[str] = None


class ParallelSummaryProcessor:
    """
    Processes multiple symbols in parallel for EOD summary generation.
    
    Features:
    - Safe concurrent execution with rate limiting
    - Isolated analyzer instances per symbol
    - Graceful error handling per symbol
    - Configurable concurrency limits
    - No shared state between analyses
    """
    
    def __init__(
        self,
        max_concurrent_analyses: int = 5,
        max_api_calls_per_second: float = 10.0,
        timeout_per_symbol: int = 30
    ):
        """
        Initialize the parallel processor.
        
        Args:
            max_concurrent_analyses: Maximum symbols to analyze simultaneously
            max_api_calls_per_second: API rate limit (calls per second)
            timeout_per_symbol: Timeout in seconds for each symbol analysis
        """
        self.max_concurrent = max_concurrent_analyses
        self.rate_limit_delay = 1.0 / max_api_calls_per_second
        self.timeout = timeout_per_symbol
        
        # Rate limiting
        self._last_api_call = 0.0
        self._api_lock = asyncio.Lock()
        
        # Concurrency control
        self._analysis_semaphore = asyncio.Semaphore(max_concurrent_analyses)
        
        logger.info(
            f"Initialized ParallelSummaryProcessor: "
            f"max_concurrent={max_concurrent_analyses}, "
            f"rate_limit={max_api_calls_per_second}/s"
        )
    
    async def process_symbols(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        strategies: List[str]
    ) -> Dict[str, Any]:
        """
        Process multiple symbols in parallel with safe concurrency.
        
        Args:
            symbols: List of symbols to analyze
            start_date: Analysis start date
            end_date: Analysis end date
            strategies: List of strategy names to apply
            
        Returns:
            Dictionary mapping symbol -> analysis results
        """
        logger.info(f"Starting parallel processing for {len(symbols)} symbols")
        start_time = datetime.now()
        
        # Create tasks for all symbols
        tasks = []
        for symbol in symbols:
            task = SymbolAnalysisTask(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                strategies=strategies
            )
            tasks.append(task)
        
        # Process tasks in parallel with controlled concurrency
        results = await self._execute_parallel_tasks(tasks)
        
        # Log summary statistics
        duration = (datetime.now() - start_time).total_seconds()
        successful = sum(1 for r in results.values() if r.get('success', False))
        logger.info(
            f"Parallel processing completed in {duration:.2f}s: "
            f"{successful}/{len(symbols)} successful"
        )
        
        return results
    
    async def _execute_parallel_tasks(
        self,
        tasks: List[SymbolAnalysisTask]
    ) -> Dict[str, Any]:
        """Execute analysis tasks with controlled concurrency."""
        
        # Create coroutines for all tasks
        coroutines = []
        for task in tasks:
            coro = self._analyze_symbol_with_semaphore(task)
            coroutines.append(coro)
        
        # Execute all coroutines concurrently
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Map results back to symbols
        symbol_results = {}
        for task, result in zip(tasks, results):
            if isinstance(result, Exception):
                symbol_results[task.symbol] = {
                    'success': False,
                    'error': f"Analysis failed: {str(result)}",
                    'symbol': task.symbol
                }
            else:
                symbol_results[task.symbol] = result
        
        return symbol_results
    
    async def _analyze_symbol_with_semaphore(
        self,
        task: SymbolAnalysisTask
    ) -> Dict[str, Any]:
        """Analyze a single symbol with concurrency control."""
        async with self._analysis_semaphore:
            try:
                # Apply timeout to prevent hanging
                return await asyncio.wait_for(
                    self._analyze_single_symbol(task),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                logger.error(f"Analysis timeout for {task.symbol}")
                return {
                    'success': False,
                    'error': f'Analysis timeout after {self.timeout}s',
                    'symbol': task.symbol
                }
            except Exception as e:
                logger.error(f"Error analyzing {task.symbol}: {e}")
                return {
                    'success': False,
                    'error': str(e),
                    'symbol': task.symbol
                }
    
    async def _analyze_single_symbol(
        self,
        task: SymbolAnalysisTask
    ) -> Dict[str, Any]:
        """
        Analyze a single symbol with all strategies.
        
        Creates isolated analyzer instances to avoid state conflicts.
        """
        from ..strategies.macd_rsi_analyzer import MACDRSIAnalyzer
        from ..strategies.sr_breakout_analyzer import SRBreakoutAnalyzer
        
        logger.debug(f"Starting analysis for {task.symbol}")
        
        # Create fresh analyzer instances (no shared state)
        analyzers = {
            'macd_rsi': MACDRSIAnalyzer(),
            'sr_breakout': SRBreakoutAnalyzer()
        }
        
        # Filter analyzers based on requested strategies
        active_analyzers = {
            name: analyzer 
            for name, analyzer in analyzers.items() 
            if name in task.strategies
        }
        
        # Analyze with each strategy in parallel
        strategy_results = await self._run_strategies_parallel(
            task.symbol,
            active_analyzers,
            task.start_date,
            task.end_date
        )
        
        # Aggregate results
        all_signals = []
        errors = []
        
        for strategy_name, result in strategy_results.items():
            if result.get('success', False):
                signals = result.get('signals', [])
                # Add strategy name to each signal
                for signal in signals:
                    signal['strategy'] = strategy_name
                all_signals.extend(signals)
            else:
                errors.append(f"{strategy_name}: {result.get('error', 'Unknown error')}")
        
        return {
            'success': True,
            'symbol': task.symbol,
            'signals': all_signals,
            'signal_count': len(all_signals),
            'errors': errors,
            'strategies_run': list(strategy_results.keys())
        }
    
    async def _run_strategies_parallel(
        self,
        symbol: str,
        analyzers: Dict[str, Any],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        """Run multiple strategies in parallel for a single symbol."""
        
        strategy_tasks = []
        
        for strategy_name, analyzer in analyzers.items():
            # Apply rate limiting before each API call
            await self._apply_rate_limit()
            
            # Create task for this strategy
            task = self._run_single_strategy(
                analyzer,
                symbol,
                start_date,
                end_date,
                strategy_name
            )
            strategy_tasks.append((strategy_name, task))
        
        # Execute all strategies concurrently
        results = {}
        
        for strategy_name, task in strategy_tasks:
            try:
                result = await task
                results[strategy_name] = {
                    'success': True,
                    'signals': result
                }
            except Exception as e:
                logger.error(f"Strategy {strategy_name} failed for {symbol}: {e}")
                results[strategy_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    async def _run_single_strategy(
        self,
        analyzer: Any,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        strategy_name: str
    ) -> List[Dict[str, Any]]:
        """Run a single strategy analysis."""
        
        logger.debug(f"Running {strategy_name} for {symbol}")
        
        # Call analyzer with consistent timeframe
        signals = await analyzer.analyze_symbol(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe='1d'  # EOD summary uses daily timeframe
        )
        
        return signals or []
    
    async def _apply_rate_limit(self):
        """Apply rate limiting to prevent API throttling."""
        async with self._api_lock:
            current_time = asyncio.get_event_loop().time()
            time_since_last = current_time - self._last_api_call
            
            if time_since_last < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - time_since_last)
            
            self._last_api_call = asyncio.get_event_loop().time()


class BatchedSummaryProcessor(ParallelSummaryProcessor):
    """
    Extended processor that handles very large symbol lists by batching.
    
    Useful when processing 50+ symbols to manage memory and API limits.
    """
    
    def __init__(
        self,
        batch_size: int = 20,
        **kwargs
    ):
        """
        Initialize with batching support.
        
        Args:
            batch_size: Number of symbols per batch
            **kwargs: Arguments for ParallelSummaryProcessor
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size
    
    async def process_symbols(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        strategies: List[str]
    ) -> Dict[str, Any]:
        """Process symbols in batches to manage resource usage."""
        
        if len(symbols) <= self.batch_size:
            # Small enough to process in one batch
            return await super().process_symbols(
                symbols, start_date, end_date, strategies
            )
        
        logger.info(
            f"Processing {len(symbols)} symbols in batches of {self.batch_size}"
        )
        
        # Process in batches
        all_results = {}
        
        for i in range(0, len(symbols), self.batch_size):
            batch = symbols[i:i + self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1}: {len(batch)} symbols")
            
            batch_results = await super().process_symbols(
                batch, start_date, end_date, strategies
            )
            
            all_results.update(batch_results)
            
            # Small delay between batches to prevent resource exhaustion
            if i + self.batch_size < len(symbols):
                await asyncio.sleep(1.0)
        
        return all_results