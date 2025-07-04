"""Backtest service for running and managing backtests."""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio

from ..backtest.engine import BacktestEngine
from ..storage.results_store import ResultsStore
from ..core.models import TimeFrame, StrategyType

logger = logging.getLogger(__name__)


class BacktestService:
    """Service for managing backtest operations."""
    
    def __init__(self):
        """Initialize the backtest service."""
        self.engine = BacktestEngine()
        self.results_store = ResultsStore()
        self.running_jobs = {}  # Track async jobs
        logger.info("BacktestService initialized")
    
    async def run_backtest(
        self,
        symbol: str,
        timeframe: TimeFrame,
        strategy: StrategyType,
        parameters: Dict[str, Any],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        initial_capital: float = 10000.0,
        commission: float = 0.0  # Default to commission-free trading
    ) -> Dict[str, Any]:
        """
        Run a backtest for a specific strategy and symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            strategy: Strategy type
            parameters: Strategy parameters
            start_date: Optional start date
            end_date: Optional end date
            initial_capital: Initial capital
            commission: Commission rate (default: 0.0 for commission-free trading)
            
        Returns:
            Dict containing backtest results
        """
        try:
            logger.info(f"Running backtest for {symbol} ({timeframe}) using {strategy}")
            
            # Run backtest using engine
            results = self.engine.run_backtest(
                symbol=symbol,
                timeframe=timeframe,
                strategy=strategy,
                parameters=parameters,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                commission=commission
            )
            
            # Generate unique ID for this backtest
            backtest_id = str(uuid.uuid4())
            
            # Store results
            backtest_data = {
                'backtest_id': backtest_id,
                'symbol': symbol,
                'timeframe': timeframe.value,
                'strategy': strategy.value,
                'parameters': parameters,
                'initial_capital': initial_capital,
                'commission': commission,
                'created_at': datetime.utcnow(),
                'metrics': results['metrics'],
                'trades': results['trades'],
                'equity_curve': results['equity_curve']
            }
            
            self.results_store.store_backtest(backtest_id, backtest_data)
            
            logger.info(f"Backtest completed with {len(results['trades'])} trades")
            
            return {
                'backtest_id': backtest_id,
                'metrics': results['metrics'],
                'trades': results['trades'],
                'equity_curve': results['equity_curve']
            }
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise
    
    async def run_backtest_async(
        self,
        symbol: str,
        timeframe: TimeFrame,
        strategy: StrategyType,
        parameters: Dict[str, Any],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        initial_capital: float = 10000.0,
        commission: float = 0.0  # Default to commission-free trading
    ) -> str:
        """
        Run a backtest asynchronously.
        
        Returns:
            Job ID for tracking the backtest
        """
        job_id = str(uuid.uuid4())
        
        # Store job info
        self.running_jobs[job_id] = {
            'status': 'running',
            'started_at': datetime.utcnow(),
            'symbol': symbol,
            'timeframe': timeframe.value,
            'strategy': strategy.value
        }
        
        # Start async task
        asyncio.create_task(self._run_backtest_async_task(
            job_id, symbol, timeframe, strategy, parameters,
            start_date, end_date, initial_capital, commission
        ))
        
        return job_id
    
    async def _run_backtest_async_task(
        self,
        job_id: str,
        symbol: str,
        timeframe: TimeFrame,
        strategy: StrategyType,
        parameters: Dict[str, Any],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        initial_capital: float,
        commission: float
    ):
        """Internal async task for running backtests."""
        try:
            # Run the backtest
            result = await self.run_backtest(
                symbol, timeframe, strategy, parameters,
                start_date, end_date, initial_capital, commission
            )
            
            # Update job status
            self.running_jobs[job_id].update({
                'status': 'completed',
                'completed_at': datetime.utcnow(),
                'backtest_id': result['backtest_id'],
                'result': result
            })
            
        except Exception as e:
            logger.error(f"Error in async backtest {job_id}: {e}")
            self.running_jobs[job_id].update({
                'status': 'failed',
                'completed_at': datetime.utcnow(),
                'error': str(e)
            })
    
    async def get_backtest_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of an async backtest job."""
        return self.running_jobs.get(job_id)
    
    async def get_backtest_results(self, backtest_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed results for a specific backtest.
        
        Args:
            backtest_id: Backtest ID
            
        Returns:
            Backtest results or None if not found
        """
        try:
            logger.info(f"Getting backtest results for {backtest_id}")
            
            # Get results from store
            results = self.results_store.get_backtest(backtest_id)
            
            if not results:
                logger.warning(f"Backtest {backtest_id} not found")
                return None
            
            logger.info(f"Retrieved backtest results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting backtest results: {e}")
            raise
    
    async def list_backtest_results(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[TimeFrame] = None,
        strategy: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        List backtest results with optional filtering.
        
        Args:
            symbol: Optional symbol filter
            timeframe: Optional timeframe filter
            strategy: Optional strategy filter
            limit: Optional record limit
            offset: Optional record offset
            
        Returns:
            Dict containing list of backtests and total count
        """
        try:
            logger.info("Listing backtest results")
            
            # Get filtered results from store
            results = self.results_store.list_backtests(
                symbol=symbol,
                timeframe=timeframe,
                strategy=strategy,
                limit=limit,
                offset=offset
            )
            
            logger.info(f"Retrieved {len(results['backtests'])} backtest results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error listing backtest results: {e}")
            raise
    
    async def delete_backtest_results(self, backtest_id: str) -> bool:
        """
        Delete a specific backtest and its results.
        
        Args:
            backtest_id: Backtest ID to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            logger.info(f"Deleting backtest {backtest_id}")
            
            # Delete from store
            success = self.results_store.delete_backtest(backtest_id)
            
            if success:
                logger.info(f"Successfully deleted backtest {backtest_id}")
            else:
                logger.warning(f"Backtest {backtest_id} not found for deletion")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting backtest results: {e}")
            raise
    
    async def delete_multiple_backtest_results(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[TimeFrame] = None,
        strategy: Optional[str] = None,
        older_than_days: Optional[int] = None
    ) -> int:
        """
        Delete multiple backtest results based on filters.
        
        Args:
            symbol: Optional symbol filter
            timeframe: Optional timeframe filter
            strategy: Optional strategy filter
            older_than_days: Optional age filter
            
        Returns:
            Number of backtests deleted
        """
        try:
            logger.info("Deleting multiple backtest results")
            
            # Build filters
            filters = {}
            if symbol:
                filters['symbol'] = symbol
            if timeframe:
                filters['timeframe'] = timeframe.value
            if strategy:
                filters['strategy'] = strategy
            if older_than_days:
                cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
                filters['older_than'] = cutoff_date
            
            # Delete from store
            deleted_count = self.results_store.delete_multiple_backtests(filters)
            
            logger.info(f"Deleted {deleted_count} backtest results")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting multiple backtest results: {e}")
            raise
    
    async def export_backtest_results(
        self,
        backtest_id: str,
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        Export backtest results in various formats.
        
        Args:
            backtest_id: Backtest ID
            format: Export format (json, csv, xlsx)
            
        Returns:
            Exported data
        """
        try:
            logger.info(f"Exporting backtest {backtest_id} in {format} format")
            
            # Get results
            results = await self.get_backtest_results(backtest_id)
            if not results:
                raise ValueError(f"Backtest {backtest_id} not found")
            
            # Export based on format
            if format.lower() == 'json':
                return results
            elif format.lower() == 'csv':
                # Convert trades to CSV format
                trades_data = []
                for trade in results['trades']:
                    trades_data.append({
                        'entry_time': trade['entry_time'],
                        'exit_time': trade.get('exit_time', ''),
                        'entry_price': trade['entry_price'],
                        'exit_price': trade.get('exit_price', ''),
                        'quantity': trade['quantity'],
                        'side': trade['side'],
                        'pnl': trade.get('pnl', ''),
                        'return_pct': trade.get('return_pct', '')
                    })
                return {
                    'trades': trades_data,
                    'metrics': results['metrics']
                }
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
        except Exception as e:
            logger.error(f"Error exporting backtest results: {e}")
            raise
    
    async def get_results_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics across all backtest results.
        
        Returns:
            Summary statistics
        """
        try:
            logger.info("Getting results summary")
            
            # Get summary from store
            summary = self.results_store.get_summary()
            
            logger.info("Retrieved results summary")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting results summary: {e}")
            raise