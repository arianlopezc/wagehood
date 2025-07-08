"""
Job Processor for Backtest Jobs

This module processes backtest job requests submitted through Redis streams,
executes the backtests, and stores results back in Redis for retrieval.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import uuid
import traceback

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from src.storage.cache import cache_manager
from src.data.providers.alpaca_provider import AlpacaProvider
from src.strategies import create_strategy
from src.backtest.engine import BacktestEngine
from src.core.models import MarketData, TimeFrame, OHLCV

logger = logging.getLogger(__name__)


class JobProcessor:
    """
    Processes backtest jobs from Redis job queue.
    
    Listens to jobs_stream for new backtest requests, executes them using
    the existing backtest engine, and stores results in Redis.
    """
    
    def __init__(self, redis_client: redis.Redis = None):
        """
        Initialize the job processor.
        
        Args:
            redis_client: Redis client instance (uses cache_manager if not provided)
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis required for job processing")
            
        self._redis_client = redis_client or cache_manager._redis_client
        self._running = False
        self._consumer_group = "job_processors"
        self._consumer_name = f"processor_{uuid.uuid4().hex[:8]}"
        
        # Initialize Alpaca provider for historical data
        self._alpaca_provider = None
        
        logger.info(f"Job processor initialized: {self._consumer_name}")
    
    async def start(self):
        """Start processing jobs from the queue."""
        self._running = True
        
        # Create consumer group
        try:
            self._redis_client.xgroup_create(
                "jobs_stream", 
                self._consumer_group, 
                id="0",
                mkstream=True
            )
            logger.info(f"Created consumer group: {self._consumer_group}")
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.debug("Consumer group already exists")
            else:
                raise
        
        # Initialize Alpaca provider
        try:
            logger.info("Initializing Alpaca provider for job processing...")
            self._alpaca_provider = AlpacaProvider()
            logger.info("Attempting to connect to Alpaca...")
            connection_result = await self._alpaca_provider.connect()
            logger.info(f"Alpaca connection result: {connection_result}")
            if self._alpaca_provider._connected:
                logger.info("✅ Connected to Alpaca for historical data")
            else:
                raise ConnectionError("Alpaca provider indicates not connected after connect() call")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Alpaca: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        # Start processing loop
        logger.info("Starting job processing loop...")
        while self._running:
            try:
                await self._process_next_job()
                await asyncio.sleep(0.1)  # Small delay between checks
            except Exception as e:
                logger.error(f"Error in job processing loop: {e}")
                await asyncio.sleep(1)
    
    async def stop(self):
        """Stop processing jobs."""
        self._running = False
        if self._alpaca_provider:
            await self._alpaca_provider.disconnect()
        logger.info("Job processor stopped")
    
    async def _process_next_job(self):
        """Process the next job from the queue."""
        # Read next job from stream
        messages = self._redis_client.xreadgroup(
            self._consumer_group,
            self._consumer_name,
            {"jobs_stream": ">"},
            count=1,
            block=1000  # Block for 1 second
        )
        
        if not messages:
            return
        
        for stream_name, stream_messages in messages:
            for message_id, data in stream_messages:
                # Parse job data
                job_data = {k.decode(): v.decode() for k, v in data.items()}
                job_id = job_data.get("job_id")
                
                if not job_id:
                    logger.error(f"Job missing job_id: {data}")
                    self._redis_client.xack("jobs_stream", self._consumer_group, message_id)
                    continue
                
                try:
                    # Process the job
                    await self._execute_job(job_id, job_data)
                    
                    # Acknowledge message
                    self._redis_client.xack("jobs_stream", self._consumer_group, message_id)
                    
                except Exception as e:
                    logger.error(f"Failed to process job {job_id}: {e}")
                    # Store error in job status
                    self._update_job_status(job_id, "failed", error=str(e))
                    # Still acknowledge to prevent reprocessing
                    self._redis_client.xack("jobs_stream", self._consumer_group, message_id)
    
    async def _execute_job(self, job_id: str, job_data: Dict[str, str]):
        """
        Execute a backtest job.
        
        Args:
            job_id: Unique job identifier
            job_data: Job parameters
        """
        logger.info(f"Executing job: {job_id}")
        
        # Update status to running
        self._update_job_status(job_id, "running", progress=0)
        
        try:
            # Parse job parameters
            params = json.loads(job_data.get("params", "{}"))
            symbol = params.get("symbol")
            timeframe = params.get("timeframe", "1d")
            strategy_name = params.get("strategy")
            start_date = datetime.fromisoformat(params.get("start_date"))
            end_date = datetime.fromisoformat(params.get("end_date"))
            
            logger.info(f"Job {job_id}: {symbol} {timeframe} {strategy_name} from {start_date} to {end_date}")
            
            # Update progress
            self._update_job_status(job_id, "running", progress=10, 
                                  message=f"Fetching historical data for {symbol}...")
            
            # Fetch historical data
            timeframe_enum = self._parse_timeframe(timeframe)
            historical_data = await self._alpaca_provider.get_historical_data(
                symbol=symbol,
                timeframe=timeframe_enum,
                start_date=start_date,
                end_date=end_date
            )
            
            if not historical_data:
                raise ValueError(f"No historical data available for {symbol}")
            
            # Update progress
            self._update_job_status(job_id, "running", progress=30,
                                  message=f"Loaded {len(historical_data)} data points")
            
            # Create MarketData object
            market_data = MarketData(
                symbol=symbol,
                timeframe=timeframe_enum,
                data=historical_data,
                indicators={},
                last_updated=datetime.now()
            )
            
            # Update progress
            self._update_job_status(job_id, "running", progress=40,
                                  message=f"Initializing {strategy_name} strategy...")
            
            # Create strategy
            strategy = create_strategy(strategy_name)
            
            # Update progress
            self._update_job_status(job_id, "running", progress=50,
                                  message="Running backtest...")
            
            # Run backtest
            engine = BacktestEngine()
            result = engine.run_backtest(
                strategy=strategy,
                data=market_data,
                initial_capital=10000  # Default capital
            )
            
            # Update progress
            self._update_job_status(job_id, "running", progress=90,
                                  message="Processing results...")
            
            # Format results
            formatted_results = self._format_results(result, symbol, timeframe, strategy_name, start_date, end_date)
            
            # Store results
            result_key = f"job:result:{job_id}"
            self._redis_client.hset(result_key, mapping=formatted_results)
            self._redis_client.expire(result_key, 86400)  # 24 hour TTL
            
            # Update status to completed
            self._update_job_status(job_id, "completed", progress=100,
                                  message="Backtest completed successfully")
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            self._update_job_status(job_id, "failed", error=str(e),
                                  message=f"Error: {str(e)}")
            raise
    
    def _update_job_status(self, job_id: str, status: str, progress: int = None,
                          message: str = None, error: str = None):
        """Update job status in Redis."""
        status_key = f"job:status:{job_id}"
        
        updates = {
            "status": status,
            "updated_at": datetime.now().isoformat()
        }
        
        if progress is not None:
            updates["progress"] = str(progress)
        if message:
            updates["message"] = message
        if error:
            updates["error"] = error
        
        self._redis_client.hset(status_key, mapping=updates)
        self._redis_client.expire(status_key, 86400)  # 24 hour TTL
        
        logger.debug(f"Updated job {job_id} status: {status} (progress: {progress})")
    
    def _parse_timeframe(self, timeframe: str) -> TimeFrame:
        """Parse timeframe string to TimeFrame enum."""
        mapping = {
            "1m": TimeFrame.MINUTE_1,
            "5m": TimeFrame.MINUTE_5,
            "15m": TimeFrame.MINUTE_15,
            "30m": TimeFrame.MINUTE_30,
            "1h": TimeFrame.HOUR_1,
            "4h": TimeFrame.HOUR_4,
            "1d": TimeFrame.DAILY,
            "1w": TimeFrame.WEEKLY,
            "1M": TimeFrame.MONTHLY
        }
        
        if timeframe not in mapping:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        return mapping[timeframe]
    
    def _format_results(self, result, symbol: str, timeframe: str, strategy: str,
                       start_date: datetime, end_date: datetime) -> Dict[str, str]:
        """Format backtest results for storage."""
        metrics = result.performance_metrics
        
        # Format trades/signals
        trades_data = []
        for trade in result.trades:
            # Get P&L - use calculated pnl or calculate it
            if hasattr(trade, 'pnl') and trade.pnl is not None:
                pnl = trade.pnl
            elif trade.exit_price and trade.exit_time:
                # Calculate P&L for closed trades
                pnl = (trade.exit_price - trade.entry_price) * abs(trade.quantity)
            else:
                # Open trade - no P&L yet
                pnl = 0.0
            
            # Calculate P&L percentage
            pnl_pct = 0.0
            if trade.entry_price > 0:
                pnl_pct = (pnl / (trade.entry_price * abs(trade.quantity))) * 100
            
            # Determine trade side (long if positive quantity, short if negative)
            side = "LONG" if trade.quantity > 0 else "SHORT"
            
            # Determine status
            status = "CLOSED" if trade.exit_time else "OPEN"
            
            trade_dict = {
                "entry_date": trade.entry_time.isoformat(),
                "exit_date": trade.exit_time.isoformat() if trade.exit_time else None,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "quantity": trade.quantity,
                "side": side,
                "profit_loss": pnl,
                "profit_loss_pct": pnl_pct,
                "status": status
            }
            trades_data.append(trade_dict)
        
        # Format signals
        signals_data = []
        for signal in result.signals:
            signal_dict = {
                "timestamp": signal.timestamp.isoformat(),
                "symbol": signal.symbol,
                "type": signal.signal_type.value,
                "price": signal.price,
                "confidence": signal.confidence,
                "strategy": signal.strategy_name,
                "metadata": signal.metadata
            }
            signals_data.append(signal_dict)
        
        # Create result dictionary
        results = {
            # Job info
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            
            # Performance metrics
            "total_return": str(metrics.total_pnl),
            "total_return_pct": str(metrics.total_return_pct),
            "win_rate": str(metrics.win_rate),
            "sharpe_ratio": str(metrics.sharpe_ratio),
            "max_drawdown": str(metrics.max_drawdown),
            "max_drawdown_pct": str(metrics.max_drawdown_pct),
            "total_trades": str(metrics.total_trades),
            "winning_trades": str(metrics.winning_trades),
            "losing_trades": str(metrics.losing_trades),
            "profit_factor": str(metrics.profit_factor),
            
            # Trade and signal data
            "trades": json.dumps(trades_data),
            "signals": json.dumps(signals_data),
            "trades_count": str(len(trades_data)),
            "signals_count": str(len(signals_data)),
            
            # Timestamps
            "completed_at": datetime.now().isoformat()
        }
        
        return results