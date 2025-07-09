"""
Job Processor for Signal Analysis Jobs

This module processes signal analysis job requests submitted through Redis streams,
executes the signal analysis, and stores results back in Redis for retrieval.
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
    Processes signal analysis jobs from Redis job queue.

    Listens to jobs_stream for new signal analysis requests, executes them using
    the existing backtest engine focused on signal generation, and stores results in Redis.
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
                "jobs_stream", self._consumer_group, id="0", mkstream=True
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
                raise ConnectionError(
                    "Alpaca provider indicates not connected after connect() call"
                )
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
            block=1000,  # Block for 1 second
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
                    self._redis_client.xack(
                        "jobs_stream", self._consumer_group, message_id
                    )
                    continue

                try:
                    # Process the job
                    await self._execute_job(job_id, job_data)

                    # Acknowledge message
                    self._redis_client.xack(
                        "jobs_stream", self._consumer_group, message_id
                    )

                except Exception as e:
                    logger.error(f"Failed to process job {job_id}: {e}")
                    # Store error in job status
                    self._update_job_status(job_id, "failed", error=str(e))
                    # Still acknowledge to prevent reprocessing
                    self._redis_client.xack(
                        "jobs_stream", self._consumer_group, message_id
                    )

    async def _execute_job(self, job_id: str, job_data: Dict[str, str]):
        """
        Execute a signal analysis job.

        Args:
            job_id: Unique job identifier
            job_data: Job parameters
        """
        logger.info(f"Executing signal analysis job: {job_id}")

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

            logger.info(
                f"Signal analysis job {job_id}: {symbol} {timeframe} {strategy_name} from {start_date} to {end_date}"
            )

            # Update progress
            self._update_job_status(
                job_id,
                "running",
                progress=10,
                message=f"Fetching historical data for {symbol}...",
            )

            # Fetch historical data
            timeframe_enum = self._parse_timeframe(timeframe)
            historical_data = await self._alpaca_provider.get_historical_data(
                symbol=symbol,
                timeframe=timeframe_enum,
                start_date=start_date,
                end_date=end_date,
            )

            if not historical_data:
                raise ValueError(f"No historical data available for {symbol}")

            # Update progress
            self._update_job_status(
                job_id,
                "running",
                progress=30,
                message=f"Loaded {len(historical_data)} data points",
            )

            # Create MarketData object
            market_data = MarketData(
                symbol=symbol,
                timeframe=timeframe_enum,
                data=historical_data,
                indicators={},
                last_updated=datetime.now(),
            )

            # Update progress
            self._update_job_status(
                job_id,
                "running",
                progress=40,
                message=f"Initializing {strategy_name} strategy...",
            )

            # Create strategy
            strategy = create_strategy(strategy_name)

            # Update progress
            self._update_job_status(
                job_id, "running", progress=50, message="Running signal analysis..."
            )

            # Run signal analysis (using backtest engine to generate signals)
            engine = BacktestEngine()
            result = engine.run_signal_analysis(strategy=strategy, data=market_data)

            # Update progress
            self._update_job_status(
                job_id,
                "running",
                progress=90,
                message="Processing signal analysis results...",
            )

            # Format results for signal analysis
            formatted_results = self._format_signal_analysis_results(
                result, symbol, timeframe, strategy_name, start_date, end_date
            )

            # Store results
            result_key = f"job:result:{job_id}"
            self._redis_client.hset(result_key, mapping=formatted_results)
            self._redis_client.expire(result_key, 86400)  # 24 hour TTL

            # Update status to completed
            self._update_job_status(
                job_id,
                "completed",
                progress=100,
                message="Signal analysis completed successfully",
            )

            logger.info(f"Signal analysis job {job_id} completed successfully")

        except Exception as e:
            logger.error(f"Signal analysis job {job_id} failed: {e}")
            self._update_job_status(
                job_id, "failed", error=str(e), message=f"Error: {str(e)}"
            )
            raise

    def _update_job_status(
        self,
        job_id: str,
        status: str,
        progress: int = None,
        message: str = None,
        error: str = None,
    ):
        """Update job status in Redis."""
        status_key = f"job:status:{job_id}"

        updates = {"status": status, "updated_at": datetime.now().isoformat()}

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
            "1M": TimeFrame.MONTHLY,
        }

        if timeframe not in mapping:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        return mapping[timeframe]

    def _format_signal_analysis_results(
        self,
        result,
        symbol: str,
        timeframe: str,
        strategy: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, str]:
        """Format signal analysis results for storage."""
        # Format signals with enhanced metadata
        signals_data = []
        for signal in result.signals:
            signal_dict = {
                "timestamp": signal.timestamp.isoformat(),
                "symbol": signal.symbol,
                "type": signal.signal_type.value,
                "price": signal.price,
                "confidence": signal.confidence,
                "strategy": signal.strategy_name,
                "metadata": signal.metadata,
            }
            signals_data.append(signal_dict)

        # Calculate signal quality metrics
        total_signals = len(signals_data)
        buy_signals = sum(1 for s in signals_data if s["type"] == "buy")
        sell_signals = sum(1 for s in signals_data if s["type"] == "sell")
        hold_signals = sum(1 for s in signals_data if s["type"] == "hold")

        # Calculate confidence statistics
        confidences = [
            float(s["confidence"]) for s in signals_data if s["confidence"] is not None
        ]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            min_confidence = min(confidences)
            max_confidence = max(confidences)
            high_confidence_count = sum(1 for c in confidences if c >= 0.7)
            medium_confidence_count = sum(1 for c in confidences if 0.4 <= c < 0.7)
            low_confidence_count = sum(1 for c in confidences if c < 0.4)
        else:
            avg_confidence = min_confidence = max_confidence = 0
            high_confidence_count = medium_confidence_count = low_confidence_count = 0

        # Calculate signal frequency
        if total_signals > 0:
            days = (end_date - start_date).days + 1
            signals_per_day = total_signals / days if days > 0 else 0
        else:
            signals_per_day = 0

        # Calculate quality score
        if total_signals > 0:
            quality_score = (
                high_confidence_count * 3
                + medium_confidence_count * 2
                + low_confidence_count * 1
            ) / total_signals
        else:
            quality_score = 0

        # Create result dictionary focused on signal analysis
        results = {
            # Job info
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            # Signal analysis metrics
            "total_signals": str(total_signals),
            "buy_signals": str(buy_signals),
            "sell_signals": str(sell_signals),
            "hold_signals": str(hold_signals),
            "buy_signal_pct": str(
                buy_signals / total_signals * 100 if total_signals > 0 else 0
            ),
            "sell_signal_pct": str(
                sell_signals / total_signals * 100 if total_signals > 0 else 0
            ),
            "hold_signal_pct": str(
                hold_signals / total_signals * 100 if total_signals > 0 else 0
            ),
            # Confidence metrics
            "avg_confidence": str(avg_confidence),
            "min_confidence": str(min_confidence),
            "max_confidence": str(max_confidence),
            "high_confidence_count": str(high_confidence_count),
            "medium_confidence_count": str(medium_confidence_count),
            "low_confidence_count": str(low_confidence_count),
            "high_confidence_pct": str(
                high_confidence_count / total_signals * 100 if total_signals > 0 else 0
            ),
            "medium_confidence_pct": str(
                medium_confidence_count / total_signals * 100
                if total_signals > 0
                else 0
            ),
            "low_confidence_pct": str(
                low_confidence_count / total_signals * 100 if total_signals > 0 else 0
            ),
            # Quality metrics
            "quality_score": str(quality_score),
            "signals_per_day": str(signals_per_day),
            # Signal data
            "signals": json.dumps(signals_data),
            "signals_count": str(len(signals_data)),
            # Timestamps
            "completed_at": datetime.now().isoformat(),
        }

        return results

    def _format_results(
        self,
        result,
        symbol: str,
        timeframe: str,
        strategy: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, str]:
        """Format backtest results for storage (legacy - kept for backward compatibility)."""
        # For backward compatibility, redirect to signal analysis results
        return self._format_signal_analysis_results(
            result, symbol, timeframe, strategy, start_date, end_date
        )
