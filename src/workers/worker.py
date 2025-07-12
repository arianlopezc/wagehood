"""
Worker process for executing trading analysis jobs.
"""

import asyncio
import logging
import os
import signal
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from src.jobs.queue import JobQueue
from src.jobs.models import Job, JobStatus

# Import strategy analyzers
from src.strategies.rsi_trend_analyzer import RSITrendAnalyzer
from src.strategies.bollinger_breakout_analyzer import BollingerBreakoutAnalyzer
from src.strategies.macd_rsi_analyzer import MACDRSIAnalyzer
from src.strategies.sr_breakout_analyzer import SRBreakoutAnalyzer


def load_env_file():
    """Load environment variables from .env file."""
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    except FileNotFoundError:
        pass

# Load environment at module level
load_env_file()


class JobWorker:
    """Worker process that executes jobs from the queue."""
    
    def __init__(self, worker_id: Optional[str] = None, queue_dir: str = ".jobs"):
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.queue = JobQueue(queue_dir)
        self.running = False
        self.current_job = None
        
        # Lazy load analyzers
        self._analyzers = None
        
        # Adaptive polling intervals
        self.min_poll_interval = 0.1
        self.max_poll_interval = 5.0
        self.current_poll_interval = self.min_poll_interval
        
        # Setup logging with proper handler
        self.logger = self._setup_logger()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup proper logger with handler."""
        logger = logging.getLogger(f"{__name__}.{self.worker_id}")
        logger.setLevel(logging.INFO)
        
        # Add handler only if not already present
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {self.worker_id} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    @property
    def analyzers(self):
        """Lazy load analyzers."""
        if self._analyzers is None:
            self._analyzers = self._load_analyzers()
        return self._analyzers
    
    def _load_analyzers(self) -> Dict[str, Any]:
        """Load analyzers dynamically."""
        return {
            'rsi_trend': RSITrendAnalyzer(),
            'bollinger_breakout': BollingerBreakoutAnalyzer(),
            'macd_rsi': MACDRSIAnalyzer(),
            'sr_breakout': SRBreakoutAnalyzer()
        }
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    async def _execute_job(self, job: Job) -> Dict[str, Any]:
        """Execute a single job using the appropriate analyzer."""
        request = job.request
        
        # Get the appropriate analyzer
        analyzer = self.analyzers.get(request.strategy)
        if not analyzer:
            raise ValueError(f"Unknown strategy: {request.strategy}")
        
        self.logger.info(f"Executing job {job.id}: {request.symbol} {request.strategy} {request.timeframe}")
        
        # Execute the analysis
        try:
            # Convert date strings to datetime objects
            start_date = datetime.strptime(request.start_date, '%Y-%m-%d')
            end_date = datetime.strptime(request.end_date, '%Y-%m-%d')
            
            signals = await analyzer.analyze_symbol(
                symbol=request.symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=request.timeframe
            )
            
            # Convert signals to JSON-serializable format
            serializable_signals = []
            for signal in signals:
                if isinstance(signal, dict):
                    # Convert any pandas Timestamp objects to ISO format strings
                    serializable_signal = {}
                    for key, value in signal.items():
                        if hasattr(value, 'isoformat'):  # pandas Timestamp or datetime
                            serializable_signal[key] = value.isoformat()
                        else:
                            serializable_signal[key] = value
                    serializable_signals.append(serializable_signal)
                else:
                    serializable_signals.append(signal)
            
            # Format results
            result = {
                'job_id': job.id,
                'symbol': request.symbol,
                'strategy': request.strategy,
                'timeframe': request.timeframe,
                'start_date': request.start_date,
                'end_date': request.end_date,
                'signals': serializable_signals,
                'signal_count': len(serializable_signals),
                'executed_at': time.time(),
                'worker_id': self.worker_id
            }
            
            self.logger.info(f"Job {job.id} completed successfully: {len(signals)} signals generated")
            return result
            
        except Exception as e:
            self.logger.error(f"Job {job.id} failed with error: {str(e)}")
            # Add context to exception
            raise type(e)(f"Job {job.id} execution failed: {str(e)}") from e
    
    async def _process_job(self, job: Job) -> None:
        """Process a single job."""
        try:
            result = await self._execute_job(job)
            self.queue.complete_job(job.id, result)
            
        except Exception as e:
            error_msg = f"Job execution failed: {str(e)}"
            self.queue.fail_job(job.id, error_msg)
            self.logger.error(f"Job {job.id} failed: {error_msg}")
        
        finally:
            self.current_job = None
    
    async def run(self) -> None:
        """Main worker loop with adaptive polling."""
        self.running = True
        self.logger.info(f"Worker {self.worker_id} started")
        
        consecutive_empty_polls = 0
        
        try:
            while self.running:
                try:
                    # Get next job from queue
                    job = self.queue.get_next_job(self.worker_id)
                    
                    if job:
                        self.current_job = job
                        consecutive_empty_polls = 0
                        self.current_poll_interval = self.min_poll_interval
                        
                        await self._process_job(job)
                    else:
                        # Adaptive polling - increase interval when no jobs
                        consecutive_empty_polls += 1
                        if consecutive_empty_polls > 10:
                            self.current_poll_interval = min(
                                self.current_poll_interval * 1.5,
                                self.max_poll_interval
                            )
                        
                        await asyncio.sleep(self.current_poll_interval)
                        
                except Exception as e:
                    self.logger.error(f"Unexpected error in worker loop: {str(e)}")
                    await asyncio.sleep(1)
        
        finally:
            self.logger.info(f"Worker {self.worker_id} stopped")
    
    def stop(self) -> None:
        """Stop the worker."""
        self.running = False
        if self.current_job:
            self.logger.info(f"Stopping worker with job {self.current_job.id} in progress")


class WorkerManager:
    """Manager for multiple worker processes."""
    
    def __init__(self, num_workers: int = 2, queue_dir: str = ".jobs"):
        self.num_workers = num_workers
        self.queue_dir = queue_dir
        self.workers = []
        self.running = False
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - WorkerManager - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down workers...")
        self.running = False
        for worker in self.workers:
            worker.stop()
    
    async def start(self) -> None:
        """Start all worker processes."""
        self.running = True
        self.logger.info(f"Starting {self.num_workers} workers")
        
        # Create and start workers
        tasks = []
        for i in range(self.num_workers):
            worker = JobWorker(f"worker-{i+1}", self.queue_dir)
            self.workers.append(worker)
            tasks.append(asyncio.create_task(worker.run()))
        
        # Wait for all workers to complete
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt, shutting down...")
        finally:
            self.logger.info("All workers stopped")
    
    def stop(self) -> None:
        """Stop all workers."""
        self.running = False
        for worker in self.workers:
            worker.stop()


async def main():
    """Main entry point for running workers."""
    manager = WorkerManager(num_workers=2)
    await manager.start()


if __name__ == "__main__":
    asyncio.run(main())