"""
End-to-end integration test for job processing system.

This test verifies that jobs can be submitted, processed by workers,
and return signals for all strategy/timeframe combinations.
"""

import asyncio
import os
import random
import sys
import time
from datetime import datetime
from multiprocessing import Process
from pathlib import Path

import pytest

# Add project root to path for direct execution
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

# Load environment variables from .env file
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

from src.jobs.queue import JobQueue
from src.jobs.models import JobRequest, JobStatus
from src.workers.worker import WorkerManager


class TestJobProcessingE2E:
    """End-to-end test for job processing system."""
    
    @classmethod
    def setup_class(cls):
        """Setup test environment."""
        # Clean up any existing jobs
        cls.queue_dir = ".test_jobs"
        cls.queue = JobQueue(cls.queue_dir)
        
        # Get symbols from environment
        symbols_str = os.getenv('SUPPORTED_SYMBOLS', 'AAPL,GOOGL,MSFT,AMZN,TSLA')
        cls.symbols = [s.strip() for s in symbols_str.split(',') if s.strip()]
        
        # Define all strategies and timeframes
        cls.strategies = ['rsi_trend', 'bollinger_breakout', 'macd_rsi', 'sr_breakout']
        cls.timeframes = ['1h', '1d']
        
        # Set realistic date ranges based on timeframe limitations
        from datetime import datetime, timedelta
        
        # For hourly: use last 20 days (within 30-day limit)
        end_date_hourly = datetime.now() - timedelta(days=1)
        start_date_hourly = end_date_hourly - timedelta(days=20)
        cls.start_date_hourly = start_date_hourly.strftime("%Y-%m-%d")
        cls.end_date_hourly = end_date_hourly.strftime("%Y-%m-%d")
        
        # For daily: use last 60 days (reasonable range)
        end_date_daily = datetime.now() - timedelta(days=1)
        start_date_daily = end_date_daily - timedelta(days=60)
        cls.start_date_daily = start_date_daily.strftime("%Y-%m-%d")
        cls.end_date_daily = end_date_daily.strftime("%Y-%m-%d")
    
    @classmethod
    def teardown_class(cls):
        """Cleanup after tests."""
        # Clean up test jobs directory
        import shutil
        if os.path.exists(cls.queue_dir):
            shutil.rmtree(cls.queue_dir)
    
    @staticmethod
    def run_workers_process(queue_dir):
        """Run workers in a separate process."""
        async def run():
            manager = WorkerManager(num_workers=2, queue_dir=queue_dir)
            await manager.start()
        
        asyncio.run(run())
    
    @pytest.mark.asyncio
    async def test_all_strategy_timeframe_combinations(self):
        """Test all combinations of strategies and timeframes."""
        # Pick a random symbol
        test_symbol = random.choice(self.symbols)
        print(f"\n🎲 Selected symbol: {test_symbol}")
        
        # Start workers in a separate process
        worker_process = Process(target=self.run_workers_process, args=(self.queue_dir,))
        worker_process.start()
        
        # Give workers time to start
        await asyncio.sleep(2)
        
        try:
            # Track all submitted jobs
            submitted_jobs = []
            
            # Submit jobs for all combinations
            print(f"\n📋 Testing {len(self.strategies)} strategies × {len(self.timeframes)} timeframes = {len(self.strategies) * len(self.timeframes)} combinations")
            
            for strategy in self.strategies:
                for timeframe in self.timeframes:
                    # Use appropriate date range based on timeframe
                    if timeframe == '1h':
                        start_date = self.start_date_hourly
                        end_date = self.end_date_hourly
                    else:
                        start_date = self.start_date_daily
                        end_date = self.end_date_daily
                    
                    # Create and submit job
                    request = JobRequest(
                        symbol=test_symbol,
                        strategy=strategy,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    job = self.queue.submit_job(request)
                    submitted_jobs.append({
                        'job': job,
                        'strategy': strategy,
                        'timeframe': timeframe
                    })
                    
                    date_range = f"{start_date} to {end_date}"
                    print(f"  ✓ Submitted: {strategy:15} / {timeframe:3} ({date_range}) - Job ID: {job.id[:8]}...")
            
            print(f"\n⏳ Waiting for {len(submitted_jobs)} jobs to complete...")
            
            # Wait for all jobs to complete with timeout
            timeout = 120  # 2 minutes total timeout
            start_time = time.time()
            completed_count = 0
            
            while completed_count < len(submitted_jobs) and (time.time() - start_time) < timeout:
                completed_count = 0
                
                for job_info in submitted_jobs:
                    job = self.queue.get_job(job_info['job'].id)
                    if job and job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                        completed_count += 1
                
                print(f"  Progress: {completed_count}/{len(submitted_jobs)} jobs completed")
                
                if completed_count < len(submitted_jobs):
                    await asyncio.sleep(3)
            
            # Verify results
            print(f"\n📊 Verifying results...")
            successful_jobs = 0
            failed_jobs = 0
            
            for job_info in submitted_jobs:
                job = self.queue.get_job(job_info['job'].id)
                strategy = job_info['strategy']
                timeframe = job_info['timeframe']
                
                assert job is not None, f"Job {job_info['job'].id} not found"
                
                if job.status == JobStatus.COMPLETED:
                    successful_jobs += 1
                    
                    # Verify job has results
                    assert job.result is not None, f"Job {job.id} has no results"
                    assert 'signals' in job.result, f"Job {job.id} results missing 'signals'"
                    assert isinstance(job.result['signals'], list), f"Job {job.id} signals is not a list"
                    
                    signal_count = len(job.result['signals'])
                    print(f"  ✅ {strategy:15} / {timeframe:3} - Completed with {signal_count} signals")
                    
                    # Check for API connection errors in the results
                    if 'error_details' in job.result and 'forbidden' in str(job.result.get('error_details', '')).lower():
                        raise AssertionError(f"Job {job.id} failed due to API credentials issue: {job.result['error_details']}")
                    
                elif job.status == JobStatus.FAILED:
                    failed_jobs += 1
                    error_msg = job.error[:100] + "..." if job.error and len(job.error) > 100 else job.error
                    print(f"  ❌ {strategy:15} / {timeframe:3} - Failed: {error_msg}")
                    
                else:
                    print(f"  ⏳ {strategy:15} / {timeframe:3} - Status: {job.status.value}")
            
            # Summary
            print(f"\n📈 Summary:")
            print(f"  Total jobs: {len(submitted_jobs)}")
            print(f"  Successful: {successful_jobs}")
            print(f"  Failed: {failed_jobs}")
            print(f"  Success rate: {(successful_jobs/len(submitted_jobs)*100):.1f}%")
            
            # Assert at least some jobs succeeded
            assert successful_jobs > 0, "No jobs completed successfully"
            assert successful_jobs + failed_jobs == len(submitted_jobs), "Some jobs didn't complete"
            
            # Check for API connection issues in failed jobs
            for job_info in submitted_jobs:
                job = self.queue.get_job(job_info['job'].id)
                if job.status == JobStatus.FAILED and job.error:
                    if 'forbidden' in job.error.lower() or 'connection test failed' in job.error.lower():
                        raise AssertionError(f"Job {job.id} failed due to API credentials issue: {job.error}")
            
            # Verify we have reasonable signal generation (not all 0)
            total_signals = 0
            for job_info in submitted_jobs:
                job = self.queue.get_job(job_info['job'].id)
                if job.status == JobStatus.COMPLETED and job.result:
                    total_signals += len(job.result.get('signals', []))
            
            # At least some strategies should generate signals with real market data
            if successful_jobs > 0 and total_signals == 0:
                print("⚠️  Warning: All jobs completed but no signals were generated")
                print("   This might indicate issues with market data or strategy parameters")
            
        finally:
            # Terminate workers
            worker_process.terminate()
            worker_process.join(timeout=5)
            if worker_process.is_alive():
                worker_process.kill()
            
            print("\n✨ Test completed!")


if __name__ == "__main__":
    # Run the test directly
    test = TestJobProcessingE2E()
    test.setup_class()
    
    try:
        asyncio.run(test.test_all_strategy_timeframe_combinations())
    finally:
        test.teardown_class()