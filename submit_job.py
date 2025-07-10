#!/usr/bin/env python3
"""
Job submission script for trading analysis.

This script allows submitting trading analysis jobs to the job queue system.
Jobs will be processed by workers and results can be tracked.
"""

import argparse
import asyncio
import sys
import time
from datetime import datetime
from typing import Dict, Any

from src.jobs.queue import JobQueue
from src.jobs.models import JobRequest, JobStatus, JobResult


def print_colored(text: str, color: str = "white") -> None:
    """Print colored text to console."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")


def submit_job(symbol: str, strategy: str, timeframe: str, start_date: str, end_date: str) -> str:
    """Submit a job to the queue."""
    try:
        # Create job request with validation
        request = JobRequest(
            symbol=symbol.upper(),
            strategy=strategy,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Submit to queue
        queue = JobQueue()
        job = queue.submit_job(request)
        
        print_colored(f"✓ Job submitted successfully!", "green")
        print_colored(f"  Job ID: {job.id}", "cyan")
        print_colored(f"  Symbol: {request.symbol}", "white")
        print_colored(f"  Strategy: {strategy}", "white")
        print_colored(f"  Timeframe: {timeframe}", "white")
        print_colored(f"  Date range: {start_date} to {end_date}", "white")
        
        return job.id
        
    except ValueError as e:
        print_colored(f"✗ Invalid input: {str(e)}", "red")
        sys.exit(1)
    except Exception as e:
        print_colored(f"✗ Failed to submit job: {str(e)}", "red")
        sys.exit(1)


def track_job_status(job_id: str, timeout: int = 300) -> JobResult:
    """Track job status until completion or timeout."""
    queue = JobQueue()
    start_time = time.time()
    
    print_colored(f"\\nTracking job {job_id}...", "yellow")
    
    while True:
        job = queue.get_job(job_id)
        
        if not job:
            print_colored(f"✗ Job {job_id} not found", "red")
            sys.exit(1)
        
        elapsed_time = time.time() - start_time
        
        if job.status == JobStatus.PENDING:
            print_colored(f"⏳ Job is pending... ({elapsed_time:.1f}s)", "yellow")
        
        elif job.status == JobStatus.PROCESSING:
            worker_info = f" (worker: {job.worker_id})" if job.worker_id else ""
            print_colored(f"⚙️  Job is processing{worker_info}... ({elapsed_time:.1f}s)", "blue")
        
        elif job.status == JobStatus.COMPLETED:
            print_colored(f"✅ Job completed successfully! ({elapsed_time:.1f}s)", "green")
            return JobResult.from_job(job)
        
        elif job.status == JobStatus.FAILED:
            print_colored(f"❌ Job failed! ({elapsed_time:.1f}s)", "red")
            return JobResult.from_job(job)
        
        # Check timeout
        if elapsed_time > timeout:
            print_colored(f"⏰ Job tracking timed out after {timeout}s", "red")
            return JobResult.from_job(job)
        
        # Wait before next check
        time.sleep(2)


def display_job_result(result: JobResult) -> None:
    """Display job result in a friendly format."""
    print_colored("\\n" + "="*60, "cyan")
    print_colored("JOB RESULT", "cyan")
    print_colored("="*60, "cyan")
    
    # Basic info
    print_colored(f"Job ID: {result.job_id}", "white")
    print_colored(f"Symbol: {result.symbol}", "white")
    print_colored(f"Strategy: {result.strategy}", "white")
    print_colored(f"Timeframe: {result.timeframe}", "white")
    print_colored(f"Status: {result.status.value.upper()}", 
                  "green" if result.status == JobStatus.COMPLETED else "red")
    
    # Timing info
    if result.created_at:
        print_colored(f"Created: {result.created_at.strftime('%Y-%m-%d %H:%M:%S')}", "white")
    
    if result.completed_at:
        print_colored(f"Completed: {result.completed_at.strftime('%Y-%m-%d %H:%M:%S')}", "white")
    
    if result.duration:
        print_colored(f"Duration: {result.duration:.2f} seconds", "white")
    
    # Results
    if result.status == JobStatus.COMPLETED:
        print_colored(f"\\n📊 ANALYSIS RESULTS:", "green")
        print_colored(f"   Signals found: {len(result.signals)}", "white")
        
        if result.signals:
            print_colored(f"\\n   Latest signals:", "white")
            for i, signal in enumerate(result.signals[-3:], 1):  # Show last 3 signals
                signal_type = signal.get('signal', 'unknown')
                confidence = signal.get('confidence', 0)
                timestamp = signal.get('timestamp', 'unknown')
                
                color = "green" if signal_type.lower() == 'buy' else "red" if signal_type.lower() == 'sell' else "yellow"
                print_colored(f"   {i}. {signal_type.upper()} - Confidence: {confidence:.2f} - Time: {timestamp}", color)
        
        print_colored(f"\\n✅ Job completed successfully!", "green")
    
    elif result.status == JobStatus.FAILED:
        print_colored(f"\\n❌ JOB FAILED:", "red")
        print_colored(f"   Error: {result.error}", "red")
    
    elif result.status == JobStatus.PROCESSING:
        print_colored(f"\\n⚙️  Job is still processing...", "blue")
    
    else:
        print_colored(f"\\n⏳ Job is pending...", "yellow")
    
    print_colored("="*60, "cyan")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Submit trading analysis jobs")
    parser.add_argument("symbol", help="Trading symbol (e.g., AAPL)")
    parser.add_argument("strategy", help="Strategy name", 
                        choices=['rsi_trend', 'bollinger_breakout', 'macd_rsi', 'sr_breakout'])
    parser.add_argument("timeframe", help="Timeframe", choices=['1h', '1d'])
    parser.add_argument("start_date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("end_date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds (default: 300)")
    
    args = parser.parse_args()
    
    # Submit job
    print_colored("🚀 Submitting trading analysis job...", "cyan")
    job_id = submit_job(args.symbol, args.strategy, args.timeframe, args.start_date, args.end_date)
    
    # Track job status and display results
    result = track_job_status(job_id, args.timeout)
    display_job_result(result)


if __name__ == "__main__":
    main()