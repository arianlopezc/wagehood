"""Wagehood CLI main entry point."""

import asyncio
import os
import sys
import click
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import ConfigManager
from ..jobs.models import JobRequest, JobStatus
from ..jobs.queue import JobQueue
from ..workers.worker import WorkerManager


def setup_env():
    """Set up environment variables from CLI configuration."""
    config_manager = ConfigManager()
    env_vars = config_manager.get_env_vars()
    
    for key, value in env_vars.items():
        os.environ[key] = value


@click.group()
@click.pass_context
def cli(ctx):
    """Wagehood - Trading Strategy Analysis CLI"""
    # Ensure configuration is available
    config_manager = ConfigManager()
    ctx.obj = config_manager
    
    # Set up environment variables for all commands
    setup_env()


@cli.command()
@click.pass_obj
def configure(config_manager: ConfigManager):
    """Configure API credentials and symbols."""
    click.echo("🔧 Wagehood Configuration")
    click.echo("=" * 40)
    
    # Get API credentials
    api_key = click.prompt("Enter your Alpaca API key", hide_input=True)
    secret_key = click.prompt("Enter your Alpaca secret key", hide_input=True)
    
    # Get symbols
    symbols_input = click.prompt(
        "Enter supported symbols (comma-separated)",
        default="AAPL,GOOGL,MSFT,AMZN,TSLA"
    )
    symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
    
    # Save configuration
    config_manager.update({
        'api_key': api_key,
        'secret_key': secret_key,
        'symbols': symbols
    })
    
    click.echo(f"\n✅ Configuration saved successfully!")
    click.echo(f"📊 Tracking {len(symbols)} symbols: {', '.join(symbols[:5])}" + 
               (" ..." if len(symbols) > 5 else ""))


@cli.command()
@click.option('--symbol', '-s', required=True, help='Trading symbol (e.g., AAPL)')
@click.option('--strategy', '-st', required=True, 
              type=click.Choice(['rsi_trend', 'bollinger_breakout', 'macd_rsi', 'sr_breakout']),
              help='Trading strategy')
@click.option('--timeframe', '-t', required=True,
              type=click.Choice(['1h', '1d']),
              help='Timeframe for analysis')
@click.option('--start-date', '-sd', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end-date', '-ed', required=True, help='End date (YYYY-MM-DD)')
@click.pass_obj
def submit(config_manager: ConfigManager, symbol: str, strategy: str, 
           timeframe: str, start_date: str, end_date: str):
    """Submit a new analysis job."""
    if not config_manager.is_configured():
        click.echo("❌ CLI not configured. Run 'wagehood configure' first.")
        return
    
    try:
        # Create job request
        request = JobRequest(
            symbol=symbol.upper(),
            strategy=strategy,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Submit job
        queue = JobQueue()
        job = queue.submit_job(request)
        
        click.echo(f"✅ Job submitted successfully!")
        click.echo(f"📋 Job ID: {job.id}")
        click.echo(f"📊 Analysis: {symbol} / {strategy} / {timeframe}")
        click.echo(f"📅 Period: {start_date} to {end_date}")
        
        # Wait for job completion and show results
        click.echo(f"\n⏳ Waiting for job to complete...")
        
        import time
        timeout = 120  # 2 minutes timeout
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            # Check job status
            current_job = queue.get_job(job.id)
            if not current_job:
                click.echo("❌ Job not found", err=True)
                sys.exit(1)
            
            if current_job.status == JobStatus.COMPLETED:
                click.echo(f"✅ Job completed successfully!")
                
                if current_job.result:
                    signals = current_job.result.get('signals', [])
                    signal_count = len(signals)
                    
                    click.echo(f"\n📊 Results:")
                    click.echo(f"   Signals found: {signal_count}")
                    click.echo(f"   Worker: {current_job.result.get('worker_id', 'unknown')}")
                    click.echo(f"   Execution time: {current_job.result.get('executed_at', 'unknown')}")
                    
                    if signals:
                        click.echo(f"\n📈 Signals:")
                        for i, signal in enumerate(signals[:10], 1):  # Show first 10 signals
                            if isinstance(signal, dict):
                                timestamp = signal.get('timestamp', 'N/A')
                                signal_type = signal.get('signal_type', 'N/A')  # Fixed: was 'signal', should be 'signal_type'
                                price = signal.get('price', 'N/A')
                                confidence = signal.get('confidence', 'N/A')
                                click.echo(f"   {i}. {timestamp} | {signal_type} | Price: {price} | Confidence: {confidence}")
                            else:
                                click.echo(f"   {i}. {signal}")
                        
                        if signal_count > 10:
                            click.echo(f"   ... and {signal_count - 10} more signals")
                        
                        click.echo(f"\n💡 Tip: Use 'wagehood jobs' to see all job details")
                    else:
                        click.echo(f"\n📭 No signals generated for this period")
                else:
                    click.echo(f"⚠️  Job completed but no results found")
                
                return
                
            elif current_job.status == JobStatus.FAILED:
                click.echo(f"❌ Job failed!")
                if current_job.error:
                    click.echo(f"   Error: {current_job.error}")
                sys.exit(1)
            
            elif current_job.status == JobStatus.PROCESSING:
                click.echo(f"🔄 Job is being processed...")
            
            # Wait before checking again
            time.sleep(3)
        
        # Timeout reached
        click.echo(f"\n⏰ Timeout reached after {timeout} seconds")
        click.echo(f"   Job is still {current_job.status.value if current_job else 'unknown'}")
        click.echo(f"   Use 'wagehood jobs' to check status later")
        sys.exit(1)
        
    except Exception as e:
        click.echo(f"❌ Error submitting job: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--status', '-s', help='Filter by status (pending/processing/completed/failed)')
@click.option('--limit', '-l', default=10, help='Number of jobs to show')
@click.pass_obj
def jobs(config_manager: ConfigManager, status: Optional[str], limit: int):
    """List all jobs."""
    if not config_manager.is_configured():
        click.echo("❌ CLI not configured. Run 'wagehood configure' first.")
        return
    
    try:
        queue = JobQueue()
        
        # Get jobs by status or all jobs
        if status:
            status_enum = getattr(JobStatus, status.upper(), None)
            if not status_enum:
                click.echo(f"❌ Invalid status: {status}. Valid options: pending, processing, completed, failed", err=True)
                return
            all_jobs = queue.get_jobs_by_status(status_enum)
        else:
            # Get all jobs by combining all statuses
            all_jobs = []
            for job_status in [JobStatus.PENDING, JobStatus.PROCESSING, JobStatus.COMPLETED, JobStatus.FAILED]:
                all_jobs.extend(queue.get_jobs_by_status(job_status))
        
        # Sort by creation time (newest first) and limit results
        all_jobs = sorted(all_jobs, key=lambda j: j.created_at, reverse=True)[:limit]
        
        if not all_jobs:
            click.echo("📭 No jobs found.")
            return
        
        stats = queue.get_queue_stats()
        total_jobs = sum(stats.values())
        click.echo(f"\n📋 Jobs (showing {len(all_jobs)} of {total_jobs} total)")
        click.echo("=" * 80)
        
        for job in all_jobs:
            status_icon = {
                'pending': '⏳',
                'processing': '🔄',
                'completed': '✅',
                'failed': '❌'
            }.get(job.status.value, '❓')
            
            click.echo(f"\n{status_icon} Job: {job.id[:8]}...")
            click.echo(f"   Symbol: {job.request.symbol} | Strategy: {job.request.strategy}")
            click.echo(f"   Timeframe: {job.request.timeframe} | Status: {job.status.value}")
            click.echo(f"   Period: {job.request.start_date} to {job.request.end_date}")
            
            if job.status.value == 'completed' and job.result:
                signal_count = job.result.get('signal_count', 0)
                click.echo(f"   Signals: {signal_count}")
            elif job.status.value == 'failed' and job.error:
                click.echo(f"   Error: {job.error[:60]}...")
        
    except Exception as e:
        click.echo(f"❌ Error listing jobs: {str(e)}", err=True)
        sys.exit(1)


@cli.group()
def symbols():
    """Manage tracked symbols."""
    pass


@symbols.command('list')
@click.pass_obj
def list_symbols(config_manager: ConfigManager):
    """List all tracked symbols."""
    symbols = config_manager.get('symbols', [])
    if not symbols:
        click.echo("📭 No symbols configured.")
        return
    
    click.echo(f"📊 Tracking {len(symbols)} symbols:")
    for i, symbol in enumerate(symbols, 1):
        click.echo(f"  {i}. {symbol}")


@symbols.command('add')
@click.argument('symbol')
@click.pass_obj
def add_symbol(config_manager: ConfigManager, symbol: str):
    """Add a new symbol to track."""
    symbols = config_manager.get('symbols', [])
    symbol = symbol.upper()
    
    if symbol in symbols:
        click.echo(f"⚠️  Symbol {symbol} is already being tracked.")
        return
    
    symbols.append(symbol)
    config_manager.set('symbols', symbols)
    click.echo(f"✅ Added {symbol} to tracked symbols.")


@symbols.command('remove')
@click.argument('symbol')
@click.pass_obj
def remove_symbol(config_manager: ConfigManager, symbol: str):
    """Remove a symbol from tracking."""
    symbols = config_manager.get('symbols', [])
    symbol = symbol.upper()
    
    if symbol not in symbols:
        click.echo(f"⚠️  Symbol {symbol} is not being tracked.")
        return
    
    symbols.remove(symbol)
    config_manager.set('symbols', symbols)
    click.echo(f"✅ Removed {symbol} from tracked symbols.")


@cli.command()
@click.pass_obj
def test(config_manager: ConfigManager):
    """Run all integration tests."""
    if not config_manager.is_configured():
        click.echo("❌ CLI not configured. Run 'wagehood configure' first.")
        return
    
    click.echo("🧪 Running integration tests...")
    
    # Find project root
    project_root = Path(__file__).parent.parent.parent
    test_script = project_root / "run_all_tests.py"
    
    if not test_script.exists():
        click.echo(f"❌ Test script not found at {test_script}")
        return
    
    try:
        # Run tests
        result = subprocess.run(
            [sys.executable, str(test_script)],
            cwd=project_root,
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            click.echo("\n✅ All tests passed!")
        else:
            click.echo(f"\n❌ Tests failed with exit code: {result.returncode}")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error running tests: {str(e)}", err=True)
        sys.exit(1)


@cli.group()
def workers():
    """Manage worker processes."""
    pass


@workers.command('start')
@click.option('--num-workers', '-n', default=2, help='Number of workers to start')
@click.pass_obj
def start_workers(config_manager: ConfigManager, num_workers: int):
    """Start worker processes."""
    if not config_manager.is_configured():
        click.echo("❌ CLI not configured. Run 'wagehood configure' first.")
        return
    
    click.echo(f"🚀 Starting {num_workers} workers...")
    
    # Check if workers are already running
    pid_file = Path.home() / '.wagehood' / 'workers.pid'
    if pid_file.exists():
        click.echo("⚠️  Workers may already be running. Use 'wagehood workers restart' to restart.")
        return
    
    # Start workers in background
    project_root = Path(__file__).parent.parent.parent
    worker_script = project_root / "start_workers.py"
    
    try:
        # Start worker process
        process = subprocess.Popen(
            [sys.executable, str(worker_script)],
            cwd=project_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        
        # Save PID
        pid_file.write_text(str(process.pid))
        
        click.echo(f"✅ Workers started with PID: {process.pid}")
        
    except Exception as e:
        click.echo(f"❌ Error starting workers: {str(e)}", err=True)
        sys.exit(1)


@workers.command('stop')
@click.pass_obj
def stop_workers(config_manager: ConfigManager):
    """Stop worker processes."""
    pid_file = Path.home() / '.wagehood' / 'workers.pid'
    
    if not pid_file.exists():
        click.echo("⚠️  No workers are running.")
        return
    
    try:
        pid = int(pid_file.read_text().strip())
        
        # Stop the process
        os.kill(pid, 15)  # SIGTERM
        
        # Remove PID file
        pid_file.unlink()
        
        click.echo(f"✅ Stopped workers (PID: {pid})")
        
    except ProcessLookupError:
        # Process already stopped
        pid_file.unlink()
        click.echo("⚠️  Workers were not running.")
    except Exception as e:
        click.echo(f"❌ Error stopping workers: {str(e)}", err=True)
        sys.exit(1)


@workers.command('restart')
@click.option('--num-workers', '-n', default=2, help='Number of workers to start')
@click.pass_context
def restart_workers(ctx, num_workers: int):
    """Restart worker processes."""
    # Stop workers
    ctx.invoke(stop_workers)
    
    # Wait a moment
    import time
    time.sleep(1)
    
    # Start workers
    ctx.invoke(start_workers, num_workers=num_workers)


@workers.command('status')
@click.pass_obj
def worker_status(config_manager: ConfigManager):
    """Check worker status."""
    pid_file = Path.home() / '.wagehood' / 'workers.pid'
    
    if not pid_file.exists():
        click.echo("❌ No workers are running.")
        return
    
    try:
        pid = int(pid_file.read_text().strip())
        
        # Check if process is running
        os.kill(pid, 0)  # Signal 0 just checks if process exists
        
        click.echo(f"✅ Workers are running (PID: {pid})")
        
        # Show queue status
        queue = JobQueue()
        stats = queue.get_queue_stats()
        
        click.echo(f"📊 Queue status: {stats.get('pending', 0)} pending, {stats.get('processing', 0)} processing, {stats.get('completed', 0)} completed")
        
    except ProcessLookupError:
        # Process not running
        pid_file.unlink()
        click.echo("❌ Workers are not running (stale PID file removed).")
    except Exception as e:
        click.echo(f"❌ Error checking worker status: {str(e)}", err=True)


def main():
    """Main entry point for CLI."""
    cli()


if __name__ == '__main__':
    main()