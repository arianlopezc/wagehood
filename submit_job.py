#!/usr/bin/env python3
"""
Wagehood Backtest Job Submission CLI

This CLI submits backtest jobs to the running production instance,
monitors their progress, and displays results including all signals and trades.

Usage:
    python submit_job.py --symbol AAPL --timeframe 1h --strategy macd_rsi \
                        --start 2024-01-01 --end 2024-12-31

Features:
    - Single command that handles submission, monitoring, and results
    - Real-time progress updates with visual progress bar
    - Detailed results including all signals and trades
    - Connects to running Docker production instance via Redis
"""

import argparse
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List
import sys
import os

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class JobSubmissionCLI:
    """CLI for submitting and monitoring backtest jobs."""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6380):
        """
        Initialize CLI with Redis connection.
        
        Args:
            redis_host: Redis host (default: localhost)
            redis_port: Redis port (default: 6380 for Docker instance)
        """
        if not REDIS_AVAILABLE:
            print("❌ Error: Redis package required. Install with: pip install redis")
            sys.exit(1)
        
        try:
            self.redis_client = redis.Redis(
                host=redis_host, 
                port=redis_port, 
                db=0, 
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            print(f"✅ Connected to Redis at {redis_host}:{redis_port}")
        except Exception as e:
            print(f"❌ Failed to connect to Redis at {redis_host}:{redis_port}: {e}")
            print("💡 Make sure the Wagehood Docker container is running with Redis on port 6380")
            sys.exit(1)
    
    def create_job(self, symbol: str, timeframe: str, strategy: str, 
                   start_date: str, end_date: str) -> Dict[str, Any]:
        """Create a new backtest job."""
        # Generate unique job ID
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Validate dates
        try:
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            if start_dt >= end_dt:
                raise ValueError("Start date must be before end date")
        except ValueError as e:
            print(f"❌ Invalid date format: {e}")
            sys.exit(1)
        
        # Create job data
        job_data = {
            "job_id": job_id,
            "type": "backtest",
            "params": json.dumps({
                "symbol": symbol.upper(),
                "timeframe": timeframe,
                "strategy": strategy,
                "start_date": start_date,
                "end_date": end_date
            }),
            "submitted_at": datetime.now().isoformat(),
            "status": "pending"
        }
        
        return job_data
    
    def submit_job(self, job_data: Dict[str, Any]) -> str:
        """Submit job to Redis job queue."""
        job_id = job_data["job_id"]
        
        # Submit to jobs stream
        self.redis_client.xadd("jobs_stream", job_data)
        
        # Initialize job status
        status_key = f"job:status:{job_id}"
        self.redis_client.hset(status_key, mapping={
            "status": "pending",
            "progress": "0",
            "message": "Job submitted and queued",
            "submitted_at": job_data["submitted_at"]
        })
        self.redis_client.expire(status_key, 86400)  # 24 hour TTL
        
        return job_id
    
    def monitor_job(self, job_id: str) -> Dict[str, Any]:
        """Monitor job progress until completion."""
        status_key = f"job:status:{job_id}"
        
        print(f"⏳ Monitoring job: {job_id}")
        
        last_progress = -1
        
        while True:
            # Get current status
            status_data = self.redis_client.hgetall(status_key)
            
            if not status_data:
                print(f"❌ Job {job_id} not found")
                return None
            
            status = status_data.get("status", "unknown")
            progress = int(status_data.get("progress", "0"))
            message = status_data.get("message", "")
            
            # Show progress if changed
            if progress != last_progress:
                self._display_progress(status, progress, message)
                last_progress = progress
            
            # Check if completed
            if status == "completed":
                print("✅ Job completed successfully!")
                return self._get_job_result(job_id)
            elif status == "failed":
                error = status_data.get("error", "Unknown error")
                print(f"❌ Job failed: {error}")
                return None
            
            # Wait before next check
            time.sleep(1)
    
    def _display_progress(self, status: str, progress: int, message: str):
        """Display progress bar and status."""
        # Create progress bar
        bar_length = 20
        filled_length = int(bar_length * progress // 100)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        # Status emoji
        status_emoji = {
            "pending": "⏳",
            "running": "🚀",
            "completed": "✅",
            "failed": "❌"
        }.get(status, "📊")
        
        print(f"\r{status_emoji} Status: {status.title()} [{bar}] {progress}% - {message}", end="", flush=True)
        
        if progress == 100 or status in ["completed", "failed"]:
            print()  # New line when done
    
    def _get_job_result(self, job_id: str) -> Dict[str, Any]:
        """Get job results from Redis."""
        result_key = f"job:result:{job_id}"
        result_data = self.redis_client.hgetall(result_key)
        
        if not result_data:
            print(f"❌ No results found for job {job_id}")
            return None
        
        return result_data
    
    def display_results(self, results: Dict[str, Any]):
        """Display formatted backtest results."""
        if not results:
            return
        
        # Parse results
        symbol = results.get("symbol", "")
        timeframe = results.get("timeframe", "")
        strategy = results.get("strategy", "")
        start_date = results.get("start_date", "")
        end_date = results.get("end_date", "")
        
        # Performance metrics
        total_return_pct = float(results.get("total_return_pct", "0"))
        win_rate = float(results.get("win_rate", "0"))
        sharpe_ratio = float(results.get("sharpe_ratio", "0"))
        max_drawdown_pct = float(results.get("max_drawdown_pct", "0"))
        total_trades = int(results.get("total_trades", "0"))
        winning_trades = int(results.get("winning_trades", "0"))
        losing_trades = int(results.get("losing_trades", "0"))
        profit_factor = float(results.get("profit_factor", "0"))
        
        # Parse trades and signals
        trades_data = json.loads(results.get("trades", "[]"))
        signals_data = json.loads(results.get("signals", "[]"))
        
        # Display header
        print("\n" + "━" * 80)
        print(f"📈 BACKTEST RESULTS")
        print("━" * 80)
        print(f"Symbol: {symbol}")
        print(f"Timeframe: {timeframe}")
        print(f"Strategy: {strategy}")
        print(f"Period: {start_date} to {end_date}")
        print("━" * 80)
        
        # Performance summary
        print(f"📊 PERFORMANCE SUMMARY")
        print("━" * 80)
        print(f"Total Return:     {total_return_pct:+8.2f}%")
        print(f"Win Rate:         {win_rate:8.1f}%")
        print(f"Sharpe Ratio:     {sharpe_ratio:8.2f}")
        print(f"Max Drawdown:     {max_drawdown_pct:8.2f}%")
        print(f"Profit Factor:    {profit_factor:8.2f}")
        print(f"Total Trades:     {total_trades:8}")
        print(f"Winning Trades:   {winning_trades:8}")
        print(f"Losing Trades:    {losing_trades:8}")
        
        # Signals summary
        print(f"\n🎯 SIGNALS SUMMARY")
        print("━" * 80)
        print(f"Total Signals:    {len(signals_data):8}")
        
        # Count signal types
        buy_signals = sum(1 for s in signals_data if s["type"] == "BUY")
        sell_signals = sum(1 for s in signals_data if s["type"] == "SELL")
        
        print(f"Buy Signals:      {buy_signals:8}")
        print(f"Sell Signals:     {sell_signals:8}")
        
        # Display all signals
        if signals_data:
            print(f"\n📋 ALL SIGNALS")
            print("━" * 80)
            print(f"{'Date':<12} {'Type':<4} {'Price':<10} {'Confidence':<10} {'Strategy'}")
            print("─" * 80)
            
            for signal in signals_data[:50]:  # Limit to first 50 signals
                date = signal["timestamp"][:10]  # Extract date
                signal_type = signal["type"]
                price = float(signal["price"])
                confidence = float(signal["confidence"])
                strategy_name = signal["strategy"]
                
                print(f"{date:<12} {signal_type:<4} ${price:<9.2f} {confidence:<9.2f} {strategy_name}")
            
            if len(signals_data) > 50:
                print(f"... and {len(signals_data) - 50} more signals")
        
        # Display all trades
        if trades_data:
            print(f"\n💰 ALL TRADES")
            print("━" * 80)
            print(f"{'Entry Date':<12} {'Exit Date':<12} {'Side':<4} {'Entry $':<10} {'Exit $':<10} {'P&L':<10} {'P&L %':<8}")
            print("─" * 80)
            
            for trade in trades_data:
                entry_date = trade["entry_date"][:10]
                exit_date = trade["exit_date"][:10] if trade["exit_date"] else "Open"
                side = trade["side"]
                entry_price = float(trade["entry_price"])
                exit_price = float(trade["exit_price"]) if trade["exit_price"] else 0
                profit_loss = float(trade["profit_loss"]) if trade["profit_loss"] else 0
                profit_loss_pct = float(trade["profit_loss_pct"]) if trade["profit_loss_pct"] else 0
                
                pnl_color = "+" if profit_loss >= 0 else ""
                
                print(f"{entry_date:<12} {exit_date:<12} {side:<4} ${entry_price:<9.2f} ${exit_price:<9.2f} {pnl_color}{profit_loss:<9.2f} {pnl_color}{profit_loss_pct:<7.1f}%")
        
        print("━" * 80)
        print("🎉 Analysis complete!")
        print("━" * 80)
    
    def run(self, symbol: str, timeframe: str, strategy: str, 
            start_date: str, end_date: str):
        """Run the complete job submission and monitoring process."""
        print(f"📊 Submitting backtest job...")
        print(f"Symbol: {symbol.upper()}")
        print(f"Timeframe: {timeframe}")
        print(f"Strategy: {strategy}")
        print(f"Period: {start_date} to {end_date}")
        print()
        
        # Create and submit job
        job_data = self.create_job(symbol, timeframe, strategy, start_date, end_date)
        job_id = self.submit_job(job_data)
        
        print(f"✅ Job submitted with ID: {job_id}")
        print()
        
        # Monitor and get results
        results = self.monitor_job(job_id)
        
        if results:
            self.display_results(results)
        else:
            print("❌ Failed to get results")
            sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Submit backtest jobs to Wagehood production instance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic backtest
  python submit_job.py --symbol AAPL --timeframe 1h --strategy macd_rsi \\
                      --start 2024-01-01 --end 2024-12-31
  
  # Short timeframe day trading
  python submit_job.py --symbol SPY --timeframe 5m --strategy rsi_trend \\
                      --start 2024-06-01 --end 2024-06-30
  
  # Long-term position trading
  python submit_job.py --symbol QQQ --timeframe 1d --strategy ma_crossover \\
                      --start 2023-01-01 --end 2024-12-31

Available Strategies:
  - macd_rsi: MACD + RSI Combined (recommended for swing trading)
  - ma_crossover: Moving Average Crossover (good for position trading)
  - rsi_trend: RSI Trend Following (good for day trading)
  - bollinger_breakout: Bollinger Band Breakout (good for volatility)
  - sr_breakout: Support/Resistance Breakout (good for breakouts)

Available Timeframes:
  - 1m, 5m, 15m, 30m: Day trading timeframes
  - 1h, 4h: Swing trading timeframes  
  - 1d, 1w, 1M: Position trading timeframes

Note: This CLI connects to the running Docker production instance on port 6380.
Make sure the Wagehood container is running before submitting jobs.
        """
    )
    
    parser.add_argument(
        "--symbol", 
        required=True,
        help="Trading symbol (e.g., AAPL, SPY, MSFT)"
    )
    parser.add_argument(
        "--timeframe", 
        required=True,
        choices=["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"],
        help="Timeframe for analysis"
    )
    parser.add_argument(
        "--strategy", 
        required=True,
        choices=["macd_rsi", "ma_crossover", "rsi_trend", "bollinger_breakout", "sr_breakout"],
        help="Trading strategy to test"
    )
    parser.add_argument(
        "--start", 
        required=True,
        help="Start date (YYYY-MM-DD format)"
    )
    parser.add_argument(
        "--end", 
        required=True,
        help="End date (YYYY-MM-DD format)"
    )
    parser.add_argument(
        "--redis-host", 
        default="localhost",
        help="Redis host (default: localhost)"
    )
    parser.add_argument(
        "--redis-port", 
        type=int,
        default=6380,
        help="Redis port (default: 6380 for Docker)"
    )
    
    args = parser.parse_args()
    
    # Create and run CLI
    cli = JobSubmissionCLI(args.redis_host, args.redis_port)
    cli.run(args.symbol, args.timeframe, args.strategy, args.start, args.end)


if __name__ == "__main__":
    main()