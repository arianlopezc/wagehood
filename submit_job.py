#!/usr/bin/env python3
"""
Wagehood Signal Analysis Job Submission CLI

This CLI submits signal analysis jobs to the running production instance,
monitors their progress, and displays comprehensive signal analysis results.

RECOMMENDED USAGE:
    The best approach is to run this script from inside the Docker container:
    
    docker exec -it wagehood-trading python submit_job.py \
        --symbol AAPL --timeframe 1h --strategy macd_rsi \
        --start 2024-01-01 --end 2024-12-31 --redis-port 6379

    This ensures proper Redis connectivity and avoids port mapping issues.

ALTERNATIVE USAGE (from host machine):
    If you have Redis installed locally and port 6380 is properly mapped:
    
    python submit_job.py --symbol AAPL --timeframe 1h --strategy macd_rsi \
                        --start 2024-01-01 --end 2024-12-31

Features:
    - Single command that handles submission, monitoring, and results
    - Real-time progress updates with visual progress bar
    - Detailed signal analysis results with quality metrics
    - Signal confidence distribution and timing analysis
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
    """CLI for submitting and monitoring signal analysis jobs."""
    
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
        
        # Check environment variables for Redis configuration
        redis_host = os.getenv('REDIS_HOST', redis_host)
        redis_port = int(os.getenv('REDIS_PORT', redis_port))
        
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
        """Create a new signal analysis job."""
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
            "type": "signal_analysis",
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
                print("✅ Signal analysis completed successfully!")
                return self._get_job_result(job_id)
            elif status == "failed":
                error = status_data.get("error", "Unknown error")
                print(f"❌ Signal analysis failed: {error}")
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
        """Display formatted signal analysis results."""
        if not results:
            return
        
        # Parse results
        symbol = results.get("symbol", "")
        timeframe = results.get("timeframe", "")
        strategy = results.get("strategy", "")
        start_date = results.get("start_date", "")
        end_date = results.get("end_date", "")
        
        # Parse signal data
        signals_data = json.loads(results.get("signals", "[]"))
        
        # Signal analysis metrics
        total_signals = len(signals_data)
        buy_signals = sum(1 for s in signals_data if s["type"] == "BUY")
        sell_signals = sum(1 for s in signals_data if s["type"] == "SELL")
        hold_signals = sum(1 for s in signals_data if s["type"] == "HOLD")
        
        # Calculate confidence statistics
        confidences = [float(s["confidence"]) for s in signals_data if s["confidence"] is not None]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            min_confidence = min(confidences)
            max_confidence = max(confidences)
            high_confidence_signals = sum(1 for c in confidences if c >= 0.7)
            medium_confidence_signals = sum(1 for c in confidences if 0.4 <= c < 0.7)
            low_confidence_signals = sum(1 for c in confidences if c < 0.4)
        else:
            avg_confidence = min_confidence = max_confidence = 0
            high_confidence_signals = medium_confidence_signals = low_confidence_signals = 0
        
        # Calculate signal frequency (signals per day)
        if total_signals > 0 and start_date and end_date:
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            days = (end_dt - start_dt).days + 1
            signals_per_day = total_signals / days if days > 0 else 0
        else:
            signals_per_day = 0
        
        # Display header
        print("\n" + "━" * 80)
        print(f"🎯 SIGNAL ANALYSIS RESULTS")
        print("━" * 80)
        print(f"Symbol: {symbol}")
        print(f"Timeframe: {timeframe}")
        print(f"Strategy: {strategy}")
        print(f"Period: {start_date} to {end_date}")
        print("━" * 80)
        
        # Signal summary
        print(f"📊 SIGNAL SUMMARY")
        print("━" * 80)
        print(f"Total Signals:    {total_signals:8}")
        print(f"Buy Signals:      {buy_signals:8} ({buy_signals/total_signals*100:.1f}%)" if total_signals > 0 else "Buy Signals:      0")
        print(f"Sell Signals:     {sell_signals:8} ({sell_signals/total_signals*100:.1f}%)" if total_signals > 0 else "Sell Signals:     0")
        print(f"Hold Signals:     {hold_signals:8} ({hold_signals/total_signals*100:.1f}%)" if total_signals > 0 else "Hold Signals:     0")
        print(f"Avg Frequency:    {signals_per_day:8.2f} signals/day")
        
        # Confidence analysis
        print(f"\n🔍 CONFIDENCE ANALYSIS")
        print("━" * 80)
        print(f"Average Confidence: {avg_confidence:6.2f}")
        print(f"Min Confidence:     {min_confidence:6.2f}")
        print(f"Max Confidence:     {max_confidence:6.2f}")
        print(f"High Confidence:    {high_confidence_signals:8} (≥0.70)")
        print(f"Medium Confidence:  {medium_confidence_signals:8} (0.40-0.69)")
        print(f"Low Confidence:     {low_confidence_signals:8} (<0.40)")
        
        # Signal quality assessment
        print(f"\n⭐ SIGNAL QUALITY ASSESSMENT")
        print("━" * 80)
        if total_signals > 0:
            quality_score = (high_confidence_signals * 3 + medium_confidence_signals * 2 + low_confidence_signals * 1) / total_signals
            print(f"Quality Score:      {quality_score:6.2f}/3.0")
            
            if quality_score >= 2.5:
                quality_rating = "Excellent"
            elif quality_score >= 2.0:
                quality_rating = "Good"
            elif quality_score >= 1.5:
                quality_rating = "Fair"
            else:
                quality_rating = "Poor"
            
            print(f"Quality Rating:     {quality_rating}")
            
            # Signal consistency check
            if signals_per_day > 0:
                if signals_per_day > 10:
                    consistency_rating = "High Activity"
                elif signals_per_day > 2:
                    consistency_rating = "Moderate Activity"
                elif signals_per_day > 0.5:
                    consistency_rating = "Low Activity"
                else:
                    consistency_rating = "Very Low Activity"
                
                print(f"Activity Level:     {consistency_rating}")
        else:
            print("Quality Score:      N/A (No signals)")
            print("Quality Rating:     N/A")
            print("Activity Level:     N/A")
        
        # Display detailed signals
        if signals_data:
            print(f"\n📋 DETAILED SIGNAL ANALYSIS")
            print("━" * 80)
            print(f"{'Date':<12} {'Time':<8} {'Type':<6} {'Price':<10} {'Conf':<6} {'Strategy':<15} {'Context'}")
            print("─" * 80)
            
            for signal in signals_data[:50]:  # Limit to first 50 signals
                timestamp = signal["timestamp"]
                date = timestamp[:10]  # Extract date
                time_str = timestamp[11:19]  # Extract time
                signal_type = signal["type"]
                price = float(signal["price"])
                confidence = float(signal["confidence"]) if signal["confidence"] is not None else 0
                strategy_name = signal["strategy"]
                
                # Extract context from metadata if available
                metadata = signal.get("metadata", {})
                context = ""
                if isinstance(metadata, dict):
                    if "trend" in metadata:
                        context += f"Trend: {metadata['trend']} "
                    if "rsi" in metadata:
                        context += f"RSI: {metadata['rsi']:.1f} "
                    if "macd" in metadata:
                        context += f"MACD: {metadata['macd']:.3f} "
                
                print(f"{date:<12} {time_str:<8} {signal_type:<6} ${price:<9.2f} {confidence:<5.2f} {strategy_name:<15} {context}")
            
            if len(signals_data) > 50:
                print(f"... and {len(signals_data) - 50} more signals")
        
        # Signal timing analysis
        if signals_data:
            print(f"\n⏰ SIGNAL TIMING ANALYSIS")
            print("━" * 80)
            
            # Group signals by hour to find patterns
            hourly_signals = {}
            for signal in signals_data:
                timestamp = signal["timestamp"]
                hour = int(timestamp[11:13])
                hourly_signals[hour] = hourly_signals.get(hour, 0) + 1
            
            # Find most active hours
            if hourly_signals:
                sorted_hours = sorted(hourly_signals.items(), key=lambda x: x[1], reverse=True)
                print(f"Most Active Hours: ", end="")
                for hour, count in sorted_hours[:3]:
                    print(f"{hour:02d}:00 ({count} signals) ", end="")
                print()
            
            # Calculate time between signals
            if len(signals_data) > 1:
                timestamps = [datetime.fromisoformat(s["timestamp"]) for s in signals_data]
                timestamps.sort()
                
                time_diffs = []
                for i in range(1, len(timestamps)):
                    diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600  # Hours
                    time_diffs.append(diff)
                
                if time_diffs:
                    avg_time_between = sum(time_diffs) / len(time_diffs)
                    print(f"Avg Time Between:  {avg_time_between:6.1f} hours")
        
        print("━" * 80)
        print("🎉 Signal analysis complete!")
        print("━" * 80)
    
    def run(self, symbol: str, timeframe: str, strategy: str, 
            start_date: str, end_date: str):
        """Run the complete job submission and monitoring process."""
        print(f"🎯 Submitting signal analysis job...")
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
        description="Submit signal analysis jobs to Wagehood production instance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic signal analysis
  python submit_job.py --symbol AAPL --timeframe 1h --strategy macd_rsi \\
                      --start 2024-01-01 --end 2024-12-31
  
  # Short timeframe signal detection
  python submit_job.py --symbol SPY --timeframe 5m --strategy rsi_trend \\
                      --start 2024-06-01 --end 2024-06-30
  
  # Long-term signal analysis
  python submit_job.py --symbol QQQ --timeframe 1d --strategy ma_crossover \\
                      --start 2023-01-01 --end 2024-12-31

Available Strategies:
  - macd_rsi: MACD + RSI Combined (excellent for trend detection)
  - ma_crossover: Moving Average Crossover (good for trend reversals)
  - rsi_trend: RSI Trend Following (good for momentum signals)
  - bollinger_breakout: Bollinger Band Breakout (good for volatility signals)
  - sr_breakout: Support/Resistance Breakout (good for breakout signals)

Available Timeframes:
  - 1m, 5m, 15m, 30m: High-frequency signal detection
  - 1h, 4h: Medium-term signal analysis
  - 1d, 1w, 1M: Long-term trend signals

Signal Analysis Features:
  - Comprehensive signal quality assessment
  - Confidence distribution analysis
  - Signal timing and frequency patterns
  - Market context and technical indicators
  - Signal validation and filtering

IMPORTANT: Redis Connection
  - RECOMMENDED: Run from inside Docker container with --redis-port 6379
    docker exec -it wagehood-trading python submit_job.py [OPTIONS] --redis-port 6379
  
  - ALTERNATIVE: Run from host machine (requires local Redis or port mapping)
    python submit_job.py [OPTIONS]  # Uses default port 6380
  
Make sure the Wagehood container is running before submitting jobs.
        """
    )
    
    parser.add_argument(
        "--symbol", 
        required=True,
        help="Symbol for signal analysis (e.g., AAPL, SPY, MSFT)"
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
        help="Signal detection strategy to analyze"
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