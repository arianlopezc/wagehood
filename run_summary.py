#!/usr/bin/env python3
"""
Direct summary generation script for end-of-day trading signal analysis.

This script generates a summary of today's trading signals for all supported symbols
using MACD+RSI and Support/Resistance strategies with 1-day timeframe.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import timezone utilities for consistent UTC handling
from src.utils.timezone_utils import utc_now

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import fast parallel processor
from src.jobs.fast_parallel_processor import FastParallelProcessor


@dataclass
class SignalSummary:
    """Summary for a single symbol."""
    symbol: str
    signal_count: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    latest_signal: Optional[Dict[str, Any]] = None
    avg_confidence: float = 0.0
    errors: List[str] = field(default_factory=list)
    strategies: List[str] = field(default_factory=list)


@dataclass
class SummaryResult:
    """Overall summary result."""
    symbols_processed: int = 0
    symbols_with_signals: int = 0
    total_signals: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    execution_duration_seconds: float = 0.0
    signal_summaries: List[SignalSummary] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=utc_now)


class SummaryGenerator:
    """Generates end-of-day summaries for trading signals."""
    
    def __init__(self):
        """Initialize the summary generator with fast parallel processing."""
        self.strategies = ['macd_rsi', 'sr_breakout']
        
        # Use fast parallel processor for better performance
        self.processor = FastParallelProcessor(
            max_concurrent=10  # Process up to 10 symbols simultaneously
        )
    
    async def generate_summary(
        self,
        symbols: Optional[List[str]] = None
    ) -> SummaryResult:
        """
        Generate a summary of today's trading signals.
        
        Args:
            symbols: List of symbols to analyze (None = all supported)
            
        Returns:
            SummaryResult with today's signal information
        """
        start_time = utc_now()
        result = SummaryResult()
        
        # Get symbols to process
        if symbols is None:
            # Get all supported symbols from environment
            symbols_str = os.getenv('SUPPORTED_SYMBOLS', 'AAPL,MSFT,GOOGL')
            symbols = [s.strip() for s in symbols_str.split(',') if s.strip()]
        
        logger.info(f"Generating today's summary for {len(symbols)} symbols")
        
        # Calculate date range - end date is today (UTC), start date provides enough data
        end_date = utc_now()
        # Use 180 days to ensure both strategies have sufficient data
        start_date = end_date - timedelta(days=180)
        
        # Always use parallel processing for better performance
        logger.info(f"Processing {len(symbols)} symbols in parallel")
        
        # Process all symbols in parallel
        symbol_results = await self.processor.process_symbols(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            strategies=self.strategies
        )
        
        # Convert parallel results to SignalSummary objects
        today = utc_now().date()
        
        for symbol, analysis_result in symbol_results.items():
            symbol_summary = self._create_summary_from_parallel_result(
                symbol, analysis_result, today
            )
            
            result.signal_summaries.append(symbol_summary)
            result.symbols_processed += 1
            
            if symbol_summary.signal_count > 0:
                result.symbols_with_signals += 1
                result.total_signals += symbol_summary.signal_count
                result.buy_signals += symbol_summary.buy_signals
                result.sell_signals += symbol_summary.sell_signals
            
            if symbol_summary.errors:
                result.errors.extend([f"{symbol}: {err}" for err in symbol_summary.errors])
        
        # Sort summaries by signal count (descending)
        result.signal_summaries.sort(key=lambda x: x.signal_count, reverse=True)
        
        # Calculate execution time
        result.execution_duration_seconds = (utc_now() - start_time).total_seconds()
        
        logger.info(f"Summary generation completed in {result.execution_duration_seconds:.2f}s")
        logger.info(f"Found {result.total_signals} today's signals across {result.symbols_with_signals} symbols")
        
        return result
    
    def _create_summary_from_parallel_result(
        self,
        symbol: str,
        analysis_result: Dict[str, Any],
        today: Any
    ) -> SignalSummary:
        """Convert parallel processing result to SignalSummary."""
        summary = SignalSummary(symbol=symbol)
        
        if not analysis_result.get('success', False):
            # Handle failed analysis
            error = analysis_result.get('error', 'Unknown error')
            summary.errors.append(error)
            return summary
        
        # Get all signals from the result
        all_signals = analysis_result.get('signals', [])
        
        # Filter for today's signals only
        todays_signals = []
        for signal in all_signals:
            signal_date = signal.get('timestamp')
            if signal_date:
                # Handle datetime objects
                if isinstance(signal_date, datetime):
                    signal_date = signal_date.date()
                # Handle string timestamps
                elif isinstance(signal_date, str):
                    try:
                        signal_date = datetime.fromisoformat(signal_date.replace('Z', '+00:00')).date()
                    except:
                        continue
                
                # Only include today's signals
                if signal_date == today:
                    todays_signals.append(signal)
        
        # Process today's signals
        if todays_signals:
            summary.signal_count = len(todays_signals)
            summary.buy_signals = sum(1 for s in todays_signals if s.get('signal_type') == 'BUY')
            summary.sell_signals = sum(1 for s in todays_signals if s.get('signal_type') == 'SELL')
            
            # Calculate average confidence
            confidences = [s.get('confidence', 0) for s in todays_signals]
            summary.avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Get latest signal
            summary.latest_signal = todays_signals[0]
            
            # Get unique strategies that generated signals
            summary.strategies = list(set(s.get('strategy', '') for s in todays_signals if s.get('strategy')))
        
        # Add any errors from the analysis
        if analysis_result.get('errors'):
            summary.errors.extend(analysis_result['errors'])
        
        return summary
    
async def main(
    symbols: Optional[List[str]] = None,
    scheduled: bool = False
) -> SummaryResult:
    """
    Main entry point for summary generation.
    
    Args:
        symbols: Optional list of symbols to analyze
        scheduled: If True, send results to Discord (for scheduled runs only)
        
    Returns:
        SummaryResult with today's signal findings
    """
    generator = SummaryGenerator()
    result = await generator.generate_summary(symbols=symbols)
    
    # Send to Discord only if this is a scheduled run
    if scheduled:
        await _send_to_discord(result)
    
    return result


async def _send_to_discord(result: SummaryResult) -> None:
    """
    Send summary results to Discord (only for scheduled runs).
    
    Args:
        result: SummaryResult to send to Discord
    """
    try:
        import os
        
        webhook_url = os.getenv('DISCORD_WEBHOOK_EOD_SUMMARY')
        if not webhook_url:
            logger.warning("DISCORD_WEBHOOK_EOD_SUMMARY not configured, skipping Discord notification")
            return
        
        # Import Discord notification components
        from src.notifications.summary_formatter import SummaryFormatter
        from src.notifications.discord_client import DiscordClient
        
        # Format the summary for Discord
        formatted_data = SummaryFormatter.format_summary(result)
        
        # Parse the formatted message (it's a JSON string)
        import json
        message_data = json.loads(formatted_data)
        
        # Send to Discord
        discord_client = DiscordClient()
        try:
            await discord_client.send_message(
                webhook_url=webhook_url,
                content=message_data.get('content', ''),
                embeds=message_data.get('embeds', []),
                username=message_data.get('username', 'Wagehood EOD Summary')
            )
            logger.info("Successfully sent summary to Discord")
        finally:
            # Always close the client to prevent unclosed session warnings
            await discord_client.close()
        
    except Exception as e:
        logger.error(f"Failed to send summary to Discord: {e}")


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate today's trading signal summary"
    )
    parser.add_argument(
        '--symbols', '-s',
        help='Comma-separated symbols (default: all supported)',
        type=str,
        default=None
    )
    parser.add_argument(
        '--scheduled',
        help='Internal flag for scheduled runs (enables Discord notifications)',
        action='store_true'
    )
    
    args = parser.parse_args()
    
    # Parse symbols if provided
    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    
    # Run the summary generation
    result = asyncio.run(main(symbols=symbols, scheduled=args.scheduled))
    
    # Print results
    print(f"\nToday's Signal Summary:")
    print(f"Generated at: {result.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Symbols processed: {result.symbols_processed}")
    print(f"Symbols with signals: {result.symbols_with_signals}")
    print(f"Total signals: {result.total_signals} ({result.buy_signals} BUY, {result.sell_signals} SELL)")
    print(f"Execution time: {result.execution_duration_seconds:.2f} seconds")
    
    if result.signal_summaries:
        print(f"\nTop Opportunities:")
        for summary in result.signal_summaries[:10]:
            if summary.signal_count > 0:
                signal_type = "BUY" if summary.buy_signals > summary.sell_signals else "SELL"
                
                # Show strategy breakdown for this symbol
                strategy_names = [s.replace('_', ' ').title() for s in summary.strategies]
                strategy_info = f" ({', '.join(strategy_names)})" if strategy_names else ""
                
                print(f"  {summary.symbol}: {summary.signal_count} signals, "
                      f"{signal_type} bias, {summary.avg_confidence:.0%} avg confidence{strategy_info}")
    
    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for error in result.errors[:5]:
            print(f"  - {error}")
        if len(result.errors) > 5:
            print(f"  ... and {len(result.errors) - 5} more errors")