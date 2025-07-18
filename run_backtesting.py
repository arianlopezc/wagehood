#!/usr/bin/env python3
"""
Direct backtesting script that replaces the job-based system.

This script accepts the same parameters as the backtesting job submit command
and uses the corresponding strategy analyzer to obtain signals directly.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import analyzers
from src.strategies.macd_rsi_analyzer import MACDRSIAnalyzer
from src.strategies.rsi_trend_analyzer import RSITrendAnalyzer
from src.strategies.bollinger_breakout_analyzer import BollingerBreakoutAnalyzer
from src.strategies.sr_breakout_analyzer import SRBreakoutAnalyzer
from src.analysis.return_calculator import ReturnCalculator
from src.data.providers.alpaca_provider import AlpacaProvider

# Strategy to analyzer mapping
STRATEGY_ANALYZER_MAP = {
    'macd_rsi': MACDRSIAnalyzer,
    'rsi_trend': RSITrendAnalyzer,
    'bollinger_breakout': BollingerBreakoutAnalyzer,
    'sr_breakout': SRBreakoutAnalyzer
}


class BacktestingRunner:
    """Direct backtesting runner that executes analysis without job queue."""
    
    def __init__(self):
        """Initialize the backtesting runner."""
        self.return_calculator = ReturnCalculator()
        self.alpaca_provider = None  # Will be initialized when needed
    
    async def run_backtesting(
        self,
        symbol: str,
        strategy: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run backtesting for the given parameters.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL')
            strategy: Strategy name ('macd_rsi', 'rsi_trend', 'bollinger_breakout', 'sr_breakout')
            timeframe: Timeframe for analysis ('1h' or '1d')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            strategy_params: Optional strategy-specific parameters
            
        Returns:
            Dictionary containing the backtesting results
        """
        logger.info(f"Starting backtesting for {symbol} using {strategy} strategy")
        logger.info(f"Parameters: timeframe={timeframe}, start={start_date}, end={end_date}")
        
        try:
            # Validate inputs
            self._validate_inputs(symbol, strategy, timeframe, start_date, end_date)
            
            # Convert date strings to datetime objects
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Get the appropriate analyzer
            analyzer_class = STRATEGY_ANALYZER_MAP.get(strategy)
            if not analyzer_class:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            # Initialize analyzer
            logger.info(f"Initializing {analyzer_class.__name__} analyzer...")
            analyzer = analyzer_class()
            
            # Run analysis
            logger.info(f"Analyzing {symbol} with {strategy} strategy...")
            logger.info("Fetching market data and calculating indicators...")
            
            analysis_result = await analyzer.analyze_symbol(
                symbol=symbol,
                start_date=start_dt,
                end_date=end_dt,
                timeframe=timeframe,
                strategy_params=strategy_params
            )
            
            # Extract signals from result
            if isinstance(analysis_result, dict):
                signals = analysis_result.get('signals', [])
                # Store any additional metrics if available
                analysis_metrics = analysis_result.get('metrics', {})
            else:
                # For backward compatibility, assume it's a list of signals
                signals = analysis_result
                analysis_metrics = {}
            
            # Log results
            logger.info(f"Analysis completed. Found {len(signals)} signals.")
            
            # Calculate returns if we have signals
            return_stats = {'total_return_pct': 0.0, 'num_trades': 0}
            if signals:
                logger.info("Calculating returns from signals...")
                
                # Get final price for open positions
                final_price = None
                final_date = None
                try:
                    # Initialize and connect Alpaca provider if needed
                    if not self.alpaca_provider:
                        self.alpaca_provider = AlpacaProvider()
                        await self.alpaca_provider.connect()
                    
                    # Fetch the most recent price
                    recent_data = await self.alpaca_provider.get_historical_data(
                        symbol=symbol,
                        start_date=end_dt - timedelta(days=5),
                        end_date=end_dt,
                        timeframe=timeframe
                    )
                    
                    if recent_data and isinstance(recent_data, list) and len(recent_data) > 0:
                        last_bar = recent_data[-1]
                        final_price = last_bar['close']
                        final_date = last_bar['timestamp']
                        logger.info(f"Using final price: ${final_price:.2f} on {final_date}")
                except Exception as e:
                    logger.warning(f"Could not fetch final price: {e}")
                
                # Calculate returns
                return_stats = self.return_calculator.calculate_returns(
                    signals=signals,
                    final_price=final_price,
                    final_date=final_date
                )
                
                logger.info(f"Return calculation completed: {return_stats['total_return_pct']:.2f}% total return")
            
            # Return results in the same format as the job system
            result = {
                'status': 'completed',
                'symbol': symbol,
                'strategy': strategy,
                'timeframe': timeframe,
                'start_date': start_date,
                'end_date': end_date,
                'signals': signals,
                'signal_count': len(signals),
                'return_stats': return_stats,
                'completed_at': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Backtesting failed: {str(e)}")
            return {
                'status': 'failed',
                'symbol': symbol,
                'strategy': strategy,
                'timeframe': timeframe,
                'start_date': start_date,
                'end_date': end_date,
                'error': str(e),
                'completed_at': datetime.now().isoformat()
            }
    
    def _validate_inputs(self, symbol: str, strategy: str, timeframe: str, 
                        start_date: str, end_date: str) -> None:
        """Validate input parameters."""
        # Validate symbol
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        # Allow crypto symbols with '/' like BTC/USD
        is_crypto = '/' in symbol
        if not is_crypto and (not symbol.isalpha() or len(symbol) > 10):
            raise ValueError(f"Invalid symbol: {symbol}. Stock symbols must be 1-10 letters only")
        
        if is_crypto:
            parts = symbol.split('/')
            if len(parts) != 2 or not all(part.isalpha() for part in parts):
                raise ValueError(f"Invalid crypto symbol: {symbol}. Must be in format BASE/QUOTE (e.g., BTC/USD)")
        
        # Validate strategy
        valid_strategies = list(STRATEGY_ANALYZER_MAP.keys())
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {strategy}. Must be one of {valid_strategies}")
        
        # Validate timeframe
        if timeframe not in ['1h', '1d']:
            raise ValueError(f"Invalid timeframe: {timeframe}. Must be '1h' or '1d'")
        
        # Validate dates
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            if start_dt >= end_dt:
                raise ValueError("start_date must be before end_date")
            # Validate reasonable date range (not more than 5 years)
            if (end_dt - start_dt).days > 1825:
                raise ValueError("Date range cannot exceed 5 years")
        except ValueError as e:
            if "time data" in str(e):
                raise ValueError("Invalid date format. Use YYYY-MM-DD")
            raise


async def main(symbol: str, strategy: str, timeframe: str, 
              start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Main entry point for direct backtesting execution.
    
    Args:
        symbol: Trading symbol
        strategy: Strategy name
        timeframe: Timeframe for analysis
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        Backtesting results
    """
    runner = BacktestingRunner()
    result = await runner.run_backtesting(
        symbol=symbol,
        strategy=strategy,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )
    return result


if __name__ == "__main__":
    # This allows the script to be run directly for testing
    if len(sys.argv) != 6:
        print("Usage: python run_backtesting.py <symbol> <strategy> <timeframe> <start_date> <end_date>")
        print("Example: python run_backtesting.py AAPL macd_rsi 1d 2024-01-01 2024-12-31")
        sys.exit(1)
    
    _, symbol, strategy, timeframe, start_date, end_date = sys.argv
    
    # Run the backtesting
    result = asyncio.run(main(
        symbol=symbol.upper(),
        strategy=strategy,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    ))
    
    # Print results
    print(f"\nBacktesting Results:")
    print(f"Status: {result['status']}")
    if result['status'] == 'completed':
        print(f"Signals found: {result['signal_count']}")
        
        # Print return statistics
        if 'return_stats' in result:
            stats = result['return_stats']
            print(f"\nReturn Statistics:")
            print(f"Total Return: {stats['total_return_pct']:.2f}%")
            print(f"Number of Trades: {stats['num_trades']}")
            if stats['num_trades'] > 0:
                print(f"Win Rate: {stats['win_rate']:.2f}%")
                print(f"Average Return per Trade: {stats['avg_return_per_trade']:.2f}%")
                print(f"Winning Trades: {stats['winning_trades']}")
                print(f"Losing Trades: {stats['losing_trades']}")
                if stats['best_trade']:
                    print(f"Best Trade: {stats['best_trade']['return_pct']:.2f}% "
                          f"({stats['best_trade']['entry_date']} to {stats['best_trade']['exit_date']})")
                if stats['worst_trade']:
                    print(f"Worst Trade: {stats['worst_trade']['return_pct']:.2f}% "
                          f"({stats['worst_trade']['entry_date']} to {stats['worst_trade']['exit_date']})")
        
        if result['signals']:
            print("\nSignals:")
            for i, signal in enumerate(result['signals'][:10]):  # Show first 10 signals
                print(f"  {i+1}. {signal['signal_type']} at {signal['price']:.2f} "
                      f"on {signal['timestamp']} (confidence: {signal['confidence']:.2f})")
            if len(result['signals']) > 10:
                print(f"  ... and {len(result['signals']) - 10} more signals")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")