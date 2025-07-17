#!/usr/bin/env python3
"""
1-Hour Analysis Trigger Script

Triggers 1-hour timeframe analysis for RSI trend and Bollinger breakout strategies
exactly as the realtime system does, using current prices and minimal historical data.
"""

import asyncio
import logging
import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.timezone_utils import utc_now
from src.data.providers.alpaca_provider import AlpacaProvider
from src.strategies.rsi_trend_analyzer import RSITrendAnalyzer
from src.strategies.bollinger_breakout_analyzer import BollingerBreakoutAnalyzer
from src.utils.data_requirements import DataRequirementsCalculator
from src.utils.market_calendar import ExtendedHoursCalendar
from src.notifications.models import NotificationMessage
from src.notifications.discord_client import DiscordNotificationSender
from src.notifications.routing import create_default_config_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Supported symbols (from environment)
SUPPORTED_SYMBOLS = os.getenv('SUPPORTED_SYMBOLS', '').split(',')
SUPPORTED_SYMBOLS = [s.strip() for s in SUPPORTED_SYMBOLS if s.strip()]

class RealTime1HAnalyzer:
    """Mimics the realtime 1-hour analysis exactly."""
    
    def __init__(self):
        """Initialize the analyzer with required components."""
        self.alpaca_provider = AlpacaProvider()
        
        # Initialize analyzers (RSI trend and Bollinger breakout for 1h only)
        self.rsi_analyzer = RSITrendAnalyzer()
        self.bollinger_analyzer = BollingerBreakoutAnalyzer()
        
        # Initialize market calendar for session checking (same as realtime system)
        self.market_calendar = ExtendedHoursCalendar()
        
        # No rate limiting needed - Alpaca supports 10,000 requests/minute
        # Our workload: 30 symbols × 3 calls = 90 requests (~9 req/sec vs 166 req/sec limit)
        
        # Calculate exact data requirements (same as realtime system)
        self.data_requirements = self._calculate_data_requirements()
        
        # Initialize deduplication tracker with persistent storage
        # Format: {('SYMBOL', 'strategy'): {'signal_type': 'BUY', 'date': '2025-07-16'}}
        self.signal_history_file = Path.home() / '.wagehood' / 'signal_history.json'
        self.signal_history_file.parent.mkdir(exist_ok=True)
        self.signal_history: Dict[Tuple[str, str], Dict[str, str]] = self._load_signal_history()
        
        # Initialize Discord notification sender
        self.discord_sender = None
        try:
            config_manager = create_default_config_manager()
            if config_manager.is_configured():
                self.discord_sender = DiscordNotificationSender(config_manager.get_all_configs())
                logger.info("Discord notification sender initialized")
            else:
                logger.warning("Discord not configured - notifications will be disabled")
        except Exception as e:
            logger.error(f"Failed to initialize Discord sender: {e}")
            self.discord_sender = None
        
        logger.info("1-Hour Analysis Trigger initialized")
        logger.info(f"RSI Trend requires: {self.data_requirements['rsi_trend']} hours")
        logger.info(f"Bollinger Breakout requires: {self.data_requirements['bollinger_breakout']} hours")
        logger.info(f"Loaded {len(self.signal_history)} signal history entries from persistent storage")
    
    def _calculate_data_requirements(self) -> Dict[str, int]:
        """Calculate exact data requirements for each strategy."""
        
        # RSI Trend requirements for 1h timeframe
        rsi_reqs = DataRequirementsCalculator.calculate_rsi_trend_requirements({})
        rsi_hours = rsi_reqs['recommended_minimum']  # This is in hours for 1h timeframe
        
        # Bollinger Breakout requirements for 1h timeframe  
        bollinger_reqs = DataRequirementsCalculator.calculate_bollinger_requirements({})
        bollinger_hours = bollinger_reqs['recommended_minimum']  # This is in hours for 1h timeframe
        
        return {
            'rsi_trend': rsi_hours,
            'bollinger_breakout': bollinger_hours
        }
    
    def _load_signal_history(self) -> Dict[Tuple[str, str], Dict[str, str]]:
        """Load signal history from persistent storage."""
        if self.signal_history_file.exists():
            try:
                with open(self.signal_history_file, 'r') as f:
                    data = json.load(f)
                    # Convert string keys back to tuples
                    result = {}
                    for key_str, value in data.items():
                        # Key format: "SYMBOL|strategy"
                        parts = key_str.split('|')
                        if len(parts) == 2:
                            # Handle backward compatibility: convert old format to new format
                            if isinstance(value, str):
                                # Old format: just signal type
                                result[(parts[0], parts[1])] = {
                                    'signal_type': value,
                                    'date': '2025-01-01'  # Old signals get old date to ensure they don't block new ones
                                }
                            else:
                                # New format: dict with signal_type and date
                                result[(parts[0], parts[1])] = value
                    return result
            except Exception as e:
                logger.warning(f"Failed to load signal history: {e}")
                return {}
        return {}
    
    def _save_signal_history(self):
        """Save signal history to persistent storage."""
        try:
            # Convert tuple keys to strings for JSON serialization
            data = {}
            for (symbol, strategy), signal_info in self.signal_history.items():
                key = f"{symbol}|{strategy}"
                data[key] = signal_info
            
            logger.info(f"Saving {len(data)} signal history entries to {self.signal_history_file}")
            with open(self.signal_history_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Successfully saved signal history to {self.signal_history_file}")
        except Exception as e:
            logger.error(f"Failed to save signal history to {self.signal_history_file}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    async def run_analysis(self):
        """Run the analysis for all symbols at the current moment."""
        trigger_time = utc_now()
        logger.info(f"🕐 Triggering 1-hour analysis at {trigger_time}")
        
        # Check if we're in extended trading hours (same as realtime system)
        if not self.market_calendar.is_extended_trading_hours(trigger_time):
            logger.warning("❌ Market is not in extended trading hours (4 AM - 8 PM EST)")
            logger.info("⏰ Extended trading hours: 4:00 AM - 8:00 PM EST on trading days")
            return
        
        # Get current trading session
        session_type = self.market_calendar.get_trading_session(trigger_time)
        logger.info(f"📈 Market session: {session_type.upper()}")
        
        if not SUPPORTED_SYMBOLS:
            logger.error("No supported symbols found in environment")
            return
        
        # Connect to Alpaca
        await self.alpaca_provider.connect()
        logger.info(f"📊 Connected to Alpaca, analyzing {len(SUPPORTED_SYMBOLS)} symbols")
        
        # Process all symbols in parallel for maximum speed
        logger.info(f"🚀 Processing {len(SUPPORTED_SYMBOLS)} symbols in parallel...")
        
        # Create tasks for all symbols
        tasks = []
        for symbol in SUPPORTED_SYMBOLS:
            task = self._analyze_symbol_parallel(symbol, trigger_time, session_type)
            tasks.append(task)
        
        # Execute all analyses in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and log
        successful_results = []
        for symbol, result in zip(SUPPORTED_SYMBOLS, results):
            if isinstance(result, Exception):
                logger.error(f"❌ Error analyzing {symbol}: {result}")
            elif result:
                successful_results.append(result)
                
                # Log results
                await self._log_results(
                    result['symbol'], 
                    result['current_price'], 
                    result['rsi_signals'], 
                    result['bollinger_signals'], 
                    result['session_type']
                )
        
        # Summary
        total_signals = sum(len(r['rsi_signals']) + len(r['bollinger_signals']) for r in successful_results)
        logger.info(f"📈 Analysis complete: {total_signals} signals generated across {len(successful_results)} symbols")
        
        # Close Discord sender if initialized
        if self.discord_sender:
            await self.discord_sender.close()
            logger.info("Discord notification sender closed")
        
        await self.alpaca_provider.disconnect()
    
    async def _analyze_symbol_parallel(self, symbol: str, trigger_time: datetime, session_type: str) -> Dict[str, Any]:
        """Analyze a single symbol in parallel with current price + both strategies."""
        try:
            # Get current price
            current_price = await self._get_current_price(symbol)
            if not current_price:
                logger.warning(f"❌ Could not get current price for {symbol}")
                return None
            
            # Run both strategies in parallel
            rsi_task = self._run_rsi_analysis(symbol, current_price, trigger_time)
            bollinger_task = self._run_bollinger_analysis(symbol, current_price, trigger_time)
            
            # Wait for both analyses to complete
            rsi_signals, bollinger_signals = await asyncio.gather(
                rsi_task, bollinger_task, return_exceptions=True
            )
            
            # Handle exceptions from strategy analyses
            if isinstance(rsi_signals, Exception):
                logger.error(f"RSI analysis failed for {symbol}: {rsi_signals}")
                rsi_signals = []
            
            if isinstance(bollinger_signals, Exception):
                logger.error(f"Bollinger analysis failed for {symbol}: {bollinger_signals}")
                bollinger_signals = []
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'rsi_signals': rsi_signals,
                'bollinger_signals': bollinger_signals,
                'session_type': session_type
            }
            
        except Exception as e:
            logger.error(f"Error in parallel analysis for {symbol}: {e}")
            return None
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current price using Alpaca bars/latest endpoint (same as realtime system)."""
        try:
            # Get latest bar (same as polling client)
            current_time = utc_now()
            data = await self.alpaca_provider.get_historical_data(
                symbol=symbol,
                start_date=current_time - timedelta(days=1),
                end_date=current_time,
                timeframe='1h'
            )
            
            if data and len(data) > 0:
                # Use the closing price of the latest bar (same as realtime system)
                # AlpacaProvider returns List[Dict[str, Any]], not DataFrame
                latest_bar = data[-1]  # Get last item from list
                return float(latest_bar['close'])
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    async def _run_rsi_analysis(self, symbol: str, current_price: float, trigger_time: datetime) -> List[Dict[str, Any]]:
        """Run RSI trend analysis with exact data requirements."""
        try:
            # Calculate start date using actual data requirements (not hardcoded)
            periods_needed = self.data_requirements['rsi_trend']
            calendar_days_needed = DataRequirementsCalculator.calculate_calendar_days_needed(
                periods_needed, '1h'
            )
            start_date = trigger_time - timedelta(days=calendar_days_needed)
            
            # Get historical data
            try:
                signals = await self.rsi_analyzer.analyze_symbol(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=trigger_time,
                    timeframe='1h'
                )
            except Exception as e:
                logger.error(f"RSI analyzer failed for {symbol}: {e}")
                return []
            
            # Filter for signals at the current moment (same as realtime system)
            current_signals = []
            for signal in signals:
                try:
                    signal_time = signal.get('timestamp')
                    if signal_time and isinstance(signal_time, datetime):
                        # Check if signal is within the last hour (current analysis window)
                        if (trigger_time - signal_time).total_seconds() < 3600:  # 1 hour
                            current_signals.append(signal)
                except Exception as e:
                    logger.warning(f"Error processing RSI signal for {symbol}: {e}")
                    continue
            
            return current_signals
            
        except Exception as e:
            logger.error(f"Error in RSI analysis for {symbol}: {e}")
            return []
    
    async def _run_bollinger_analysis(self, symbol: str, current_price: float, trigger_time: datetime) -> List[Dict[str, Any]]:
        """Run Bollinger breakout analysis with exact data requirements."""
        try:
            # Calculate start date using actual data requirements (not hardcoded)
            periods_needed = self.data_requirements['bollinger_breakout']
            calendar_days_needed = DataRequirementsCalculator.calculate_calendar_days_needed(
                periods_needed, '1h'
            )
            start_date = trigger_time - timedelta(days=calendar_days_needed)
            
            # Get historical data
            try:
                signals = await self.bollinger_analyzer.analyze_symbol(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=trigger_time,
                    timeframe='1h'
                )
            except Exception as e:
                logger.error(f"Bollinger analyzer failed for {symbol}: {e}")
                return []
            
            # Filter for signals at the current moment (same as realtime system)
            current_signals = []
            for signal in signals:
                try:
                    signal_time = signal.get('timestamp')
                    if signal_time and isinstance(signal_time, datetime):
                        # Check if signal is within the last hour (current analysis window)
                        if (trigger_time - signal_time).total_seconds() < 3600:  # 1 hour
                            current_signals.append(signal)
                except Exception as e:
                    logger.warning(f"Error processing Bollinger signal for {symbol}: {e}")
                    continue
            
            return current_signals
            
        except Exception as e:
            logger.error(f"Error in Bollinger analysis for {symbol}: {e}")
            return []
    
    async def _log_results(self, symbol: str, current_price: float, rsi_signals: List, bollinger_signals: List, session_type: str):
        """Log analysis results in the same format as realtime system and send notifications."""
        
        # Session indicator (same as realtime system)
        session_indicator = {
            'premarket': '🌅',
            'regular': '🏢', 
            'afterhours': '🌙'
        }.get(session_type, '❓')
        
        # Log current price with session indicator
        logger.info(f"💰 {symbol}: Current price ${current_price:.2f} {session_indicator}")
        
        # Log and notify RSI signals
        if rsi_signals:
            for signal in rsi_signals:
                signal_type = signal.get('signal_type', 'UNKNOWN')
                confidence = signal.get('confidence', 0)
                price = signal.get('price', current_price)
                
                emoji = "🟢" if signal_type == "BUY" else "🔴" if signal_type == "SELL" else "⚪"
                logger.info(f"{emoji} {symbol} RSI_TREND 1h: {signal_type} at ${price:.2f} (confidence: {confidence * 100:.1f}%) {session_indicator}")
                
                # Send notification if conditions are met
                await self._send_notification_if_needed(
                    symbol, 'rsi_trend', signal_type, price, confidence, session_type
                )
        else:
            logger.info(f"⚪ {symbol} RSI_TREND 1h: No signals {session_indicator}")
        
        # Log and notify Bollinger signals
        if bollinger_signals:
            for signal in bollinger_signals:
                signal_type = signal.get('signal_type', 'UNKNOWN')
                confidence = signal.get('confidence', 0)
                price = signal.get('price', current_price)
                
                emoji = "🟢" if signal_type == "BUY" else "🔴" if signal_type == "SELL" else "⚪"
                logger.info(f"{emoji} {symbol} BOLLINGER_BREAKOUT 1h: {signal_type} at ${price:.2f} (confidence: {confidence * 100:.1f}%) {session_indicator}")
                
                # Send notification if conditions are met
                await self._send_notification_if_needed(
                    symbol, 'bollinger_breakout', signal_type, price, confidence, session_type
                )
        else:
            logger.info(f"⚪ {symbol} BOLLINGER_BREAKOUT 1h: No signals {session_indicator}")
    
    async def _send_notification_if_needed(self, symbol: str, strategy: str, signal_type: str, 
                                         price: float, confidence: float, session_type: str):
        """Send notification if deduplication conditions are met."""
        if not self.discord_sender:
            return
        
        # Create key for tracking
        key = (symbol, strategy)
        
        # Get current date for comparison
        current_date = utc_now().date().isoformat()  # Format: '2025-07-16'
        
        # Check deduplication conditions
        last_signal_info = self.signal_history.get(key)
        
        # Send notification if:
        # 1. Never sent for this symbol/strategy combination
        # 2. Signal type is different from the last one sent
        # 3. Same signal type but different date (new occurrence)
        should_send = False
        if last_signal_info is None:
            should_send = True
            logger.info(f"📤 Sending notification for {symbol} {strategy} {signal_type} (first time)")
        elif last_signal_info['signal_type'] != signal_type:
            should_send = True
            logger.info(f"📤 Sending notification for {symbol} {strategy} {signal_type} (different signal: {last_signal_info['signal_type']} -> {signal_type})")
        elif last_signal_info['date'] != current_date:
            should_send = True
            logger.info(f"📤 Sending notification for {symbol} {strategy} {signal_type} (new date: {last_signal_info['date']} -> {current_date})")
        
        if should_send:
            try:
                # Create notification content
                emoji = "🟢" if signal_type == "BUY" else "🔴" if signal_type == "SELL" else "⚪"
                session_emoji = {
                    'premarket': '🌅',
                    'regular': '🏢',
                    'afterhours': '🌙'
                }.get(session_type, '❓')
                
                content = (
                    f"{emoji} **{symbol}** {signal_type} Signal\n"
                    f"📈 Strategy: {strategy.replace('_', ' ').title()}\n"
                    f"💰 Price: ${price:.2f}\n"
                    f"📊 Confidence: {confidence:.1%}\n"
                    f"⏰ Session: {session_type.title()} {session_emoji}"
                )
                
                # Create notification message
                notification = NotificationMessage.create_signal_notification(
                    symbol=symbol,
                    strategy=strategy,
                    timeframe='1h',
                    content=content
                )
                
                # Send notification
                success = await self.discord_sender.send_notification(notification)
                
                if success:
                    # Update signal history and persist
                    self.signal_history[key] = {
                        'signal_type': signal_type,
                        'date': current_date
                    }
                    self._save_signal_history()
                    logger.info(f"✅ Notification sent for {symbol} {strategy} {signal_type}")
                else:
                    logger.error(f"❌ Failed to send notification for {symbol} {strategy}")
                    
            except Exception as e:
                logger.error(f"Error sending notification for {symbol} {strategy}: {e}")
                # Don't crash - continue with other notifications
        else:
            # Notification skipped due to deduplication
            logger.info(f"⏭️ Skipping notification for {symbol} {strategy} {signal_type} (same signal on same date: {current_date})")

async def main():
    """Main function to run the 1-hour analysis."""
    try:
        analyzer = RealTime1HAnalyzer()
        await analyzer.run_analysis()
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())