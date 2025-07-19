#!/usr/bin/env python3
"""
1-Day Analysis Trigger Script

Triggers 1-day timeframe analysis for MACD+RSI and Support/Resistance strategies
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
from src.strategies.macd_rsi_analyzer import MACDRSIAnalyzer
from src.strategies.sr_breakout_analyzer import SRBreakoutAnalyzer
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

class RealTime1DAnalyzer:
    """Mimics the realtime 1-day analysis exactly."""
    
    def __init__(self):
        """Initialize the analyzer with required components."""
        self.alpaca_provider = AlpacaProvider()
        
        # Initialize analyzers (MACD+RSI and Support/Resistance for 1d only)
        self.macd_rsi_analyzer = MACDRSIAnalyzer()
        self.sr_analyzer = SRBreakoutAnalyzer()
        
        # Initialize market calendar for session checking (same as realtime system)
        self.market_calendar = ExtendedHoursCalendar()
        
        # Calculate exact data requirements (same as realtime system)
        self.data_requirements = self._calculate_data_requirements()
        
        # Initialize deduplication tracker with persistent storage
        # Format: {('SYMBOL', 'strategy'): 'BUY'}
        self.signal_history_file = Path.home() / '.wagehood' / 'signal_history_1d.json'
        self.signal_history_file.parent.mkdir(exist_ok=True)
        self.signal_history: Dict[Tuple[str, str], str] = self._load_signal_history()
        
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
        
        logger.info("1-Day Analysis Trigger initialized")
        logger.info(f"MACD+RSI requires: {self.data_requirements['macd_rsi']} days")
        logger.info(f"Support/Resistance requires: {self.data_requirements['sr_breakout']} days")
        logger.info(f"Loaded {len(self.signal_history)} signal history entries from persistent storage")
    
    def _calculate_data_requirements(self) -> Dict[str, int]:
        """Calculate exact data requirements for each strategy."""
        
        # MACD+RSI requirements for 1d timeframe
        macd_rsi_reqs = DataRequirementsCalculator.calculate_macd_rsi_requirements({})
        macd_rsi_days = macd_rsi_reqs['recommended_minimum']  # This is in days for 1d timeframe
        
        # Support/Resistance requirements for 1d timeframe  
        sr_reqs = DataRequirementsCalculator.calculate_sr_requirements({})
        sr_days = sr_reqs['recommended_minimum']  # This is in days for 1d timeframe
        
        return {
            'macd_rsi': macd_rsi_days,
            'sr_breakout': sr_days
        }
    
    def _load_signal_history(self) -> Dict[Tuple[str, str], str]:
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
                            # Handle backward compatibility
                            if isinstance(value, str):
                                # Old format: just signal type
                                result[(parts[0], parts[1])] = value
                            elif isinstance(value, dict) and 'signal_type' in value:
                                # Was dict format, extract signal_type only
                                result[(parts[0], parts[1])] = value['signal_type']
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
            for (symbol, strategy), signal_type in self.signal_history.items():
                key = f"{symbol}|{strategy}"
                data[key] = signal_type
            
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
        logger.info(f"📅 Triggering 1-day analysis at {trigger_time}")
        
        # For daily analysis, we check if it's a trading day
        if not self.market_calendar._is_trading_day(trigger_time.date()):
            logger.warning("❌ Today is not a trading day")
            return
        
        # Get current trading session for context
        session_type = self.market_calendar.get_trading_session(trigger_time)
        if session_type:
            logger.info(f"📈 Market session: {session_type.upper()}")
        else:
            logger.info("📈 Market is closed, using last available daily data")
            session_type = "closed"
        
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
                    result['macd_rsi_signals'], 
                    result['sr_signals'], 
                    result['session_type']
                )
        
        # Summary
        total_signals = sum(len(r['macd_rsi_signals']) + len(r['sr_signals']) for r in successful_results)
        logger.info(f"📈 Analysis complete: {total_signals} signals generated across {len(successful_results)} symbols")
        
        # Close Discord sender if initialized
        if self.discord_sender:
            await self.discord_sender.close()
            logger.info("Discord notification sender closed")
        
        await self.alpaca_provider.disconnect()
    
    async def _analyze_symbol_parallel(self, symbol: str, trigger_time: datetime, session_type: str) -> Dict[str, Any]:
        """Analyze a single symbol in parallel with current price + both strategies."""
        try:
            # Get current price (using latest daily bar)
            current_price = await self._get_current_price(symbol)
            if not current_price:
                logger.warning(f"❌ Could not get current price for {symbol}")
                return None
            
            # Run both strategies in parallel
            macd_rsi_task = self._run_macd_rsi_analysis(symbol, current_price, trigger_time)
            sr_task = self._run_sr_analysis(symbol, current_price, trigger_time)
            
            # Wait for both analyses to complete
            macd_rsi_signals, sr_signals = await asyncio.gather(
                macd_rsi_task, sr_task, return_exceptions=True
            )
            
            # Handle exceptions from strategy analyses
            if isinstance(macd_rsi_signals, Exception):
                logger.error(f"MACD+RSI analysis failed for {symbol}: {macd_rsi_signals}")
                macd_rsi_signals = []
            
            if isinstance(sr_signals, Exception):
                logger.error(f"Support/Resistance analysis failed for {symbol}: {sr_signals}")
                sr_signals = []
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'macd_rsi_signals': macd_rsi_signals,
                'sr_signals': sr_signals,
                'session_type': session_type
            }
            
        except Exception as e:
            logger.error(f"Error in parallel analysis for {symbol}: {e}")
            return None
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current price using Alpaca bars/latest endpoint (same as realtime system)."""
        try:
            # For daily data, get the latest daily bar
            current_time = utc_now()
            data = await self.alpaca_provider.get_historical_data(
                symbol=symbol,
                start_date=current_time - timedelta(days=5),  # Get last few days
                end_date=current_time,
                timeframe='1d'
            )
            
            if data and len(data) > 0:
                # Use the closing price of the latest bar
                # AlpacaProvider returns List[Dict[str, Any]], not DataFrame
                latest_bar = data[-1]  # Get last item from list
                return float(latest_bar['close'])
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    async def _run_macd_rsi_analysis(self, symbol: str, current_price: float, trigger_time: datetime) -> List[Dict[str, Any]]:
        """Run MACD+RSI analysis with exact data requirements."""
        try:
            # Calculate start date using actual data requirements
            periods_needed = self.data_requirements['macd_rsi']
            calendar_days_needed = DataRequirementsCalculator.calculate_calendar_days_needed(
                periods_needed, '1d'
            )
            start_date = trigger_time - timedelta(days=calendar_days_needed)
            
            # Get historical data
            try:
                signals = await self.macd_rsi_analyzer.analyze_symbol(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=trigger_time,
                    timeframe='1d'
                )
            except Exception as e:
                logger.error(f"MACD+RSI analyzer failed for {symbol}: {e}")
                return []
            
            # Filter for signals at the current moment (most recent day)
            current_signals = []
            for signal in signals:
                try:
                    signal_time = signal.get('timestamp')
                    if signal_time and isinstance(signal_time, datetime):
                        # Check if signal is from today or yesterday (for daily timeframe)
                        days_diff = (trigger_time.date() - signal_time.date()).days
                        if days_diff <= 1:  # Today or yesterday
                            current_signals.append(signal)
                except Exception as e:
                    logger.warning(f"Error processing MACD+RSI signal for {symbol}: {e}")
                    continue
            
            return current_signals
            
        except Exception as e:
            logger.error(f"Error in MACD+RSI analysis for {symbol}: {e}")
            return []
    
    async def _run_sr_analysis(self, symbol: str, current_price: float, trigger_time: datetime) -> List[Dict[str, Any]]:
        """Run Support/Resistance analysis with exact data requirements."""
        try:
            # Calculate start date using actual data requirements
            periods_needed = self.data_requirements['sr_breakout']
            calendar_days_needed = DataRequirementsCalculator.calculate_calendar_days_needed(
                periods_needed, '1d'
            )
            start_date = trigger_time - timedelta(days=calendar_days_needed)
            
            # Get historical data
            try:
                signals = await self.sr_analyzer.analyze_symbol(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=trigger_time,
                    timeframe='1d'
                )
            except Exception as e:
                logger.error(f"Support/Resistance analyzer failed for {symbol}: {e}")
                return []
            
            # Filter for signals at the current moment (most recent day)
            current_signals = []
            for signal in signals:
                try:
                    signal_time = signal.get('timestamp')
                    if signal_time and isinstance(signal_time, datetime):
                        # Check if signal is from today or yesterday (for daily timeframe)
                        days_diff = (trigger_time.date() - signal_time.date()).days
                        if days_diff <= 1:  # Today or yesterday
                            current_signals.append(signal)
                except Exception as e:
                    logger.warning(f"Error processing S/R signal for {symbol}: {e}")
                    continue
            
            return current_signals
            
        except Exception as e:
            logger.error(f"Error in Support/Resistance analysis for {symbol}: {e}")
            return []
    
    async def _log_results(self, symbol: str, current_price: float, macd_rsi_signals: List, sr_signals: List, session_type: str):
        """Log analysis results in the same format as realtime system and send notifications."""
        
        # Session indicator for daily timeframe
        session_indicator = {
            'premarket': '🌅',
            'regular': '🏢', 
            'afterhours': '🌙',
            'closed': '🔒'
        }.get(session_type, '❓')
        
        # Log current price with session indicator
        logger.info(f"💰 {symbol}: Current price ${current_price:.2f} {session_indicator}")
        
        # Log and notify MACD+RSI signals
        if macd_rsi_signals:
            for signal in macd_rsi_signals:
                signal_type = signal.get('signal_type', 'UNKNOWN')
                confidence = signal.get('confidence', 0)
                price = signal.get('price', current_price)
                
                emoji = "🟢" if signal_type == "BUY" else "🔴" if signal_type == "SELL" else "⚪"
                logger.info(f"{emoji} {symbol} MACD_RSI 1d: {signal_type} at ${price:.2f} (confidence: {confidence * 100:.1f}%) {session_indicator}")
                
                # Send notification if conditions are met
                await self._send_notification_if_needed(
                    symbol, 'macd_rsi', signal_type, price, confidence, session_type
                )
        else:
            logger.info(f"⚪ {symbol} MACD_RSI 1d: No signals {session_indicator}")
        
        # Log and notify Support/Resistance signals
        if sr_signals:
            for signal in sr_signals:
                signal_type = signal.get('signal_type', 'UNKNOWN')
                confidence = signal.get('confidence', 0)
                price = signal.get('price', current_price)
                
                emoji = "🟢" if signal_type == "BUY" else "🔴" if signal_type == "SELL" else "⚪"
                logger.info(f"{emoji} {symbol} SR_BREAKOUT 1d: {signal_type} at ${price:.2f} (confidence: {confidence * 100:.1f}%) {session_indicator}")
                
                # Send notification if conditions are met
                await self._send_notification_if_needed(
                    symbol, 'sr_breakout', signal_type, price, confidence, session_type
                )
        else:
            logger.info(f"⚪ {symbol} SR_BREAKOUT 1d: No signals {session_indicator}")
    
    async def _send_notification_if_needed(self, symbol: str, strategy: str, signal_type: str, 
                                         price: float, confidence: float, session_type: str):
        """Send notification if deduplication conditions are met."""
        if not self.discord_sender:
            return
        
        # Create key for tracking
        key = (symbol, strategy)
        
        # Check deduplication conditions
        last_signal_type = self.signal_history.get(key)
        
        # Send notification if:
        # 1. Never sent for this symbol/strategy combination
        # 2. Signal type is different from the last one sent
        should_send = False
        if last_signal_type is None:
            should_send = True
            logger.info(f"📤 Sending notification for {symbol} {strategy} {signal_type} (first time)")
        elif last_signal_type != signal_type:
            should_send = True
            logger.info(f"📤 Sending notification for {symbol} {strategy} {signal_type} (signal changed: {last_signal_type} -> {signal_type})")
        
        if should_send:
            try:
                # Create notification content
                emoji = "🟢" if signal_type == "BUY" else "🔴" if signal_type == "SELL" else "⚪"
                session_emoji = {
                    'premarket': '🌅',
                    'regular': '🏢',
                    'afterhours': '🌙',
                    'closed': '🔒'
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
                    timeframe='1d',
                    content=content
                )
                
                # Send notification
                success = await self.discord_sender.send_notification(notification)
                
                if success:
                    # Update signal history and persist
                    self.signal_history[key] = signal_type
                    self._save_signal_history()
                    logger.info(f"✅ Notification sent for {symbol} {strategy} {signal_type}")
                else:
                    logger.error(f"❌ Failed to send notification for {symbol} {strategy}")
                    
            except Exception as e:
                logger.error(f"Error sending notification for {symbol} {strategy}: {e}")
                # Don't crash - continue with other notifications
        else:
            # Notification skipped due to deduplication
            logger.info(f"⏭️ Skipping notification for {symbol} {strategy} {signal_type} (same signal type)")

async def main():
    """Main function to run the 1-day analysis."""
    try:
        analyzer = RealTime1DAnalyzer()
        await analyzer.run_analysis()
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())