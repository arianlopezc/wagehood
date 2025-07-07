"""
Multi-Channel Notification Service

Enhanced notification service that routes signals to strategy-specific Discord channels
while maintaining compatibility with the existing notification system.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional

from .notification_service import NotificationService
from .multi_channel_config import MultiChannelNotificationConfig
from .multi_channel_notifier import MultiChannelDiscordNotifier
from .config import NotificationConfig

logger = logging.getLogger(__name__)


class MultiChannelNotificationService(NotificationService):
    """
    Enhanced notification service with multi-channel Discord support.
    
    Extends the existing NotificationService to add strategy-specific routing
    while maintaining backward compatibility.
    """
    
    def __init__(self, config: Optional[NotificationConfig] = None):
        """
        Initialize multi-channel notification service.
        
        Args:
            config: Legacy notification configuration (optional)
        """
        # Load multi-channel configuration
        self.multi_config = MultiChannelNotificationConfig.from_environment()
        
        # Initialize based on multi-channel settings
        if self.multi_config.enabled and self.multi_config.multi_channel_enabled:
            # Use multi-channel mode
            logger.info("Initializing multi-channel Discord notification service")
            self._initialize_multi_channel_mode()
        else:
            # Fall back to single-channel mode
            logger.info("Multi-channel disabled, using single-channel notification service")
            super().__init__(config)
            return
        
        # Initialize base class components (Redis, etc.) but not Discord notifier
        self.config = config or NotificationConfig.from_environment()
        self.running = False
        self.tasks = []
        
        # Initialize Redis connection
        self.redis_client = None
        self._last_notification_times = {}
        
        logger.info("Multi-channel notification service initialized")
    
    def _initialize_multi_channel_mode(self):
        """Initialize multi-channel Discord components."""
        self.multi_channel_notifier = MultiChannelDiscordNotifier(self.multi_config)
        self.is_multi_channel_mode = True
        
        logger.info(f"Multi-channel mode initialized with {len(self.multi_config.strategy_channels)} strategy channels")
    
    async def _process_signal_event(self, msg_id: str, fields: Dict[str, str]):
        """
        Process a signal event from Redis stream with multi-channel routing.
        
        Args:
            msg_id: Redis message ID
            fields: Event fields
        """
        try:
            # Parse event data
            event_data = self._parse_signal_event(fields)
            if not event_data:
                return
            
            # Check if this is a signal we should notify about
            if not self._should_notify_signal(event_data):
                logger.debug(f"Skipping notification for {event_data.get('symbol', 'unknown')}")
                return
            
            # Route to appropriate channel(s)
            success = await self._route_signal_notification(event_data)
            
            if success:
                symbol = event_data.get('symbol', 'unknown')
                strategy = event_data.get('strategy', 'unknown')
                signal_type = event_data.get('signal', 'unknown')
                self._last_notification_times[symbol] = time.time()
                logger.info(f"Sent {strategy} {signal_type} notification for {symbol}")
            else:
                logger.error("Failed to send multi-channel notification")
                
        except Exception as e:
            logger.error(f"Error processing signal event: {e}")
    
    async def _route_signal_notification(self, signal_data: Dict[str, Any]) -> bool:
        """
        Route signal notification to appropriate strategy channel.
        
        Args:
            signal_data: Signal data dictionary
            
        Returns:
            True if notification sent successfully
        """
        try:
            if not hasattr(self, 'is_multi_channel_mode') or not self.is_multi_channel_mode:
                # Fall back to single-channel mode
                return super().discord_notifier.send_signal_notification(signal_data)
            
            # Extract strategy information
            strategy = self._extract_strategy_name(signal_data)
            
            if not strategy:
                logger.warning(f"No strategy identified for signal: {signal_data}")
                strategy = "unknown"
            
            # Send to strategy-specific channel
            return self.multi_channel_notifier.send_strategy_notification(strategy, signal_data)
            
        except Exception as e:
            logger.error(f"Error routing signal notification: {e}")
            return False
    
    def _extract_strategy_name(self, signal_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract strategy name from signal data.
        
        Args:
            signal_data: Signal data dictionary
            
        Returns:
            Strategy name or None if not identifiable
        """
        # Try to get strategy from signal data
        strategy = signal_data.get('strategy', '')
        
        if strategy:
            # Normalize strategy name for channel routing
            strategy_lower = strategy.lower()
            
            # Map various strategy name formats to channel keys
            if 'macd' in strategy_lower and 'rsi' in strategy_lower:
                return 'macd_rsi'
            elif 'rsi' in strategy_lower and ('trend' in strategy_lower or 'following' in strategy_lower):
                return 'rsi_trend'
            elif 'bollinger' in strategy_lower:
                return 'bollinger_breakout'
            elif 'support' in strategy_lower or 'resistance' in strategy_lower or 'sr' in strategy_lower:
                return 'sr_breakout'
            elif 'moving' in strategy_lower and 'average' in strategy_lower:
                # Moving Average Crossover - not configured for notifications
                logger.debug("Moving Average Crossover strategy filtered out (not recommended for swing trading)")
                return None
        
        # Try to extract from details or other fields
        details = signal_data.get('details', {})
        if details:
            # Check for strategy-specific indicators in details
            if 'macd_signal' in details or 'rsi_value' in details:
                return 'macd_rsi'
            elif 'band_position' in details or 'volatility' in details:
                return 'bollinger_breakout'
            elif 'level_type' in details or 'level_strength' in details:
                return 'sr_breakout'
        
        # If we can't identify the strategy, log it and return None for fallback
        logger.debug(f"Could not identify strategy from signal data: {signal_data.get('strategy', 'unknown')}")
        return None
    
    def _should_notify_signal(self, signal_data: Dict[str, Any]) -> bool:
        """
        Enhanced signal filtering for multi-channel mode.
        
        Args:
            signal_data: Signal data dictionary
            
        Returns:
            True if notification should be sent
        """
        try:
            symbol = signal_data.get('symbol', '')
            confidence = signal_data.get('confidence', 0.0)
            signal_type = signal_data.get('signal', '').upper()
            timeframe = signal_data.get('timeframe', 'unknown')
            strategy = self._extract_strategy_name(signal_data)
            
            # Check symbol filter (using base config)
            if hasattr(self, 'config') and self.config and not self.config.should_notify_symbol(symbol):
                logger.debug(f"Symbol {symbol} not in notification list")
                return False
            
            # Check timeframe filter (CRITICAL: Only notify for allowed timeframes)
            if hasattr(self, 'config') and self.config and not self.config.should_notify_timeframe(timeframe):
                logger.debug(f"Timeframe {timeframe} not in notification list - skipping Discord notification")
                return False
            
            # Skip Moving Average Crossover (not recommended for swing trading)
            if strategy is None:
                # This includes MA Crossover and other unidentified strategies
                logger.debug(f"Strategy not configured for notifications: {signal_data.get('strategy', 'unknown')}")
                return False
            
            # Check if strategy has a configured channel
            if hasattr(self, 'multi_config') and not self.multi_config.has_strategy_channel(strategy):
                logger.debug(f"No channel configured for strategy: {strategy}")
                return False
            
            # In multi-channel mode, we trust the core signal logic (no confidence filtering)
            # Each strategy channel can have its own rate limiting
            
            # Only notify for BUY/SELL signals, not HOLD
            if signal_type == 'HOLD':
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking notification criteria: {e}")
            return False
    
    async def _send_startup_notification(self):
        """Send startup notification with multi-channel information."""
        try:
            if not hasattr(self, 'is_multi_channel_mode') or not self.is_multi_channel_mode:
                # Use base class startup notification
                return await super()._send_startup_notification()
            
            # Send startup notification to each configured strategy channel
            startup_count = 0
            enabled_strategies = self.multi_config.get_enabled_strategies()
            
            for strategy_key in enabled_strategies:
                strategy_config = self.multi_config.strategy_channels[strategy_key]
                
                status_data = {
                    'service': f'Discord Notifications - {strategy_config.strategy_name}',
                    'status': 'running',
                    'strategy': strategy_config.strategy_name,
                    'timeframe': '1d (swing trading)',
                    'rate_limit': f"{strategy_config.max_notifications_per_hour}/hour",
                    'message': f'Monitoring {strategy_config.strategy_name} swing trading signals'
                }
                
                # Create startup embed
                embed = {
                    "title": f"{strategy_config.emoji} {strategy_config.strategy_name} Monitor Started",
                    "color": strategy_config.channel_color,
                    "fields": [
                        {"name": "📊 Strategy", "value": strategy_config.strategy_name, "inline": True},
                        {"name": "⏰ Timeframe", "value": "1d (Daily)", "inline": True},
                        {"name": "📈 Profile", "value": "Swing Trading", "inline": True},
                        {"name": "🎯 Rate Limit", "value": f"{strategy_config.max_notifications_per_hour}/hour", "inline": True},
                        {"name": "💾 Data Source", "value": "Alpaca Markets", "inline": True},
                        {"name": "✅ Status", "value": "Active & Monitoring", "inline": True}
                    ],
                    "footer": {"text": f"Wagehood • {strategy_config.strategy_name} Channel"},
                    "timestamp": datetime.now().isoformat()
                }
                
                # Send to strategy channel
                notifier = self.multi_channel_notifier.strategy_notifiers.get(strategy_key)
                if notifier:
                    success = notifier.send_embed(embed)
                    if success:
                        startup_count += 1
                        logger.info(f"Sent startup notification to {strategy_config.strategy_name} channel")
            
            logger.info(f"Sent startup notifications to {startup_count}/{len(enabled_strategies)} strategy channels")
            
        except Exception as e:
            logger.error(f"Error sending multi-channel startup notification: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get enhanced statistics including multi-channel information.
        
        Returns:
            Statistics dictionary
        """
        if hasattr(self, 'is_multi_channel_mode') and self.is_multi_channel_mode:
            # Multi-channel statistics
            base_stats = {
                'enabled': self.multi_config.enabled,
                'running': self.running,
                'mode': 'multi_channel',
                'multi_channel_enabled': self.multi_config.multi_channel_enabled,
                'last_notifications': self._last_notification_times.copy()
            }
            
            # Add multi-channel specific stats
            if hasattr(self, 'multi_channel_notifier'):
                strategy_stats = self.multi_channel_notifier.get_strategy_stats()
                base_stats.update(strategy_stats)
            
            return base_stats
        else:
            # Single-channel statistics
            stats = super().get_stats()
            stats['mode'] = 'single_channel'
            return stats