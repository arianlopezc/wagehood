"""
Multi-Channel Discord Notifier

Routes trading signals to strategy-specific Discord channels with individual
rate limiting and configuration per strategy.
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime

from .discord_notifier import DiscordNotifier
from .multi_channel_config import MultiChannelNotificationConfig, StrategyChannelConfig
from .config import NotificationRateLimiter

logger = logging.getLogger(__name__)


class MultiChannelDiscordNotifier:
    """
    Multi-channel Discord notifier that routes signals to strategy-specific channels.
    """
    
    def __init__(self, config: MultiChannelNotificationConfig):
        """
        Initialize multi-channel Discord notifier.
        
        Args:
            config: Multi-channel notification configuration
        """
        self.config = config
        self.strategy_notifiers: Dict[str, DiscordNotifier] = {}
        self.strategy_rate_limiters: Dict[str, NotificationRateLimiter] = {}
        
        # Initialize strategy-specific notifiers and rate limiters
        self._initialize_strategy_notifiers()
        
        # Fallback notifier for unknown strategies
        self.fallback_notifier = None
        if self.config.fallback_webhook_url:
            self.fallback_notifier = DiscordNotifier(self.config.fallback_webhook_url)
            self.fallback_rate_limiter = NotificationRateLimiter(self.config.fallback_rate_limit)
            logger.info("Fallback Discord notifier initialized")
    
    def _initialize_strategy_notifiers(self):
        """Initialize Discord notifiers and rate limiters for each strategy."""
        for strategy_key, strategy_config in self.config.strategy_channels.items():
            if strategy_config.enabled and strategy_config.webhook_url:
                # Create Discord notifier for this strategy
                notifier = DiscordNotifier(strategy_config.webhook_url)
                self.strategy_notifiers[strategy_key] = notifier
                
                # Create rate limiter for this strategy
                rate_limiter = NotificationRateLimiter(strategy_config.max_notifications_per_hour)
                self.strategy_rate_limiters[strategy_key] = rate_limiter
                
                logger.info(f"Initialized Discord notifier for {strategy_config.strategy_name}")
    
    def send_strategy_notification(self, strategy_name: str, signal_data: Dict[str, Any]) -> bool:
        """
        Send notification to strategy-specific channel.
        
        Args:
            strategy_name: Name of the strategy
            signal_data: Signal data dictionary
            
        Returns:
            True if notification sent successfully
        """
        try:
            # Get strategy configuration
            strategy_config = self.config.get_strategy_config(strategy_name)
            
            if not strategy_config:
                logger.debug(f"No channel configured for strategy: {strategy_name}")
                return self._send_fallback_notification(signal_data, f"Unknown strategy: {strategy_name}")
            
            # Get normalized strategy key
            strategy_key = self.config._normalize_strategy_name(strategy_name)
            
            # Check if strategy is enabled
            if not strategy_config.enabled:
                logger.debug(f"Strategy channel disabled: {strategy_name}")
                return False
            
            # Get notifier and rate limiter for this strategy
            notifier = self.strategy_notifiers.get(strategy_key)
            rate_limiter = self.strategy_rate_limiters.get(strategy_key)
            
            if not notifier or not rate_limiter:
                logger.error(f"Notifier or rate limiter not found for strategy: {strategy_name}")
                return self._send_fallback_notification(signal_data, f"Configuration error: {strategy_name}")
            
            # Check rate limiting for this strategy
            if not rate_limiter.can_send_notification():
                logger.warning(f"Rate limit exceeded for strategy {strategy_name}")
                return False
            
            # Create strategy-specific embed
            embed = self._create_strategy_embed(strategy_config, signal_data)
            
            # Send notification
            success = notifier.send_embed(embed)
            
            if success:
                rate_limiter.record_notification()
                logger.info(f"Sent {strategy_name} notification to Discord channel")
            else:
                logger.error(f"Failed to send {strategy_name} notification")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending strategy notification for {strategy_name}: {e}")
            return False
    
    def _send_fallback_notification(self, signal_data: Dict[str, Any], reason: str) -> bool:
        """
        Send notification to fallback channel.
        
        Args:
            signal_data: Signal data dictionary
            reason: Reason for using fallback
            
        Returns:
            True if fallback notification sent successfully
        """
        if not self.fallback_notifier or not self.fallback_rate_limiter:
            logger.debug(f"No fallback notifier configured: {reason}")
            return False
        
        # Check fallback rate limiting
        if not self.fallback_rate_limiter.can_send_notification():
            logger.warning("Fallback rate limit exceeded")
            return False
        
        try:
            # Create generic embed for fallback
            embed = self._create_fallback_embed(signal_data, reason)
            
            # Send to fallback channel
            success = self.fallback_notifier.send_embed(embed)
            
            if success:
                self.fallback_rate_limiter.record_notification()
                logger.info(f"Sent fallback notification: {reason}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending fallback notification: {e}")
            return False
    
    def _create_strategy_embed(self, strategy_config: StrategyChannelConfig, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create strategy-specific Discord embed.
        
        Args:
            strategy_config: Strategy channel configuration
            signal_data: Signal data dictionary
            
        Returns:
            Discord embed dictionary
        """
        # Extract signal information
        symbol = signal_data.get('symbol', 'UNKNOWN')
        signal_type = signal_data.get('signal', 'UNKNOWN').upper()
        price = signal_data.get('price', 0.0)
        price_change_pct = signal_data.get('price_change_pct', 0.0)
        confidence = signal_data.get('confidence', 0.0)
        timeframe = signal_data.get('timeframe', '1d')
        timestamp = signal_data.get('timestamp', datetime.now())
        
        # Format confidence as percentage
        confidence_pct = int(confidence * 100) if confidence <= 1.0 else int(confidence)
        
        # Format price change
        price_change_sign = "+" if price_change_pct >= 0 else ""
        price_display = f"${price:.2f} ({price_change_sign}{price_change_pct:.1f}%)"
        
        # Get company name if available
        company_name = signal_data.get('company_name', '')
        symbol_display = f"{symbol} - {company_name}" if company_name else symbol
        
        # Create strategy-specific embed
        embed = {
            "title": f"{strategy_config.emoji} {strategy_config.strategy_name.upper()} SIGNAL",
            "color": strategy_config.channel_color,
            "fields": [
                {
                    "name": f"🎯 {symbol_display}",
                    "value": f"**{signal_type} SIGNAL**",
                    "inline": False
                },
                {
                    "name": "💰 Price",
                    "value": price_display,
                    "inline": True
                },
                {
                    "name": "🎯 Confidence", 
                    "value": f"{confidence_pct}%",
                    "inline": True
                },
                {
                    "name": "⏰ Timeframe",
                    "value": timeframe.upper(),
                    "inline": True
                }
            ],
            "footer": {
                "text": f"Paper Trading • Not Financial Advice • Wagehood • {strategy_config.strategy_name}"
            },
            "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else datetime.now().isoformat()
        }
        
        # Add strategy-specific details if available
        details = signal_data.get('details', {})
        if details:
            detail_text = self._format_strategy_details(strategy_config.strategy_name, details)
            if detail_text:
                embed["fields"].append({
                    "name": "📋 Signal Details",
                    "value": detail_text,
                    "inline": False
                })
        
        return embed
    
    def _format_strategy_details(self, strategy_name: str, details: Dict[str, Any]) -> str:
        """
        Format strategy-specific details for embed.
        
        Args:
            strategy_name: Name of the strategy
            details: Details dictionary
            
        Returns:
            Formatted details string
        """
        if not details:
            return ""
        
        # Strategy-specific detail formatting
        if 'macd' in strategy_name.lower() and 'rsi' in strategy_name.lower():
            # MACD+RSI specific details
            parts = []
            if 'macd_signal' in details:
                parts.append(f"• MACD: {details['macd_signal']}")
            if 'rsi_value' in details:
                parts.append(f"• RSI: {details['rsi_value']}")
            if 'macd_histogram' in details:
                parts.append(f"• Histogram: {details['macd_histogram']}")
            return "\n".join(parts)
        
        elif 'rsi' in strategy_name.lower() and 'trend' in strategy_name.lower():
            # RSI Trend specific details
            parts = []
            if 'rsi_value' in details:
                parts.append(f"• RSI: {details['rsi_value']}")
            if 'trend_direction' in details:
                parts.append(f"• Trend: {details['trend_direction']}")
            if 'momentum' in details:
                parts.append(f"• Momentum: {details['momentum']}")
            return "\n".join(parts)
        
        elif 'bollinger' in strategy_name.lower():
            # Bollinger Bands specific details
            parts = []
            if 'band_position' in details:
                parts.append(f"• Band Position: {details['band_position']}")
            if 'volatility' in details:
                parts.append(f"• Volatility: {details['volatility']}")
            if 'squeeze' in details:
                parts.append(f"• Squeeze: {details['squeeze']}")
            return "\n".join(parts)
        
        elif 'resistance' in strategy_name.lower() or 'support' in strategy_name.lower():
            # Support/Resistance specific details
            parts = []
            if 'level_type' in details:
                parts.append(f"• Level: {details['level_type']}")
            if 'level_strength' in details:
                parts.append(f"• Strength: {details['level_strength']}")
            if 'breakout_volume' in details:
                parts.append(f"• Volume: {details['breakout_volume']}")
            return "\n".join(parts)
        
        # Generic details formatting
        return "• " + "\n• ".join([f"{k}: {v}" for k, v in details.items()][:3])
    
    def _create_fallback_embed(self, signal_data: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """
        Create fallback Discord embed for unknown strategies.
        
        Args:
            signal_data: Signal data dictionary
            reason: Reason for fallback
            
        Returns:
            Discord embed dictionary
        """
        symbol = signal_data.get('symbol', 'UNKNOWN')
        signal_type = signal_data.get('signal', 'UNKNOWN').upper()
        price = signal_data.get('price', 0.0)
        confidence = signal_data.get('confidence', 0.0)
        strategy = signal_data.get('strategy', 'Unknown Strategy')
        timestamp = signal_data.get('timestamp', datetime.now())
        
        confidence_pct = int(confidence * 100) if confidence <= 1.0 else int(confidence)
        
        embed = {
            "title": "🔄 FALLBACK TRADING SIGNAL",
            "color": 8421504,  # Gray color for fallback
            "fields": [
                {
                    "name": f"📈 {symbol}",
                    "value": f"**{signal_type} SIGNAL**",
                    "inline": False
                },
                {
                    "name": "💰 Price",
                    "value": f"${price:.2f}",
                    "inline": True
                },
                {
                    "name": "🎯 Confidence",
                    "value": f"{confidence_pct}%",
                    "inline": True
                },
                {
                    "name": "📊 Strategy",
                    "value": strategy,
                    "inline": True
                },
                {
                    "name": "⚠️ Fallback Reason",
                    "value": reason,
                    "inline": False
                }
            ],
            "footer": {
                "text": "Paper Trading • Not Financial Advice • Wagehood • Fallback Channel"
            },
            "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else datetime.now().isoformat()
        }
        
        return embed
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all strategy channels.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            'enabled': self.config.enabled,
            'multi_channel_enabled': self.config.multi_channel_enabled,
            'total_strategies': len(self.config.strategy_channels),
            'enabled_strategies': len(self.config.get_enabled_strategies()),
            'strategy_stats': {}
        }
        
        # Get stats for each strategy
        for strategy_key, strategy_config in self.config.strategy_channels.items():
            rate_limiter = self.strategy_rate_limiters.get(strategy_key)
            strategy_stats = {
                'enabled': strategy_config.enabled,
                'webhook_configured': bool(strategy_config.webhook_url),
                'max_notifications_per_hour': strategy_config.max_notifications_per_hour,
                'channel_color': strategy_config.channel_color,
                'emoji': strategy_config.emoji
            }
            
            if rate_limiter:
                limiter_stats = rate_limiter.get_status()
                strategy_stats.update(limiter_stats)
            
            stats['strategy_stats'][strategy_key] = strategy_stats
        
        return stats