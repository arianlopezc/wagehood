"""
Discord webhook client for sending notifications.
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

from .models import NotificationMessage, ChannelConfig
from .summary_formatter import SummaryFormatter

logger = logging.getLogger(__name__)


class DiscordWebhookError(Exception):
    """Discord webhook specific error."""
    pass


class DiscordRateLimitError(DiscordWebhookError):
    """Discord rate limit exceeded error."""
    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"Rate limited, retry after {retry_after} seconds")


class DiscordClient:
    """
    Discord webhook client with error handling and rate limiting.
    
    Provides reliable message delivery to Discord channels via webhooks
    with automatic retry logic and proper error handling.
    """
    
    def __init__(self, timeout: float = 30.0, max_retries: int = 3):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def send_message(self, webhook_url: str, content: str, 
                          embeds: Optional[list] = None,
                          username: Optional[str] = None) -> bool:
        """
        Send message to Discord via webhook.
        
        Args:
            webhook_url: Discord webhook URL
            content: Message content (up to 2000 characters)
            embeds: Optional list of embed objects
            username: Optional custom username
            
        Returns:
            True if message sent successfully, False otherwise
        """
        if not webhook_url:
            logger.error("No webhook URL provided")
            return False
        
        # Prepare payload
        payload = {"content": content[:2000]}  # Discord limit
        
        if embeds:
            payload["embeds"] = embeds
        if username:
            payload["username"] = username
        
        session = await self._get_session()
        
        for attempt in range(self.max_retries):
            try:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 204:
                        # Success
                        logger.debug(f"Message sent to Discord successfully")
                        return True
                    elif response.status == 429:
                        # Rate limited
                        retry_after = float(response.headers.get('Retry-After', 1))
                        logger.warning(f"Discord rate limited, retry after {retry_after}s")
                        
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(retry_after)
                            continue
                        else:
                            raise DiscordRateLimitError(retry_after)
                    else:
                        # Other error
                        error_text = await response.text()
                        logger.error(f"Discord webhook error {response.status}: {error_text}")
                        
                        if response.status >= 500:
                            # Server error, retry
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                continue
                        
                        return False
                        
            except asyncio.TimeoutError:
                logger.error(f"Timeout sending to Discord (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return False
            except Exception as e:
                logger.error(f"Error sending to Discord (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return False
        
        return False
    
    async def test_webhook(self, webhook_url: str) -> bool:
        """
        Test webhook connectivity.
        
        Args:
            webhook_url: Discord webhook URL to test
            
        Returns:
            True if webhook is accessible, False otherwise
        """
        try:
            test_message = f"🧪 Webhook test at {datetime.now().strftime('%H:%M:%S')}"
            return await self.send_message(webhook_url, test_message, username="Wagehood Test")
        except Exception as e:
            logger.error(f"Webhook test failed: {e}")
            return False


class NotificationFormatter:
    """
    Formats notification messages for Discord with appropriate styling.
    """
    
    @staticmethod
    def format_signal_message(message: NotificationMessage) -> Dict[str, Any]:
        """
        Format signal notification for Discord.
        
        Args:
            message: Signal notification message
            
        Returns:
            Dictionary with formatted content and embeds
        """
        # Determine emoji based on signal type
        signal_emoji = "📈"  # Default
        if "BUY" in message.content.upper():
            signal_emoji = "🟢"
        elif "SELL" in message.content.upper():
            signal_emoji = "🔴"
        
        # Create embed for rich formatting
        embed = {
            "title": f"{signal_emoji} {message.strategy.upper()} Signal",
            "description": message.content,
            "color": 0x00ff00 if "BUY" in message.content.upper() else 0xff0000,
            "fields": [
                {
                    "name": "Symbol",
                    "value": message.symbol or "N/A",
                    "inline": True
                },
                {
                    "name": "Timeframe", 
                    "value": message.timeframe or "N/A",
                    "inline": True
                },
                {
                    "name": "Strategy",
                    "value": message.strategy or "N/A",
                    "inline": True
                }
            ],
            "timestamp": message.created_at.isoformat(),
            "footer": {
                "text": "Wagehood Trading Signals"
            }
        }
        
        return {
            "content": f"{signal_emoji} **{message.symbol}** signal detected",
            "embeds": [embed],
            "username": "Wagehood Signals"
        }
    
    @staticmethod
    def format_service_message(message: NotificationMessage) -> Dict[str, Any]:
        """
        Format service notification for Discord.
        
        Args:
            message: Service notification message
            
        Returns:
            Dictionary with formatted content and embeds
        """
        # Determine emoji and color based on message content
        if "error" in message.content.lower() or "failed" in message.content.lower():
            emoji = "🚨"
            color = 0xff0000  # Red
        elif "warning" in message.content.lower():
            emoji = "⚠️"
            color = 0xffa500  # Orange
        elif "started" in message.content.lower() or "success" in message.content.lower():
            emoji = "✅"
            color = 0x00ff00  # Green
        else:
            emoji = "ℹ️"
            color = 0x0099ff  # Blue
        
        embed = {
            "title": f"{emoji} Service Notification",
            "description": message.content,
            "color": color,
            "timestamp": message.created_at.isoformat(),
            "footer": {
                "text": "Wagehood Infrastructure"
            }
        }
        
        # Add priority field for high-priority messages
        if message.priority == 1:
            embed["fields"] = [
                {
                    "name": "Priority",
                    "value": "🔥 HIGH",
                    "inline": True
                }
            ]
        
        return {
            "content": f"{emoji} Service update",
            "embeds": [embed],
            "username": "Wagehood System"
        }
    
    @staticmethod
    def format_summary_message(message: NotificationMessage) -> Dict[str, Any]:
        """
        Format end-of-day summary notification for Discord.
        
        Args:
            message: Summary notification message
            
        Returns:
            Dictionary with formatted content and embeds
        """
        try:
            # Parse the JSON content which contains the formatted summary data
            summary_data = json.loads(message.content)
            return summary_data
        except json.JSONDecodeError:
            # Fallback for plain text summary
            logger.warning("Failed to parse summary JSON, using plain text fallback")
            
            embed = {
                "title": "📊 Daily Trading Summary",
                "description": message.content,
                "color": 0x0099ff,
                "timestamp": message.created_at.isoformat(),
                "footer": {
                    "text": "Wagehood EOD Summary"
                }
            }
            
            return {
                "content": "📊 **Daily Trading Summary**",
                "embeds": [embed],
                "username": "Wagehood EOD Summary"
            }


class ChannelRouter:
    """
    Routes messages to appropriate Discord channels based on strategy and timeframe.
    """
    
    def __init__(self, channel_configs: Dict[str, ChannelConfig]):
        self.channel_configs = channel_configs
        
    def get_webhook_url(self, channel_name: str) -> Optional[str]:
        """
        Get webhook URL for channel.
        
        Args:
            channel_name: Name of the channel
            
        Returns:
            Webhook URL if channel exists and is enabled, None otherwise
        """
        config = self.channel_configs.get(channel_name)
        if config and config.enabled:
            return config.webhook_url
        return None
    
    def is_channel_enabled(self, channel_name: str) -> bool:
        """
        Check if channel is enabled.
        
        Args:
            channel_name: Name of the channel
            
        Returns:
            True if channel exists and is enabled
        """
        config = self.channel_configs.get(channel_name)
        return config is not None and config.enabled
    
    def validate_message_routing(self, message: NotificationMessage) -> bool:
        """
        Validate that message can be routed to its target channel.
        
        Args:
            message: Notification message
            
        Returns:
            True if message can be routed successfully
        """
        if not self.is_channel_enabled(message.channel):
            logger.warning(f"Channel {message.channel} is not enabled")
            return False
        
        webhook_url = self.get_webhook_url(message.channel)
        if not webhook_url:
            logger.warning(f"No webhook URL configured for channel {message.channel}")
            return False
        
        return True
    
    def update_channel_config(self, channel_name: str, config: ChannelConfig):
        """
        Update channel configuration.
        
        Args:
            channel_name: Name of the channel
            config: New channel configuration
        """
        self.channel_configs[channel_name] = config
        logger.info(f"Updated config for channel {channel_name}")


class DiscordNotificationSender:
    """
    High-level Discord notification sender that combines all components.
    """
    
    def __init__(self, channel_configs: Dict[str, ChannelConfig]):
        self.client = DiscordClient()
        self.router = ChannelRouter(channel_configs)
        self.formatter = NotificationFormatter()
    
    async def send_notification(self, message: NotificationMessage) -> bool:
        """
        Send notification message to Discord.
        
        Args:
            message: Notification message to send
            
        Returns:
            True if sent successfully, False otherwise
        """
        # Validate routing
        if not self.router.validate_message_routing(message):
            return False
        
        # Get webhook URL
        webhook_url = self.router.get_webhook_url(message.channel)
        if not webhook_url:
            logger.error(f"No webhook URL for channel {message.channel}")
            return False
        
        # Format message
        try:
            if message.type.value == "signal":
                formatted = self.formatter.format_signal_message(message)
            elif message.type.value == "summary":
                formatted = self.formatter.format_summary_message(message)
            else:
                formatted = self.formatter.format_service_message(message)
            
            # Send to Discord
            return await self.client.send_message(
                webhook_url=webhook_url,
                content=formatted["content"],
                embeds=formatted.get("embeds"),
                username=formatted.get("username")
            )
            
        except Exception as e:
            logger.error(f"Error formatting or sending message {message.id}: {e}")
            return False
    
    async def test_all_channels(self) -> Dict[str, bool]:
        """
        Test connectivity to all configured channels.
        
        Returns:
            Dictionary mapping channel names to test results
        """
        results = {}
        
        for channel_name, config in self.router.channel_configs.items():
            if config.enabled:
                logger.info(f"Testing channel: {channel_name}")
                results[channel_name] = await self.client.test_webhook(config.webhook_url)
            else:
                results[channel_name] = False
                
        return results
    
    async def close(self):
        """Close the Discord client session."""
        await self.client.close()