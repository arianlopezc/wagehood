"""
Notification Service Module

Provides Discord webhook integration for sharing swing trading signals
with public channels. Supports both single-channel and multi-channel
strategy-specific routing. Uses existing Redis streams and requires no
additional dependencies beyond Python standard library.
"""

from .discord_notifier import DiscordNotifier
from .notification_service import NotificationService
from .message_formatter import MessageFormatter
from .config import NotificationConfig, NotificationRateLimiter

# Multi-channel components
from .multi_channel_config import MultiChannelNotificationConfig, StrategyChannelConfig
from .multi_channel_notifier import MultiChannelDiscordNotifier
from .multi_channel_service import MultiChannelNotificationService

__all__ = [
    # Single-channel components
    'DiscordNotifier',
    'NotificationService', 
    'MessageFormatter',
    'NotificationConfig',
    'NotificationRateLimiter',
    # Multi-channel components
    'MultiChannelNotificationConfig',
    'StrategyChannelConfig', 
    'MultiChannelDiscordNotifier',
    'MultiChannelNotificationService'
]